"""Strategy to IR translator.

Converts vibe-trade Strategy/Card models to StrategyIR for the LEAN runtime.
This replaces string-based code generation with typed data structures.
"""

import logging
from typing import Any

from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.archetypes import parse_archetype


logger = logging.getLogger(__name__)
from .ir import (
    ADX,
    ATR,
    EMA,
    SMA,
    VWAP,
    AllOfCondition,
    AnchoredVWAP,
    AnyOfCondition,
    BandField,
    BollingerBands,
    CompareCondition,
    CompareOp,
    Condition,
    DonchianChannel,
    EntryRule,
    ExitRule,
    ExpressionValue,
    Gap,
    Gate,
    Indicator,
    IndicatorBandValue,
    IndicatorProperty,
    IndicatorPropertyValue,
    IndicatorValue,
    IncrementStateOp,
    KeltnerChannel,
    LiquidateAction,
    LiteralValue,
    Maximum,
    MaxStateOp,
    Minimum,
    NotCondition,
    Overlay,
    Percentile,
    PriceField,
    PriceValue,
    RateOfChange,
    RegimeCondition,
    Resolution,
    RollingMinMax,
    RollingWindow,
    RollingWindowValue,
    SessionHighLow,
    SetHoldingsAction,
    SetStateFromConditionOp,
    SetStateOp,
    StateOp,
    StateType,
    StateValue,
    StateVar,
    StrategyIR,
    TimeValue,
    VolumeSMA,
    VolumeValue,
    VWAPBands,
)
from .ir_validator import validate_ir


class TranslationError(Exception):
    """Raised when strategy translation fails."""

    pass


class IRTranslator:
    """Translates vibe-trade strategies to StrategyIR."""

    # Map comparison operator strings to CompareOp enum
    OP_MAP = {
        "<": CompareOp.LT,
        "<=": CompareOp.LTE,
        ">": CompareOp.GT,
        ">=": CompareOp.GTE,
        "==": CompareOp.EQ,
        "!=": CompareOp.NEQ,
    }

    # Map resolution strings to Resolution enum
    RESOLUTION_MAP = {
        "1m": Resolution.MINUTE,
        "1min": Resolution.MINUTE,
        "minute": Resolution.MINUTE,
        "1h": Resolution.HOUR,
        "hour": Resolution.HOUR,
        "1d": Resolution.DAILY,
        "daily": Resolution.DAILY,
    }

    def __init__(self, strategy: Strategy, cards: dict[str, Card]):
        """Initialize translator.

        Args:
            strategy: The Strategy model with attachments
            cards: Dict mapping card_id to Card objects
        """
        self.strategy = strategy
        self.cards = cards

        # Get first symbol from universe (MVP: single asset)
        self.symbol = strategy.universe[0] if strategy.universe else "BTC-USD"

        # Track indicators to avoid duplicates
        self._indicators: dict[str, Indicator] = {}
        self._state_vars: dict[str, StateVar] = {}
        self._on_bar_hooks: list[StateOp] = []
        self._on_bar_invested_ops: list[StateOp] = []
        self._additional_symbols: list[str] = []
        self._exit_counter = 0
        self._sequence_counter = 0


    def _try_archetype_to_ir(self, archetype: str, slots: dict[str, Any]) -> Condition | None:
        """Try to get IR condition directly from archetype's to_ir() method.

        This is the preferred path for archetypes that implement to_ir().
        Falls back to None if the archetype doesn't support direct IR conversion.

        Args:
            archetype: The archetype type_id (e.g., "entry.trend_pullback")
            slots: The slot values for the archetype

        Returns:
            Condition if the archetype supports to_ir() and produced a valid condition,
            None otherwise (caller should use legacy translation path).
        """
        result = self._translate_archetype_full(archetype, slots)
        return result[0] if result else None

    def _translate_archetype_full(
        self, archetype: str, slots: dict[str, Any]
    ) -> tuple[Condition, list[StateOp], list[StateOp], list[StateOp]] | None:
        """Translate archetype using uniform interface: condition + state + hooks.

        This is the preferred translation path. It calls:
        - to_ir() for the condition
        - get_state_vars() for state declarations
        - get_on_fill_ops() for fill operations
        - get_on_bar_ops() for per-bar operations
        - get_on_bar_invested_ops() for invested-state operations

        Args:
            archetype: The archetype type_id (e.g., "entry.trend_pullback")
            slots: The slot values for the archetype

        Returns:
            Tuple of (condition, on_fill_ops, on_bar_ops, on_bar_invested_ops),
            or None if the archetype doesn't support to_ir().
        """
        from pydantic import TypeAdapter

        try:
            # Parse the archetype into a typed model
            typed_archetype = parse_archetype(archetype, slots)

            # Try to get IR directly
            ir_condition = typed_archetype.to_ir()
            if ir_condition is None:
                # Archetype doesn't support to_ir() yet
                return None

            # Convert condition to dict and process inline indicators
            condition_dict = ir_condition.model_dump()
            self._process_inline_indicators(condition_dict)

            # Use TypeAdapter to validate the union type
            adapter = TypeAdapter(Condition)
            local_condition = adapter.validate_python(condition_dict)

            # Collect state declarations from archetype
            for state_var in typed_archetype.get_state_vars():
                var_type = {
                    "float": StateType.FLOAT,
                    "int": StateType.INT,
                    "bool": StateType.BOOL,
                }.get(state_var.var_type, StateType.FLOAT)
                self._add_state(state_var.id, var_type, state_var.default)

            # Collect hooks from archetype and convert to local types
            on_fill_ops = self._convert_state_ops(typed_archetype.get_on_fill_ops())
            on_bar_ops = self._convert_state_ops(typed_archetype.get_on_bar_ops())
            on_bar_invested_ops = self._convert_state_ops(typed_archetype.get_on_bar_invested_ops())

            # Register global hooks
            self._on_bar_hooks.extend(on_bar_ops)
            self._on_bar_invested_ops.extend(on_bar_invested_ops)

            logger.debug(f"Successfully used to_ir() for {archetype}")
            return (local_condition, on_fill_ops, on_bar_ops, on_bar_invested_ops)

        except Exception as e:
            # to_ir() failed - fall back to legacy condition translation
            logger.debug(f"to_ir() not available for {archetype}: {e}")
            return None

    def _convert_state_ops(self, shared_ops: list[Any]) -> list[StateOp]:
        """Convert shared library StateOps to local IR StateOps.

        The shared library uses different class names (SetStateAction vs SetStateOp),
        but the serialized form is compatible. Convert via dict round-trip.
        """
        from pydantic import TypeAdapter

        if not shared_ops:
            return []

        result = []
        adapter = TypeAdapter(StateOp)
        for op in shared_ops:
            op_dict = op.model_dump()
            # Process any inline indicators in the value
            if "value" in op_dict:
                self._process_inline_indicators(op_dict["value"])
            result.append(adapter.validate_python(op_dict))
        return result

    def _process_inline_indicators(self, obj: dict[str, Any]) -> None:
        """Process a condition dict, converting inline IndicatorRefs to indicator_id references.

        This walks the dict tree and:
        1. Finds inline IndicatorRef patterns (type="indicator" with indicator_type + params)
        2. Creates corresponding Indicator objects and registers them
        3. Mutates the dict to use indicator_id references instead
        4. Removes shared library-specific fields that the local IR doesn't support
        5. Converts RuntimeCondition.indicators_required from list[IndicatorRef] to list[str]
        """
        if not isinstance(obj, dict):
            return

        # Remove semantic_label field (shared library debug field not in local IR)
        if "semantic_label" in obj:
            del obj["semantic_label"]

        # Check for inline IndicatorRef pattern
        if obj.get("type") == "indicator" and obj.get("indicator_type") is not None:
            ind_type = obj["indicator_type"]
            params = obj.get("params") or {}
            field = obj.get("field", "value")

            # Generate indicator ID
            ind_id = self._generate_indicator_id(ind_type, params)

            # Create and register the indicator
            indicator = self._create_indicator_from_spec(ind_type, ind_id, params)
            if indicator:
                self._add_indicator(indicator)

            # Handle band field specially - convert to indicator_band type
            if field in ("upper", "middle", "lower"):
                # Convert to indicator_band reference
                obj.clear()
                obj["type"] = "indicator_band"
                obj["indicator_id"] = ind_id
                obj["band"] = field
            else:
                # Regular indicator reference
                obj.clear()
                obj["type"] = "indicator"
                obj["indicator_id"] = ind_id
        else:
            # Handle RegimeCondition - create required indicators based on metric
            if obj.get("type") == "regime":
                metric = obj.get("metric")
                if metric == "trend_ma_relation":
                    # Regime needs EMA indicators
                    fast = obj.get("ma_fast") or 20
                    slow = obj.get("ma_slow") or 50
                    self._add_indicator(EMA(id=f"ema_{fast}", period=fast))
                    self._add_indicator(EMA(id=f"ema_{slow}", period=slow))
                elif metric == "ret_pct":
                    # Regime needs ROC indicator
                    lookback = obj.get("lookback_bars") or 1
                    self._add_indicator(RateOfChange(id=f"roc_{lookback}", period=lookback))
                elif metric == "vol_atr_pct":
                    # Regime needs ATR indicator
                    lookback = obj.get("lookback_bars") or 14
                    self._add_indicator(ATR(id=f"atr_{lookback}", period=lookback))
                elif metric == "vol_bb_width_pctile":
                    # Regime needs BB width percentile
                    length = obj.get("lookback_bars") or 20
                    self._add_indicator(
                        BollingerBands(id=f"bb_{length}", period=length, num_std_dev=2.0)
                    )
                elif metric == "trend_adx":
                    # Regime needs ADX indicator
                    lookback = obj.get("lookback_bars") or 14
                    self._add_indicator(ADX(id=f"adx_{lookback}", period=lookback))
                elif metric == "volume_pctile":
                    # Volume percentile needs volume SMA for comparison
                    lookback = obj.get("lookback_bars") or 20
                    self._add_indicator(VolumeSMA(id=f"vol_sma_{lookback}", period=lookback))
                elif metric == "liquidity_sweep":
                    # Liquidity sweep needs rolling min/max for level detection
                    lookback = obj.get("lookback_bars") or 20
                    self._add_indicator(
                        RollingMinMax(id=f"rmm_low_{lookback}", period=lookback, mode="min")
                    )
                elif metric == "dist_from_vwap_pct":
                    # Regime needs VWAP indicator
                    self._add_indicator(VWAP(id="vwap"))
                elif metric == "trend_regime":
                    # Alias for trend_ma_relation - uses EMA comparison
                    fast = obj.get("ma_fast") or 20
                    slow = obj.get("ma_slow") or 50
                    self._add_indicator(EMA(id=f"ema_{fast}", period=fast))
                    self._add_indicator(EMA(id=f"ema_{slow}", period=slow))
                elif metric == "vol_regime":
                    # Volatility regime uses BB width
                    length = obj.get("lookback_bars") or 20
                    self._add_indicator(
                        BollingerBands(id=f"bb_{length}", period=length, num_std_dev=2.0)
                    )
                elif metric == "bb_width_pctile":
                    # Alias for vol_bb_width_pctile
                    length = obj.get("lookback_bars") or 20
                    self._add_indicator(
                        BollingerBands(id=f"bb_{length}", period=length, num_std_dev=2.0)
                    )
                elif metric == "risk_event_prob":
                    # Risk event probability requires external calendar data
                    # No indicators needed - evaluator will pass through without calendar
                    pass
                elif metric == "pennant_pattern":
                    # Pennant pattern needs momentum and consolidation indicators
                    momentum_bars = obj.get("pennant_momentum_bars") or 5
                    consolidation_bars = obj.get("pennant_consolidation_bars") or 10
                    self._add_indicator(RateOfChange(id="momentum_roc", period=momentum_bars))
                    self._add_indicator(ATR(id="pattern_atr", period=consolidation_bars))
                    self._add_indicator(Maximum(id="pattern_max", period=consolidation_bars))
                    self._add_indicator(Minimum(id="pattern_min", period=consolidation_bars))
                elif metric == "flag_pattern":
                    # Flag pattern needs similar indicators to pennant
                    momentum_bars = obj.get("flag_momentum_bars") or 5
                    consolidation_bars = obj.get("flag_consolidation_bars") or 10
                    self._add_indicator(RateOfChange(id="momentum_roc", period=momentum_bars))
                    self._add_indicator(ATR(id="pattern_atr", period=consolidation_bars))
                    self._add_indicator(Maximum(id="pattern_max", period=consolidation_bars))
                    self._add_indicator(Minimum(id="pattern_min", period=consolidation_bars))
                elif metric == "session_phase":
                    # Session phase uses TimeValue - no indicators needed
                    pass
                elif metric == "volume_spike":
                    # Volume spike needs volume SMA for comparison
                    lookback = obj.get("lookback_bars") or 20
                    self._add_indicator(VolumeSMA(id=f"vol_sma_{lookback}", period=lookback))
                elif metric == "volume_dip":
                    # Volume dip needs volume SMA for comparison
                    lookback = obj.get("lookback_bars") or 20
                    self._add_indicator(VolumeSMA(id=f"vol_sma_{lookback}", period=lookback))
                elif metric == "price_level_touch":
                    # Price level touch needs rolling min/max for dynamic levels
                    lookback = obj.get("lookback_bars") or 20
                    self._add_indicator(
                        RollingMinMax(id=f"rmm_low_{lookback}", period=lookback, mode="min")
                    )
                    self._add_indicator(
                        RollingMinMax(id=f"rmm_high_{lookback}", period=lookback, mode="max")
                    )
                elif metric == "price_level_cross":
                    # Price level cross needs rolling min/max for dynamic levels
                    lookback = obj.get("lookback_bars") or 20
                    self._add_indicator(
                        RollingMinMax(id=f"rmm_low_{lookback}", period=lookback, mode="min")
                    )
                    self._add_indicator(
                        RollingMinMax(id=f"rmm_high_{lookback}", period=lookback, mode="max")
                    )
                elif metric == "gap_pct":
                    # Gap percentage needs rolling window to access previous close
                    self._add_indicator(RollingWindow(id="prev_close_rw", period=2))

            # Handle RuntimeCondition - create indicators based on condition type
            elif obj.get("type") == "runtime":
                cond_type = obj.get("condition_type") or ""
                params = obj.get("params") or {}

                # Band events need band indicators
                if cond_type.startswith("band_"):
                    band_type = params.get("band_type") or "bollinger"
                    length = params.get("length") or 20
                    mult = params.get("mult") or 2.0

                    if band_type == "bollinger":
                        self._add_indicator(
                            BollingerBands(id=f"bb_{length}", period=length, num_std_dev=mult)
                        )
                    elif band_type == "keltner":
                        self._add_indicator(
                            KeltnerChannel(id=f"kc_{length}", period=length, atr_mult=mult)
                        )
                    elif band_type == "donchian":
                        self._add_indicator(DonchianChannel(id=f"dc_{length}", period=length))
                    elif band_type == "vwap_band":
                        # VWAP band uses anchored VWAP
                        anchor = params.get("anchor") or "session"
                        self._add_indicator(
                            AnchoredVWAP(id="avwap", anchor=anchor, anchor_datetime=None)
                        )

                # Breakout needs Donchian channel
                elif cond_type == "breakout":
                    lookback = params.get("lookback_bars") or 50
                    self._add_indicator(DonchianChannel(id=f"dc_{lookback}", period=lookback))

                # Intermarket trigger needs additional symbol subscription
                elif cond_type == "intermarket_trigger":
                    leader_symbol = params.get("leader_symbol")
                    if leader_symbol and leader_symbol not in self._additional_symbols:
                        self._additional_symbols.append(leader_symbol)

                # Handle indicators_required list
                if "indicators_required" in obj:
                    indicator_ids = []
                    for ind_ref in obj["indicators_required"]:
                        if isinstance(ind_ref, dict) and ind_ref.get("indicator_type"):
                            ind_type = ind_ref["indicator_type"]
                            ind_params = ind_ref.get("params") or {}
                            ind_id = self._generate_indicator_id(ind_type, ind_params)

                            # Create and register the indicator
                            indicator = self._create_indicator_from_spec(ind_type, ind_id, ind_params)
                            if indicator:
                                self._add_indicator(indicator)

                            indicator_ids.append(ind_id)
                        elif isinstance(ind_ref, str):
                            # Already a string ID
                            indicator_ids.append(ind_ref)
                    obj["indicators_required"] = indicator_ids

            # Recurse into nested dicts and lists
            for value in list(obj.values()):  # Use list() to avoid mutation during iteration
                if isinstance(value, dict):
                    self._process_inline_indicators(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._process_inline_indicators(item)

    def _generate_indicator_id(self, ind_type: str, params: dict[str, Any]) -> str:
        """Generate a deterministic indicator ID from type and params."""
        parts = [ind_type.lower()]
        for key in sorted(params.keys()):
            val = params[key]
            if isinstance(val, float):
                val_str = str(val).replace(".", "_")
            else:
                val_str = str(val)
            parts.append(val_str)
        return "_".join(parts)

    def _create_indicator_from_spec(
        self, ind_type: str, ind_id: str, params: dict[str, Any]
    ) -> Indicator | None:
        """Create a typed Indicator from a type string and params."""
        type_map = {
            "EMA": lambda: EMA(id=ind_id, period=params.get("period", 20)),
            "SMA": lambda: SMA(id=ind_id, period=params.get("period", 20)),
            "BB": lambda: BollingerBands(
                id=ind_id,
                period=params.get("period", 20),
                multiplier=params.get("multiplier", 2.0),
            ),
            "KC": lambda: KeltnerChannel(
                id=ind_id,
                period=params.get("period", 20),
                multiplier=params.get("multiplier", 2.0),
            ),
            "DC": lambda: DonchianChannel(id=ind_id, period=params.get("period", 20)),
            "ATR": lambda: ATR(id=ind_id, period=params.get("period", 14)),
            "MAX": lambda: Maximum(id=ind_id, period=params.get("period", 50)),
            "MIN": lambda: Minimum(id=ind_id, period=params.get("period", 50)),
            "ROC": lambda: RateOfChange(id=ind_id, period=params.get("period", 14)),
            "ADX": lambda: ADX(id=ind_id, period=params.get("period", 14)),
            "VWAP": lambda: VWAP(id=ind_id, period=params.get("period", 0)),
            "AVWAP": lambda: AnchoredVWAP(
                id=ind_id,
                anchor=params.get("anchor", "session"),
                anchor_datetime=params.get("anchor_datetime"),
            ),
            "PCTILE": lambda: Percentile(
                id=ind_id,
                period=params.get("period", 100),
                percentile=params.get("percentile", 10.0),
                source=params.get("source", "close"),
            ),
            "GAP": lambda: Gap(
                id=ind_id,
                session=params.get("session", "us"),
            ),
        }

        factory = type_map.get(ind_type.upper())
        if factory:
            return factory()

        logger.warning(f"Unknown indicator type: {ind_type}")
        return None

    def translate(self) -> StrategyIR:
        """Translate the strategy to IR.

        Returns:
            StrategyIR - valid, executable strategy IR

        Raises:
            TranslationError: If translation fails for any reason
        """
        entry: EntryRule | None = None
        exits: list[ExitRule] = []
        gates: list[Gate] = []
        overlays: list[Overlay] = []

        # Process attachments by role
        for attachment in self.strategy.attachments:
            if not attachment.enabled:
                continue

            card = self.cards.get(attachment.card_id)
            if not card:
                raise TranslationError(f"Card not found: {attachment.card_id}")

            # Merge card slots with attachment overrides
            slots = {**card.slots, **attachment.overrides}

            # Dispatch based on role and card type
            if attachment.role == "entry":
                translated = self._translate_entry(card.type, slots)
                if translated:
                    entry = translated
            elif attachment.role == "exit":
                translated = self._translate_exit(card.type, slots)
                if translated:
                    exits.append(translated)
            elif attachment.role == "gate":
                translated = self._translate_gate(card.type, slots)
                if translated:
                    gates.append(translated)
            elif attachment.role == "overlay":
                translated = self._translate_overlay(card.type, slots)
                if translated:
                    overlays.append(translated)

        # Determine resolution from first entry card's context
        resolution = Resolution.HOUR
        for attachment in self.strategy.attachments:
            if attachment.role == "entry" and attachment.enabled:
                card = self.cards.get(attachment.card_id)
                if card:
                    context = card.slots.get("context", {})
                    tf = context.get("tf", "1h")
                    resolution = self.RESOLUTION_MAP.get(tf.lower(), Resolution.HOUR)
                    break

        ir = StrategyIR(
            strategy_id=self.strategy.id,
            strategy_name=self.strategy.name,
            symbol=self.symbol,
            resolution=resolution,
            additional_symbols=self._additional_symbols,
            indicators=list(self._indicators.values()),
            state=list(self._state_vars.values()),
            gates=gates,
            overlays=overlays,
            entry=entry,
            exits=exits,
            on_bar=self._on_bar_hooks,
            on_bar_invested=self._on_bar_invested_ops,
        )

        # Validate IR for referential integrity - fail fast if invalid
        validation_result = validate_ir(ir)
        if not validation_result.is_valid:
            errors = "; ".join(f"{e.path}: {e.message}" for e in validation_result.errors)
            raise TranslationError(f"Invalid IR produced: {errors}")

        return ir

    # =========================================================================
    # Entry Translation
    # =========================================================================

    def _translate_entry(self, archetype: str, slots: dict[str, Any]) -> EntryRule | None:
        """Translate an entry card to EntryRule.

        Translation paths:
        1. to_ir() + state methods - uniform archetype interface (preferred)
        2. Legacy condition translation - for arbitrary conditions via rule_trigger

        The uniform interface collects condition, state vars, and hooks from
        the archetype's methods. EntryArchetype base class provides standard
        state (entry_price, bars_since_entry) by default.
        """
        # 1. Try uniform archetype interface first (preferred path)
        result = self._translate_archetype_full(archetype, slots)
        if result is not None:
            condition, on_fill_ops, _, _ = result  # on_bar ops already registered
            action = slots.get("action", {})
            direction = action.get("direction", "long")
            allocation = 0.95 if direction == "long" else -0.95

            return EntryRule(
                condition=condition,
                action=SetHoldingsAction(allocation=allocation),
                on_fill=on_fill_ops,  # From archetype.get_on_fill_ops()
            )

        raise TranslationError(f"Unsupported entry archetype: {archetype}")

    # =========================================================================
    # Exit Translation
    # =========================================================================

    def _translate_exit(self, archetype: str, slots: dict[str, Any]) -> ExitRule | None:
        """Translate an exit card to ExitRule.

        Uses the uniform archetype interface: to_ir() + state methods.
        TrailingStop uses this to declare its state (highest_since_entry)
        and hooks (MaxStateOp on each bar).
        """
        result = self._translate_archetype_full(archetype, slots)
        if result is not None:
            condition, _, _, _ = result  # on_bar ops already registered; exits don't use on_fill
            self._exit_counter += 1
            return ExitRule(
                id=f"exit_{self._exit_counter}",
                condition=condition,
                action=LiquidateAction(),
                priority=self._exit_counter,
            )

        raise TranslationError(f"Unsupported exit archetype: {archetype}")

    # =========================================================================
    # Gate Translation
    # =========================================================================

    def _translate_gate(self, archetype: str, slots: dict[str, Any]) -> Gate | None:
        """Translate a gate card.

        Translation priority:
        1. Try archetype's to_ir() method (preferred path)
        2. Fall back to primitive handlers
        """
        # 1. Try to_ir() first - this is the preferred path
        ir_condition = self._try_archetype_to_ir(archetype, slots)
        if ir_condition is not None:
            action = slots.get("action", {})
            mode = action.get("mode", "allow")
            target_roles = action.get("target_roles", ["entry"])

            return Gate(
                id=f"gate_{len(self._indicators)}",
                condition=ir_condition,
                mode=mode,
                target_roles=target_roles,
            )

        raise TranslationError(f"Unsupported gate archetype: {archetype}")

    # =========================================================================
    # Overlay Translation
    # =========================================================================

    def _translate_overlay(self, archetype: str, slots: dict[str, Any]) -> Overlay | None:
        """Translate an overlay card.

        Translation priority:
        1. Try archetype's to_ir() method (preferred path)
        2. Fall back raises error (no legacy handlers)
        """
        # 1. Try to_ir() - this is the preferred path
        ir_condition = self._try_archetype_to_ir(archetype, slots)
        if ir_condition is not None:
            action = slots.get("action", {})

            # Extract scaling factors with defaults
            scale_risk_frac = action.get("scale_risk_frac", 1.0)
            scale_size_frac = action.get("scale_size_frac", 1.0)

            # Extract targeting options
            target_roles = action.get("target_roles", ["entry", "exit"])
            target_tags = action.get("target_tags", [])
            target_ids = action.get("target_ids", [])

            return Overlay(
                id=f"overlay_{len(self._indicators)}",
                condition=ir_condition,
                scale_risk_frac=scale_risk_frac,
                scale_size_frac=scale_size_frac,
                target_roles=target_roles,
                target_tags=target_tags,
                target_ids=target_ids,
            )

        raise TranslationError(f"Unsupported overlay archetype: {archetype}")

    # =========================================================================
    # Helpers
    # =========================================================================

    def _add_indicator(self, indicator: Indicator) -> None:
        """Add an indicator, avoiding duplicates."""
        if indicator.id not in self._indicators:
            self._indicators[indicator.id] = indicator

    def _add_state(self, state_id: str, var_type: StateType, default: Any) -> None:
        """Add a state variable, avoiding duplicates."""
        if state_id not in self._state_vars:
            self._state_vars[state_id] = StateVar(
                id=state_id,
                var_type=var_type,
                default=default,
            )

    def _invert_op(self, op: CompareOp) -> CompareOp:
        """Invert a comparison operator."""
        inversions = {
            CompareOp.GT: CompareOp.LT,
            CompareOp.LT: CompareOp.GT,
            CompareOp.GTE: CompareOp.LTE,
            CompareOp.LTE: CompareOp.GTE,
            CompareOp.EQ: CompareOp.NEQ,
            CompareOp.NEQ: CompareOp.EQ,
        }
        return inversions.get(op, op)
