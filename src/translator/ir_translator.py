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
            # to_ir() not supported or failed - fall back to legacy path
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
            # Handle RuntimeCondition.indicators_required specially - convert list[IndicatorRef] to list[str]
            if obj.get("type") == "runtime" and "indicators_required" in obj:
                indicator_ids = []
                for ind_ref in obj["indicators_required"]:
                    if isinstance(ind_ref, dict) and ind_ref.get("indicator_type"):
                        ind_type = ind_ref["indicator_type"]
                        params = ind_ref.get("params") or {}
                        ind_id = self._generate_indicator_id(ind_type, params)

                        # Create and register the indicator
                        indicator = self._create_indicator_from_spec(ind_type, ind_id, params)
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
        2. Primitive handler - entry.rule_trigger (wraps arbitrary conditions)
        3. Custom handler - archetypes requiring special logic (e.g., intermarket)

        The uniform interface collects condition, state vars, and hooks from
        the archetype's methods. EntryArchetype base class provides standard
        state (entry_price, bars_since_entry) by default.
        """
        # Handlers for specific archetypes that don't use to_ir()
        handlers = {
            "entry.rule_trigger": self._entry_rule_trigger,
            "entry.intermarket_trigger": self._entry_intermarket_trigger,
        }

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

        # 2. Try handler for this archetype
        handler = handlers.get(archetype)
        if handler:
            return handler(slots)

        raise TranslationError(f"Unsupported entry archetype: {archetype}")

    def _entry_rule_trigger(self, slots: dict[str, Any]) -> EntryRule | None:
        """Translate entry.rule_trigger archetype."""
        event = slots.get("event", {})
        action = slots.get("action", {})

        condition_spec = event.get("condition", {})
        direction = action.get("direction", "long")

        condition = self._translate_condition(condition_spec)
        if condition is None:
            raise TranslationError("Could not translate condition for entry.rule_trigger")

        # Build action based on direction
        allocation = 0.95 if direction == "long" else -0.95

        # Add state tracking for entry
        self._add_state("entry_price", StateType.FLOAT, None)
        self._add_state("bars_since_entry", StateType.INT, 0)

        return EntryRule(
            condition=condition,
            action=SetHoldingsAction(allocation=allocation),
            on_fill=[
                SetStateOp(state_id="entry_price", value=PriceValue(field=PriceField.CLOSE)),
                SetStateOp(state_id="bars_since_entry", value=LiteralValue(value=0)),
            ],
        )

    def _entry_intermarket_trigger(self, slots: dict[str, Any]) -> EntryRule | None:
        """Translate entry.intermarket_trigger archetype.

        Enter follower symbol based on leader symbol's movement.
        """
        event = slots.get("event", {})
        action = slots.get("action", {})

        lead_follow = event.get("lead_follow", {})
        if not lead_follow:
            raise TranslationError("entry.intermarket_trigger requires event.lead_follow")

        # Pass to intermarket condition translator
        condition = self._translate_intermarket(lead_follow)
        if condition is None:
            raise TranslationError("Could not translate intermarket condition")

        # Determine direction from entry_side_map or action
        direction = action.get("direction", "auto")
        if direction == "auto":
            entry_side_map = lead_follow.get("entry_side_map", {})
            # Default to long if leader up
            direction = entry_side_map.get("leader_up", "long")

        allocation = 0.95 if direction == "long" else -0.95

        self._add_state("entry_price", StateType.FLOAT, None)

        return EntryRule(
            condition=condition,
            action=SetHoldingsAction(allocation=allocation),
            on_fill=[
                SetStateOp(state_id="entry_price", value=PriceValue(field=PriceField.CLOSE)),
            ],
        )

    # =========================================================================
    # Exit Translation
    # =========================================================================

    def _translate_exit(self, archetype: str, slots: dict[str, Any]) -> ExitRule | None:
        """Translate an exit card to ExitRule.

        Translation paths:
        1. to_ir() + state methods - uniform archetype interface (preferred)
        2. Primitive handler - exit.rule_trigger (wraps arbitrary conditions)

        The uniform interface collects condition, state vars, and hooks from
        the archetype's methods. TrailingStop uses this to declare its state
        (highest_since_entry) and hooks (MaxStateOp on each bar).
        """
        # Handlers for archetypes that don't use to_ir()
        handlers = {
            "exit.rule_trigger": self._exit_rule_trigger,
        }

        # 1. Try uniform archetype interface first (preferred path)
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

        # 2. Try handler for this archetype
        handler = handlers.get(archetype)
        if handler:
            return handler(slots)

        raise TranslationError(f"Unsupported exit archetype: {archetype}")

    def _exit_rule_trigger(self, slots: dict[str, Any]) -> ExitRule | None:
        """Translate exit.rule_trigger archetype."""
        event = slots.get("event", {})
        risk = slots.get("risk", {})

        condition_spec = event.get("condition", {})
        condition = self._translate_condition(condition_spec)

        if condition is None:
            # Fall back to risk-based exit
            sl_pct = risk.get("sl_pct")
            tp_pct = risk.get("tp_pct")

            if sl_pct and tp_pct:
                raise TranslationError("Percentage-based TP/SL not yet implemented in IR")
            else:
                raise TranslationError("Could not translate condition for exit.rule_trigger")

        self._exit_counter += 1

        return ExitRule(
            id=f"exit_{self._exit_counter}",
            condition=condition,
            action=LiquidateAction(),
            priority=self._exit_counter,
        )

    # NOTE: _exit_trailing_stop handler removed - now uses uniform archetype interface
    # TrailingStop.to_ir() returns the condition, get_state_vars() and
    # get_on_bar_invested_ops() provide state management.

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
    # Condition Translation
    # =========================================================================

    def _translate_condition(self, spec: dict[str, Any]) -> Condition | None:
        """Translate a ConditionSpec to IR Condition."""
        if not spec:
            return None

        cond_type = spec.get("type", "regime")

        if cond_type == "regime":
            return self._translate_regime_condition(spec.get("regime", spec))
        elif cond_type == "composite":
            # Handle composite wrapper: {"type": "composite", "composite": {"op": "allOf", "conditions": [...]}}
            composite_spec = spec.get("composite", {})
            composite_op = composite_spec.get("op", "allOf")
            conditions = composite_spec.get("conditions", [])
            translated = [self._translate_condition(c) for c in conditions]
            translated = [c for c in translated if c is not None]
            if not translated:
                return None
            if composite_op == "allOf":
                return AllOfCondition(conditions=translated)
            elif composite_op == "anyOf":
                return AnyOfCondition(conditions=translated)
            else:
                # Default to allOf
                return AllOfCondition(conditions=translated)
        elif cond_type == "allOf":
            conditions = [self._translate_condition(c) for c in spec.get("allOf", [])]
            conditions = [c for c in conditions if c is not None]
            if not conditions:
                return None
            return AllOfCondition(conditions=conditions)
        elif cond_type == "anyOf":
            conditions = [self._translate_condition(c) for c in spec.get("anyOf", [])]
            conditions = [c for c in conditions if c is not None]
            if not conditions:
                return None
            return AnyOfCondition(conditions=conditions)
        elif cond_type == "not":
            inner = self._translate_condition(spec.get("not", {}))
            if inner is None:
                return None
            return NotCondition(condition=inner)
        elif cond_type == "band_event":
            return self._translate_band_event(spec.get("band_event", spec))
        elif cond_type == "sequence":
            return self._translate_sequence(spec.get("sequence", []))
        elif cond_type == "breakout":
            return self._translate_breakout(spec.get("breakout", spec))
        elif cond_type == "squeeze":
            return self._translate_squeeze(spec.get("squeeze", spec))
        elif cond_type == "time_filter":
            return self._translate_time_filter(spec.get("time_filter", spec))
        elif cond_type == "intermarket":
            return self._translate_intermarket(spec.get("intermarket", spec))
        else:
            # Try as regime directly
            return self._translate_regime_condition(spec)

    def _translate_regime_condition(self, regime: dict[str, Any]) -> Condition | None:
        """Translate a RegimeSpec to IR Condition.

        All supported metrics are lowered to CompareCondition with appropriate indicators.
        Unsupported metrics raise a warning and return None (condition passes).
        """
        metric = regime.get("metric")
        op_str = regime.get("op", "==")
        value = regime.get("value")

        op = self.OP_MAP.get(op_str, CompareOp.EQ)

        # =================================================================
        # Return / Momentum Metrics
        # =================================================================
        if metric == "ret_pct":
            # Return percentage - use ROC indicator
            # ROC returns decimal (0.05 = 5%), but schema uses percentage (-2.0 = -2%)
            lookback = regime.get("lookback_bars", 1)
            roc_id = f"roc_{lookback}"
            self._add_indicator(RateOfChange(id=roc_id, period=lookback))
            # Multiply ROC by 100 to convert to percentage
            return CompareCondition(
                left=ExpressionValue(
                    op="*",
                    left=IndicatorValue(indicator_id=roc_id),
                    right=LiteralValue(value=100.0),
                ),
                op=op,
                right=LiteralValue(value=float(value)),
            )

        # =================================================================
        # Trend Metrics
        # =================================================================
        elif metric == "trend_ma_relation":
            # MA relation: compare fast vs slow
            fast = regime.get("ma_fast", 20)
            slow = regime.get("ma_slow", 50)
            fast_id = f"ema_{fast}"
            slow_id = f"ema_{slow}"
            self._add_indicator(EMA(id=fast_id, period=fast))
            self._add_indicator(EMA(id=slow_id, period=slow))

            # For "EMA fast > EMA slow", value=0 and op=">"
            if value == 0:
                return CompareCondition(
                    left=IndicatorValue(indicator_id=fast_id),
                    op=op,
                    right=IndicatorValue(indicator_id=slow_id),
                )
            else:
                # Compare difference to threshold
                return CompareCondition(
                    left=ExpressionValue(
                        op="-",
                        left=IndicatorValue(indicator_id=fast_id),
                        right=IndicatorValue(indicator_id=slow_id),
                    ),
                    op=op,
                    right=LiteralValue(value=float(value)),
                )

        elif metric == "trend_regime":
            # Trend regime classification: "up", "down", or numeric
            fast = regime.get("ma_fast", 20)
            slow = regime.get("ma_slow", 50)
            fast_id = f"ema_{fast}"
            slow_id = f"ema_{slow}"
            self._add_indicator(EMA(id=fast_id, period=fast))
            self._add_indicator(EMA(id=slow_id, period=slow))

            if value == "up" or (isinstance(value, (int, float)) and value > 0):
                return CompareCondition(
                    left=IndicatorValue(indicator_id=fast_id),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id=slow_id),
                )
            elif value == "down" or (isinstance(value, (int, float)) and value < 0):
                return CompareCondition(
                    left=IndicatorValue(indicator_id=fast_id),
                    op=CompareOp.LT,
                    right=IndicatorValue(indicator_id=slow_id),
                )
            elif isinstance(value, (int, float)) and value == 0:
                if op in (CompareOp.GT, CompareOp.GTE):
                    return CompareCondition(
                        left=IndicatorValue(indicator_id=fast_id),
                        op=CompareOp.GT,
                        right=IndicatorValue(indicator_id=slow_id),
                    )
                else:
                    return CompareCondition(
                        left=IndicatorValue(indicator_id=fast_id),
                        op=CompareOp.LT,
                        right=IndicatorValue(indicator_id=slow_id),
                    )
            else:
                return None  # Neutral - always passes

        elif metric == "trend_adx":
            # ADX trend strength (schema uses trend_adx, not adx_strength)
            period = regime.get("period", 14)
            self._add_indicator(ADX(id="adx", period=period))
            return CompareCondition(
                left=IndicatorValue(indicator_id="adx"),
                op=op,
                right=LiteralValue(value=float(value)),
            )

        # =================================================================
        # Volatility Metrics
        # =================================================================
        elif metric == "vol_bb_width_pctile":
            # Bollinger Band width as percentile (approximated)
            # BB width = (upper - lower) / middle
            # We compare width to threshold, not true percentile (would need rolling rank)
            period = regime.get("lookback_bars", 20)
            mult = regime.get("mult", 2.0)
            self._add_indicator(BollingerBands(id="bb", period=period, multiplier=mult))
            # BandWidth = (Upper - Lower) / Middle * 100
            return CompareCondition(
                left=IndicatorPropertyValue(
                    indicator_id="bb", property=IndicatorProperty.BAND_WIDTH
                ),
                op=op,
                right=LiteralValue(value=float(value)),
            )

        elif metric == "vol_atr_pct":
            # ATR as percentage of price
            period = regime.get("period", 14)
            self._add_indicator(ATR(id="atr", period=period))
            # ATR% = ATR / Close * 100
            return CompareCondition(
                left=ExpressionValue(
                    op="*",
                    left=ExpressionValue(
                        op="/",
                        left=IndicatorValue(indicator_id="atr"),
                        right=PriceValue(field=PriceField.CLOSE),
                    ),
                    right=LiteralValue(value=100.0),
                ),
                op=op,
                right=LiteralValue(value=float(value)),
            )

        elif metric == "vol_regime":
            # Volatility regime: "quiet", "normal", "high"
            # Based on BB width percentile thresholds
            period = regime.get("lookback_bars", 20)
            self._add_indicator(BollingerBands(id="bb", period=period))
            low_thresh = regime.get("vol_threshold_low", 25)
            high_thresh = regime.get("vol_threshold_high", 75)

            if value == "quiet":
                # BB width below low threshold
                return CompareCondition(
                    left=IndicatorPropertyValue(
                        indicator_id="bb", property=IndicatorProperty.BAND_WIDTH
                    ),
                    op=CompareOp.LT,
                    right=LiteralValue(value=float(low_thresh)),
                )
            elif value == "high":
                # BB width above high threshold
                return CompareCondition(
                    left=IndicatorPropertyValue(
                        indicator_id="bb", property=IndicatorProperty.BAND_WIDTH
                    ),
                    op=CompareOp.GT,
                    right=LiteralValue(value=float(high_thresh)),
                )
            else:  # "normal"
                # BB width between thresholds
                return AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=IndicatorPropertyValue(
                                indicator_id="bb", property=IndicatorProperty.BAND_WIDTH
                            ),
                            op=CompareOp.GTE,
                            right=LiteralValue(value=float(low_thresh)),
                        ),
                        CompareCondition(
                            left=IndicatorPropertyValue(
                                indicator_id="bb", property=IndicatorProperty.BAND_WIDTH
                            ),
                            op=CompareOp.LTE,
                            right=LiteralValue(value=float(high_thresh)),
                        ),
                    ]
                )

        # =================================================================
        # VWAP Metrics
        # =================================================================
        elif metric == "dist_from_vwap_pct":
            # Distance from VWAP as percentage: (Close - VWAP) / VWAP * 100
            self._add_indicator(VWAP(id="vwap", period=0))  # Intraday VWAP
            return CompareCondition(
                left=ExpressionValue(
                    op="*",
                    left=ExpressionValue(
                        op="/",
                        left=ExpressionValue(
                            op="-",
                            left=PriceValue(field=PriceField.CLOSE),
                            right=IndicatorValue(indicator_id="vwap"),
                        ),
                        right=IndicatorValue(indicator_id="vwap"),
                    ),
                    right=LiteralValue(value=100.0),
                ),
                op=op,
                right=LiteralValue(value=float(value)),
            )

        # =================================================================
        # Gap Metrics
        # =================================================================
        elif metric == "gap_pct":
            # Gap percentage: (Open - PrevClose) / PrevClose * 100
            # Use RollingWindow to access previous close
            self._add_indicator(RollingWindow(id="prev_close_rw", period=2, field=PriceField.CLOSE))
            # gap_pct = (open - prev_close) / prev_close * 100
            return CompareCondition(
                left=ExpressionValue(
                    op="*",
                    left=ExpressionValue(
                        op="/",
                        left=ExpressionValue(
                            op="-",
                            left=PriceValue(field=PriceField.OPEN),
                            right=RollingWindowValue(indicator_id="prev_close_rw", offset=1),
                        ),
                        right=RollingWindowValue(indicator_id="prev_close_rw", offset=1),
                    ),
                    right=LiteralValue(value=100.0),
                ),
                op=op,
                right=LiteralValue(value=float(value)),
            )

        # =================================================================
        # Volume Metrics
        # =================================================================
        elif metric == "volume_pctile":
            # Volume percentile - approximated using ratio to average
            # True percentile requires rolling rank, this uses relative volume
            lookback = regime.get("lookback_bars", 20)
            self._add_indicator(VolumeSMA(id="vol_sma", period=lookback))
            # Approximate percentile as volume / avg_volume * 50
            # If volume = avg, result = 50 (median)
            # If volume = 2x avg, result = 100 (high percentile)
            return CompareCondition(
                left=ExpressionValue(
                    op="*",
                    left=ExpressionValue(
                        op="/",
                        left=VolumeValue(),
                        right=IndicatorValue(indicator_id="vol_sma"),
                    ),
                    right=LiteralValue(value=50.0),
                ),
                op=op,
                right=LiteralValue(value=float(value)),
            )

        elif metric == "volume_spike":
            # Volume spike - volume exceeds threshold percentile of average
            # volume_threshold_pctile: e.g., 80 means 80th percentile = 1.6x average
            lookback = regime.get("lookback_bars", 20)
            threshold = regime.get("volume_threshold_pctile", 80)
            self._add_indicator(VolumeSMA(id="vol_sma", period=lookback))
            # Convert percentile to multiplier: 80th pctile ≈ 1.6x, 90th ≈ 2x
            # Approximation: multiplier = 1 + (threshold - 50) / 50
            multiplier = 1.0 + (threshold - 50) / 50
            return CompareCondition(
                left=VolumeValue(),
                op=CompareOp.GT,
                right=ExpressionValue(
                    op="*",
                    left=IndicatorValue(indicator_id="vol_sma"),
                    right=LiteralValue(value=multiplier),
                ),
            )

        elif metric == "volume_dip":
            # Volume dip - volume below threshold percentile of average
            lookback = regime.get("lookback_bars", 20)
            threshold = regime.get("volume_threshold_pctile", 20)
            self._add_indicator(VolumeSMA(id="vol_sma", period=lookback))
            # Convert percentile to multiplier: 20th pctile ≈ 0.6x, 10th ≈ 0.4x
            multiplier = threshold / 50
            return CompareCondition(
                left=VolumeValue(),
                op=CompareOp.LT,
                right=ExpressionValue(
                    op="*",
                    left=IndicatorValue(indicator_id="vol_sma"),
                    right=LiteralValue(value=multiplier),
                ),
            )

        # =================================================================
        # Session/Time Metrics
        # =================================================================
        elif metric == "session_phase":
            # Session phase: 'open', 'close', 'overnight'
            # Requires session parameter to determine hours
            session = regime.get("session", "us")
            phase = value  # 'open', 'close', 'overnight'

            # Session hours (approximate, in UTC)
            session_hours = {
                "us": {"open_start": 14, "open_end": 16, "close_start": 20, "close_end": 21},
                "eu": {"open_start": 8, "open_end": 10, "close_start": 16, "close_end": 17},
                "asia": {"open_start": 0, "open_end": 2, "close_start": 6, "close_end": 7},
            }
            hours = session_hours.get(session, session_hours["us"])

            if phase == "open":
                # Market open period
                return AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=TimeValue(component="hour"),
                            op=CompareOp.GTE,
                            right=LiteralValue(value=float(hours["open_start"])),
                        ),
                        CompareCondition(
                            left=TimeValue(component="hour"),
                            op=CompareOp.LT,
                            right=LiteralValue(value=float(hours["open_end"])),
                        ),
                    ]
                )
            elif phase == "close":
                # Market close period
                return AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=TimeValue(component="hour"),
                            op=CompareOp.GTE,
                            right=LiteralValue(value=float(hours["close_start"])),
                        ),
                        CompareCondition(
                            left=TimeValue(component="hour"),
                            op=CompareOp.LT,
                            right=LiteralValue(value=float(hours["close_end"])),
                        ),
                    ]
                )
            elif phase == "overnight":
                # Outside regular session (overnight)
                # Inverted: NOT (during regular hours)
                return NotCondition(
                    condition=AllOfCondition(
                        conditions=[
                            CompareCondition(
                                left=TimeValue(component="hour"),
                                op=CompareOp.GTE,
                                right=LiteralValue(value=float(hours["open_start"])),
                            ),
                            CompareCondition(
                                left=TimeValue(component="hour"),
                                op=CompareOp.LT,
                                right=LiteralValue(value=float(hours["close_end"])),
                            ),
                        ]
                    )
                )
            else:
                raise TranslationError(f"Unknown session_phase value: {phase}")

        # =================================================================
        # Price Level Metrics
        # =================================================================
        elif metric == "price_level_touch":
            # Price touches a level (high/low reaches the level)
            level_price = regime.get("level_price")
            level_ref = regime.get("level_reference")
            lookback = regime.get("lookback_bars", 20)

            if level_price is not None:
                # Fixed price level - check if high >= level >= low
                return AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceValue(field=PriceField.HIGH),
                            op=CompareOp.GTE,
                            right=LiteralValue(value=float(level_price)),
                        ),
                        CompareCondition(
                            left=PriceValue(field=PriceField.LOW),
                            op=CompareOp.LTE,
                            right=LiteralValue(value=float(level_price)),
                        ),
                    ]
                )
            elif level_ref:
                # Dynamic level reference
                return self._price_level_touch_dynamic(level_ref, lookback)
            else:
                raise TranslationError("price_level_touch requires level_price or level_reference")

        elif metric == "price_level_cross":
            # Price crosses a level in specified direction
            level_price = regime.get("level_price")
            level_ref = regime.get("level_reference")
            direction = regime.get("direction", "up")
            lookback = regime.get("lookback_bars", 20)

            if level_price is not None:
                # Fixed price level crossing
                if direction == "up":
                    # Close above level (implies crossed up)
                    return CompareCondition(
                        left=PriceValue(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=LiteralValue(value=float(level_price)),
                    )
                else:  # down
                    return CompareCondition(
                        left=PriceValue(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=LiteralValue(value=float(level_price)),
                    )
            elif level_ref:
                return self._price_level_cross_dynamic(level_ref, direction, lookback)
            else:
                raise TranslationError("price_level_cross requires level_price or level_reference")

        # =================================================================
        # Complex Pattern Metrics (require runtime state machines)
        # =================================================================
        elif metric == "liquidity_sweep":
            # Break-below-then-reclaim pattern for stop hunts
            # This requires multi-bar state tracking in the runtime
            level_ref = regime.get("level_reference", "previous_low")
            reclaim_bars = regime.get("reclaim_within_bars", 3)
            lookback = regime.get("lookback_bars", 20)

            # Set up required indicators for the runtime
            if level_ref in ("previous_low", "session_low"):
                self._add_indicator(
                    RollingMinMax(id="level_min", period=lookback, mode="min", field=PriceField.LOW)
                )
            elif level_ref in ("previous_high", "session_high"):
                self._add_indicator(
                    RollingMinMax(
                        id="level_max", period=lookback, mode="max", field=PriceField.HIGH
                    )
                )

            # Pass to runtime as RegimeCondition with metadata
            # value is the level_ref for pattern metrics
            return RegimeCondition(
                metric=metric,
                op=op,
                value=level_ref,
                lookback_bars=reclaim_bars,
            )

        elif metric == "flag_pattern":
            # Flag consolidation pattern
            # Requires: initial momentum + narrowing range + breakout
            momentum_bars = regime.get("flag_momentum_bars", 5)
            consolidation_bars = regime.get("flag_consolidation_bars", 10)
            breakout_dir = regime.get("flag_breakout_direction", "same")

            # Set up indicators for pattern detection
            self._add_indicator(RateOfChange(id="momentum_roc", period=momentum_bars))
            self._add_indicator(ATR(id="pattern_atr", period=consolidation_bars))
            self._add_indicator(Maximum(id="pattern_max", period=consolidation_bars))
            self._add_indicator(Minimum(id="pattern_min", period=consolidation_bars))

            # Pass to runtime
            return RegimeCondition(
                metric=metric,
                op=op,
                value=breakout_dir,
                lookback_bars=consolidation_bars,
            )

        elif metric == "pennant_pattern":
            # Pennant consolidation pattern (similar to flag but triangular)
            momentum_bars = regime.get("pennant_momentum_bars", 5)
            consolidation_bars = regime.get("pennant_consolidation_bars", 10)
            breakout_dir = regime.get("pennant_breakout_direction", "same")

            # Set up indicators
            self._add_indicator(RateOfChange(id="momentum_roc", period=momentum_bars))
            self._add_indicator(ATR(id="pattern_atr", period=consolidation_bars))
            self._add_indicator(Maximum(id="pattern_max", period=consolidation_bars))
            self._add_indicator(Minimum(id="pattern_min", period=consolidation_bars))

            # Pass to runtime
            return RegimeCondition(
                metric=metric,
                op=op,
                value=breakout_dir,
                lookback_bars=consolidation_bars,
            )

        # =================================================================
        # External Data Metrics (not implementable without data source)
        # =================================================================
        elif metric == "risk_event_prob":
            # Risk event probability requires external data feed
            raise TranslationError(
                "risk_event_prob metric requires external calendar data. "
                "This metric will always pass."
            )
            return None  # Always pass - no filter

        else:
            # Unknown metric - warn and ignore
            raise TranslationError(f"Unknown metric '{metric}' - condition will always pass.")

    def _price_level_touch_dynamic(self, level_ref: str, lookback: int) -> Condition | None:
        """Handle dynamic level reference for price_level_touch."""
        if level_ref == "previous_low":
            # Touch the lookback low
            self._add_indicator(
                RollingMinMax(id="level_min", period=lookback, mode="min", field=PriceField.LOW)
            )
            return AllOfCondition(
                conditions=[
                    CompareCondition(
                        left=PriceValue(field=PriceField.HIGH),
                        op=CompareOp.GTE,
                        right=IndicatorValue(indicator_id="level_min"),
                    ),
                    CompareCondition(
                        left=PriceValue(field=PriceField.LOW),
                        op=CompareOp.LTE,
                        right=IndicatorValue(indicator_id="level_min"),
                    ),
                ]
            )
        elif level_ref == "previous_high":
            # Touch the lookback high
            self._add_indicator(
                RollingMinMax(id="level_max", period=lookback, mode="max", field=PriceField.HIGH)
            )
            return AllOfCondition(
                conditions=[
                    CompareCondition(
                        left=PriceValue(field=PriceField.HIGH),
                        op=CompareOp.GTE,
                        right=IndicatorValue(indicator_id="level_max"),
                    ),
                    CompareCondition(
                        left=PriceValue(field=PriceField.LOW),
                        op=CompareOp.LTE,
                        right=IndicatorValue(indicator_id="level_max"),
                    ),
                ]
            )
        elif level_ref == "session_low":
            # Session low: lowest low of current session
            self._add_indicator(SessionHighLow(id="session_low", mode="low"))
            return AnyOfCondition(
                conditions=[
                    CompareCondition(
                        left=PriceValue(field=PriceField.LOW),
                        op=CompareOp.LTE,
                        right=IndicatorValue(indicator_id="session_low"),
                    ),
                    CompareCondition(
                        left=PriceValue(field=PriceField.HIGH),
                        op=CompareOp.GTE,
                        right=IndicatorValue(indicator_id="session_low"),
                    ),
                ]
            )
        elif level_ref == "session_high":
            # Session high: highest high of current session
            self._add_indicator(SessionHighLow(id="session_high", mode="high"))
            return AnyOfCondition(
                conditions=[
                    CompareCondition(
                        left=PriceValue(field=PriceField.HIGH),
                        op=CompareOp.GTE,
                        right=IndicatorValue(indicator_id="session_high"),
                    ),
                    CompareCondition(
                        left=PriceValue(field=PriceField.LOW),
                        op=CompareOp.LTE,
                        right=IndicatorValue(indicator_id="session_high"),
                    ),
                ]
            )
        elif level_ref in ("recent_support", "recent_resistance"):
            # These require more complex level detection
            raise TranslationError(
                f"level_reference '{level_ref}' for price_level_touch requires runtime support"
            )
        else:
            raise TranslationError(f"Unknown level_reference: {level_ref}")

    def _price_level_cross_dynamic(
        self, level_ref: str, direction: str, lookback: int
    ) -> Condition | None:
        """Handle dynamic level reference for price_level_cross."""
        if level_ref == "previous_low":
            self._add_indicator(
                RollingMinMax(id="level_min", period=lookback, mode="min", field=PriceField.LOW)
            )
            if direction == "up":
                # Cross above the previous low (recovery)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="level_min"),
                )
            else:
                # Cross below the previous low (breakdown)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=IndicatorValue(indicator_id="level_min"),
                )
        elif level_ref == "previous_high":
            self._add_indicator(
                RollingMinMax(id="level_max", period=lookback, mode="max", field=PriceField.HIGH)
            )
            if direction == "up":
                # Cross above the previous high (breakout)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="level_max"),
                )
            else:
                # Cross below the previous high (rejection)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=IndicatorValue(indicator_id="level_max"),
                )
        elif level_ref == "session_low":
            # Cross session low
            self._add_indicator(SessionHighLow(id="session_low", mode="low"))
            if direction == "up":
                # Cross above session low (recovery)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="session_low"),
                )
            else:
                # Cross below session low (breakdown)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=IndicatorValue(indicator_id="session_low"),
                )
        elif level_ref == "session_high":
            # Cross session high
            self._add_indicator(SessionHighLow(id="session_high", mode="high"))
            if direction == "up":
                # Cross above session high (breakout)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="session_high"),
                )
            else:
                # Cross below session high (rejection)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=IndicatorValue(indicator_id="session_high"),
                )
        elif level_ref == "recent_support":
            # Recent support approximated as rolling minimum low
            # In practice, this is the lowest low in the lookback period
            self._add_indicator(
                RollingMinMax(
                    id="recent_support", period=lookback, mode="min", field=PriceField.LOW
                )
            )
            if direction == "up":
                # Cross above support (bounce/recovery)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="recent_support"),
                )
            else:
                # Cross below support (breakdown)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=IndicatorValue(indicator_id="recent_support"),
                )
        elif level_ref == "recent_resistance":
            # Recent resistance approximated as rolling maximum high
            # In practice, this is the highest high in the lookback period
            self._add_indicator(
                RollingMinMax(
                    id="recent_resistance", period=lookback, mode="max", field=PriceField.HIGH
                )
            )
            if direction == "up":
                # Cross above resistance (breakout)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="recent_resistance"),
                )
            else:
                # Cross below resistance (rejection)
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=IndicatorValue(indicator_id="recent_resistance"),
                )
        else:
            raise TranslationError(f"Unknown level_reference: {level_ref}")

    def _translate_band_event(self, band_event: dict[str, Any]) -> Condition | None:
        """Translate a BandEvent to IR Condition.

        BandEvent supports:
        - edge_event (touch): price crosses a band edge
        - distance (z-score): how far price is from band in std devs
        - cross_in/cross_out: directional crossing of band
        - reentry: price exits band and re-enters through middle
        """
        band_spec = band_event.get("band", {})
        kind = band_event.get("kind", "edge_event")

        # Create band indicator
        band_id = self._create_band_indicator(band_spec)
        if not band_id:
            return None

        if kind == "distance":
            return self._band_distance_condition(band_event, band_id, band_spec)
        elif kind == "edge_event":
            return self._band_edge_event_condition(band_event, band_id)
        elif kind == "reentry":
            return self._band_reentry_condition(band_event, band_id)
        else:
            raise TranslationError(f"Unknown band_event kind: {kind}")

    def _create_band_indicator(self, band_spec: dict[str, Any]) -> str | None:
        """Create a band indicator and return its ID.

        Supported bands:
        - bollinger: Bollinger Bands (middle = SMA, bands = std dev)
        - keltner: Keltner Channel (middle = EMA, bands = ATR)
        - donchian: Donchian Channel (high/low of N bars)

        Not yet supported:
        - vwap_band: VWAP with standard deviation bands (requires custom implementation)
        - vwap: Plain VWAP without bands (use IndicatorValue instead)
        """
        band_type = band_spec.get("band", "bollinger")
        length = band_spec.get("length", 20)
        mult = band_spec.get("mult", 2.0)

        # Generate param-based ID using same scheme as _generate_indicator_id
        # Band type maps: bollinger->bb, keltner->kc, donchian->dc
        type_abbrev = {"bollinger": "bb", "keltner": "kc", "donchian": "dc"}.get(
            band_type, band_type
        )
        mult_str = str(mult).replace(".", "_")
        band_id = f"{type_abbrev}_{mult_str}_{length}"

        if band_type == "bollinger":
            self._add_indicator(BollingerBands(id=band_id, period=length, multiplier=mult))
        elif band_type == "keltner":
            self._add_indicator(KeltnerChannel(id=band_id, period=length, multiplier=mult))
        elif band_type == "donchian":
            self._add_indicator(DonchianChannel(id=band_id, period=length))
        elif band_type in ("vwap", "vwap_band"):
            # VWAP bands: VWAP as middle line with standard deviation bands
            anchor = band_spec.get("anchor", "session")
            # Map various anchor formats to IR-supported values
            anchor_map = {
                "session_open": "session",
                "month_start": "month",
                "ytd": "ytd",
                "week_start": "week",
            }
            ir_anchor = anchor_map.get(
                anchor,
                anchor if anchor in ("session", "week", "month", "ytd", "custom") else "session",
            )
            self._add_indicator(VWAPBands(id=band_id, anchor=ir_anchor, multiplier=mult))
        else:
            raise TranslationError(f"Unknown band type: {band_type}")

        return band_id

    def _band_edge_event_condition(
        self, band_event: dict[str, Any], band_id: str
    ) -> Condition | None:
        """Translate edge_event (touch, cross_in, cross_out) to condition."""
        event = band_event.get("event", "touch")
        edge = band_event.get("edge", "lower")

        band_field = BandField.UPPER if edge == "upper" else BandField.LOWER

        if event == "touch":
            # Price touches or crosses band edge
            if edge == "upper":
                op = CompareOp.GTE
            else:
                op = CompareOp.LTE

            return CompareCondition(
                left=PriceValue(field=PriceField.CLOSE),
                op=op,
                right=IndicatorBandValue(indicator_id=band_id, band=band_field),
            )

        elif event == "cross_in":
            # Price crosses INTO the band (from outside)
            # Need state tracking: was outside, now inside
            return self._band_cross_condition(band_id, edge, direction="in")

        elif event == "cross_out":
            # Price crosses OUT OF the band (from inside)
            # Need state tracking: was inside, now outside
            return self._band_cross_condition(band_id, edge, direction="out")

        else:
            raise TranslationError(f"Unknown edge_event type: {event}")

    def _band_cross_condition(self, band_id: str, edge: str, direction: str) -> Condition:
        """Create condition for band crossing with state tracking.

        For cross_in (upper band):
        - was_above_upper = True (previous bar)
        - price < upper_band (current bar, now inside)

        For cross_out (upper band):
        - was_inside_upper = True (previous bar, price < upper)
        - price >= upper_band (current bar, now outside)
        """
        band_field = BandField.UPPER if edge == "upper" else BandField.LOWER
        state_id = f"was_{direction}side_{edge}_{band_id}"

        # Add state variable for tracking
        self._add_state(state_id, StateType.BOOL, False)

        if direction == "in":
            # Cross IN means: was outside, now inside
            if edge == "upper":
                # Was above upper, now below upper (crossed back in from above)
                current_op = CompareOp.LT
                was_outside_condition = CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GTE,
                    right=IndicatorBandValue(indicator_id=band_id, band=band_field),
                )
            else:
                # Was below lower, now above lower (crossed back in from below)
                current_op = CompareOp.GT
                was_outside_condition = CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.LTE,
                    right=IndicatorBandValue(indicator_id=band_id, band=band_field),
                )
        else:
            # Cross OUT means: was inside, now outside
            if edge == "upper":
                # Was below upper (inside), now above upper (outside)
                current_op = CompareOp.GTE
                was_outside_condition = CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=IndicatorBandValue(indicator_id=band_id, band=band_field),
                )
            else:
                # Was above lower (inside), now below lower (outside)
                current_op = CompareOp.LTE
                was_outside_condition = CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorBandValue(indicator_id=band_id, band=band_field),
                )

        # Add on_bar hook to track state
        self._on_bar_hooks.append(
            SetStateFromConditionOp(
                state_id=state_id,
                condition=was_outside_condition,
            )
        )

        # Current condition: state was set AND price now on other side
        return AllOfCondition(
            conditions=[
                CompareCondition(
                    left=StateValue(state_id=state_id),
                    op=CompareOp.EQ,
                    right=LiteralValue(value=1.0),  # True as float
                ),
                CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=current_op,
                    right=IndicatorBandValue(indicator_id=band_id, band=band_field),
                ),
            ]
        )

    def _band_distance_condition(
        self, band_event: dict[str, Any], band_id: str, band_spec: dict[str, Any]
    ) -> Condition:
        """Translate distance (z-score) condition.

        Z-score = (price - middle) / standard_deviation
        For Bollinger Bands, we can access the StandardDeviation property.
        """
        op_str = band_event.get("op", ">")
        value = band_event.get("value", 2.0)
        op = self.OP_MAP.get(op_str, CompareOp.GT)

        band_type = band_spec.get("band", "bollinger")

        if band_type == "bollinger":
            # Z-score: (close - middle) / std_dev
            return CompareCondition(
                left=ExpressionValue(
                    op="/",
                    left=ExpressionValue(
                        op="-",
                        left=PriceValue(field=PriceField.CLOSE),
                        right=IndicatorBandValue(indicator_id=band_id, band=BandField.MIDDLE),
                    ),
                    right=IndicatorPropertyValue(
                        indicator_id=band_id,
                        property=IndicatorProperty.STANDARD_DEVIATION,
                    ),
                ),
                op=op,
                right=LiteralValue(value=float(value)),
            )
        else:
            # For non-Bollinger bands, approximate using band width
            # (price - middle) / ((upper - lower) / 4)
            # This gives a rough z-score equivalent
            return CompareCondition(
                left=ExpressionValue(
                    op="/",
                    left=ExpressionValue(
                        op="-",
                        left=PriceValue(field=PriceField.CLOSE),
                        right=IndicatorBandValue(indicator_id=band_id, band=BandField.MIDDLE),
                    ),
                    right=ExpressionValue(
                        op="/",
                        left=ExpressionValue(
                            op="-",
                            left=IndicatorBandValue(indicator_id=band_id, band=BandField.UPPER),
                            right=IndicatorBandValue(indicator_id=band_id, band=BandField.LOWER),
                        ),
                        right=LiteralValue(value=4.0),
                    ),
                ),
                op=op,
                right=LiteralValue(value=float(value)),
            )

    def _band_reentry_condition(self, band_event: dict[str, Any], band_id: str) -> Condition:
        """Translate reentry condition.

        Reentry: price crossed outside band, then came back through middle.
        Tracks state: was_outside_{edge}, triggers when price crosses middle.
        """
        edge = band_event.get("edge", "lower")
        state_id = f"was_outside_{edge}_{band_id}"

        # Add state variable for tracking
        self._add_state(state_id, StateType.BOOL, False)

        if edge == "lower":
            # Was below lower band
            was_outside_condition = CompareCondition(
                left=PriceValue(field=PriceField.CLOSE),
                op=CompareOp.LT,
                right=IndicatorBandValue(indicator_id=band_id, band=BandField.LOWER),
            )
            # Now crosses above middle (reentry complete)
            reentry_op = CompareOp.GT
        else:
            # Was above upper band
            was_outside_condition = CompareCondition(
                left=PriceValue(field=PriceField.CLOSE),
                op=CompareOp.GT,
                right=IndicatorBandValue(indicator_id=band_id, band=BandField.UPPER),
            )
            # Now crosses below middle (reentry complete)
            reentry_op = CompareOp.LT

        # Add on_bar hook to track if price went outside band
        # Note: This sets state to True when outside, needs to persist until reentry
        self._on_bar_hooks.append(
            SetStateFromConditionOp(
                state_id=state_id,
                condition=was_outside_condition,
            )
        )

        # Trigger when: was_outside AND price crosses middle
        return AllOfCondition(
            conditions=[
                CompareCondition(
                    left=StateValue(state_id=state_id),
                    op=CompareOp.EQ,
                    right=LiteralValue(value=1.0),  # True as float
                ),
                CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=reentry_op,
                    right=IndicatorBandValue(indicator_id=band_id, band=BandField.MIDDLE),
                ),
            ]
        )

    # =========================================================================
    # Sequence Translation
    # =========================================================================

    def _translate_sequence(self, steps: list[dict[str, Any]]) -> Condition | None:
        """Translate a sequence ConditionSpec to IR.

        A sequence is a series of conditions that must be satisfied in order,
        optionally with timing constraints (within_bars).

        Example: [{"cond": A}, {"cond": B, "within_bars": 5}]
        Means: A must be true, then within 5 bars B must be true.

        Implementation uses boolean latches per step:
        - step_N_done: latched true when step N satisfied (after previous steps)
        - bars_since_step_N: increments after step_N_done, for timeout checking
        """
        if len(steps) < 2:
            raise TranslationError("Sequence must have at least 2 steps")

        # Generate unique ID for this sequence
        seq_id = f"seq_{self._sequence_counter}"
        self._sequence_counter += 1

        # Translate each step's condition
        step_conditions: list[Condition] = []
        for i, step in enumerate(steps):
            cond = self._translate_condition(step.get("cond", {}))
            if cond is None:
                raise TranslationError(f"Failed to translate sequence step {i} condition")
            step_conditions.append(cond)

        # For each step (except last), create state tracking
        # Step 0: track when condition first becomes true (latch)
        # Step 1+: track when condition becomes true AFTER previous step done

        for i in range(len(steps) - 1):
            trigger_state = f"{seq_id}_step_{i}_trigger"  # 1.0 when condition true now
            done_state = f"{seq_id}_step_{i}_done"  # latched 1.0 when step completed
            bars_since = f"{seq_id}_bars_since_{i}"  # bars elapsed since step done

            self._add_state(trigger_state, StateType.FLOAT, 0.0)
            self._add_state(done_state, StateType.FLOAT, 0.0)
            self._add_state(bars_since, StateType.INT, 0)

            # Build the trigger condition for this step
            if i == 0:
                # Step 0: just check if condition is true
                trigger_cond = step_conditions[0]
            else:
                # Step i>0: previous step must be done AND condition true
                prev_done = f"{seq_id}_step_{i - 1}_done"
                within_bars = steps[i].get("within_bars")

                if within_bars:
                    prev_bars_since = f"{seq_id}_bars_since_{i - 1}"
                    # Previous step done AND within timeout AND this condition true
                    trigger_cond = AllOfCondition(
                        conditions=[
                            CompareCondition(
                                left=StateValue(state_id=prev_done),
                                op=CompareOp.EQ,
                                right=LiteralValue(value=1.0),
                            ),
                            CompareCondition(
                                left=StateValue(state_id=prev_bars_since),
                                op=CompareOp.LTE,
                                right=LiteralValue(value=float(within_bars)),
                            ),
                            step_conditions[i],
                        ]
                    )
                else:
                    # Previous step done AND this condition true (no timeout)
                    trigger_cond = AllOfCondition(
                        conditions=[
                            CompareCondition(
                                left=StateValue(state_id=prev_done),
                                op=CompareOp.EQ,
                                right=LiteralValue(value=1.0),
                            ),
                            step_conditions[i],
                        ]
                    )

            # on_bar: Set trigger state based on condition
            self._on_bar_hooks.append(
                SetStateFromConditionOp(state_id=trigger_state, condition=trigger_cond)
            )

            # on_bar: Latch done state using MaxStateOp (once 1.0, stays 1.0)
            self._on_bar_hooks.append(
                MaxStateOp(state_id=done_state, value=StateValue(state_id=trigger_state))
            )

            # on_bar: Increment bars_since when done (using expression)
            # bars_since = bars_since + done_state (only increments when done=1.0)
            self._on_bar_hooks.append(
                SetStateOp(
                    state_id=bars_since,
                    value=ExpressionValue(
                        op="+",
                        left=StateValue(state_id=bars_since),
                        right=StateValue(state_id=done_state),
                    ),
                )
            )

        # Entry condition: all previous steps done AND final condition true
        final_idx = len(steps) - 1
        entry_conditions: list[Condition] = []

        # Check all previous steps are done
        for i in range(final_idx):
            done_state = f"{seq_id}_step_{i}_done"
            entry_conditions.append(
                CompareCondition(
                    left=StateValue(state_id=done_state),
                    op=CompareOp.EQ,
                    right=LiteralValue(value=1.0),
                )
            )

        # Check final step's timeout if specified
        if final_idx > 0:
            within_bars = steps[final_idx].get("within_bars")
            if within_bars:
                prev_bars_since = f"{seq_id}_bars_since_{final_idx - 1}"
                entry_conditions.append(
                    CompareCondition(
                        left=StateValue(state_id=prev_bars_since),
                        op=CompareOp.LTE,
                        right=LiteralValue(value=float(within_bars)),
                    )
                )

        # Final step's condition must be true now
        entry_conditions.append(step_conditions[final_idx])

        return AllOfCondition(conditions=entry_conditions)

    def _translate_breakout(self, spec: dict[str, Any]) -> Condition | None:
        """Translate a BreakoutSpec to IR Condition.

        Breakout: price breaks above N-bar high (long) or below N-bar low (short).
        Lowers to CompareCondition comparing Close to MAX/MIN indicators.
        """
        lookback_bars = spec.get("lookback_bars", 50)
        buffer_bps = spec.get("buffer_bps", 0)
        direction = spec.get("direction", "long")  # long = breakout above, short = below

        # Create MAX and MIN indicators for N-bar high/low
        max_id = f"highest_{lookback_bars}"
        min_id = f"lowest_{lookback_bars}"

        self._add_indicator(Maximum(id=max_id, period=lookback_bars))
        self._add_indicator(Minimum(id=min_id, period=lookback_bars))

        # Apply buffer if specified (buffer_bps = basis points above/below level)
        buffer_mult = 1.0 + (buffer_bps / 10000.0) if buffer_bps else 1.0

        if direction == "long":
            # Breakout above: Close > MAX * buffer
            if buffer_bps:
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=ExpressionValue(
                        op="*",
                        left=IndicatorValue(indicator_id=max_id),
                        right=LiteralValue(value=buffer_mult),
                    ),
                )
            return CompareCondition(
                left=PriceValue(field=PriceField.CLOSE),
                op=CompareOp.GT,
                right=IndicatorValue(indicator_id=max_id),
            )
        else:
            # Breakout below: Close < MIN * (1 - buffer)
            buffer_mult_low = 1.0 - (buffer_bps / 10000.0) if buffer_bps else 1.0
            if buffer_bps:
                return CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=ExpressionValue(
                        op="*",
                        left=IndicatorValue(indicator_id=min_id),
                        right=LiteralValue(value=buffer_mult_low),
                    ),
                )
            return CompareCondition(
                left=PriceValue(field=PriceField.CLOSE),
                op=CompareOp.LT,
                right=IndicatorValue(indicator_id=min_id),
            )

    def _translate_squeeze(self, spec: dict[str, Any]) -> Condition | None:
        """Translate a SqueezeSpec to IR Condition.

        Squeeze: BB width percentile below threshold indicates compression.
        Lowers to CompareCondition using Percentile indicator for BB width.
        """
        pctile_threshold = spec.get("pctile_min", 10)
        bb_period = spec.get("bb_period", 20)
        pctile_period = spec.get("pctile_period", 100)

        # Create Bollinger Bands for width calculation
        bb_id = f"bb_{bb_period}"
        self._add_indicator(BollingerBands(id=bb_id, period=bb_period, multiplier=2.0))

        # Create Percentile indicator for BB width ranking
        pctile_id = f"bb_width_pctile_{bb_period}_{pctile_period}"
        self._add_indicator(
            Percentile(id=pctile_id, period=pctile_period, percentile=float(pctile_threshold), source="bb_width")
        )

        # Squeeze condition: current BB width percentile < threshold
        # The percentile indicator returns 0-100 ranking
        return CompareCondition(
            left=IndicatorValue(indicator_id=pctile_id),
            op=CompareOp.LT,
            right=LiteralValue(value=float(pctile_threshold)),
        )

    def _translate_time_filter(self, spec: dict[str, Any]) -> Condition | None:
        """Translate a TimeFilterSpec to IR Condition.

        Time filter: check day of week, time window, or day of month.
        Lowers to CompareCondition/AnyOfCondition using TimeValue.
        """
        raw_days = spec.get("days_of_week", [])
        time_window = spec.get("time_window", "")

        # Convert day names to integers if needed (0=Monday, 6=Sunday)
        day_name_map = {
            "monday": 0, "mon": 0,
            "tuesday": 1, "tue": 1,
            "wednesday": 2, "wed": 2,
            "thursday": 3, "thu": 3,
            "friday": 4, "fri": 4,
            "saturday": 5, "sat": 5,
            "sunday": 6, "sun": 6,
        }
        days_of_week = []
        for day in raw_days:
            if isinstance(day, int):
                days_of_week.append(day)
            elif isinstance(day, str):
                day_int = day_name_map.get(day.lower())
                if day_int is not None:
                    days_of_week.append(day_int)

        conditions: list[Condition] = []

        # Days of week filter: any of the specified days
        if days_of_week:
            day_conditions = [
                CompareCondition(
                    left=TimeValue(component="day_of_week"),
                    op=CompareOp.EQ,
                    right=LiteralValue(value=float(day)),
                )
                for day in days_of_week
            ]
            if len(day_conditions) == 1:
                conditions.append(day_conditions[0])
            else:
                conditions.append(AnyOfCondition(conditions=day_conditions))

        # Time window filter: parse "HH:MM-HH:MM" format
        if time_window and "-" in time_window:
            try:
                start_str, end_str = time_window.split("-")
                start_hour = int(start_str.split(":")[0])
                end_hour = int(end_str.split(":")[0])

                # Hour >= start AND Hour < end
                time_conditions = [
                    CompareCondition(
                        left=TimeValue(component="hour"),
                        op=CompareOp.GTE,
                        right=LiteralValue(value=float(start_hour)),
                    ),
                    CompareCondition(
                        left=TimeValue(component="hour"),
                        op=CompareOp.LT,
                        right=LiteralValue(value=float(end_hour)),
                    ),
                ]
                conditions.append(AllOfCondition(conditions=time_conditions))
            except (ValueError, IndexError):
                pass  # Invalid time window format, skip

        # Combine all conditions
        if not conditions:
            return None  # No valid filter specified
        if len(conditions) == 1:
            return conditions[0]
        return AllOfCondition(conditions=conditions)

    def _translate_intermarket(self, spec: dict[str, Any]) -> Condition | None:
        """Translate an intermarket trigger condition.

        Intermarket: enter follower based on leader symbol's movement.

        Supports two modes:
        1. Single leader: leader_symbol + trigger_feature + trigger_threshold
        2. Multi-leader: leaders + leader_aggregate (with feature, op, threshold)

        Supported trigger_features: ret_pct, ma_flip, band_break
        """
        leader_symbol = spec.get("leader_symbol")
        leaders = spec.get("leaders")
        leader_aggregate = spec.get("leader_aggregate")
        window_bars = spec.get("window_bars", 24)

        # Multi-leader mode
        if leaders and leader_aggregate:
            return self._translate_intermarket_multi_leader(
                leaders=leaders,
                aggregate=leader_aggregate,
                window_bars=window_bars,
            )

        # Single leader mode
        if not leader_symbol:
            raise TranslationError(
                "intermarket condition requires leader_symbol or leaders+leader_aggregate"
            )

        trigger_feature = spec.get("trigger_feature", "ret_pct")
        trigger_threshold = spec.get("trigger_threshold", 2.0)

        # Add leader symbol to additional_symbols for subscription
        if leader_symbol not in self._additional_symbols:
            self._additional_symbols.append(leader_symbol)

        # Create indicator on leader symbol
        if trigger_feature == "ret_pct":
            # Leader's return percentage over window
            indicator_id = f"leader_roc_{leader_symbol.replace('-', '_')}"
            self._add_indicator(
                RateOfChange(id=indicator_id, period=window_bars, symbol=leader_symbol)
            )
            # ROC returns decimal, convert to percentage and compare
            return CompareCondition(
                left=ExpressionValue(
                    op="*",
                    left=IndicatorValue(indicator_id=indicator_id),
                    right=LiteralValue(value=100.0),
                ),
                op=CompareOp.GTE,
                right=LiteralValue(value=float(trigger_threshold)),
            )
        elif trigger_feature == "ma_flip":
            # Leader's MA crossover - fast crosses above slow
            fast_id = f"leader_ema_fast_{leader_symbol.replace('-', '_')}"
            slow_id = f"leader_ema_slow_{leader_symbol.replace('-', '_')}"
            # Use standard fast/slow periods
            self._add_indicator(EMA(id=fast_id, period=12))
            self._add_indicator(EMA(id=slow_id, period=26))
            # TODO: Add symbol field to EMA for multi-symbol support
            # For now, this works conceptually but needs runtime support
            return CompareCondition(
                left=IndicatorValue(indicator_id=fast_id),
                op=CompareOp.GT,
                right=IndicatorValue(indicator_id=slow_id),
            )
        elif trigger_feature == "band_break":
            # Leader breaks above upper band
            band_id = f"leader_bb_{leader_symbol.replace('-', '_')}"
            self._add_indicator(BollingerBands(id=band_id, period=20, multiplier=2.0))
            return CompareCondition(
                left=PriceValue(field=PriceField.CLOSE),
                op=CompareOp.GT,
                right=IndicatorBandValue(indicator_id=band_id, band=BandField.UPPER),
            )
        else:
            raise TranslationError(f"Unknown intermarket trigger_feature: {trigger_feature}")

    def _translate_intermarket_multi_leader(
        self,
        leaders: list[str],
        aggregate: dict[str, Any],
        window_bars: int,
    ) -> Condition:
        """Translate multi-leader intermarket condition.

        Creates indicators for each leader and aggregates them.
        Currently supports ret_pct feature with avg aggregation.
        """
        feature = aggregate.get("feature", "ret_pct")
        agg_op = aggregate.get("op", "avg")
        threshold = aggregate.get("threshold", 2.0)

        if feature != "ret_pct":
            raise TranslationError(
                f"Multi-leader intermarket only supports ret_pct feature, got: {feature}"
            )

        # Create ROC indicator for each leader
        indicator_ids = []
        for leader in leaders:
            if leader not in self._additional_symbols:
                self._additional_symbols.append(leader)

            indicator_id = f"leader_roc_{leader.replace('-', '_')}"
            self._add_indicator(
                RateOfChange(id=indicator_id, period=window_bars, symbol=leader)
            )
            indicator_ids.append(indicator_id)

        # Build aggregation expression
        # For avg: (ind1 + ind2 + ...) / n
        if agg_op == "avg":
            # Convert each to percentage and sum
            sum_expr: ExpressionValue | None = None
            for ind_id in indicator_ids:
                pct_expr = ExpressionValue(
                    op="*",
                    left=IndicatorValue(indicator_id=ind_id),
                    right=LiteralValue(value=100.0),
                )
                if sum_expr is None:
                    sum_expr = pct_expr
                else:
                    sum_expr = ExpressionValue(
                        op="+",
                        left=sum_expr,
                        right=pct_expr,
                    )

            # Divide by count to get average
            avg_expr = ExpressionValue(
                op="/",
                left=sum_expr,
                right=LiteralValue(value=float(len(indicator_ids))),
            )

            return CompareCondition(
                left=avg_expr,
                op=CompareOp.GTE,
                right=LiteralValue(value=float(threshold)),
            )
        else:
            raise TranslationError(
                f"Multi-leader intermarket only supports avg aggregation, got: {agg_op}"
            )

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
