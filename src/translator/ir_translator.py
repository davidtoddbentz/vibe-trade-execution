"""Strategy to IR translator.

Converts vibe-trade Strategy/Card models to StrategyIR for the LEAN runtime.
This replaces string-based code generation with typed data structures.
"""

from dataclasses import dataclass
from typing import Any

from vibe_trade_shared.models import Card, Strategy

from .archetype_expander import ArchetypeExpander
from .ir import (
    ADX,
    ATR,
    EMA,
    VWAP,
    AllOfCondition,
    AnyOfCondition,
    BandField,
    BollingerBands,
    BreakoutCondition,
    CompareCondition,
    CompareOp,
    Condition,
    DonchianChannel,
    EntryRule,
    ExitRule,
    ExpressionValue,
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
    PriceField,
    PriceValue,
    RateOfChange,
    RegimeCondition,
    Resolution,
    RollingMinMax,
    RollingWindow,
    SetHoldingsAction,
    SetStateFromConditionOp,
    SetStateOp,
    SqueezeCondition,
    StateOp,
    StateType,
    StateValue,
    StateVar,
    StrategyIR,
    TimeFilterCondition,
    TimeValue,
    VolumeSMA,
    VolumeValue,
)


@dataclass
class IRTranslationResult:
    """Result of translating a strategy to IR."""

    ir: StrategyIR
    warnings: list[str]


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
        self.warnings: list[str] = []

        # Get first symbol from universe (MVP: single asset)
        self.symbol = strategy.universe[0] if strategy.universe else "BTC-USD"

        # Track indicators to avoid duplicates
        self._indicators: dict[str, Indicator] = {}
        self._state_vars: dict[str, StateVar] = {}
        self._on_bar_hooks: list[StateOp] = []
        self._exit_counter = 0
        self._sequence_counter = 0

        # Archetype expander for converting specialized archetypes to primitives
        self._expander = ArchetypeExpander()

    def translate(self) -> IRTranslationResult:
        """Translate the strategy to IR.

        Returns:
            IRTranslationResult with the IR and any warnings
        """
        entry: EntryRule | None = None
        exits: list[ExitRule] = []
        gates: list[Gate] = []
        on_bar_invested: list[StateOp] = []

        # Process attachments by role
        for attachment in self.strategy.attachments:
            if not attachment.enabled:
                continue

            card = self.cards.get(attachment.card_id)
            if not card:
                self.warnings.append(f"Card not found: {attachment.card_id}")
                continue

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
                self.warnings.append(f"Overlay translation not yet implemented: {card.type}")

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
            indicators=list(self._indicators.values()),
            state=list(self._state_vars.values()),
            gates=gates,
            entry=entry,
            exits=exits,
            on_bar=self._on_bar_hooks,
            on_bar_invested=on_bar_invested,
        )

        return IRTranslationResult(ir=ir, warnings=self.warnings)

    # =========================================================================
    # Entry Translation
    # =========================================================================

    def _expand_and_translate(
        self,
        archetype: str,
        slots: dict[str, Any],
        primitive_handler: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Expand an archetype to its primitive form if needed.

        Args:
            archetype: The archetype type_id
            slots: The card slots
            primitive_handler: The handler function for the primitive type

        Returns:
            Tuple of (expanded_archetype, expanded_slots)
        """
        if self._expander.is_primitive(archetype):
            return archetype, slots

        # Build a card-like dict for expansion
        card_dict = {"type_id": archetype, "slots": slots}
        expanded, provenance = self._expander.expand(card_dict)

        if provenance:
            # Log the expansion for debugging
            self.warnings.append(
                f"Expanded {archetype} → {expanded['type_id']} "
                f"(provenance: {provenance['source_archetype']})"
            )

        return expanded["type_id"], expanded.get("slots", slots)

    def _translate_entry(self, archetype: str, slots: dict[str, Any]) -> EntryRule | None:
        """Translate an entry card to EntryRule.

        First attempts to expand specialized archetypes to their primitive form,
        then dispatches to the appropriate handler.
        """
        # Handlers for primitive types
        primitive_handlers = {
            "entry.rule_trigger": self._entry_rule_trigger,
        }

        # Handlers for archetypes that have custom translation logic
        custom_handlers = {
            "entry.trend_pullback": self._entry_trend_pullback,
            "entry.breakout_trendfollow": self._entry_breakout,
        }

        # First try custom handlers (for archetypes with optimized translation)
        if archetype in custom_handlers:
            return custom_handlers[archetype](slots)

        # Try to expand to primitive form
        expanded_archetype, expanded_slots = self._expand_and_translate(
            archetype, slots, primitive_handlers.get("entry.rule_trigger")
        )

        # Now dispatch to primitive handler
        handler = primitive_handlers.get(expanded_archetype)
        if handler:
            return handler(expanded_slots)
        else:
            self.warnings.append(f"Unsupported entry archetype: {archetype}")
            return None

    def _entry_rule_trigger(self, slots: dict[str, Any]) -> EntryRule | None:
        """Translate entry.rule_trigger archetype."""
        event = slots.get("event", {})
        action = slots.get("action", {})

        condition_spec = event.get("condition", {})
        direction = action.get("direction", "long")

        condition = self._translate_condition(condition_spec)
        if condition is None:
            self.warnings.append("Could not translate condition for entry.rule_trigger")
            return None

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

    def _entry_trend_pullback(self, slots: dict[str, Any]) -> EntryRule | None:
        """Translate entry.trend_pullback archetype."""
        event = slots.get("event", {})
        action = slots.get("action", {})

        dip_band = event.get("dip_band", {})
        trend_gate = event.get("trend_gate", {})
        direction = action.get("direction", "long")

        # Get band parameters
        dip_band.get("band", "bollinger")
        dip_band.get("length", 20)
        dip_band.get("mult", 2.0)

        # Get trend gate parameters
        fast_ma = trend_gate.get("fast", 20)
        slow_ma = trend_gate.get("slow", 50)
        ma_op = trend_gate.get("op", ">")

        # Add indicators
        self._add_indicator(EMA(id="ema_fast", period=fast_ma))
        self._add_indicator(EMA(id="ema_slow", period=slow_ma))

        # Create band indicator using consistent naming convention
        band_id = self._create_band_indicator(dip_band)
        if not band_id:
            return None

        # Build condition
        trend_op = self.OP_MAP.get(ma_op, CompareOp.GT)

        if direction == "long":
            condition = AllOfCondition(
                conditions=[
                    CompareCondition(
                        left=IndicatorValue(indicator_id="ema_fast"),
                        op=trend_op,
                        right=IndicatorValue(indicator_id="ema_slow"),
                    ),
                    CompareCondition(
                        left=PriceValue(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=IndicatorBandValue(indicator_id=band_id, band=BandField.LOWER),
                    ),
                ]
            )
            allocation = 0.95
        else:
            inverted_op = self._invert_op(trend_op)
            condition = AllOfCondition(
                conditions=[
                    CompareCondition(
                        left=IndicatorValue(indicator_id="ema_fast"),
                        op=inverted_op,
                        right=IndicatorValue(indicator_id="ema_slow"),
                    ),
                    CompareCondition(
                        left=PriceValue(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=IndicatorBandValue(indicator_id=band_id, band=BandField.UPPER),
                    ),
                ]
            )
            allocation = -0.95

        self._add_state("entry_price", StateType.FLOAT, None)

        return EntryRule(
            condition=condition,
            action=SetHoldingsAction(allocation=allocation),
            on_fill=[
                SetStateOp(state_id="entry_price", value=PriceValue(field=PriceField.CLOSE)),
            ],
        )

    def _entry_breakout(self, slots: dict[str, Any]) -> EntryRule | None:
        """Translate entry.breakout_trendfollow archetype."""
        event = slots.get("event", {})
        action = slots.get("action", {})

        breakout = event.get("breakout", {})
        lookback = breakout.get("lookback_bars", 50)
        direction = action.get("direction", "long")

        # Add indicators
        self._add_indicator(Maximum(id="highest", period=lookback))
        self._add_indicator(Minimum(id="lowest", period=lookback))

        if direction == "long":
            condition = CompareCondition(
                left=PriceValue(field=PriceField.CLOSE),
                op=CompareOp.GTE,
                right=IndicatorValue(indicator_id="highest"),
            )
            allocation = 0.95
        else:
            condition = CompareCondition(
                left=PriceValue(field=PriceField.CLOSE),
                op=CompareOp.LTE,
                right=IndicatorValue(indicator_id="lowest"),
            )
            allocation = -0.95

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

        Supports archetype expansion for specialized exit types like
        exit.structure_break and exit.squeeze_compression.
        """
        # Try archetype expansion first
        if not self._expander.is_primitive(archetype):
            card_dict = {"type_id": archetype, "slots": slots}
            expanded, provenance = self._expander.expand(card_dict)
            if provenance:
                self.warnings.append(f"Expanded {archetype} -> {expanded['type_id']}")
                archetype = expanded["type_id"]
                slots = expanded["slots"]

        handlers = {
            "exit.rule_trigger": self._exit_rule_trigger,
            "exit.trailing_stop": self._exit_trailing_stop,
            "exit.band_exit": self._exit_band,
        }

        handler = handlers.get(archetype)
        if handler:
            return handler(slots)
        else:
            self.warnings.append(f"Unsupported exit archetype: {archetype}")
            return None

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
                # Note: For percentage-based stops, we'd need portfolio access
                # For now, use a placeholder condition
                self.warnings.append("Percentage-based TP/SL not yet implemented in IR")
                return None
            else:
                self.warnings.append("Could not translate condition for exit.rule_trigger")
                return None

        self._exit_counter += 1

        return ExitRule(
            id=f"exit_{self._exit_counter}",
            condition=condition,
            action=LiquidateAction(),
            priority=self._exit_counter,
        )

    def _exit_trailing_stop(self, slots: dict[str, Any]) -> ExitRule | None:
        """Translate exit.trailing_stop archetype."""
        event = slots.get("event", {})

        trail_band = event.get("trail_band", {})
        band_length = trail_band.get("length", 20)
        band_mult = trail_band.get("mult", 2.0)

        # Add ATR indicator
        self._add_indicator(ATR(id="atr", period=band_length))

        # Add state for tracking highest since entry
        self._add_state("highest_since_entry", StateType.FLOAT, None)

        # Trailing stop: exit if price < highest - (mult * ATR)
        condition = CompareCondition(
            left=PriceValue(field=PriceField.CLOSE),
            op=CompareOp.LT,
            right=ExpressionValue(
                op="-",
                left=IndicatorValue(indicator_id="highest_since_entry_state"),
                # This is a simplification - in reality we'd use state value
                right=ExpressionValue(
                    op="*",
                    left=IndicatorValue(indicator_id="atr"),
                    right=LiteralValue(value=band_mult),
                ),
            ),
        )

        self._exit_counter += 1

        return ExitRule(
            id=f"trailing_stop_{self._exit_counter}",
            condition=condition,
            action=LiquidateAction(),
            priority=self._exit_counter,
        )

    def _exit_band(self, slots: dict[str, Any]) -> ExitRule | None:
        """Translate exit.band_exit archetype."""
        event = slots.get("event", {})

        exit_band = event.get("exit_band", {})
        exit_trigger = event.get("exit_trigger", {})

        band_type = exit_band.get("band", "bollinger")
        band_length = exit_band.get("length", 20)
        band_mult = exit_band.get("mult", 2.0)
        edge = exit_trigger.get("edge", "upper")

        # Add indicator
        if band_type == "bollinger":
            self._add_indicator(
                BollingerBands(id="exit_bb", period=band_length, multiplier=band_mult)
            )
            band_id = "exit_bb"
        else:
            self._add_indicator(
                KeltnerChannel(id="exit_kc", period=band_length, multiplier=band_mult)
            )
            band_id = "exit_kc"

        band_field = BandField.UPPER if edge == "upper" else BandField.LOWER
        op = CompareOp.GTE if edge == "upper" else CompareOp.LTE

        condition = CompareCondition(
            left=PriceValue(field=PriceField.CLOSE),
            op=op,
            right=IndicatorBandValue(indicator_id=band_id, band=band_field),
        )

        self._exit_counter += 1

        return ExitRule(
            id=f"band_exit_{self._exit_counter}",
            condition=condition,
            action=LiquidateAction(),
            priority=self._exit_counter,
        )

    # =========================================================================
    # Gate Translation
    # =========================================================================

    def _translate_gate(self, archetype: str, slots: dict[str, Any]) -> Gate | None:
        """Translate a gate card."""
        if archetype == "gate.regime":
            return self._gate_regime(slots)
        else:
            self.warnings.append(f"Unsupported gate archetype: {archetype}")
            return None

    def _gate_regime(self, slots: dict[str, Any]) -> Gate | None:
        """Translate gate.regime archetype."""
        event = slots.get("event", {})
        action = slots.get("action", {})

        # Gate schema uses "regime" for the condition spec, not "condition"
        condition_spec = event.get("regime", event.get("condition", {}))
        condition = self._translate_condition(condition_spec)

        if condition is None:
            return None

        mode = action.get("mode", "allow")
        target_roles = action.get("target_roles", ["entry"])

        return Gate(
            id=f"regime_gate_{len(self._indicators)}",
            condition=condition,
            mode=mode,
            target_roles=target_roles,
        )

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
            self._add_indicator(RateOfChange(id="roc", period=lookback))
            # Multiply ROC by 100 to convert to percentage
            return CompareCondition(
                left=ExpressionValue(
                    op="*",
                    left=IndicatorValue(indicator_id="roc"),
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
            self._add_indicator(EMA(id="ema_fast", period=fast))
            self._add_indicator(EMA(id="ema_slow", period=slow))

            # For "EMA fast > EMA slow", value=0 and op=">"
            if value == 0:
                return CompareCondition(
                    left=IndicatorValue(indicator_id="ema_fast"),
                    op=op,
                    right=IndicatorValue(indicator_id="ema_slow"),
                )
            else:
                # Compare difference to threshold
                return CompareCondition(
                    left=ExpressionValue(
                        op="-",
                        left=IndicatorValue(indicator_id="ema_fast"),
                        right=IndicatorValue(indicator_id="ema_slow"),
                    ),
                    op=op,
                    right=LiteralValue(value=float(value)),
                )

        elif metric == "trend_regime":
            # Trend regime classification: "up", "down", or numeric
            fast = regime.get("ma_fast", 20)
            slow = regime.get("ma_slow", 50)
            self._add_indicator(EMA(id="ema_fast", period=fast))
            self._add_indicator(EMA(id="ema_slow", period=slow))

            if value == "up" or (isinstance(value, (int, float)) and value > 0):
                return CompareCondition(
                    left=IndicatorValue(indicator_id="ema_fast"),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="ema_slow"),
                )
            elif value == "down" or (isinstance(value, (int, float)) and value < 0):
                return CompareCondition(
                    left=IndicatorValue(indicator_id="ema_fast"),
                    op=CompareOp.LT,
                    right=IndicatorValue(indicator_id="ema_slow"),
                )
            elif isinstance(value, (int, float)) and value == 0:
                if op in (CompareOp.GT, CompareOp.GTE):
                    return CompareCondition(
                        left=IndicatorValue(indicator_id="ema_fast"),
                        op=CompareOp.GT,
                        right=IndicatorValue(indicator_id="ema_slow"),
                    )
                else:
                    return CompareCondition(
                        left=IndicatorValue(indicator_id="ema_fast"),
                        op=CompareOp.LT,
                        right=IndicatorValue(indicator_id="ema_slow"),
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
            # Requires tracking previous close via rolling window
            self._add_indicator(RollingWindow(id="prev_close", period=2, field=PriceField.CLOSE))
            # The rolling window stores [current, previous] - we need index 1
            # For now, we use a state-based approach with the runtime
            # This metric requires special runtime support
            self.warnings.append(
                "gap_pct metric requires runtime state tracking. "
                "Ensure StrategyRuntime supports RollingWindow indicator."
            )
            return RegimeCondition(
                metric=metric,
                op=op,
                value=float(value) if isinstance(value, (int, float)) else value,
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
                self.warnings.append(f"Unknown session_phase value: {phase}")
                return None

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
                self.warnings.append("price_level_touch requires level_price or level_reference")
                return None

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
                self.warnings.append("price_level_cross requires level_price or level_reference")
                return None

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
            self.warnings.append(
                "risk_event_prob metric requires external calendar data. "
                "This metric will always pass."
            )
            return None  # Always pass - no filter

        else:
            # Unknown metric - warn and ignore
            self.warnings.append(f"Unknown metric '{metric}' - condition will always pass.")
            return None

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
        elif level_ref in ("recent_support", "recent_resistance", "session_low", "session_high"):
            # These require more complex level detection
            self.warnings.append(
                f"level_reference '{level_ref}' for price_level_touch requires runtime support"
            )
            return RegimeCondition(
                metric="price_level_touch",
                op=CompareOp.EQ,
                value=level_ref,
                lookback_bars=lookback,
            )
        else:
            self.warnings.append(f"Unknown level_reference: {level_ref}")
            return None

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
        elif level_ref in ("recent_support", "recent_resistance", "session_low", "session_high"):
            self.warnings.append(
                f"level_reference '{level_ref}' for price_level_cross requires runtime support"
            )
            return RegimeCondition(
                metric="price_level_cross",
                op=CompareOp.EQ,
                value=f"{level_ref}_{direction}",
                lookback_bars=lookback,
            )
        else:
            self.warnings.append(f"Unknown level_reference: {level_ref}")
            return None

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
            self.warnings.append(f"Unknown band_event kind: {kind}")
            return None

    def _create_band_indicator(self, band_spec: dict[str, Any]) -> str | None:
        """Create a band indicator and return its ID."""
        band_type = band_spec.get("band", "bollinger")
        length = band_spec.get("length", 20)
        mult = band_spec.get("mult", 2.0)

        # Generate unique ID based on parameters
        band_id = f"band_{band_type}_{length}_{mult}".replace(".", "_")

        if band_type == "bollinger":
            self._add_indicator(BollingerBands(id=band_id, period=length, multiplier=mult))
        elif band_type == "keltner":
            self._add_indicator(KeltnerChannel(id=band_id, period=length, multiplier=mult))
        elif band_type == "donchian":
            self._add_indicator(DonchianChannel(id=band_id, period=length))
        else:
            self.warnings.append(f"Unsupported band type: {band_type}")
            return None

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
            self.warnings.append(f"Unknown edge_event type: {event}")
            return None

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
            self.warnings.append("Sequence must have at least 2 steps")
            return None

        # Generate unique ID for this sequence
        seq_id = f"seq_{self._sequence_counter}"
        self._sequence_counter += 1

        # Translate each step's condition
        step_conditions: list[Condition] = []
        for i, step in enumerate(steps):
            cond = self._translate_condition(step.get("cond", {}))
            if cond is None:
                self.warnings.append(f"Failed to translate sequence step {i} condition")
                return None
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
        Returns a BreakoutCondition for runtime evaluation.
        """
        lookback_bars = spec.get("lookback_bars", 50)
        buffer_bps = spec.get("buffer_bps", 0)

        # Create MAX and MIN indicators for N-bar high/low
        max_id = "highest"
        min_id = "lowest"

        self._add_indicator(Maximum(id=max_id, period=lookback_bars))
        self._add_indicator(Minimum(id=min_id, period=lookback_bars))

        return BreakoutCondition(
            lookback_bars=lookback_bars,
            buffer_bps=buffer_bps,
            max_indicator=max_id,
            min_indicator=min_id,
        )

    def _translate_squeeze(self, spec: dict[str, Any]) -> Condition | None:
        """Translate a SqueezeSpec to IR Condition.

        Squeeze: BB width percentile below threshold indicates compression.
        """
        metric = spec.get("metric", "bb_width_pctile")
        pctile_min = spec.get("pctile_min", 10)
        break_rule = spec.get("break_rule", "donchian")
        with_trend = spec.get("with_trend", False)

        return SqueezeCondition(
            squeeze_metric=metric,
            pctile_threshold=float(pctile_min),
            break_rule=break_rule,
            with_trend=with_trend,
        )

    def _translate_time_filter(self, spec: dict[str, Any]) -> Condition | None:
        """Translate a TimeFilterSpec to IR Condition.

        Time filter: check day of week, time window, or day of month.
        """
        raw_days = spec.get("days_of_week", [])
        time_window = spec.get("time_window", "")
        days_of_month = spec.get("days_of_month", [])
        timezone = spec.get("timezone", "UTC")

        # Convert day names to integers if needed (0=Monday, 6=Sunday)
        day_name_map = {
            "monday": 0,
            "mon": 0,
            "tuesday": 1,
            "tue": 1,
            "wednesday": 2,
            "wed": 2,
            "thursday": 3,
            "thu": 3,
            "friday": 4,
            "fri": 4,
            "saturday": 5,
            "sat": 5,
            "sunday": 6,
            "sun": 6,
        }
        days_of_week = []
        for day in raw_days:
            if isinstance(day, int):
                days_of_week.append(day)
            elif isinstance(day, str):
                day_int = day_name_map.get(day.lower())
                if day_int is not None:
                    days_of_week.append(day_int)

        return TimeFilterCondition(
            days_of_week=days_of_week,
            time_window=time_window,
            days_of_month=days_of_month,
            timezone=timezone,
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
