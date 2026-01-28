"""New IRTranslator implementation with clean architecture.

Uses building blocks:
- TranslationContext for state accumulation
- ConditionPipeline for condition processing (lowering -> collection -> extraction)
- ActionBuilder for building actions
- Typed visitors instead of dict manipulation

Translation Flow:
    Strategy + Cards -> IRTranslator -> StrategyIR

For each enabled attachment:
1. Get card from cards dict
2. Parse archetype using parse_archetype
3. Call archetype.to_ir() to get condition
4. Process condition through ConditionPipeline
5. Build appropriate rule type (EntryRule, ExitRule, GateRule, OverlayRule)
6. Collect state vars and hooks from archetype
7. Return validated StrategyIR
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import TypeAdapter
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.archetypes import BaseArchetype, parse_archetype
from vibe_trade_shared.models.ir import Condition

from .builders import ActionBuilder
from .context import TranslationContext
from .ir import (
    EntryRule,
    ExitRule,
    GateRule,
    Indicator,
    LiquidateAction,
    OverlayRule,
    Resolution,
    StateOp,
    StrategyIR,
)
from .errors import TranslationError
from .ir_validator import validate_ir
from .pipeline import ConditionPipeline

logger = logging.getLogger(__name__)


# Map timeframe strings to Resolution enum
RESOLUTION_MAP = {
    "1m": Resolution.MINUTE,
    "1min": Resolution.MINUTE,
    "minute": Resolution.MINUTE,
    "5m": Resolution.MINUTE,
    "15m": Resolution.MINUTE,
    "1h": Resolution.HOUR,
    "hour": Resolution.HOUR,
    "4h": Resolution.HOUR,
    "1d": Resolution.DAILY,
    "daily": Resolution.DAILY,
}


class IRTranslator:
    """Translates vibe-trade strategies to StrategyIR using clean architecture.

    This implementation uses:
    - TranslationContext for accumulating state during translation
    - ConditionPipeline for processing conditions through lowering, collection, extraction
    - ActionBuilder for building action objects

    Example:
        translator = IRTranslator(strategy, cards)
        ir = translator.translate()
    """

    def __init__(self, strategy: Strategy, cards: dict[str, Card]) -> None:
        """Initialize translator.

        Args:
            strategy: The Strategy model with attachments
            cards: Dict mapping card_id to Card objects
        """
        self.strategy = strategy
        self.cards = cards

        # Get first symbol from universe (MVP: single asset)
        self.symbol = strategy.universe[0] if strategy.universe else "BTC-USD"

        # Translation context accumulates artifacts
        self.ctx = TranslationContext(symbol=self.symbol)

        # Pipeline for processing conditions
        self.pipeline = ConditionPipeline(self.ctx)

        # Exit counter for unique IDs
        self._exit_counter = 0

    def translate(self) -> StrategyIR:
        """Translate the strategy to IR.

        Returns:
            StrategyIR - valid, executable strategy IR

        Raises:
            TranslationError: If translation fails for any reason
        """
        entry: EntryRule | None = None
        exits: list[ExitRule] = []
        gates: list[GateRule] = []
        overlays: list[OverlayRule] = []

        # Process attachments by role
        for attachment in self.strategy.attachments:
            if not attachment.enabled:
                continue

            card = self.cards.get(attachment.card_id)
            if not card:
                raise TranslationError(f"Card not found: {attachment.card_id}")

            # Merge card slots with attachment overrides
            slots = {**card.slots, **attachment.overrides}

            # Dispatch based on role
            if attachment.role == "entry":
                translated = self._build_entry(card.type, slots)
                if translated:
                    entry = translated
            elif attachment.role == "exit":
                translated = self._build_exit(card.type, slots)
                if translated:
                    exits.append(translated)
            elif attachment.role == "gate":
                translated = self._build_gate(card.type, slots)
                if translated:
                    gates.append(translated)
            elif attachment.role == "overlay":
                translated = self._build_overlay(card.type, slots)
                if translated:
                    overlays.append(translated)

        # Determine resolution from first entry card's context
        resolution = self._determine_resolution()

        # Merge on_fill ops from exits into entry rule
        # This is critical for trailing stops which need state init on entry fill
        if entry is not None and self.ctx.on_fill_ops:
            entry = self._merge_on_fill_ops(entry)

        ir = StrategyIR(
            strategy_id=self.strategy.id,
            strategy_name=self.strategy.name,
            symbol=self.symbol,
            resolution=resolution,
            additional_symbols=self.ctx.additional_symbols,
            indicators=list(self.ctx.indicators.values()),
            state=list(self.ctx.state_vars.values()),
            gates=gates,
            overlays=overlays,
            entry=entry,
            exits=exits,
            on_bar=self.ctx.on_bar_hooks,
            on_bar_invested=self.ctx.on_bar_invested_ops,
        )

        # Validate IR for referential integrity
        validation_result = validate_ir(ir)
        if not validation_result.is_valid:
            errors = "; ".join(f"{e.path}: {e.message}" for e in validation_result.errors)
            raise TranslationError(f"Invalid IR produced: {errors}")

        return ir

    # =========================================================================
    # Rule Builders
    # =========================================================================

    def _build_entry(self, archetype: str, slots: dict[str, Any]) -> EntryRule | None:
        """Build an EntryRule from archetype and slots.

        Args:
            archetype: The archetype type_id (e.g., "entry.rule_trigger")
            slots: The slot values with any overrides merged

        Returns:
            EntryRule ready for IR, or None if translation fails
        """
        condition, on_fill_ops = self._translate_archetype(archetype, slots)
        if condition is None:
            raise TranslationError(f"Unsupported entry archetype: {archetype}")

        # Get action config and build holdings action
        action_spec = slots.get("action", {})
        direction = action_spec.get("direction", "long")
        sizing = action_spec.get("sizing")

        holdings_action = ActionBuilder.build_holdings_action(
            action_spec, direction, sizing
        )

        return EntryRule(
            condition=condition,
            action=holdings_action,
            on_fill=on_fill_ops,
        )

    def _build_exit(self, archetype: str, slots: dict[str, Any]) -> ExitRule | None:
        """Build an ExitRule from archetype and slots.

        Args:
            archetype: The archetype type_id (e.g., "exit.trailing_stop")
            slots: The slot values with any overrides merged

        Returns:
            ExitRule ready for IR, or None if translation fails
        """
        condition, _ = self._translate_archetype(archetype, slots)
        if condition is None:
            raise TranslationError(f"Unsupported exit archetype: {archetype}")

        self._exit_counter += 1
        return ExitRule(
            id=f"exit_{self._exit_counter}",
            condition=condition,
            action=LiquidateAction(),
            priority=self._exit_counter,
        )

    def _build_gate(self, archetype: str, slots: dict[str, Any]) -> GateRule | None:
        """Build a GateRule from archetype and slots.

        Args:
            archetype: The archetype type_id (e.g., "gate.regime")
            slots: The slot values with any overrides merged

        Returns:
            GateRule ready for IR, or None if translation fails
        """
        condition, _ = self._translate_archetype(archetype, slots)
        if condition is None:
            raise TranslationError(f"Unsupported gate archetype: {archetype}")

        action_spec = slots.get("action", {})
        mode = action_spec.get("mode", "allow")
        target_roles = action_spec.get("target_roles", ["entry"])

        return GateRule(
            id=f"gate_{len(self.ctx.indicators)}",
            condition=condition,
            mode=mode,
            target_roles=target_roles,
        )

    def _build_overlay(self, archetype: str, slots: dict[str, Any]) -> OverlayRule | None:
        """Build an OverlayRule from archetype and slots.

        Args:
            archetype: The archetype type_id (e.g., "overlay.regime_scaler")
            slots: The slot values with any overrides merged

        Returns:
            OverlayRule ready for IR, or None if translation fails
        """
        condition, _ = self._translate_archetype(archetype, slots)
        if condition is None:
            raise TranslationError(f"Unsupported overlay archetype: {archetype}")

        action_spec = slots.get("action", {})
        scale_risk_frac = action_spec.get("scale_risk_frac", 1.0)
        scale_size_frac = action_spec.get("scale_size_frac", 1.0)
        target_roles = action_spec.get("target_roles", ["entry", "exit"])
        target_tags = action_spec.get("target_tags", [])
        target_ids = action_spec.get("target_ids", [])

        return OverlayRule(
            id=f"overlay_{len(self.ctx.indicators)}",
            condition=condition,
            scale_risk_frac=scale_risk_frac,
            scale_size_frac=scale_size_frac,
            target_roles=target_roles,
            target_tags=target_tags,
            target_ids=target_ids,
        )

    # =========================================================================
    # Core Translation
    # =========================================================================

    def _translate_archetype(
        self, archetype_id: str, slots: dict[str, Any]
    ) -> tuple[Condition | None, list[StateOp]]:
        """Translate an archetype to IR condition and state ops.

        This is the core translation method that:
        1. Parses the archetype from slots
        2. Gets the IR condition via to_ir()
        3. Processes the condition through the pipeline (lowering, collection, extraction)
        4. Resolves inline IndicatorRefs to indicator_id refs
        5. Collects state vars and hooks from the archetype

        Args:
            archetype_id: The archetype type_id (e.g., "entry.rule_trigger")
            slots: The slot values

        Returns:
            Tuple of (condition, on_fill_ops)

        Raises:
            TranslationError: If archetype parsing or translation fails
        """
        from pydantic import ValidationError

        # Step 1: Parse archetype
        try:
            typed_archetype = parse_archetype(archetype_id, slots)
        except KeyError as e:
            raise TranslationError(f"Unknown archetype: {archetype_id}") from e
        except ValidationError as e:
            raise TranslationError(f"Invalid slots for {archetype_id}: {e}") from e

        # Step 2: Get IR condition from archetype
        ir_condition = typed_archetype.to_ir()
        if ir_condition is None:
            return (None, [])

        # Step 3: Process condition through pipeline
        # The pipeline lowers regime conditions, collects indicators, extracts state
        processed_condition = self.pipeline.process(ir_condition)

        # Step 4: Resolve inline IndicatorRefs to indicator_id refs
        # Convert to dict, process inline indicators, convert back to typed Condition
        resolved_condition = self._resolve_inline_indicators(processed_condition)

        # Step 5: Collect state and hooks from archetype
        self._collect_archetype_state(typed_archetype)

        # Step 6: Get on_fill_ops from archetype
        on_fill_ops = self._convert_state_ops(typed_archetype.get_on_fill_ops())

        logger.debug(f"Translated archetype {archetype_id}")
        return (resolved_condition, on_fill_ops)

    def _collect_archetype_state(self, archetype: BaseArchetype) -> None:
        """Collect state variables and hooks from archetype.

        Args:
            archetype: The parsed archetype instance
        """
        # Collect state variable declarations
        for state_var in archetype.get_state_vars():
            self.ctx.add_state_var(state_var)

        # Collect on_bar hooks
        on_bar_ops = self._convert_state_ops(archetype.get_on_bar_ops())
        self.ctx.on_bar_hooks.extend(on_bar_ops)

        # Collect on_bar_invested hooks
        on_bar_invested_ops = self._convert_state_ops(archetype.get_on_bar_invested_ops())
        self.ctx.on_bar_invested_ops.extend(on_bar_invested_ops)

        # Collect on_fill ops for merging into entry rule later
        on_fill_ops = self._convert_state_ops(archetype.get_on_fill_ops())
        self.ctx.on_fill_ops.extend(on_fill_ops)

    def _convert_state_ops(self, shared_ops: list[Any]) -> list[StateOp]:
        """Convert shared library StateOps to local IR StateOps.

        The shared library uses different class names but compatible serialization.

        Args:
            shared_ops: List of state operations from archetype

        Returns:
            List of local IR StateOp objects
        """
        if not shared_ops:
            return []

        result = []
        adapter = TypeAdapter(StateOp)
        for op in shared_ops:
            op_dict = op.model_dump()
            # Process any inline indicators in the op's value
            self._process_inline_indicators_in_dict(op_dict)
            result.append(adapter.validate_python(op_dict))
        return result

    def _resolve_inline_indicators(self, condition: Condition) -> Condition:
        """Resolve inline IndicatorRefs to indicator_id references.

        Converts condition to dict, processes inline indicators (which creates
        indicators and sets indicator_id), then parses back to typed Condition.

        Args:
            condition: The condition with possibly inline IndicatorRefs

        Returns:
            Condition with all IndicatorRefs resolved to indicator_id refs
        """
        # Convert to dict
        condition_dict = condition.model_dump()

        # Process inline indicators recursively
        self._process_inline_indicators_in_dict(condition_dict)

        # Parse back to typed Condition
        adapter = TypeAdapter(Condition)
        return adapter.validate_python(condition_dict)

    def _process_inline_indicators_in_dict(self, obj: dict[str, Any]) -> None:
        """Process inline IndicatorRefs in a dict, registering them as indicators.

        This handles the pattern where IndicatorRef has inline indicator_type + params,
        which need to be converted to indicator_id references.

        Args:
            obj: Dict to process (mutated in place)
        """
        if not isinstance(obj, dict):
            return

        # Remove semantic_label (shared library debug field)
        if "semantic_label" in obj:
            del obj["semantic_label"]

        # Check for inline IndicatorRef pattern
        if obj.get("type") == "indicator" and obj.get("indicator_type") is not None:
            ind_type = obj["indicator_type"]
            params = obj.get("params") or {}
            field = obj.get("field", "value")

            # Create the indicator and register it
            indicator = self._create_indicator(ind_type, params)
            if indicator:
                self.ctx.add_indicator(indicator)

                # Handle band fields specially
                if field in ("upper", "middle", "lower"):
                    obj.clear()
                    obj["type"] = "indicator_band"
                    obj["indicator_id"] = indicator.id
                    obj["band"] = field
                else:
                    obj.clear()
                    obj["type"] = "indicator"
                    obj["indicator_id"] = indicator.id
                    obj["field"] = field
        else:
            # Handle indicator_band references - create the band indicator if not exists
            # This handles references like {"type": "indicator_band", "indicator_id": "bollinger_20", "band": "lower"}
            if obj.get("type") == "indicator_band":
                ind_id = obj.get("indicator_id", "")
                # Parse the indicator ID to determine band type and period
                # Format: "{band_type}_{period}" e.g., "bollinger_20", "keltner_20"
                if ind_id and "_" in ind_id and ind_id not in self.ctx.indicators:
                    parts = ind_id.rsplit("_", 1)
                    if len(parts) == 2:
                        band_type, period_str = parts
                        try:
                            period = int(period_str)
                            indicator = self._create_band_indicator(band_type, ind_id, period)
                            if indicator:
                                self.ctx.add_indicator(indicator)
                        except ValueError:
                            pass  # Invalid period, skip

            # Recurse into nested dicts and lists
            for value in list(obj.values()):
                if isinstance(value, dict):
                    self._process_inline_indicators_in_dict(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self._process_inline_indicators_in_dict(item)

    def _create_indicator(self, ind_type: str, params: dict[str, Any]) -> Indicator | None:
        """Create an Indicator from type and params.

        Args:
            ind_type: Indicator type string (e.g., "EMA", "ATR")
            params: Indicator parameters

        Returns:
            Typed Indicator instance, or None if type unknown
        """
        from .ir import (
            ADX,
            ATR,
            EMA,
            RSI,
            SMA,
            VWAP,
            AnchoredVWAP,
            BollingerBands,
            DonchianChannel,
            KeltnerChannel,
            Maximum,
            Minimum,
            Percentile,
            RateOfChange,
        )

        # Generate indicator ID
        ind_id = self._generate_indicator_id(ind_type, params)

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
            "RSI": lambda: RSI(id=ind_id, period=params.get("period", 14)),
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
        }

        factory = type_map.get(ind_type.upper())
        if factory:
            return factory()

        logger.warning(f"Unknown indicator type: {ind_type}")
        return None

    def _generate_indicator_id(self, ind_type: str, params: dict[str, Any]) -> str:
        """Generate a deterministic indicator ID from type and params.

        Args:
            ind_type: Indicator type (e.g., "EMA")
            params: Indicator parameters

        Returns:
            Unique indicator ID string
        """
        parts = [ind_type.lower()]
        for key in sorted(params.keys()):
            val = params[key]
            if isinstance(val, float):
                val_str = str(val).replace(".", "_")
            else:
                val_str = str(val)
            parts.append(val_str)
        return "_".join(parts)

    def _create_band_indicator(self, band_type: str, ind_id: str, period: int) -> Indicator | None:
        """Create a band indicator from type, id, and period.

        Args:
            band_type: Band type string (e.g., "bollinger", "keltner", "donchian")
            ind_id: The indicator ID to use
            period: The period for the indicator

        Returns:
            Typed Indicator instance, or None if type unknown
        """
        from .ir import (
            BollingerBands,
            DonchianChannel,
            KeltnerChannel,
        )

        if band_type == "bollinger":
            return BollingerBands(id=ind_id, period=period, num_std_dev=2.0)
        elif band_type == "keltner":
            return KeltnerChannel(id=ind_id, period=period, multiplier=2.0)
        elif band_type == "donchian":
            return DonchianChannel(id=ind_id, period=period)

        logger.warning(f"Unknown band type: {band_type}")
        return None

    # =========================================================================
    # Helpers
    # =========================================================================

    def _determine_resolution(self) -> Resolution:
        """Determine resolution from entry card's context.

        Returns:
            Resolution enum value
        """
        for attachment in self.strategy.attachments:
            if attachment.role == "entry" and attachment.enabled:
                card = self.cards.get(attachment.card_id)
                if card:
                    context = card.slots.get("context", {})
                    tf = context.get("tf", "1h")
                    return RESOLUTION_MAP.get(tf.lower(), Resolution.HOUR)
        return Resolution.HOUR

    def _merge_on_fill_ops(self, entry: EntryRule) -> EntryRule:
        """Merge accumulated on_fill_ops from exits into entry rule.

        Trailing stops and other exits may need state initialized on entry fill.
        This deduplicates ops by state_id.

        Args:
            entry: The entry rule to merge into

        Returns:
            New EntryRule with merged on_fill ops
        """
        seen_ops: set[str] = set()
        merged_on_fill: list[StateOp] = []

        # Entry's ops first
        for op in entry.on_fill:
            op_key = f"{op.type}:{op.state_id}"
            if op_key not in seen_ops:
                seen_ops.add(op_key)
                merged_on_fill.append(op)

        # Then accumulated ops from exits
        for op in self.ctx.on_fill_ops:
            op_key = f"{op.type}:{op.state_id}"
            if op_key not in seen_ops:
                seen_ops.add(op_key)
                merged_on_fill.append(op)

        return EntryRule(
            condition=entry.condition,
            action=entry.action,
            on_fill=merged_on_fill,
        )
