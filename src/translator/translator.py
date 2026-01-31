"""New IRTranslator implementation with clean architecture.

Uses building blocks:
- CompilationContext for state accumulation
- Compiler package for condition building, indicator resolution, state collection
- RegimeLowerer visitor for lowering regime conditions
- ActionBuilder for building actions

Translation Flow:
    Strategy + Cards -> IRTranslator -> StrategyIR

For each enabled attachment:
1. Get card from cards dict
2. Parse archetype using parse_archetype
3. Build condition via compiler (build_condition)
4. Lower regime conditions (RegimeLowerer)
5. Resolve inline indicators (resolve_condition)
6. Collect state vars and hooks (collect_state)
7. Resolve state ops (resolve_state_ops)
8. Build appropriate rule type (EntryRule, ExitRule, GateRule, OverlayRule)
9. Return validated StrategyIR
"""

from __future__ import annotations

import logging
from typing import Any

from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.archetypes import parse_archetype
from vibe_trade_shared.models.archetypes.primitives import FixedTargetsSpec
from vibe_trade_shared.models.ir import Condition

from .builders import ActionBuilder
from .compiler.condition_builder import _compile_fixed_targets
from .compiler import (
    CompilationContext,
    build_condition,
    collect_state,
    resolve_condition,
    resolve_state_ops,
)
from .errors import TranslationError
from .ir import (
    EntryRule,
    ExitRule,
    GateRule,
    OverlayRule,
    Resolution,
    StateOp,
    StrategyIR,
)
from .ir_validator import validate_ir
from .visitors import RegimeLowerer

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
    - CompilationContext for accumulating state during translation
    - Compiler package for condition building, indicator resolution, state collection
    - RegimeLowerer for lowering regime conditions to primitives
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

        # Compilation context accumulates artifacts
        self.ctx = CompilationContext(symbol=self.symbol)

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

        # Track auto-generated risk exits from entry cards
        auto_risk_exits: list[ExitRule] = []

        # Process attachments by role
        for attachment in self.strategy.attachments:
            if not attachment.enabled:
                continue

            card = self.cards.get(attachment.card_id)
            if not card:
                raise TranslationError(f"Card not found: {attachment.card_id}")

            slots = card.slots

            # Dispatch based on role
            if attachment.role == "entry":
                translated = self._build_entry(card.type, slots)
                if translated:
                    entry = translated

                # Auto-generate exit rules from entry card's risk specification
                # This ensures sl_pct/tp_pct on entries actually work
                risk_exits = self._build_risk_exits_from_entry(slots)
                auto_risk_exits.extend(risk_exits)

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

        # Append auto-generated risk exits to explicit exits
        # They come after explicit exits so user-defined exits have priority
        exits.extend(auto_risk_exits)

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
        # Execution params (order_type, limit_price, etc.) handled by ActionBuilder.

        # close_confirm is a no-op: engine evaluates on bar close by default.
        # Reject truly unsupported confirm modes (future: "immediate" etc.)
        confirm = action_spec.get("confirm", "none")
        if confirm not in ("none", "close_confirm"):
            raise TranslationError(
                f"Entry confirm mode '{confirm}' is not yet supported."
            )

        # Map cooldown_bars → position_policy.min_bars_between
        cooldown_bars = action_spec.get("cooldown_bars")
        if cooldown_bars is not None:
            policy = action_spec.get("position_policy") or {}
            if isinstance(policy, dict):
                policy = {**policy}
            else:
                policy = {}
            if policy.get("min_bars_between") is None:
                policy["min_bars_between"] = cooldown_bars
            action_spec = {**action_spec, "position_policy": policy}

        # Map max_entries_per_day → position_policy.max_entries_per_day
        max_entries = action_spec.get("max_entries_per_day")
        if max_entries is not None:
            policy = action_spec.get("position_policy") or {}
            if isinstance(policy, dict):
                policy = {**policy}
            else:
                policy = {}
            if policy.get("max_entries_per_day") is None:
                policy["max_entries_per_day"] = max_entries
            action_spec = {**action_spec, "position_policy": policy}

        direction = action_spec.get("direction", "long")
        sizing = action_spec.get("sizing")

        holdings_action = ActionBuilder.build_holdings_action(action_spec, direction, sizing)

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

        # Build exit action from slots (supports partial exits via size_frac)
        action_spec = slots.get("action", {})

        # close_confirm is a no-op: engine evaluates on bar close by default.
        exit_confirm = action_spec.get("confirm", "none")
        if exit_confirm not in ("none", "close_confirm"):
            raise TranslationError(
                f"Exit confirm mode '{exit_confirm}' is not yet supported."
            )

        action = ActionBuilder.build_exit_action(action_spec)

        return ExitRule(
            id=f"exit_{self._exit_counter}",
            condition=condition,
            action=action,
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

        scale_size_frac = action_spec.get("scale_size_frac", 1.0)
        target_roles = action_spec.get("target_roles", ["entry", "exit"])

        return OverlayRule(
            id=f"overlay_{len(self.ctx.indicators)}",
            condition=condition,
            scale_size_frac=scale_size_frac,
            target_roles=target_roles,
        )

    def _build_risk_exits_from_entry(self, slots: dict[str, Any]) -> list[ExitRule]:
        """Auto-generate exit rules from entry card's risk specification.

        If the entry card has risk parameters (sl_pct, tp_pct, time_stop_bars),
        automatically create corresponding exit rules. This ensures user
        expectation that risk = protection is met.

        Args:
            slots: The entry card's slot values

        Returns:
            List of ExitRule objects (may be empty if no risk params set)
        """
        risk_spec = slots.get("risk")
        if not risk_spec:
            return []

        # Handle both dict and PositionRiskSpec-like objects
        if hasattr(risk_spec, "model_dump"):
            risk_spec = risk_spec.model_dump()

        # Extract risk parameters (percentage-based)
        tp_pct = risk_spec.get("tp_pct")
        sl_pct = risk_spec.get("sl_pct")
        time_stop_bars = risk_spec.get("time_stop_bars")

        # TODO: Future enhancement - handle tp_rr and sl_atr
        # These require additional context (entry price for RR, ATR indicator for ATR-based)
        # For now, log a warning if they're used
        if risk_spec.get("tp_rr") is not None or risk_spec.get("sl_atr") is not None:
            logger.warning(
                "Entry risk spec contains tp_rr or sl_atr which are not yet auto-translated. "
                "Use exit.fixed_targets or exit.trailing_stop cards for these features."
            )

        # Check if any percentage-based params are set
        if tp_pct is None and sl_pct is None and time_stop_bars is None:
            return []

        # Build FixedTargetsSpec from risk parameters
        try:
            spec = FixedTargetsSpec(
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                time_stop_bars=time_stop_bars,
            )
        except ValueError as e:
            logger.warning(f"Invalid risk spec, skipping auto-exit generation: {e}")
            return []

        # Compile to condition using existing infrastructure
        condition = _compile_fixed_targets(spec)

        # Build exit rule
        self._exit_counter += 1
        exit_rule = ExitRule(
            id=f"auto_risk_exit_{self._exit_counter}",
            condition=condition,
            action=ActionBuilder.build_exit_action({"mode": "close"}),
            priority=self._exit_counter,
        )

        logger.info(
            f"Auto-generated exit rule from entry risk spec: "
            f"tp={tp_pct}%, sl={sl_pct}%, time_stop={time_stop_bars}"
        )

        return [exit_rule]

    # =========================================================================
    # Core Translation
    # =========================================================================

    def _translate_archetype(
        self, archetype_id: str, slots: dict[str, Any]
    ) -> tuple[Condition | None, list[StateOp]]:
        """Translate an archetype to IR condition and state ops.

        This is the core translation method that uses the compiler package:
        1. Parses the archetype from slots
        2. Builds condition via build_condition
        3. Lowers regime conditions via RegimeLowerer
        4. Resolves inline IndicatorRefs via resolve_condition
        5. Collects state vars and hooks via collect_state
        6. Resolves state ops via resolve_state_ops

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

        # Step 2: Build condition via compiler
        condition = build_condition(typed_archetype, self.ctx)

        # Step 3: Lower regime conditions to primitives
        condition = RegimeLowerer().visit(condition)

        # Step 4: Resolve inline IndicatorRefs to indicator_id refs (typed walk)
        condition = resolve_condition(condition, self.ctx)

        # Step 5: Collect state vars and hooks from archetype + condition tree
        collect_state(typed_archetype, condition, self.ctx)

        # Step 6: Resolve inline indicators in on_fill_ops
        on_fill_ops = resolve_state_ops(typed_archetype.get_on_fill_ops(), self.ctx)

        logger.debug(f"Translated archetype {archetype_id}")
        return (condition, on_fill_ops)

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
