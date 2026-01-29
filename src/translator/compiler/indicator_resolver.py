"""Typed indicator resolver for IR condition trees.

Replaces the dict-mutation _resolve_inline_indicators() pattern with a typed
recursive walk that creates new Pydantic model instances instead of mutating dicts.

Key design rules:
- NO model_dump() anywhere
- Creates typed Indicator objects (EMA, BollingerBands, etc.)
- Registers indicators in CompilationContext
- Returns new typed Condition/ValueRef instances when changes are needed
- Also resolves inline indicators inside StateOp value fields
- Handles special condition types that implicitly require indicators
  (BreakoutCondition, SqueezeCondition, TrailingBreakoutCondition, etc.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_trade_shared.models.ir import (
    AllOfCondition,
    AnyOfCondition,
    BreakoutCondition,
    CompareCondition,
    Condition,
    CrossCondition,
    EventWindowCondition,
    FlagPatternCondition,
    GapCondition,
    IndicatorBandRef,
    IndicatorPropertyRef,
    IndicatorRef,
    IntermarketCondition,
    IRExpression,
    LiquiditySweepCondition,
    LiteralRef,
    MaxStateAction,
    MinStateAction,
    MultiLeaderIntermarketCondition,
    NotCondition,
    PennantPatternCondition,
    PriceRef,
    RegimeCondition,
    SequenceCondition,
    SequenceStep,
    SetStateAction,
    SetStateFromConditionAction,
    SpreadCondition,
    SqueezeCondition,
    StateCondition,
    StateOp,
    StateRef,
    TimeFilterCondition,
    TrailingBreakoutCondition,
    TrailingStateCondition,
    ValueRef,
    VolumeRef,
)
from vibe_trade_shared.models.ir.indicator_ids import indicator_id_from_type_params
from vibe_trade_shared.models.ir.indicators import (
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    Maximum,
    Minimum,
)

if TYPE_CHECKING:
    from vibe_trade_shared.models.ir.indicators import IndicatorUnion

    from src.translator.compiler.context import CompilationContext


# =============================================================================
# Indicator factory (creates typed Indicator from type string + params)
# =============================================================================

# Lazy import to avoid circular dependencies at module level
_INDICATOR_FACTORIES = None


def _get_indicator_factories() -> dict:
    """Get indicator factory registry, lazy-loaded."""
    global _INDICATOR_FACTORIES  # noqa: PLW0603
    if _INDICATOR_FACTORIES is None:
        from vibe_trade_shared.models.ir.indicators import (
            ADX,
            ATR,
            EMA,
            RSI,
            SMA,
            VWAP,
            AnchoredVWAP,
            Gap,
            Percentile,
            RateOfChange,
            RollingWindow,
            VolumeSMA,
        )

        _INDICATOR_FACTORIES = {
            "EMA": lambda ind_id, p: EMA(id=ind_id, period=p.get("period", 20)),
            "SMA": lambda ind_id, p: SMA(id=ind_id, period=p.get("period", 20)),
            "BB": lambda ind_id, p: BollingerBands(
                id=ind_id,
                period=p.get("period", 20),
                multiplier=p.get("multiplier", 2.0),
            ),
            "KC": lambda ind_id, p: KeltnerChannel(
                id=ind_id,
                period=p.get("period", 20),
                multiplier=p.get("multiplier", 2.0),
            ),
            "DC": lambda ind_id, p: DonchianChannel(id=ind_id, period=p.get("period", 20)),
            "ATR": lambda ind_id, p: ATR(id=ind_id, period=p.get("period", 14)),
            "RSI": lambda ind_id, p: RSI(id=ind_id, period=p.get("period", 14)),
            "MAX": lambda ind_id, p: Maximum(id=ind_id, period=p.get("period", 50)),
            "MIN": lambda ind_id, p: Minimum(id=ind_id, period=p.get("period", 50)),
            "ROC": lambda ind_id, p: RateOfChange(id=ind_id, period=p.get("period", 14)),
            "ADX": lambda ind_id, p: ADX(id=ind_id, period=p.get("period", 14)),
            "VWAP": lambda ind_id, p: VWAP(id=ind_id, period=p.get("period", 0)),
            "AVWAP": lambda ind_id, p: AnchoredVWAP(
                id=ind_id,
                anchor=p.get("anchor", "session"),
                anchor_datetime=p.get("anchor_datetime"),
            ),
            "PCTILE": lambda ind_id, p: Percentile(
                id=ind_id,
                period=p.get("period", 100),
                percentile=p.get("percentile", 10.0),
                source=p.get("source", "close"),
            ),
            "GAP": lambda ind_id, p: Gap(id=ind_id, session=p.get("session", "us")),
            "RW": lambda ind_id, p: RollingWindow(id=ind_id, period=p.get("period", 2)),
            "VOL_SMA": lambda ind_id, p: VolumeSMA(id=ind_id, period=p.get("period", 20)),
        }
    return _INDICATOR_FACTORIES


def _create_typed_indicator(ind_type: str, params: dict) -> IndicatorUnion:
    """Create a typed indicator from type string and params.

    Args:
        ind_type: Indicator type string (e.g., "EMA", "BB").
        params: Indicator parameters.

    Returns:
        Typed indicator instance.

    Raises:
        ValueError: If indicator type is unknown.
    """
    factories = _get_indicator_factories()
    factory = factories.get(ind_type.upper())
    if factory is None:
        raise ValueError(f"Unknown indicator type: {ind_type}")

    canonical_id = indicator_id_from_type_params(ind_type, params)
    return factory(canonical_id, params)


# =============================================================================
# Public API
# =============================================================================


def resolve_condition(cond: Condition, ctx: CompilationContext) -> Condition:
    """Walk typed condition tree, resolve inline IndicatorRefs to indicator_id refs.

    For each inline IndicatorRef (has indicator_type, no indicator_id):
    1. Create typed Indicator object (EMA, BollingerBands, etc.)
    2. Register in ctx.indicators
    3. Replace with IndicatorRef(indicator_id=...) or IndicatorBandRef for band fields

    Also handles special condition types that implicitly require indicators
    (BreakoutCondition, SqueezeCondition, etc.).

    Args:
        cond: The condition tree to resolve.
        ctx: Compilation context for indicator registration.

    Returns:
        New Condition with all inline indicators resolved.
    """
    # Composite conditions - recurse
    if isinstance(cond, AllOfCondition):
        new_children = [resolve_condition(c, ctx) for c in cond.conditions]
        return AllOfCondition(
            conditions=new_children,
            semantic_label=cond.semantic_label,
        )

    if isinstance(cond, AnyOfCondition):
        new_children = [resolve_condition(c, ctx) for c in cond.conditions]
        return AnyOfCondition(
            conditions=new_children,
            semantic_label=cond.semantic_label,
        )

    if isinstance(cond, NotCondition):
        return NotCondition(
            condition=resolve_condition(cond.condition, ctx),
            semantic_label=cond.semantic_label,
        )

    if isinstance(cond, SequenceCondition):
        new_steps = [
            SequenceStep(
                condition=resolve_condition(step.condition, ctx),
                hold_bars=step.hold_bars,
                within_bars=step.within_bars,
            )
            for step in cond.steps
        ]
        return SequenceCondition(
            steps=new_steps,
            semantic_label=cond.semantic_label,
        )

    # Leaf conditions with ValueRef fields
    if isinstance(cond, CompareCondition):
        new_left = resolve_value_ref(cond.left, ctx)
        new_right = resolve_value_ref(cond.right, ctx)
        if new_left is not cond.left or new_right is not cond.right:
            return CompareCondition(
                left=new_left,
                op=cond.op,
                right=new_right,
                semantic_label=cond.semantic_label,
            )
        return cond

    if isinstance(cond, CrossCondition):
        new_left = resolve_value_ref(cond.left, ctx)
        new_right = resolve_value_ref(cond.right, ctx)
        if new_left is not cond.left or new_right is not cond.right:
            return CrossCondition(
                left=new_left,
                right=new_right,
                direction=cond.direction,
                semantic_label=cond.semantic_label,
            )
        return cond

    if isinstance(cond, StateCondition):
        changed = False
        new_outside = cond.outside_condition
        new_inside = cond.inside_condition
        new_current = cond.current_condition

        if cond.outside_condition:
            new_outside = resolve_condition(cond.outside_condition, ctx)
            changed = changed or new_outside is not cond.outside_condition
        if cond.inside_condition:
            new_inside = resolve_condition(cond.inside_condition, ctx)
            changed = changed or new_inside is not cond.inside_condition
        if cond.current_condition:
            new_current = resolve_condition(cond.current_condition, ctx)
            changed = changed or new_current is not cond.current_condition

        if changed:
            return StateCondition(
                state_var=cond.state_var,
                trigger_on_transition=cond.trigger_on_transition,
                outside_condition=new_outside,
                inside_condition=new_inside,
                current_condition=new_current,
                semantic_label=cond.semantic_label,
            )
        return cond

    # Special conditions that implicitly require indicators
    if isinstance(cond, RegimeCondition):
        _register_regime_indicators(cond, ctx)
        return cond

    if isinstance(cond, BreakoutCondition):
        _register_breakout_indicators(cond, ctx)
        return cond

    if isinstance(cond, SqueezeCondition):
        _register_squeeze_indicators(cond, ctx)
        return cond

    if isinstance(cond, TrailingBreakoutCondition):
        _register_trailing_breakout_indicators(cond, ctx)
        return cond

    if isinstance(cond, SpreadCondition):
        _register_spread_symbols(cond, ctx)
        return cond

    if isinstance(cond, IntermarketCondition):
        _register_intermarket_symbols(cond, ctx)
        return cond

    if isinstance(cond, MultiLeaderIntermarketCondition):
        _register_multi_leader_symbols(cond, ctx)
        return cond

    # Pass-through conditions (no ValueRef fields to resolve)
    if isinstance(
        cond,
        (
            TimeFilterCondition,
            TrailingStateCondition,
            EventWindowCondition,
            GapCondition,
            LiquiditySweepCondition,
            FlagPatternCondition,
            PennantPatternCondition,
        ),
    ):
        return cond

    # Unknown condition type - return as-is
    return cond


def resolve_value_ref(ref: ValueRef, ctx: CompilationContext) -> ValueRef:
    """Resolve a single ValueRef, creating indicators for inline IndicatorRefs.

    Args:
        ref: The ValueRef to resolve.
        ctx: Compilation context.

    Returns:
        Resolved ValueRef (may be a new instance if changes were needed).
    """
    if isinstance(ref, IndicatorRef):
        return _resolve_indicator_ref(ref, ctx)

    if isinstance(ref, IRExpression):
        new_left = resolve_value_ref(ref.left, ctx)
        new_right = resolve_value_ref(ref.right, ctx)
        if new_left is not ref.left or new_right is not ref.right:
            return IRExpression(
                op=ref.op,
                left=new_left,
                right=new_right,
            )
        return ref

    # These types don't need resolution
    if isinstance(
        ref,
        (
            PriceRef,
            LiteralRef,
            VolumeRef,
            StateRef,
            IndicatorBandRef,
            IndicatorPropertyRef,
        ),
    ):
        return ref

    # Unknown ref type - return as-is
    return ref


def resolve_state_ops(ops: list[StateOp], ctx: CompilationContext) -> list[StateOp]:
    """Resolve inline indicators inside state operations.

    State operations can contain ValueRef fields that have inline indicators:
    - SetStateAction.value
    - MaxStateAction.value
    - MinStateAction.value
    - SetStateFromConditionAction.condition

    Args:
        ops: List of state operations to resolve.
        ctx: Compilation context.

    Returns:
        New list of state operations with resolved indicators.
    """
    result = []
    for op in ops:
        result.append(_resolve_state_op(op, ctx))
    return result


# =============================================================================
# Internal helpers
# =============================================================================


def _resolve_indicator_ref(ref: IndicatorRef, ctx: CompilationContext) -> ValueRef:
    """Resolve an inline IndicatorRef to a pre-declared reference.

    If the IndicatorRef has indicator_type (inline mode), creates a typed
    Indicator object, registers it, and returns a new IndicatorRef with
    indicator_id (or IndicatorBandRef for band fields).

    If the IndicatorRef already has indicator_id, returns it unchanged.

    Args:
        ref: IndicatorRef to resolve.
        ctx: Compilation context.

    Returns:
        Resolved ValueRef.
    """
    if ref.indicator_id is not None:
        # Already resolved - but check if field indicates a band ref
        return ref

    if ref.indicator_type is None:
        return ref

    # Inline mode: create typed indicator and register
    ind_type = ref.indicator_type.upper()
    params = ref.params or {}
    field = ref.field or "value"

    indicator = _create_typed_indicator(ind_type, params)
    ctx.add_indicator(indicator)

    # Band fields (upper/lower/middle) -> IndicatorBandRef
    if field in ("upper", "lower", "middle"):
        return IndicatorBandRef(
            indicator_id=indicator.id,
            band=field,
        )

    # Regular indicator value -> IndicatorRef with indicator_id
    return IndicatorRef(
        indicator_id=indicator.id,
        field=field,
    )


def _resolve_state_op(op: StateOp, ctx: CompilationContext) -> StateOp:
    """Resolve inline indicators in a single state operation.

    Args:
        op: State operation.
        ctx: Compilation context.

    Returns:
        Resolved state operation.
    """
    if isinstance(op, SetStateAction):
        new_value = resolve_value_ref(op.value, ctx)
        if new_value is not op.value:
            return SetStateAction(
                state_id=op.state_id,
                value=new_value,
            )
        return op

    if isinstance(op, MaxStateAction):
        new_value = resolve_value_ref(op.value, ctx)
        if new_value is not op.value:
            return MaxStateAction(
                state_id=op.state_id,
                value=new_value,
            )
        return op

    if isinstance(op, MinStateAction):
        new_value = resolve_value_ref(op.value, ctx)
        if new_value is not op.value:
            return MinStateAction(
                state_id=op.state_id,
                value=new_value,
            )
        return op

    if isinstance(op, SetStateFromConditionAction):
        new_cond = resolve_condition(op.condition, ctx)
        if new_cond is not op.condition:
            return SetStateFromConditionAction(
                state_id=op.state_id,
                condition=new_cond,
            )
        return op

    # IncrementStateAction and others don't have ValueRef fields to resolve
    return op


# =============================================================================
# Special condition indicator registration
# =============================================================================


def _register_regime_indicators(cond: RegimeCondition, ctx: CompilationContext) -> None:
    """Register indicators required by a RegimeCondition.

    Uses the regime registry from the execution service to determine which
    indicators are needed for this specific regime metric.

    Args:
        cond: RegimeCondition.
        ctx: Compilation context.
    """
    from src.translator.registries.regimes import get_regime_indicators

    try:
        indicators = get_regime_indicators(cond)
        for indicator in indicators:
            ctx.add_indicator(indicator)
    except (KeyError, ValueError):
        # Unknown metric - pass through for runtime handling
        pass


def _register_breakout_indicators(cond: BreakoutCondition, ctx: CompilationContext) -> None:
    """Register MAX/MIN indicators for BreakoutCondition.

    Args:
        cond: BreakoutCondition.
        ctx: Compilation context.
    """
    lookback = cond.lookback_bars
    ctx.add_indicator(Maximum(id=f"max_{lookback}", period=lookback))
    ctx.add_indicator(Minimum(id=f"min_{lookback}", period=lookback))


def _register_squeeze_indicators(cond: SqueezeCondition, ctx: CompilationContext) -> None:
    """Register BB/KC indicators for SqueezeCondition.

    Args:
        cond: SqueezeCondition.
        ctx: Compilation context.
    """
    period = 20
    ctx.add_indicator(BollingerBands(id=f"bb_{period}", period=period, multiplier=2.0))
    ctx.add_indicator(KeltnerChannel(id=f"kc_{period}", period=period, multiplier=2.0))


def _register_trailing_breakout_indicators(
    cond: TrailingBreakoutCondition, ctx: CompilationContext
) -> None:
    """Register band indicator for TrailingBreakoutCondition.

    Args:
        cond: TrailingBreakoutCondition.
        ctx: Compilation context.
    """
    period = cond.band_length
    band_type = cond.band_type

    if band_type == "bollinger":
        ctx.add_indicator(
            BollingerBands(
                id=f"bb_{period}",
                period=period,
                multiplier=cond.band_mult,
            )
        )
    elif band_type == "keltner":
        ctx.add_indicator(
            KeltnerChannel(
                id=f"kc_{period}",
                period=period,
                multiplier=cond.band_mult,
            )
        )
    elif band_type == "donchian":
        ctx.add_indicator(DonchianChannel(id=f"dc_{period}", period=period))


def _register_spread_symbols(cond: SpreadCondition, ctx: CompilationContext) -> None:
    """Register additional symbols from SpreadCondition.

    Args:
        cond: SpreadCondition.
        ctx: Compilation context.
    """
    if cond.symbol_a != ctx.symbol:
        ctx.add_additional_symbol(cond.symbol_a)
    if cond.symbol_b != ctx.symbol:
        ctx.add_additional_symbol(cond.symbol_b)


def _register_intermarket_symbols(cond: IntermarketCondition, ctx: CompilationContext) -> None:
    """Register leader symbol from IntermarketCondition.

    Args:
        cond: IntermarketCondition.
        ctx: Compilation context.
    """
    if cond.leader_symbol != ctx.symbol:
        ctx.add_additional_symbol(cond.leader_symbol)


def _register_multi_leader_symbols(
    cond: MultiLeaderIntermarketCondition, ctx: CompilationContext
) -> None:
    """Register leader symbols from MultiLeaderIntermarketCondition.

    Args:
        cond: MultiLeaderIntermarketCondition.
        ctx: Compilation context.
    """
    for symbol in cond.leader_symbols:
        if symbol != ctx.symbol:
            ctx.add_additional_symbol(symbol)
