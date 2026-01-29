"""Condition builder for compiling archetypes to typed IR conditions.

This module extracts the condition compilation logic from two sources:

1. ConditionSpec.to_ir() in vibe-trade-shared/models/archetypes/primitives.py
   - Handles all ConditionSpec condition types (regime, band_event, breakout, etc.)

2. Direct-IR archetypes that bypass ConditionSpec
   - TrailingStop, BreakoutRetest, etc. build Condition directly

Key differences from the original ConditionSpec.to_ir():
- Returns TYPED ValueRef objects, never .model_dump() dicts
- Creates typed Indicator objects with full params (e.g., BollingerBands with multiplier)
- Registers indicators in CompilationContext during band event expansion
- Band indicators use canonical IDs via indicator_id_from_type_params
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_trade_shared.models.archetypes.base import BaseArchetype
from vibe_trade_shared.models.archetypes.primitives import (
    BandEventSpec,
    BandSpec,
    ConditionSpec,
    SignalRef,
)
from vibe_trade_shared.models.ir import (
    AllOfCondition,
    AnyOfCondition,
    BreakoutCondition,
    CompareCondition,
    CompareOp,
    Condition,
    CrossCondition,
    EventWindowCondition,
    GapCondition,
    IndicatorBandRef,
    IndicatorPropertyRef,
    IndicatorRef,
    IntermarketCondition,
    IRExpression,
    LiteralRef,
    MultiLeaderIntermarketCondition,
    NotCondition,
    PriceRef,
    RegimeCondition,
    SequenceCondition,
    SequenceStep,
    SpreadCondition,
    SqueezeCondition,
    StateCondition,
    StateRef,
    TimeFilterCondition,
    TrailingBreakoutCondition,
    TrailingStateCondition,
    ValueRef,
)
from vibe_trade_shared.models.ir.enums import IndicatorProperty, PriceField
from vibe_trade_shared.models.ir.indicators import (
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    VWAPBands,
)

if TYPE_CHECKING:
    from src.translator.compiler.context import CompilationContext


# =============================================================================
# Field maps
# =============================================================================

_PRICE_FIELD_MAP = {
    "open": PriceField.OPEN,
    "high": PriceField.HIGH,
    "low": PriceField.LOW,
    "close": PriceField.CLOSE,
}

_COMPARE_OP_MAP = {
    ">": CompareOp.GT,
    ">=": CompareOp.GTE,
    "<": CompareOp.LT,
    "<=": CompareOp.LTE,
    "==": CompareOp.EQ,
    "!=": CompareOp.NEQ,
}

_DAY_MAP = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


# =============================================================================
# Public API
# =============================================================================


def build_condition(archetype: BaseArchetype, ctx: CompilationContext) -> Condition:
    """Build IR Condition from an archetype.

    Dispatches: if archetype has to_condition_spec(), compile that.
    Otherwise, fall back to archetype.to_ir() (for direct-IR archetypes
    like TrailingStop).

    Args:
        archetype: Parsed archetype instance.
        ctx: Compilation context for registering indicators, state, etc.

    Returns:
        Typed IR Condition.

    Raises:
        ValueError: If archetype cannot be compiled.
    """
    if hasattr(archetype, "to_condition_spec"):
        try:
            spec = archetype.to_condition_spec()
        except (AttributeError, NotImplementedError):
            spec = None
        if spec is not None:
            return compile_condition_spec(spec, ctx)
    # Fall back to direct IR
    return _dispatch_direct_builder(archetype, ctx)


def compile_condition_spec(spec: ConditionSpec, ctx: CompilationContext) -> Condition:
    """Compile a ConditionSpec to a typed IR Condition.

    This is the extraction of ConditionSpec.to_ir() from shared/primitives.py.
    Key differences from the original:
    - Returns typed ValueRef objects, never model_dump() dicts
    - Creates typed Indicator objects (BollingerBands, EMA, etc.) with full params
    - Registers indicators in ctx immediately during band event expansion

    Args:
        spec: The ConditionSpec to compile.
        ctx: Compilation context.

    Returns:
        Typed IR Condition.

    Raises:
        ValueError: If condition type is unknown.
    """
    # Composite conditions - recurse
    if spec.type == "allOf" and spec.allOf:
        return AllOfCondition(
            conditions=[compile_condition_spec(c, ctx) for c in spec.allOf],
            semantic_label=f"allOf[{len(spec.allOf)}]",
        )

    if spec.type == "anyOf" and spec.anyOf:
        return AnyOfCondition(
            conditions=[compile_condition_spec(c, ctx) for c in spec.anyOf],
            semantic_label=f"anyOf[{len(spec.anyOf)}]",
        )

    if spec.type == "not" and spec.not_:
        return NotCondition(
            condition=compile_condition_spec(spec.not_, ctx),
            semantic_label="not",
        )

    if spec.type == "sequence" and spec.sequence:
        return SequenceCondition(
            steps=[
                SequenceStep(
                    condition=compile_condition_spec(step.cond, ctx),
                    within_bars=step.within_bars,
                )
                for step in spec.sequence
            ],
            semantic_label=f"sequence[{len(spec.sequence)}]",
        )

    # Atomic conditions
    if spec.type == "regime" and spec.regime:
        return _compile_regime(spec, ctx)

    if spec.type == "band_event" and spec.band_event:
        return _compile_band_event(spec.band_event, ctx)

    if spec.type == "breakout" and spec.breakout:
        b = spec.breakout
        return BreakoutCondition(
            lookback_bars=b.lookback_bars,
            buffer_bps=int(b.buffer_bps),
            semantic_label=f"breakout: {b.lookback_bars}-bar",
        )

    if spec.type == "squeeze" and spec.squeeze:
        s = spec.squeeze
        return SqueezeCondition(
            squeeze_metric=s.metric,
            pctile_threshold=float(s.pctile_min),
            break_rule=s.break_rule,
            with_trend=s.with_trend,
        )

    if spec.type == "time_filter" and spec.time_filter:
        return _compile_time_filter(spec)

    if spec.type == "cross" and spec.cross:
        return _compile_cross(spec, ctx)

    if spec.type == "compare" and spec.compare:
        return _compile_compare(spec, ctx)

    if spec.type == "trailing_state" and spec.trailing_state:
        return _compile_trailing_state(spec)

    if spec.type == "spread" and spec.spread:
        return _compile_spread(spec, ctx)

    if spec.type == "event" and spec.event:
        return _compile_event(spec)

    raise ValueError(f"Unknown condition type: {spec.type}")


# =============================================================================
# Signal ref conversion (typed, no model_dump())
# =============================================================================


def _signal_ref_to_value_ref(ref: SignalRef | float | int) -> ValueRef:
    """Convert archetype SignalRef to typed IR ValueRef.

    Returns typed ValueRef objects instead of .model_dump() dicts.

    Args:
        ref: SignalRef or numeric literal.

    Returns:
        Typed ValueRef instance.
    """
    if isinstance(ref, (int, float)):
        return LiteralRef(value=float(ref))

    if ref.type == "price":
        field = _PRICE_FIELD_MAP.get(ref.field, PriceField.CLOSE) if ref.field else PriceField.CLOSE
        return PriceRef(field=field)

    if ref.type == "indicator":
        ind_type = ref.indicator.upper() if ref.indicator else "EMA"
        period = ref.period or 20
        params: dict[str, int] = {"period": period}

        # Handle band indicators (bb_upper, bb_lower, bb_mid)
        if ind_type in ("BB_UPPER", "BB_LOWER", "BB_MID"):
            field_name = ind_type.split("_")[1].lower()
            return IndicatorRef(
                indicator_type="BB",
                params=params,
                field=field_name,
            )

        return IndicatorRef(
            indicator_type=ind_type,
            params=params,
        )

    if ref.type == "constant":
        return LiteralRef(value=float(ref.value or 0.0))

    raise ValueError(f"Unknown SignalRef type: {ref.type}")


# =============================================================================
# Band event compilation (typed indicators, canonical IDs)
# =============================================================================


def _make_band_indicator(band: BandSpec, ctx: CompilationContext) -> str:
    """Create and register a typed band indicator in the context.

    Returns the canonical indicator ID. Uses full params (including multiplier)
    for correct canonical IDs, unlike the old f"{band.band}_{band.length}" format.

    Args:
        band: BandSpec with band type, length, and multiplier.
        ctx: Compilation context to register the indicator.

    Returns:
        Canonical indicator ID string.
    """
    from vibe_trade_shared.models.ir.indicators import indicator_id_from_spec

    if band.band == "bollinger":
        indicator = BollingerBands(
            id="",
            period=band.length,
            multiplier=band.mult if band.mult else 2.0,
        )
    elif band.band == "keltner":
        indicator = KeltnerChannel(
            id="",
            period=band.length,
            multiplier=band.mult if band.mult else 2.0,
        )
    elif band.band == "donchian":
        indicator = DonchianChannel(id="", period=band.length)
    elif band.band == "vwap_band":
        indicator = VWAPBands(
            id="",
            anchor=band.anchor or "session",
            multiplier=band.mult if band.mult else 2.0,
        )
    else:
        raise ValueError(f"Unknown band type: {band.band}")

    # Set canonical ID and register
    canonical_id = indicator_id_from_spec(indicator)
    indicator = indicator.model_copy(update={"id": canonical_id})
    ctx.add_indicator(indicator)
    return canonical_id


def _make_band_ref(band_id: str, edge: str) -> IndicatorBandRef:
    """Create IndicatorBandRef for a band edge.

    Args:
        band_id: The indicator ID.
        edge: Edge name ("upper", "lower", "mid").

    Returns:
        IndicatorBandRef instance.
    """
    band_map = {"upper": "upper", "lower": "lower", "mid": "middle"}
    return IndicatorBandRef(
        indicator_id=band_id,
        band=band_map.get(edge, "lower"),
    )


def _compile_band_event(be: BandEventSpec, ctx: CompilationContext) -> Condition:
    """Compile a band event specification to typed IR.

    Handles all band event kinds: edge_event (touch, cross_in, cross_out),
    reentry, and distance.

    Args:
        be: BandEventSpec to compile.
        ctx: Compilation context.

    Returns:
        Typed IR Condition.
    """
    band = be.band
    band_id = _make_band_indicator(band, ctx)

    if be.kind == "edge_event" and be.edge and be.event:
        band_ref = _make_band_ref(band_id, be.edge)

        if be.event == "touch":
            # Touch: bar range intersects the band line
            # bar.low <= band AND bar.high >= band
            return AllOfCondition(
                conditions=[
                    CompareCondition(
                        left=PriceRef(field=PriceField.LOW),
                        op=CompareOp.LTE,
                        right=band_ref,
                        semantic_label=f"low <= {be.edge} band",
                    ),
                    CompareCondition(
                        left=PriceRef(field=PriceField.HIGH),
                        op=CompareOp.GTE,
                        right=band_ref,
                        semantic_label=f"high >= {be.edge} band",
                    ),
                ],
                semantic_label=f"touch {band.band} {be.edge}",
            )

        if be.event == "cross_in":
            # Cross in: price crosses into the band from outside
            # For lower band: was below, now above
            # For upper band: was above, now below
            direction = "above" if be.edge == "lower" else "below"
            return CrossCondition(
                left=PriceRef(field=PriceField.CLOSE),
                right=band_ref,
                direction=direction,
                semantic_label=f"cross {direction} {band.band} {be.edge}",
            )

        if be.event == "cross_out":
            # Cross out: price crosses out of the band
            # For lower band: was above, now below
            # For upper band: was below, now above
            direction = "below" if be.edge == "lower" else "above"
            return CrossCondition(
                left=PriceRef(field=PriceField.CLOSE),
                right=band_ref,
                direction=direction,
                semantic_label=f"cross {direction} {band.band} {be.edge}",
            )

    if be.kind == "reentry" and be.edge:
        band_ref = _make_band_ref(band_id, be.edge)

        if be.edge == "lower":
            # Outside: close < lower band, Inside: close >= lower band
            return StateCondition(
                state_var=f"outside_{band_id}_{be.edge}",
                trigger_on_transition=True,
                outside_condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=band_ref,
                ),
                inside_condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GTE,
                    right=band_ref,
                ),
                semantic_label=f"reentry from below {band.band} {be.edge}",
            )
        else:  # upper
            # Outside: close > upper band, Inside: close <= upper band
            return StateCondition(
                state_var=f"outside_{band_id}_{be.edge}",
                trigger_on_transition=True,
                outside_condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=band_ref,
                ),
                inside_condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LTE,
                    right=band_ref,
                ),
                semantic_label=f"reentry from above {band.band} {be.edge}",
            )

    # Distance-based events
    if be.kind == "distance" and be.mode and be.side and be.thresh:
        middle_ref = IndicatorBandRef(indicator_id=band_id, band="middle")

        if be.mode == "z":
            # z-score: (close - middle) / stddev
            stddev_ref = IndicatorPropertyRef(
                indicator_id=band_id,
                property=IndicatorProperty.STANDARD_DEVIATION,
            )
            distance_expr = IRExpression(
                op="/",
                left=IRExpression(
                    op="-",
                    left=PriceRef(field=PriceField.CLOSE),
                    right=middle_ref,
                ),
                right=stddev_ref,
            )
        else:  # band_mult
            # Band multiple: (close - middle) / (upper - middle)
            upper_ref = IndicatorBandRef(indicator_id=band_id, band="upper")
            half_width = IRExpression(
                op="-",
                left=upper_ref,
                right=middle_ref,
            )
            distance_expr = IRExpression(
                op="/",
                left=IRExpression(
                    op="-",
                    left=PriceRef(field=PriceField.CLOSE),
                    right=middle_ref,
                ),
                right=half_width,
            )

        # Determine comparison based on side
        if be.side == "away_upper":
            return CompareCondition(
                left=distance_expr,
                op=CompareOp.GT,
                right=LiteralRef(value=be.thresh),
                semantic_label=f"band_distance: {be.mode} > {be.thresh} (away_upper)",
            )
        else:  # away_lower
            return CompareCondition(
                left=distance_expr,
                op=CompareOp.LT,
                right=LiteralRef(value=-be.thresh),
                semantic_label=f"band_distance: {be.mode} < -{be.thresh} (away_lower)",
            )

    # Fallback for band_event that does not match any known kind
    raise ValueError(f"Unsupported band_event kind: {be.kind}")


# =============================================================================
# Atomic condition compilers
# =============================================================================


def _compile_regime(spec: ConditionSpec, ctx: CompilationContext) -> Condition:
    """Compile a regime condition."""
    r = spec.regime
    assert r is not None
    return RegimeCondition(
        metric=r.metric,
        op=_COMPARE_OP_MAP.get(r.op, CompareOp.GT),
        value=r.value,
        ma_fast=r.ma_fast,
        ma_slow=r.ma_slow,
        lookback_bars=r.lookback_bars,
        level_reference=r.level_reference,
        semantic_label=f"regime: {r.metric} {r.op} {r.value}",
    )


def _compile_time_filter(spec: ConditionSpec) -> Condition:
    """Compile a time filter condition."""
    tf = spec.time_filter
    assert tf is not None
    days_int = [_DAY_MAP[d.lower()] for d in (tf.days_of_week or [])]
    return TimeFilterCondition(
        days_of_week=days_int,
        time_window=tf.time_window or "",
        days_of_month=tf.days_of_month or [],
        timezone=tf.timezone,
        semantic_label=f"time_filter: {tf.days_of_week or 'all'} {tf.time_window or ''}",
    )


def _compile_cross(spec: ConditionSpec, ctx: CompilationContext) -> Condition:
    """Compile a cross condition."""
    c = spec.cross
    assert c is not None
    direction = "above" if c.direction == "cross_above" else "below"
    return CrossCondition(
        left=_signal_ref_to_value_ref(c.lhs),
        right=_signal_ref_to_value_ref(c.rhs),
        direction=direction,
        semantic_label=f"cross: {direction}",
    )


def _compile_compare(spec: ConditionSpec, ctx: CompilationContext) -> Condition:
    """Compile a compare condition."""
    cmp = spec.compare
    assert cmp is not None
    return CompareCondition(
        left=_signal_ref_to_value_ref(cmp.lhs),
        op=_COMPARE_OP_MAP.get(cmp.op, CompareOp.GT),
        right=_signal_ref_to_value_ref(cmp.rhs),
        semantic_label=f"compare: {cmp.op}",
    )


def _compile_trailing_state(spec: ConditionSpec) -> Condition:
    """Compile a trailing state condition."""
    ts = spec.trailing_state
    assert ts is not None
    return TrailingStateCondition(
        state_id=ts.state_id,
        update_rule=ts.update_rule,
        update_price=ts.update_price,
        trigger_op=ts.trigger_op,
        trigger_price=ts.trigger_price,
        atr_period=ts.atr_period,
        atr_mult=ts.atr_mult,
        semantic_label=f"trailing_state: {ts.trigger_op} {ts.atr_mult}x ATR",
    )


def _compile_spread(spec: ConditionSpec, ctx: CompilationContext) -> Condition:
    """Compile a spread condition and register additional symbols."""
    sp = spec.spread
    assert sp is not None
    ctx.add_additional_symbol(sp.symbol_a)
    ctx.add_additional_symbol(sp.symbol_b)
    return SpreadCondition(
        symbol_a=sp.symbol_a,
        symbol_b=sp.symbol_b,
        calc_type=sp.calc_type,
        window_bars=sp.window_bars,
        trigger_op=sp.trigger_op,
        threshold=sp.threshold,
        hedge_ratio=sp.hedge_ratio,
        semantic_label=f"spread: {sp.symbol_a}/{sp.symbol_b} {sp.trigger_op} {sp.threshold}",
    )


def _compile_event(spec: ConditionSpec) -> Condition:
    """Compile an event condition."""
    ev = spec.event
    assert ev is not None
    if ev.trigger_type == "pre_event":
        pre_bars = ev.bars_offset
        post_bars = 0
    elif ev.trigger_type == "post_event":
        pre_bars = 0
        post_bars = ev.bars_offset
    else:  # in_window
        pre_bars = ev.bars_offset
        post_bars = ev.window_bars or ev.bars_offset

    return EventWindowCondition(
        event_types=[ev.event_kind],
        pre_window_bars=pre_bars,
        post_window_bars=post_bars,
        mode="within",
        semantic_label=f"event: {ev.event_kind} {ev.trigger_type}",
    )


# =============================================================================
# Direct-IR archetype dispatching
# =============================================================================


def _dispatch_direct_builder(archetype: BaseArchetype, ctx: CompilationContext) -> Condition:
    """Dispatch to archetype-specific builder for direct-IR archetypes.

    These archetypes build Condition directly without going through ConditionSpec.
    We call their to_ir() method since they already produce typed Condition objects.

    Args:
        archetype: Parsed archetype instance.
        ctx: Compilation context.

    Returns:
        Typed IR Condition.

    Raises:
        ValueError: If archetype has no to_ir() or it returns None.
    """
    # Direct-IR archetypes already produce typed Condition from to_ir()
    # They include: TrailingStop, BreakoutRetest, AvwapReversion, GapPlay,
    # PairsTrade, IntermarketTrigger, EventFollowthrough, TrailingBreakout,
    # VwapReversion, EventRiskWindow, RegimeScaler, etc.
    condition = archetype.to_ir()
    if condition is None:
        raise ValueError(
            f"Archetype {archetype.__class__.__name__} returned None from to_ir()"
        )
    return condition
