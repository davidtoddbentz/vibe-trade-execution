"""Condition builder for compiling archetypes to typed IR conditions.

All archetypes produce a ConditionSpec via to_condition_spec(), which is compiled
here into typed IR Condition trees by compile_condition_spec().

This is the single compilation path for all archetype types (entry, exit, gate, overlay).

Key properties:
- Returns TYPED ValueRef objects, never .model_dump() dicts
- Creates typed Indicator objects with full params (e.g., BollingerBands with multiplier)
- Registers indicators in CompilationContext during band event expansion
- Band indicators use canonical IDs via indicator_id_from_type_params
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_trade_shared.models.archetypes.base import BaseArchetype
from vibe_trade_shared.models.archetypes.primitives import (
    AVWAPReversionSpec,
    BandEventSpec,
    BandSpec,
    BreakoutRetestSpec,
    ConditionSpec,
    FixedTargetsSpec,
    GapSpec,
    IntermarketSpec,
    SignalRef,
    TrailingBreakoutSpec,
    TrailingStopSpec,
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
    TimeRef,
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

    All archetypes produce a ConditionSpec via to_condition_spec(), which is
    then compiled to a typed IR Condition by compile_condition_spec().

    Args:
        archetype: Parsed archetype instance.
        ctx: Compilation context for registering indicators, state, etc.

    Returns:
        Typed IR Condition.

    Raises:
        ValueError: If archetype cannot produce a ConditionSpec.
    """
    spec = archetype.to_condition_spec()
    return compile_condition_spec(spec, ctx)


def compile_condition_spec(spec: ConditionSpec, ctx: CompilationContext) -> Condition:
    """Compile a ConditionSpec to a typed IR Condition.

    Dispatches on spec.type to the appropriate handler, producing typed
    ValueRef objects, typed Indicator objects, and registering indicators
    in the compilation context.

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

    if spec.type == "gap" and spec.gap:
        return _compile_gap(spec.gap, ctx)

    if spec.type == "trailing_breakout" and spec.trailing_breakout:
        return _compile_trailing_breakout(spec.trailing_breakout)

    if spec.type == "breakout_retest" and spec.breakout_retest:
        return _compile_breakout_retest(spec.breakout_retest, ctx)

    if spec.type == "avwap_reversion" and spec.avwap_reversion:
        return _compile_avwap_reversion(spec.avwap_reversion)

    if spec.type == "fixed_targets" and spec.fixed_targets:
        return _compile_fixed_targets(spec.fixed_targets)

    if spec.type == "trailing_stop" and spec.trailing_stop:
        return _compile_trailing_stop(spec.trailing_stop)

    if spec.type == "intermarket" and spec.intermarket:
        return _compile_intermarket(spec.intermarket, ctx)

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
        mode=ev.mode,
        semantic_label=f"event: {ev.event_kind} {ev.trigger_type} ({ev.mode})",
    )


# =============================================================================
# New condition type compilers (Phase 2: unify all archetypes onto ConditionSpec)
# =============================================================================


def _compile_gap(spec: GapSpec, ctx: CompilationContext) -> Condition:
    """Compile a gap play condition.

    Creates AllOf(GapCondition, time filter) — a gap detection combined with
    a time window filter for the session open.
    """
    gap_cond = GapCondition(
        session=spec.session,
        mode=spec.mode,
        direction=spec.direction,
        semantic_label=f"gap_play: {spec.mode} during {spec.session} session",
    )

    # Time window filter: hour >= start AND hour < end
    # Only add +1 to end_hour if end_minute > 0
    end_hour_bound = spec.window_end_hour + (1 if spec.window_end_minute > 0 else 0)
    time_conds: list[Condition] = [
        CompareCondition(
            left=TimeRef(component="hour"),
            op=CompareOp.GTE,
            right=LiteralRef(value=float(spec.window_start_hour)),
            semantic_label=f"hour >= {spec.window_start_hour}",
        ),
        CompareCondition(
            left=TimeRef(component="hour"),
            op=CompareOp.LT,
            right=LiteralRef(value=float(end_hour_bound)),
            semantic_label=f"hour < {end_hour_bound}",
        ),
    ]

    time_filter = AllOfCondition(
        conditions=time_conds,
        semantic_label=f"time: {spec.window_start_hour}-{spec.window_end_hour}",
    )

    return AllOfCondition(
        conditions=[gap_cond, time_filter],
        semantic_label=f"gap_play: {spec.mode} at {spec.session} open",
    )


def _compile_trailing_breakout(spec: TrailingBreakoutSpec) -> Condition:
    """Compile a trailing breakout condition.

    Creates TrailingBreakoutCondition from the spec's band and trigger parameters.
    """
    return TrailingBreakoutCondition(
        band_type=spec.band.band,
        band_length=spec.band.length,
        band_mult=spec.band.mult if spec.band.mult else 2.0,
        update_rule=spec.update_rule,
        band_edge=spec.band_edge,
        trigger_direction=spec.trigger_direction,
        semantic_label=f"trailing_breakout: {spec.band.band}({spec.band.length}, {spec.band.mult})",
    )


def _compile_breakout_retest(spec: BreakoutRetestSpec, ctx: CompilationContext) -> Condition:
    """Compile a breakout-retest sequence condition.

    Creates a SequenceCondition: first a Donchian channel breakout,
    then an ATR-based pullback within 20 bars.
    For direction="auto", wraps in AnyOfCondition with long + short sequences.
    """
    lookback = spec.break_lookback_bars
    pullback_atr = spec.pullback_depth_atr
    dc_params: dict[str, int] = {"period": lookback}
    atr_params: dict[str, int] = {"period": 14}

    atr_ref = IndicatorRef(indicator_type="ATR", params=atr_params)
    close_ref = PriceRef(field=PriceField.CLOSE)

    def _make_sequence(direction: str) -> SequenceCondition:
        if direction == "long":
            # Step 1: close > DC upper (breakout)
            breakout_cond = CompareCondition(
                left=close_ref,
                op=CompareOp.GT,
                right=IndicatorRef(indicator_type="DC", params=dc_params, field="upper"),
                semantic_label=f"breakout: close > DC({lookback}).upper",
            )
            # Step 2: pullback within ATR distance of DC upper
            pullback_cond = CompareCondition(
                left=IRExpression(
                    op="-",
                    left=close_ref,
                    right=IndicatorRef(indicator_type="DC", params=dc_params, field="upper"),
                ),
                op=CompareOp.LTE,
                right=IRExpression(
                    op="*",
                    left=atr_ref,
                    right=LiteralRef(value=pullback_atr),
                ),
                semantic_label=f"pullback: within {pullback_atr}x ATR of DC upper",
            )
        else:  # short
            # Step 1: close < DC lower (breakdown)
            breakout_cond = CompareCondition(
                left=close_ref,
                op=CompareOp.LT,
                right=IndicatorRef(indicator_type="DC", params=dc_params, field="lower"),
                semantic_label=f"breakout: close < DC({lookback}).lower",
            )
            # Step 2: pullback within ATR distance of DC lower
            pullback_cond = CompareCondition(
                left=IRExpression(
                    op="-",
                    left=IndicatorRef(indicator_type="DC", params=dc_params, field="lower"),
                    right=close_ref,
                ),
                op=CompareOp.LTE,
                right=IRExpression(
                    op="*",
                    left=atr_ref,
                    right=LiteralRef(value=pullback_atr),
                ),
                semantic_label=f"pullback: within {pullback_atr}x ATR of DC lower",
            )

        return SequenceCondition(
            steps=[
                SequenceStep(condition=breakout_cond, hold_bars=1),
                SequenceStep(condition=pullback_cond, within_bars=20),
            ],
            semantic_label=f"breakout_retest: {direction} DC({lookback})",
        )

    if spec.direction == "auto":
        return AnyOfCondition(
            conditions=[_make_sequence("long"), _make_sequence("short")],
            semantic_label="breakout_retest: auto (long or short)",
        )
    return _make_sequence(spec.direction)


def _compile_avwap_reversion(spec: AVWAPReversionSpec) -> Condition:
    """Compile an AVWAP reversion condition.

    Creates z-score CompareCondition using IRExpression tree:
    z = (close - AVWAP.value) / AVWAP.std_dev
    """
    avwap_params: dict[str, str] = {"anchor": spec.anchor}
    vwap_ref = IndicatorRef(indicator_type="AVWAP", params=avwap_params, field="value")
    vwap_std_ref = IndicatorRef(indicator_type="AVWAP", params=avwap_params, field="std_dev")
    close_ref = PriceRef(field=PriceField.CLOSE)

    z_score = IRExpression(
        op="/",
        left=IRExpression(op="-", left=close_ref, right=vwap_ref),
        right=vwap_std_ref,
    )

    threshold = spec.dist_sigma

    if spec.direction == "long":
        return CompareCondition(
            left=z_score,
            op=CompareOp.LT,
            right=LiteralRef(value=-threshold),
            semantic_label=f"avwap_reversion: z < -{threshold} (long entry)",
        )

    if spec.direction == "short":
        return CompareCondition(
            left=z_score,
            op=CompareOp.GT,
            right=LiteralRef(value=threshold),
            semantic_label=f"avwap_reversion: z > {threshold} (short entry)",
        )

    if spec.direction == "exit":
        # Exit when z-score drops to threshold (price near VWAP).
        # Single z <= threshold check for exit direction.
        return CompareCondition(
            left=z_score,
            op=CompareOp.LTE,
            right=LiteralRef(value=threshold),
            semantic_label=f"vwap_reversion_exit: |z-score| <= {threshold} (price near VWAP)",
        )

    # direction == "auto": either long or short entry
    return AnyOfCondition(
        conditions=[
            CompareCondition(
                left=z_score,
                op=CompareOp.LT,
                right=LiteralRef(value=-threshold),
                semantic_label=f"avwap_reversion: z < -{threshold} (long)",
            ),
            CompareCondition(
                left=z_score,
                op=CompareOp.GT,
                right=LiteralRef(value=threshold),
                semantic_label=f"avwap_reversion: z > {threshold} (short)",
            ),
        ],
        semantic_label=f"avwap_reversion: |z| > {threshold} (auto)",
    )


def _compile_fixed_targets(spec: FixedTargetsSpec) -> Condition:
    """Compile fixed price/time target conditions for exits.

    Creates CompareCondition(s) using StateRef(entry_price) and
    StateRef(bars_since_entry). Multiple targets are wrapped in AnyOfCondition.
    """
    conditions: list[Condition] = []

    if spec.tp_pct is not None:
        tp_multiplier = 1.0 + spec.tp_pct / 100.0
        conditions.append(
            CompareCondition(
                left=PriceRef(field=PriceField.CLOSE),
                op=CompareOp.GT,
                right=IRExpression(
                    op="*",
                    left=StateRef(state_id="entry_price"),
                    right=LiteralRef(value=tp_multiplier),
                ),
                semantic_label=f"fixed_targets: close > entry_price * {tp_multiplier:.4f}",
            )
        )

    if spec.sl_pct is not None:
        sl_multiplier = 1.0 - spec.sl_pct / 100.0
        conditions.append(
            CompareCondition(
                left=PriceRef(field=PriceField.CLOSE),
                op=CompareOp.LT,
                right=IRExpression(
                    op="*",
                    left=StateRef(state_id="entry_price"),
                    right=LiteralRef(value=sl_multiplier),
                ),
                semantic_label=f"fixed_targets: close < entry_price * {sl_multiplier:.4f}",
            )
        )

    if spec.time_stop_bars is not None:
        conditions.append(
            CompareCondition(
                left=StateRef(state_id="bars_since_entry"),
                op=CompareOp.GTE,
                right=LiteralRef(value=float(spec.time_stop_bars)),
                semantic_label=f"fixed_targets: bars_since_entry >= {spec.time_stop_bars}",
            )
        )

    if len(conditions) == 0:
        raise ValueError("FixedTargetsSpec must have at least one target (tp_pct, sl_pct, or time_stop_bars)")
    if len(conditions) == 1:
        return conditions[0]
    return AnyOfCondition(
        conditions=conditions,
        semantic_label="fixed_targets: any exit trigger",
    )


def _compile_trailing_stop(spec: TrailingStopSpec) -> Condition:
    """Compile a trailing stop condition.

    Creates CompareCondition: close < highest_since_entry - trail_mult * ATR(atr_period).
    The IRExpression tree is built from the spec's named fields.
    """
    atr_ref = IndicatorRef(
        indicator_type="ATR",
        params={"period": spec.atr_period},
    )

    return CompareCondition(
        left=PriceRef(field=PriceField.CLOSE),
        op=CompareOp.LT,
        right=IRExpression(
            op="-",
            left=StateRef(state_id="highest_since_entry"),
            right=IRExpression(
                op="*",
                left=atr_ref,
                right=LiteralRef(value=spec.trail_mult),
            ),
        ),
        semantic_label=f"trailing_stop: close < highest - {spec.trail_mult} * ATR({spec.atr_period})",
    )


def _compile_intermarket(spec: IntermarketSpec, ctx: CompilationContext) -> Condition:
    """Compile an intermarket trigger condition.

    Creates IntermarketCondition (single leader) or
    MultiLeaderIntermarketCondition (multi-leader with aggregation).
    """
    feature_map = {
        "ret_pct": "ret_pct",
        "ma_flip": "price_cross_ma",
        "band_break": "volume_spike",
    }

    # Register additional symbols
    ctx.add_additional_symbol(spec.follower_symbol)

    if spec.leaders and spec.leader_aggregate:
        # Multi-leader mode
        for leader in spec.leaders:
            ctx.add_additional_symbol(leader)
        return MultiLeaderIntermarketCondition(
            leader_symbols=spec.leaders,
            follower_symbol=spec.follower_symbol,
            aggregate_feature=feature_map.get(
                spec.leader_aggregate.feature, spec.leader_aggregate.feature
            ),
            aggregate_op=spec.leader_aggregate.op,
            trigger_threshold=spec.leader_aggregate.threshold,
            window_bars=spec.window_bars,
            direction=spec.direction,
            semantic_label=f"intermarket: multi-leader -> {spec.follower_symbol}",
        )

    # Single leader mode
    if spec.leader_symbol:
        ctx.add_additional_symbol(spec.leader_symbol)
        return IntermarketCondition(
            leader_symbol=spec.leader_symbol,
            follower_symbol=spec.follower_symbol,
            trigger_feature=feature_map.get(
                spec.trigger_feature, spec.trigger_feature
            ) if spec.trigger_feature else "ret_pct",
            trigger_threshold=spec.trigger_threshold or 1.0,
            window_bars=spec.window_bars,
            direction=spec.direction,
            semantic_label=f"intermarket: {spec.leader_symbol} -> {spec.follower_symbol}",
        )

    raise ValueError("IntermarketSpec must have either leader_symbol or leaders")


