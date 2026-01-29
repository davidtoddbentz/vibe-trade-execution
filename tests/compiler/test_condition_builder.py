"""Tests for condition_builder module.

Tests compile_condition_spec with various ConditionSpec objects, and verifies
that _signal_ref_to_value_ref returns typed objects (not dicts) and that band
parameter preservation works correctly.
"""

from __future__ import annotations

import pytest
from vibe_trade_shared.models.archetypes.primitives import (
    BandEventSpec,
    BandSpec,
    BreakoutSpec,
    CompareSpec,
    ConditionSpec,
    EventConditionSpec,
    RegimeSpec,
    SequenceStep,
    SignalRef,
    SpreadConditionSpec,
    SqueezeSpec,
    TimeFilterSpec,
    TrailingStateSpec,
)
from vibe_trade_shared.models.archetypes.primitives import (
    CrossCondition as CrossConditionSpec,
)
from vibe_trade_shared.models.ir import (
    AllOfCondition,
    AnyOfCondition,
    BreakoutCondition,
    CompareCondition,
    CompareOp,
    CrossCondition,
    EventWindowCondition,
    IndicatorRef,
    IRExpression,
    LiteralRef,
    NotCondition,
    PriceRef,
    RegimeCondition,
    SequenceCondition,
    SpreadCondition,
    SqueezeCondition,
    StateCondition,
    TimeFilterCondition,
    TrailingStateCondition,
)
from vibe_trade_shared.models.ir.enums import PriceField
from vibe_trade_shared.models.ir.indicators import BollingerBands, KeltnerChannel

from src.translator.compiler.condition_builder import (
    _signal_ref_to_value_ref,
    compile_condition_spec,
)
from src.translator.compiler.context import CompilationContext

# =============================================================================
# Helpers
# =============================================================================


def _make_ctx() -> CompilationContext:
    return CompilationContext(symbol="BTC-USD")


# =============================================================================
# _signal_ref_to_value_ref tests (typed, not dicts)
# =============================================================================


class TestSignalRefToValueRef:
    """Verify _signal_ref_to_value_ref returns typed Pydantic models."""

    def test_numeric_literal(self):
        result = _signal_ref_to_value_ref(42.5)
        assert isinstance(result, LiteralRef)
        assert result.value == 42.5

    def test_int_literal(self):
        result = _signal_ref_to_value_ref(100)
        assert isinstance(result, LiteralRef)
        assert result.value == 100.0

    def test_price_close(self):
        ref = SignalRef(type="price", field="close")
        result = _signal_ref_to_value_ref(ref)
        assert isinstance(result, PriceRef)
        assert result.field == PriceField.CLOSE

    def test_price_high(self):
        ref = SignalRef(type="price", field="high")
        result = _signal_ref_to_value_ref(ref)
        assert isinstance(result, PriceRef)
        assert result.field == PriceField.HIGH

    def test_indicator_ema(self):
        ref = SignalRef(type="indicator", indicator="ema", period=20)
        result = _signal_ref_to_value_ref(ref)
        assert isinstance(result, IndicatorRef)
        assert result.indicator_type == "EMA"
        assert result.params == {"period": 20}

    def test_indicator_bb_upper(self):
        ref = SignalRef(type="indicator", indicator="bb_upper", period=20)
        result = _signal_ref_to_value_ref(ref)
        assert isinstance(result, IndicatorRef)
        assert result.indicator_type == "BB"
        assert result.field == "upper"

    def test_constant(self):
        ref = SignalRef(type="constant", value=50.0)
        result = _signal_ref_to_value_ref(ref)
        assert isinstance(result, LiteralRef)
        assert result.value == 50.0

    def test_returns_pydantic_model_not_dict(self):
        """Ensure we get typed models, not model_dump() dicts."""
        ref = SignalRef(type="price", field="close")
        result = _signal_ref_to_value_ref(ref)
        # Must be a Pydantic model, not a dict
        assert not isinstance(result, dict)
        assert hasattr(result, "model_dump")


# =============================================================================
# compile_condition_spec: atomic conditions
# =============================================================================


class TestCompileCompare:
    """Test compare condition compilation."""

    def test_compare_close_gt_literal(self):
        spec = ConditionSpec(
            type="compare",
            compare=CompareSpec(
                lhs=SignalRef(type="price", field="close"),
                op=">",
                rhs=100.0,
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, CompareCondition)
        assert result.op == CompareOp.GT
        assert isinstance(result.left, PriceRef)
        assert isinstance(result.right, LiteralRef)
        assert result.right.value == 100.0

    def test_compare_indicator_vs_indicator(self):
        spec = ConditionSpec(
            type="compare",
            compare=CompareSpec(
                lhs=SignalRef(type="indicator", indicator="ema", period=20),
                op=">=",
                rhs=SignalRef(type="indicator", indicator="sma", period=50),
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, CompareCondition)
        assert result.op == CompareOp.GTE
        assert isinstance(result.left, IndicatorRef)
        assert isinstance(result.right, IndicatorRef)


class TestCompileCross:
    """Test cross condition compilation."""

    def test_cross_above(self):
        spec = ConditionSpec(
            type="cross",
            cross=CrossConditionSpec(
                lhs=SignalRef(type="indicator", indicator="ema", period=20),
                rhs=SignalRef(type="indicator", indicator="sma", period=50),
                direction="cross_above",
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, CrossCondition)
        assert result.direction == "above"
        assert isinstance(result.left, IndicatorRef)
        assert isinstance(result.right, IndicatorRef)

    def test_cross_below(self):
        spec = ConditionSpec(
            type="cross",
            cross=CrossConditionSpec(
                lhs=SignalRef(type="price", field="close"),
                rhs=50.0,
                direction="cross_below",
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, CrossCondition)
        assert result.direction == "below"


class TestCompileRegime:
    """Test regime condition compilation."""

    def test_trend_ma_relation(self):
        spec = ConditionSpec(
            type="regime",
            regime=RegimeSpec(
                metric="trend_ma_relation",
                op=">",
                value=1,
                ma_fast=20,
                ma_slow=50,
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, RegimeCondition)
        assert result.metric == "trend_ma_relation"
        assert result.op == CompareOp.GT
        assert result.ma_fast == 20
        assert result.ma_slow == 50


class TestCompileBreakout:
    """Test breakout condition compilation."""

    def test_breakout(self):
        spec = ConditionSpec(
            type="breakout",
            breakout=BreakoutSpec(lookback_bars=50, buffer_bps=10),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, BreakoutCondition)
        assert result.lookback_bars == 50
        assert result.buffer_bps == 10


class TestCompileSqueeze:
    """Test squeeze condition compilation."""

    def test_squeeze(self):
        spec = ConditionSpec(
            type="squeeze",
            squeeze=SqueezeSpec(
                metric="bb_width_pctile",
                pctile_min=10,
                break_rule="donchian",
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, SqueezeCondition)
        assert result.squeeze_metric == "bb_width_pctile"
        assert result.pctile_threshold == 10.0
        assert result.break_rule == "donchian"


class TestCompileTimeFilter:
    """Test time_filter condition compilation."""

    def test_time_filter(self):
        spec = ConditionSpec(
            type="time_filter",
            time_filter=TimeFilterSpec(
                days_of_week=["monday", "wednesday", "friday"],
                time_window="0930-1600",
                timezone="US/Eastern",
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, TimeFilterCondition)
        assert result.days_of_week == [0, 2, 4]
        assert result.time_window == "0930-1600"
        assert result.timezone == "US/Eastern"


class TestCompileTrailingState:
    """Test trailing_state condition compilation."""

    def test_trailing_state(self):
        spec = ConditionSpec(
            type="trailing_state",
            trailing_state=TrailingStateSpec(
                state_id="trailing_high",
                update_rule="max",
                update_price="high",
                trigger_op="below",
                trigger_price="close",
                atr_period=20,
                atr_mult=2.0,
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, TrailingStateCondition)
        assert result.state_id == "trailing_high"
        assert result.update_rule == "max"
        assert result.trigger_op == "below"
        assert result.atr_mult == 2.0


class TestCompileSpread:
    """Test spread condition compilation."""

    def test_spread_registers_symbols(self):
        spec = ConditionSpec(
            type="spread",
            spread=SpreadConditionSpec(
                symbol_a="ETH-USD",
                symbol_b="BTC-USD",
                calc_type="zscore",
                window_bars=100,
                trigger_op="above",
                threshold=2.0,
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, SpreadCondition)
        assert result.symbol_a == "ETH-USD"
        assert result.symbol_b == "BTC-USD"
        # Should register additional symbols
        assert "ETH-USD" in ctx.additional_symbols
        assert "BTC-USD" in ctx.additional_symbols


class TestCompileEvent:
    """Test event condition compilation."""

    def test_pre_event(self):
        spec = ConditionSpec(
            type="event",
            event=EventConditionSpec(
                event_kind="earnings",
                trigger_type="pre_event",
                bars_offset=5,
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, EventWindowCondition)
        assert result.event_types == ["earnings"]
        assert result.pre_window_bars == 5
        assert result.post_window_bars == 0

    def test_post_event(self):
        spec = ConditionSpec(
            type="event",
            event=EventConditionSpec(
                event_kind="macro",
                trigger_type="post_event",
                bars_offset=10,
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, EventWindowCondition)
        assert result.pre_window_bars == 0
        assert result.post_window_bars == 10

    def test_in_window(self):
        spec = ConditionSpec(
            type="event",
            event=EventConditionSpec(
                event_kind="earnings",
                trigger_type="in_window",
                bars_offset=5,
                window_bars=20,
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, EventWindowCondition)
        assert result.pre_window_bars == 5
        assert result.post_window_bars == 20


# =============================================================================
# compile_condition_spec: band_event conditions
# =============================================================================


class TestCompileBandEvent:
    """Test band_event compilation (touch, cross_in, reentry, distance)."""

    def test_touch_lower(self):
        spec = ConditionSpec(
            type="band_event",
            band_event=BandEventSpec(
                band=BandSpec(band="bollinger", length=20, mult=2.0),
                kind="edge_event",
                edge="lower",
                event="touch",
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, AllOfCondition)
        assert len(result.conditions) == 2
        # Both children should be CompareConditions referencing band
        assert isinstance(result.conditions[0], CompareCondition)
        assert isinstance(result.conditions[1], CompareCondition)
        # Should register BB indicator
        assert len(ctx.indicators) == 1

    def test_cross_in_lower(self):
        spec = ConditionSpec(
            type="band_event",
            band_event=BandEventSpec(
                band=BandSpec(band="bollinger", length=20, mult=2.0),
                kind="edge_event",
                edge="lower",
                event="cross_in",
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, CrossCondition)
        assert result.direction == "above"
        assert len(ctx.indicators) == 1

    def test_cross_out_upper(self):
        spec = ConditionSpec(
            type="band_event",
            band_event=BandEventSpec(
                band=BandSpec(band="bollinger", length=20, mult=2.0),
                kind="edge_event",
                edge="upper",
                event="cross_out",
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, CrossCondition)
        assert result.direction == "above"

    def test_reentry_lower(self):
        spec = ConditionSpec(
            type="band_event",
            band_event=BandEventSpec(
                band=BandSpec(band="bollinger", length=20, mult=2.0),
                kind="reentry",
                edge="lower",
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, StateCondition)
        assert result.trigger_on_transition is True
        assert result.outside_condition is not None
        assert result.inside_condition is not None

    def test_reentry_upper(self):
        spec = ConditionSpec(
            type="band_event",
            band_event=BandEventSpec(
                band=BandSpec(band="keltner", length=20, mult=1.5),
                kind="reentry",
                edge="upper",
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, StateCondition)
        # Should register KC indicator
        assert len(ctx.indicators) == 1
        ind = list(ctx.indicators.values())[0]
        assert isinstance(ind, KeltnerChannel)

    def test_distance_z_away_upper(self):
        spec = ConditionSpec(
            type="band_event",
            band_event=BandEventSpec(
                band=BandSpec(band="bollinger", length=20, mult=2.0),
                kind="distance",
                mode="z",
                side="away_upper",
                thresh=2.0,
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, CompareCondition)
        assert result.op == CompareOp.GT
        assert isinstance(result.left, IRExpression)
        assert isinstance(result.right, LiteralRef)
        assert result.right.value == 2.0

    def test_distance_band_mult_away_lower(self):
        spec = ConditionSpec(
            type="band_event",
            band_event=BandEventSpec(
                band=BandSpec(band="bollinger", length=20, mult=2.0),
                kind="distance",
                mode="band_mult",
                side="away_lower",
                thresh=1.5,
            ),
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, CompareCondition)
        assert result.op == CompareOp.LT
        assert isinstance(result.right, LiteralRef)
        assert result.right.value == -1.5

    def test_band_param_preservation_custom_mult(self):
        """Band multiplier 2.5 must be preserved in the registered indicator."""
        spec = ConditionSpec(
            type="band_event",
            band_event=BandEventSpec(
                band=BandSpec(band="bollinger", length=20, mult=2.5),
                kind="edge_event",
                edge="lower",
                event="touch",
            ),
        )
        ctx = _make_ctx()
        compile_condition_spec(spec, ctx)

        # Should have exactly one indicator
        assert len(ctx.indicators) == 1
        ind = list(ctx.indicators.values())[0]
        assert isinstance(ind, BollingerBands)
        assert ind.period == 20
        assert ind.multiplier == 2.5  # NOT the default 2.0


# =============================================================================
# compile_condition_spec: composite conditions
# =============================================================================


class TestCompileComposite:
    """Test composite condition compilation (allOf, anyOf, not, sequence)."""

    def test_allOf(self):
        spec = ConditionSpec(
            type="allOf",
            allOf=[
                ConditionSpec(
                    type="regime",
                    regime=RegimeSpec(metric="trend_ma_relation", op=">", value=1, ma_fast=20, ma_slow=50),
                ),
                ConditionSpec(
                    type="compare",
                    compare=CompareSpec(
                        lhs=SignalRef(type="price", field="close"),
                        op=">",
                        rhs=100.0,
                    ),
                ),
            ],
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, AllOfCondition)
        assert len(result.conditions) == 2
        assert isinstance(result.conditions[0], RegimeCondition)
        assert isinstance(result.conditions[1], CompareCondition)

    def test_anyOf(self):
        spec = ConditionSpec(
            type="anyOf",
            anyOf=[
                ConditionSpec(
                    type="breakout",
                    breakout=BreakoutSpec(lookback_bars=50),
                ),
                ConditionSpec(
                    type="squeeze",
                    squeeze=SqueezeSpec(metric="bb_width_pctile", pctile_min=10),
                ),
            ],
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, AnyOfCondition)
        assert len(result.conditions) == 2

    def test_not(self):
        spec = ConditionSpec(
            type="not",
            **{
                "not": ConditionSpec(
                    type="regime",
                    regime=RegimeSpec(metric="trend_adx", op=">", value=25),
                ),
            },
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, NotCondition)
        assert isinstance(result.condition, RegimeCondition)

    def test_sequence(self):
        spec = ConditionSpec(
            type="sequence",
            sequence=[
                SequenceStep(
                    cond=ConditionSpec(
                        type="squeeze",
                        squeeze=SqueezeSpec(metric="bb_width_pctile", pctile_min=10),
                    ),
                    within_bars=None,
                ),
                SequenceStep(
                    cond=ConditionSpec(
                        type="breakout",
                        breakout=BreakoutSpec(lookback_bars=50),
                    ),
                    within_bars=10,
                ),
            ],
        )
        ctx = _make_ctx()
        result = compile_condition_spec(spec, ctx)

        assert isinstance(result, SequenceCondition)
        assert len(result.steps) == 2
        assert result.steps[1].within_bars == 10


# =============================================================================
# Error handling
# =============================================================================


class TestErrors:
    """Test error handling."""

    def test_unknown_type_raises(self):
        spec = ConditionSpec.__new__(ConditionSpec)
        object.__setattr__(spec, "type", "nonexistent")
        object.__setattr__(spec, "regime", None)
        object.__setattr__(spec, "band_event", None)
        object.__setattr__(spec, "breakout", None)
        object.__setattr__(spec, "squeeze", None)
        object.__setattr__(spec, "time_filter", None)
        object.__setattr__(spec, "cross", None)
        object.__setattr__(spec, "compare", None)
        object.__setattr__(spec, "trailing_state", None)
        object.__setattr__(spec, "spread", None)
        object.__setattr__(spec, "event", None)
        object.__setattr__(spec, "allOf", None)
        object.__setattr__(spec, "anyOf", None)
        object.__setattr__(spec, "not_", None)
        object.__setattr__(spec, "sequence", None)

        ctx = _make_ctx()
        with pytest.raises(ValueError, match="Unknown condition type"):
            compile_condition_spec(spec, ctx)
