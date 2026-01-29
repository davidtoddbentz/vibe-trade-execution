"""Tests for indicator_resolver module.

Tests resolve_condition with various condition types, verifying that inline
IndicatorRefs are resolved to pre-declared indicator references and that
special conditions (Breakout, Squeeze, etc.) register required indicators.
"""

from __future__ import annotations

import pytest
from vibe_trade_shared.models.ir import (
    AllOfCondition,
    BreakoutCondition,
    CompareCondition,
    CompareOp,
    CrossCondition,
    IndicatorBandRef,
    IndicatorRef,
    IntermarketCondition,
    IRExpression,
    LiteralRef,
    MaxStateAction,
    MinStateAction,
    MultiLeaderIntermarketCondition,
    NotCondition,
    PriceRef,
    SequenceCondition,
    SequenceStep,
    SetStateAction,
    SpreadCondition,
    SqueezeCondition,
    StateCondition,
    TimeFilterCondition,
    TrailingBreakoutCondition,
)
from vibe_trade_shared.models.ir.enums import PriceField
from vibe_trade_shared.models.ir.indicators import (
    EMA,
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    Maximum,
    Minimum,
)

from src.translator.compiler.context import CompilationContext
from src.translator.compiler.indicator_resolver import (
    resolve_condition,
    resolve_state_ops,
    resolve_value_ref,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_ctx() -> CompilationContext:
    return CompilationContext(symbol="BTC-USD")


def _inline_ema(period: int = 20) -> IndicatorRef:
    """Create an inline IndicatorRef for EMA."""
    return IndicatorRef(indicator_type="EMA", params={"period": period})


def _inline_bb_lower(period: int = 20) -> IndicatorRef:
    """Create an inline IndicatorRef for BB lower band."""
    return IndicatorRef(indicator_type="BB", params={"period": period}, field="lower")


# =============================================================================
# resolve_value_ref tests
# =============================================================================


class TestResolveValueRef:
    """Test resolving individual ValueRefs."""

    def test_inline_indicator_creates_indicator(self):
        """Inline IndicatorRef should create typed indicator and return id-based ref."""
        ctx = _make_ctx()
        ref = _inline_ema(20)

        resolved = resolve_value_ref(ref, ctx)

        assert isinstance(resolved, IndicatorRef)
        assert resolved.indicator_id == "ema_20"
        assert resolved.indicator_type is None  # No longer inline
        assert "ema_20" in ctx.indicators
        assert isinstance(ctx.indicators["ema_20"], EMA)
        assert ctx.indicators["ema_20"].period == 20

    def test_inline_bb_band_field_creates_band_ref(self):
        """Inline BB with lower field should produce IndicatorBandRef."""
        ctx = _make_ctx()
        ref = IndicatorRef(
            indicator_type="BB",
            params={"period": 20, "multiplier": 2.0},
            field="lower",
        )

        resolved = resolve_value_ref(ref, ctx)

        assert isinstance(resolved, IndicatorBandRef)
        assert resolved.band == "lower"
        assert "bb_multiplier_2_period_20" in ctx.indicators
        ind = ctx.indicators["bb_multiplier_2_period_20"]
        assert isinstance(ind, BollingerBands)

    def test_already_resolved_passes_through(self):
        """IndicatorRef with indicator_id should pass through unchanged."""
        ctx = _make_ctx()
        ref = IndicatorRef(indicator_id="ema_20")

        resolved = resolve_value_ref(ref, ctx)

        assert resolved is ref
        assert len(ctx.indicators) == 0

    def test_price_ref_passes_through(self):
        """PriceRef should pass through unchanged."""
        ctx = _make_ctx()
        ref = PriceRef(field=PriceField.CLOSE)

        resolved = resolve_value_ref(ref, ctx)

        assert resolved is ref

    def test_literal_ref_passes_through(self):
        """LiteralRef should pass through unchanged."""
        ctx = _make_ctx()
        ref = LiteralRef(value=42.0)

        resolved = resolve_value_ref(ref, ctx)

        assert resolved is ref

    def test_indicator_band_ref_passes_through(self):
        """IndicatorBandRef should pass through unchanged."""
        ctx = _make_ctx()
        ref = IndicatorBandRef(indicator_id="bb_20", band="upper")

        resolved = resolve_value_ref(ref, ctx)

        assert resolved is ref

    def test_ir_expression_recursion(self):
        """IRExpression should recursively resolve inner refs."""
        ctx = _make_ctx()
        expr = IRExpression(
            op="/",
            left=_inline_ema(20),
            right=PriceRef(field=PriceField.CLOSE),
        )

        resolved = resolve_value_ref(expr, ctx)

        assert isinstance(resolved, IRExpression)
        assert resolved.op == "/"
        assert isinstance(resolved.left, IndicatorRef)
        assert resolved.left.indicator_id == "ema_20"
        assert isinstance(resolved.right, PriceRef)
        assert "ema_20" in ctx.indicators

    def test_ir_expression_unchanged_if_no_inline(self):
        """IRExpression with no inline refs should return same object."""
        ctx = _make_ctx()
        expr = IRExpression(
            op="+",
            left=PriceRef(field=PriceField.HIGH),
            right=LiteralRef(value=10.0),
        )

        resolved = resolve_value_ref(expr, ctx)

        assert resolved is expr


# =============================================================================
# resolve_condition: CompareCondition and CrossCondition
# =============================================================================


class TestResolveLeafConditions:
    """Test resolving leaf conditions with inline ValueRefs."""

    def test_compare_with_inline_refs(self):
        ctx = _make_ctx()
        cond = CompareCondition(
            left=_inline_ema(20),
            op=CompareOp.GT,
            right=_inline_ema(50),
        )

        resolved = resolve_condition(cond, ctx)

        assert isinstance(resolved, CompareCondition)
        assert isinstance(resolved.left, IndicatorRef)
        assert resolved.left.indicator_id == "ema_20"
        assert isinstance(resolved.right, IndicatorRef)
        assert resolved.right.indicator_id == "ema_50"
        assert len(ctx.indicators) == 2

    def test_compare_no_inline_passes_through(self):
        ctx = _make_ctx()
        cond = CompareCondition(
            left=PriceRef(field=PriceField.CLOSE),
            op=CompareOp.GT,
            right=LiteralRef(value=100.0),
        )

        resolved = resolve_condition(cond, ctx)

        # Should return the same object (no changes needed)
        assert resolved is cond

    def test_cross_with_inline_refs(self):
        ctx = _make_ctx()
        cond = CrossCondition(
            left=_inline_ema(20),
            right=_inline_ema(50),
            direction="above",
        )

        resolved = resolve_condition(cond, ctx)

        assert isinstance(resolved, CrossCondition)
        assert isinstance(resolved.left, IndicatorRef)
        assert resolved.left.indicator_id == "ema_20"


# =============================================================================
# resolve_condition: composite conditions
# =============================================================================


class TestResolveCompositeConditions:
    """Test resolving composite conditions (AllOf, AnyOf, Not, Sequence)."""

    def test_allOf_recurse(self):
        ctx = _make_ctx()
        cond = AllOfCondition(
            conditions=[
                CompareCondition(
                    left=_inline_ema(20),
                    op=CompareOp.GT,
                    right=PriceRef(field=PriceField.CLOSE),
                ),
                CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=100.0),
                ),
            ],
        )

        resolved = resolve_condition(cond, ctx)

        assert isinstance(resolved, AllOfCondition)
        assert len(resolved.conditions) == 2
        # First child should have resolved the inline ref
        assert isinstance(resolved.conditions[0].left, IndicatorRef)
        assert resolved.conditions[0].left.indicator_id == "ema_20"
        assert "ema_20" in ctx.indicators

    def test_not_recurse(self):
        ctx = _make_ctx()
        cond = NotCondition(
            condition=CompareCondition(
                left=_inline_ema(20),
                op=CompareOp.GT,
                right=LiteralRef(value=50.0),
            ),
        )

        resolved = resolve_condition(cond, ctx)

        assert isinstance(resolved, NotCondition)
        inner = resolved.condition
        assert isinstance(inner, CompareCondition)
        assert isinstance(inner.left, IndicatorRef)
        assert inner.left.indicator_id == "ema_20"

    def test_sequence_recurse(self):
        ctx = _make_ctx()
        cond = SequenceCondition(
            steps=[
                SequenceStep(
                    condition=CompareCondition(
                        left=_inline_ema(20),
                        op=CompareOp.GT,
                        right=LiteralRef(value=50.0),
                    ),
                    within_bars=5,
                ),
            ],
        )

        resolved = resolve_condition(cond, ctx)

        assert isinstance(resolved, SequenceCondition)
        step_cond = resolved.steps[0].condition
        assert isinstance(step_cond, CompareCondition)
        assert isinstance(step_cond.left, IndicatorRef)
        assert step_cond.left.indicator_id == "ema_20"


# =============================================================================
# resolve_condition: special conditions (register implicit indicators)
# =============================================================================


class TestResolveSpecialConditions:
    """Test that special conditions register their required indicators."""

    def test_breakout_registers_max_min(self):
        ctx = _make_ctx()
        cond = BreakoutCondition(lookback_bars=50)

        resolve_condition(cond, ctx)

        assert "max_50" in ctx.indicators
        assert "min_50" in ctx.indicators
        assert isinstance(ctx.indicators["max_50"], Maximum)
        assert isinstance(ctx.indicators["min_50"], Minimum)

    def test_squeeze_registers_bb_kc(self):
        ctx = _make_ctx()
        cond = SqueezeCondition(
            squeeze_metric="bb_width_pctile",
            pctile_threshold=10.0,
        )

        resolve_condition(cond, ctx)

        assert "bb_20" in ctx.indicators
        assert "kc_20" in ctx.indicators
        assert isinstance(ctx.indicators["bb_20"], BollingerBands)
        assert isinstance(ctx.indicators["kc_20"], KeltnerChannel)

    def test_trailing_breakout_registers_band(self):
        ctx = _make_ctx()
        cond = TrailingBreakoutCondition(
            band_type="bollinger",
            band_length=20,
            band_mult=2.0,
            update_rule="min",
            band_edge="upper",
            trigger_direction="above",
        )

        resolve_condition(cond, ctx)

        assert "bb_20" in ctx.indicators
        assert isinstance(ctx.indicators["bb_20"], BollingerBands)

    def test_trailing_breakout_donchian(self):
        ctx = _make_ctx()
        cond = TrailingBreakoutCondition(
            band_type="donchian",
            band_length=50,
            update_rule="min",
            band_edge="upper",
            trigger_direction="above",
        )

        resolve_condition(cond, ctx)

        assert "dc_50" in ctx.indicators
        assert isinstance(ctx.indicators["dc_50"], DonchianChannel)

    def test_spread_registers_symbols(self):
        ctx = _make_ctx()
        cond = SpreadCondition(
            symbol_a="ETH-USD",
            symbol_b="BTC-USD",
            trigger_op="above",
            threshold=2.0,
        )

        resolve_condition(cond, ctx)

        assert "ETH-USD" in ctx.additional_symbols
        # BTC-USD is the primary symbol, so it should not be in additional
        assert "BTC-USD" not in ctx.additional_symbols

    def test_intermarket_registers_leader(self):
        ctx = _make_ctx()
        cond = IntermarketCondition(
            leader_symbol="ETH-USD",
            follower_symbol="BTC-USD",
        )

        resolve_condition(cond, ctx)

        assert "ETH-USD" in ctx.additional_symbols

    def test_multi_leader_registers_all_leaders(self):
        ctx = _make_ctx()
        cond = MultiLeaderIntermarketCondition(
            leader_symbols=["ETH-USD", "SOL-USD"],
            follower_symbol="BTC-USD",
        )

        resolve_condition(cond, ctx)

        assert "ETH-USD" in ctx.additional_symbols
        assert "SOL-USD" in ctx.additional_symbols

    def test_time_filter_passes_through(self):
        ctx = _make_ctx()
        cond = TimeFilterCondition(
            days_of_week=[0, 1, 2],
            time_window="0930-1600",
        )

        resolved = resolve_condition(cond, ctx)

        assert resolved is cond


# =============================================================================
# resolve_condition: StateCondition recursion
# =============================================================================


class TestResolveStateCondition:
    """Test StateCondition inner conditions are resolved."""

    def test_state_condition_resolves_inner(self):
        ctx = _make_ctx()
        cond = StateCondition(
            state_var="outside_bb",
            trigger_on_transition=True,
            outside_condition=CompareCondition(
                left=PriceRef(field=PriceField.CLOSE),
                op=CompareOp.LT,
                right=_inline_ema(20),  # Inline ref to resolve
            ),
            inside_condition=CompareCondition(
                left=PriceRef(field=PriceField.CLOSE),
                op=CompareOp.GTE,
                right=IndicatorBandRef(indicator_id="bb_20", band="lower"),
            ),
        )

        resolved = resolve_condition(cond, ctx)

        assert isinstance(resolved, StateCondition)
        # outside_condition should have resolved the inline ref
        assert isinstance(resolved.outside_condition.right, IndicatorRef)
        assert resolved.outside_condition.right.indicator_id == "ema_20"
        assert "ema_20" in ctx.indicators


# =============================================================================
# resolve_state_ops tests
# =============================================================================


class TestResolveStateOps:
    """Test resolving inline indicators in state operations."""

    def test_set_state_resolves_inline(self):
        ctx = _make_ctx()
        ops = [
            SetStateAction(
                state_id="trailing_high",
                value=_inline_ema(20),
            ),
        ]

        resolved = resolve_state_ops(ops, ctx)

        assert len(resolved) == 1
        assert isinstance(resolved[0], SetStateAction)
        assert isinstance(resolved[0].value, IndicatorRef)
        assert resolved[0].value.indicator_id == "ema_20"
        assert "ema_20" in ctx.indicators

    def test_max_state_resolves_inline(self):
        ctx = _make_ctx()
        ops = [
            MaxStateAction(
                state_id="highest",
                value=_inline_ema(50),
            ),
        ]

        resolved = resolve_state_ops(ops, ctx)

        assert isinstance(resolved[0], MaxStateAction)
        assert isinstance(resolved[0].value, IndicatorRef)
        assert resolved[0].value.indicator_id == "ema_50"

    def test_min_state_resolves_inline(self):
        ctx = _make_ctx()
        ops = [
            MinStateAction(
                state_id="lowest",
                value=_inline_ema(30),
            ),
        ]

        resolved = resolve_state_ops(ops, ctx)

        assert isinstance(resolved[0], MinStateAction)
        assert isinstance(resolved[0].value, IndicatorRef)
        assert resolved[0].value.indicator_id == "ema_30"

    def test_no_inline_passes_through(self):
        ctx = _make_ctx()
        ops = [
            SetStateAction(
                state_id="entry_price",
                value=PriceRef(field=PriceField.CLOSE),
            ),
        ]

        resolved = resolve_state_ops(ops, ctx)

        assert resolved[0] is ops[0]  # Same object - no changes


# =============================================================================
# Unknown indicator type
# =============================================================================


class TestUnknownIndicator:
    """Test handling of unknown indicator types."""

    def test_unknown_type_raises(self):
        ctx = _make_ctx()
        ref = IndicatorRef(indicator_type="NONEXISTENT", params={"period": 20})

        with pytest.raises(ValueError, match="Unknown indicator type"):
            resolve_value_ref(ref, ctx)
