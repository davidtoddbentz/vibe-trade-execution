"""Tests for regime condition lowering to primitives."""
from vibe_trade_shared.models.ir import (
    CompareCondition,
    CompareOp,
    IndicatorRef,
    IRExpression,
    RegimeCondition,
)

from src.translator.regime_lowering import lower_regime_condition


class TestRegimeLowering:
    """Test that RegimeCondition is lowered to primitives."""

    def test_trend_ma_relation_lowered(self):
        """trend_ma_relation becomes: (EMA_fast - EMA_slow) op value."""
        regime = RegimeCondition(
            metric="trend_ma_relation",
            op=CompareOp.GT,
            value=0.0,
            ma_fast=20,
            ma_slow=50,
        )

        result = lower_regime_condition(regime)

        assert isinstance(result, CompareCondition)
        assert isinstance(result.left, IRExpression)
        assert result.left.op == "-"
        assert result.op == CompareOp.GT
        assert result.right.value == 0.0

    def test_trend_adx_lowered(self):
        """trend_adx becomes: ADX op value."""
        regime = RegimeCondition(
            metric="trend_adx",
            op=CompareOp.GT,
            value=25.0,
        )

        result = lower_regime_condition(regime)

        assert isinstance(result, CompareCondition)
        assert isinstance(result.left, IndicatorRef)
        assert result.left.indicator_type == "ADX"

    def test_vol_atr_pct_lowered(self):
        """vol_atr_pct becomes: (ATR / close * 100) op value."""
        regime = RegimeCondition(
            metric="vol_atr_pct",
            op=CompareOp.LT,
            value=2.0,
        )

        result = lower_regime_condition(regime)

        assert isinstance(result, CompareCondition)
        assert isinstance(result.left, IRExpression)

    def test_pattern_metrics_not_lowered(self):
        """Pattern metrics (liquidity_sweep, flag_pattern) remain as RegimeCondition."""
        regime = RegimeCondition(
            metric="liquidity_sweep",
            op=CompareOp.EQ,
            value=1.0,
        )

        result = lower_regime_condition(regime)

        # Pattern metrics stay as RegimeCondition (runtime handles them)
        assert isinstance(result, RegimeCondition)
