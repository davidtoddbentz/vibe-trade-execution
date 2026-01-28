"""Tests for RegimeLowerer visitor."""

from vibe_trade_shared.models.ir import (
    AllOfCondition,
    CompareCondition,
    CompareOp,
    IRExpression,
    LiteralRef,
    PriceField,
    PriceRef,
    RegimeCondition,
)

from src.translator.visitors.regime_lowerer import RegimeLowerer


def test_lowers_trend_ma_relation():
    """trend_ma_relation is lowered to CompareCondition."""
    lowerer = RegimeLowerer()

    cond = RegimeCondition(
        metric="trend_ma_relation",
        op=CompareOp.GT,
        value=0,
        ma_fast=10,
        ma_slow=30,
    )

    result = lowerer.visit(cond)

    assert isinstance(result, CompareCondition)
    assert isinstance(result.left, IRExpression)
    assert result.left.op == "-"
    assert result.op == CompareOp.GT


def test_lowers_trend_adx():
    """trend_adx is lowered to CompareCondition."""
    lowerer = RegimeLowerer()

    cond = RegimeCondition(
        metric="trend_adx",
        op=CompareOp.GT,
        value=25,
    )

    result = lowerer.visit(cond)

    assert isinstance(result, CompareCondition)


def test_lowers_vol_atr_pct():
    """vol_atr_pct is lowered to CompareCondition."""
    lowerer = RegimeLowerer()

    cond = RegimeCondition(
        metric="vol_atr_pct",
        op=CompareOp.LT,
        value=2.0,
    )

    result = lowerer.visit(cond)

    assert isinstance(result, CompareCondition)
    assert isinstance(result.left, IRExpression)


def test_passes_through_non_lowerable():
    """Non-lowerable metrics pass through unchanged."""
    lowerer = RegimeLowerer()

    cond = RegimeCondition(
        metric="liquidity_sweep",
        op=CompareOp.GT,
        value=0,
    )

    result = lowerer.visit(cond)

    assert isinstance(result, RegimeCondition)
    assert result == cond


def test_lowers_nested_in_allof():
    """Lowering works for nested RegimeConditions."""
    lowerer = RegimeLowerer()

    inner = RegimeCondition(
        metric="trend_ma_relation",
        op=CompareOp.GT,
        value=0,
    )
    outer = AllOfCondition(conditions=[inner])

    result = lowerer.visit(outer)

    assert isinstance(result, AllOfCondition)
    assert len(result.conditions) == 1
    assert isinstance(result.conditions[0], CompareCondition)


def test_non_regime_passes_through():
    """Non-RegimeCondition passes through unchanged."""
    lowerer = RegimeLowerer()

    cond = CompareCondition(
        left=PriceRef(field=PriceField.CLOSE),
        op=CompareOp.GT,
        right=LiteralRef(value=100.0),
    )

    result = lowerer.visit(cond)

    assert result == cond
