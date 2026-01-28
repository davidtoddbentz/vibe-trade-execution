"""Tests for IndicatorCollector visitor."""

from vibe_trade_shared.models.ir import (
    AllOfCondition,
    BreakoutCondition,
    CompareCondition,
    CompareOp,
    IndicatorRef,
    IRExpression,
    LiteralRef,
    PriceField,
    PriceRef,
    RegimeCondition,
    SqueezeCondition,
)

from src.translator.context import TranslationContext
from src.translator.ir import ATR, EMA, Maximum, Minimum
from src.translator.visitors.indicator_collector import IndicatorCollector


def test_collects_ema_from_compare_left():
    """Collects indicator from CompareCondition left side."""
    ctx = TranslationContext(symbol="BTC-USD")
    collector = IndicatorCollector(ctx)

    cond = CompareCondition(
        left=IndicatorRef(indicator_type="EMA", params={"period": 20}),
        op=CompareOp.GT,
        right=LiteralRef(value=50.0),
    )

    collector.visit(cond)

    assert "ema_20" in ctx.indicators
    assert isinstance(ctx.indicators["ema_20"], EMA)
    assert ctx.indicators["ema_20"].period == 20


def test_collects_from_ir_expression():
    """Collects indicators from nested IRExpression."""
    ctx = TranslationContext(symbol="BTC-USD")
    collector = IndicatorCollector(ctx)

    # ATR / price * 100
    cond = CompareCondition(
        left=IRExpression(
            op="*",
            left=IRExpression(
                op="/",
                left=IndicatorRef(indicator_type="ATR", params={"period": 14}),
                right=PriceRef(field=PriceField.CLOSE),
            ),
            right=LiteralRef(value=100.0),
        ),
        op=CompareOp.LT,
        right=LiteralRef(value=2.0),
    )

    collector.visit(cond)

    assert "atr_14" in ctx.indicators
    assert isinstance(ctx.indicators["atr_14"], ATR)


def test_collects_from_regime_condition():
    """Collects indicators required by RegimeCondition."""
    ctx = TranslationContext(symbol="BTC-USD")
    collector = IndicatorCollector(ctx)

    cond = RegimeCondition(
        metric="trend_ma_relation",
        op=CompareOp.GT,
        value=0,
        ma_fast=10,
        ma_slow=30,
    )

    collector.visit(cond)

    assert "ema_10" in ctx.indicators
    assert "ema_30" in ctx.indicators


def test_collects_from_breakout_condition():
    """Collects MAX/MIN from BreakoutCondition."""
    ctx = TranslationContext(symbol="BTC-USD")
    collector = IndicatorCollector(ctx)

    cond = BreakoutCondition(
        lookback_bars=50,
    )

    collector.visit(cond)

    assert "max_50" in ctx.indicators
    assert "min_50" in ctx.indicators
    assert isinstance(ctx.indicators["max_50"], Maximum)
    assert isinstance(ctx.indicators["min_50"], Minimum)


def test_collects_from_squeeze_condition():
    """Collects BB/KC from SqueezeCondition."""
    ctx = TranslationContext(symbol="BTC-USD")
    collector = IndicatorCollector(ctx)

    cond = SqueezeCondition(squeeze_metric="kc_comp")

    collector.visit(cond)

    assert "bb" in ctx.indicators or "bb_20" in ctx.indicators
    assert "kc" in ctx.indicators or "kc_20" in ctx.indicators


def test_collects_from_nested_allof():
    """Collects from all conditions in AllOfCondition."""
    ctx = TranslationContext(symbol="BTC-USD")
    collector = IndicatorCollector(ctx)

    cond = AllOfCondition(
        conditions=[
            CompareCondition(
                left=IndicatorRef(indicator_type="EMA", params={"period": 20}),
                op=CompareOp.GT,
                right=IndicatorRef(indicator_type="EMA", params={"period": 50}),
            ),
            CompareCondition(
                left=IndicatorRef(indicator_type="RSI", params={"period": 14}),
                op=CompareOp.LT,
                right=LiteralRef(value=70.0),
            ),
        ]
    )

    collector.visit(cond)

    assert "ema_20" in ctx.indicators
    assert "ema_50" in ctx.indicators
    assert "rsi_14" in ctx.indicators


def test_returns_condition_unchanged():
    """Visitor returns the condition (for pipeline chaining)."""
    ctx = TranslationContext(symbol="BTC-USD")
    collector = IndicatorCollector(ctx)

    cond = CompareCondition(
        left=PriceRef(field=PriceField.CLOSE),
        op=CompareOp.GT,
        right=LiteralRef(value=100.0),
    )

    result = collector.visit(cond)

    assert result == cond
