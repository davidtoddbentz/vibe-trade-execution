"""Tests for ConditionPipeline."""

from vibe_trade_shared.models.ir import (
    AllOfCondition,
    CompareCondition,
    CompareOp,
    IndicatorBandRef,
    IndicatorRef,
    LiteralRef,
    PriceField,
    PriceRef,
    RegimeCondition,
    StateCondition,
)

from src.translator.context import TranslationContext
from src.translator.ir import EMA
from src.translator.pipeline import ConditionPipeline


def test_pipeline_lowers_regime():
    """RegimeCondition gets lowered to CompareCondition."""
    ctx = TranslationContext(symbol="BTC-USD")
    pipeline = ConditionPipeline(ctx)

    cond = RegimeCondition(
        metric="trend_ma_relation",
        op=CompareOp.GT,
        value=0,
        ma_fast=10,
        ma_slow=30,
    )

    result = pipeline.process(cond)

    # RegimeCondition was lowered to CompareCondition
    assert isinstance(result, CompareCondition)
    assert result.op == CompareOp.GT


def test_pipeline_collects_indicators():
    """Indicators from IndicatorRef get collected into context."""
    ctx = TranslationContext(symbol="BTC-USD")
    pipeline = ConditionPipeline(ctx)

    cond = CompareCondition(
        left=IndicatorRef(indicator_type="EMA", params={"period": 20}),
        op=CompareOp.GT,
        right=IndicatorRef(indicator_type="EMA", params={"period": 50}),
    )

    pipeline.process(cond)

    # Indicators were collected
    assert "ema_20" in ctx.indicators
    assert "ema_50" in ctx.indicators
    assert isinstance(ctx.indicators["ema_20"], EMA)
    assert ctx.indicators["ema_20"].period == 20


def test_pipeline_extracts_state():
    """StateCondition's state_var gets extracted into context."""
    ctx = TranslationContext(symbol="BTC-USD")
    pipeline = ConditionPipeline(ctx)

    cond = StateCondition(
        state_var="below_band",
        trigger_on_transition=True,
        outside_condition=CompareCondition(
            left=PriceRef(field=PriceField.CLOSE),
            op=CompareOp.LT,
            right=IndicatorBandRef(indicator_id="bb_20", band="lower"),
        ),
        inside_condition=CompareCondition(
            left=PriceRef(field=PriceField.CLOSE),
            op=CompareOp.GT,
            right=IndicatorBandRef(indicator_id="bb_20", band="lower"),
        ),
    )

    pipeline.process(cond)

    # State variable was extracted
    assert "below_band" in ctx.state_vars
    assert ctx.state_vars["below_band"].var_type == "bool"
    assert ctx.state_vars["below_band"].default is False


def test_pipeline_handles_complex_tree():
    """AllOf with multiple condition types works correctly."""
    ctx = TranslationContext(symbol="BTC-USD")
    pipeline = ConditionPipeline(ctx)

    cond = AllOfCondition(
        conditions=[
            # A regime condition that will be lowered
            RegimeCondition(
                metric="trend_ma_relation",
                op=CompareOp.GT,
                value=0,
                ma_fast=10,
                ma_slow=30,
            ),
            # An indicator comparison
            CompareCondition(
                left=IndicatorRef(indicator_type="RSI", params={"period": 14}),
                op=CompareOp.LT,
                right=LiteralRef(value=70.0),
            ),
            # A state condition
            StateCondition(
                state_var="my_state",
                trigger_on_transition=True,
                outside_condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=LiteralRef(value=100.0),
                ),
            ),
        ]
    )

    result = pipeline.process(cond)

    # Result is still AllOfCondition
    assert isinstance(result, AllOfCondition)
    assert len(result.conditions) == 3

    # First condition was lowered from RegimeCondition to CompareCondition
    assert isinstance(result.conditions[0], CompareCondition)

    # Indicators were collected from both lowered regime and RSI condition
    # The lowered regime creates EMA indicators
    assert "ema_10" in ctx.indicators
    assert "ema_30" in ctx.indicators
    # The RSI condition creates RSI indicator
    assert "rsi_14" in ctx.indicators

    # State variable was extracted
    assert "my_state" in ctx.state_vars
    assert ctx.state_vars["my_state"].var_type == "bool"
