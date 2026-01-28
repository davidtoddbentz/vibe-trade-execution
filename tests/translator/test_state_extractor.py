"""Tests for StateExtractor visitor."""

from vibe_trade_shared.models.ir import (
    AllOfCondition,
    CompareCondition,
    CompareOp,
    IndicatorBandRef,
    LiteralRef,
    PriceField,
    PriceRef,
    StateCondition,
)

from src.translator.context import TranslationContext
from src.translator.visitors.state_extractor import StateExtractor


def test_extracts_state_from_state_condition():
    """StateCondition declares bool state variable."""
    ctx = TranslationContext(symbol="BTC-USD")
    extractor = StateExtractor(ctx)

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

    extractor.visit(cond)

    assert "below_band" in ctx.state_vars
    assert ctx.state_vars["below_band"].var_type == "bool"
    assert ctx.state_vars["below_band"].default is False


def test_extracts_from_nested_conditions():
    """Extracts state from nested conditions."""
    ctx = TranslationContext(symbol="BTC-USD")
    extractor = StateExtractor(ctx)

    inner = StateCondition(
        state_var="inner_state",
        trigger_on_transition=True,
    )
    outer = AllOfCondition(conditions=[inner])

    extractor.visit(outer)

    assert "inner_state" in ctx.state_vars
    assert ctx.state_vars["inner_state"].var_type == "bool"


def test_ignores_non_state_conditions():
    """Non-StateConditions don't add state vars."""
    ctx = TranslationContext(symbol="BTC-USD")
    extractor = StateExtractor(ctx)

    cond = CompareCondition(
        left=PriceRef(field=PriceField.CLOSE),
        op=CompareOp.GT,
        right=LiteralRef(value=100.0),
    )

    extractor.visit(cond)

    assert len(ctx.state_vars) == 0


def test_deduplicates_state_vars():
    """Same state_var only declared once."""
    ctx = TranslationContext(symbol="BTC-USD")
    extractor = StateExtractor(ctx)

    # Same state_var used twice
    cond1 = StateCondition(state_var="shared_state", trigger_on_transition=True)
    cond2 = StateCondition(state_var="shared_state", trigger_on_transition=False)

    extractor.visit(cond1)
    extractor.visit(cond2)

    # Should only have one entry
    assert len(ctx.state_vars) == 1
    assert "shared_state" in ctx.state_vars


def test_extracts_multiple_state_vars():
    """Multiple StateConditions declare multiple state vars."""
    ctx = TranslationContext(symbol="BTC-USD")
    extractor = StateExtractor(ctx)

    cond1 = StateCondition(state_var="state_a", trigger_on_transition=True)
    cond2 = StateCondition(state_var="state_b", trigger_on_transition=True)

    # Process both via AllOf
    outer = AllOfCondition(conditions=[cond1, cond2])
    extractor.visit(outer)

    assert len(ctx.state_vars) == 2
    assert "state_a" in ctx.state_vars
    assert "state_b" in ctx.state_vars
