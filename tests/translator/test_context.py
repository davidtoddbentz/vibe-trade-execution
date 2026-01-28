"""Tests for TranslationContext."""

from src.translator.context import TranslationContext
from src.translator.ir import EMA, StateVarSpec


def test_context_add_indicator_registers_once():
    """Adding same indicator twice only registers once."""
    ctx = TranslationContext(symbol="BTC-USD")

    ema = EMA(id="ema_20", period=20)
    ctx.add_indicator(ema)
    ctx.add_indicator(ema)

    assert len(ctx.indicators) == 1
    assert "ema_20" in ctx.indicators


def test_context_add_state_var_registers_once():
    """Adding same state var twice only registers once."""
    ctx = TranslationContext(symbol="BTC-USD")

    spec = StateVarSpec(id="entry_price", var_type="float", default=0.0)
    ctx.add_state_var(spec)
    ctx.add_state_var(spec)

    assert len(ctx.state_vars) == 1
    assert "entry_price" in ctx.state_vars


def test_context_starts_empty():
    """Fresh context has empty collections."""
    ctx = TranslationContext(symbol="BTC-USD")

    assert ctx.indicators == {}
    assert ctx.state_vars == {}
    assert ctx.on_bar_hooks == []
    assert ctx.on_bar_invested_ops == []
    assert ctx.on_fill_ops == []
    assert ctx.additional_symbols == []
