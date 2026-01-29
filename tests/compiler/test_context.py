"""Tests for CompilationContext."""

from vibe_trade_shared.models.ir.indicators import EMA, SMA
from vibe_trade_shared.models.ir.strategy_ir import StateVarSpec

from src.translator.compiler.context import CompilationContext


def test_add_indicator_deduplicates():
    """Adding same indicator ID twice keeps first."""
    ctx = CompilationContext(symbol="BTC-USD")

    ema1 = EMA(id="ema_20", period=20)
    ema2 = EMA(id="ema_20", period=50)
    ctx.add_indicator(ema1)
    ctx.add_indicator(ema2)

    assert len(ctx.indicators) == 1
    assert ctx.indicators["ema_20"].period == 20


def test_add_indicator_returns_id():
    """Returns the indicator's ID."""
    ctx = CompilationContext(symbol="BTC-USD")

    ema = EMA(id="ema_20", period=20)
    result = ctx.add_indicator(ema)

    assert result == "ema_20"


def test_add_state_var_deduplicates():
    """Adding same state var ID twice keeps first."""
    ctx = CompilationContext(symbol="BTC-USD")

    var1 = StateVarSpec(id="entry_price", var_type="float", default=0.0)
    var2 = StateVarSpec(id="entry_price", var_type="float", default=999.0)
    ctx.add_state_var(var1)
    ctx.add_state_var(var2)

    assert len(ctx.state_vars) == 1
    assert ctx.state_vars["entry_price"].default == 0.0


def test_add_additional_symbol_deduplicates():
    """Adding same symbol twice keeps first."""
    ctx = CompilationContext(symbol="BTC-USD")

    ctx.add_additional_symbol("ETH-USD")
    ctx.add_additional_symbol("ETH-USD")

    assert len(ctx.additional_symbols) == 1
    assert ctx.additional_symbols[0] == "ETH-USD"


def test_empty_context():
    """Fresh context has empty collections."""
    ctx = CompilationContext(symbol="BTC-USD")

    assert ctx.indicators == {}
    assert ctx.state_vars == {}
    assert ctx.on_bar_hooks == []
    assert ctx.on_bar_invested_ops == []
    assert ctx.on_fill_ops == []
    assert ctx.additional_symbols == []


def test_add_different_indicators():
    """Different IDs all stored."""
    ctx = CompilationContext(symbol="BTC-USD")

    ema = EMA(id="ema_20", period=20)
    sma = SMA(id="sma_50", period=50)
    ctx.add_indicator(ema)
    ctx.add_indicator(sma)

    assert len(ctx.indicators) == 2
    assert "ema_20" in ctx.indicators
    assert "sma_50" in ctx.indicators
