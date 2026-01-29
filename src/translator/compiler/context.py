"""Compilation context for strategy translation.

Accumulates indicators, state variables, hooks, and additional symbols
during the compilation phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vibe_trade_shared.models.ir.indicators import IndicatorUnion
from vibe_trade_shared.models.ir.strategy_ir import StateOp, StateVarSpec


@dataclass
class CompilationContext:
    """Accumulation context for all compilation artifacts.

    Used by condition_builder, indicator_resolver, and state_collector
    to register discovered artifacts during compilation.
    """

    symbol: str
    indicators: dict[str, IndicatorUnion] = field(default_factory=dict)
    state_vars: dict[str, StateVarSpec] = field(default_factory=dict)
    on_bar_hooks: list[StateOp] = field(default_factory=list)
    on_bar_invested_ops: list[StateOp] = field(default_factory=list)
    on_fill_ops: list[StateOp] = field(default_factory=list)
    additional_symbols: list[str] = field(default_factory=list)

    def add_indicator(self, indicator: IndicatorUnion) -> str:
        """Add indicator, deduplicate by ID. Returns the indicator's ID."""
        self.indicators.setdefault(indicator.id, indicator)
        return indicator.id

    def add_state_var(self, var: StateVarSpec) -> None:
        """Add state variable, deduplicate by ID."""
        self.state_vars.setdefault(var.id, var)

    def add_additional_symbol(self, symbol: str) -> None:
        """Add an additional symbol to subscribe to."""
        if symbol not in self.additional_symbols:
            self.additional_symbols.append(symbol)
