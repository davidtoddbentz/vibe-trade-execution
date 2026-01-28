"""Translation context for accumulating artifacts during IR translation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .ir import Indicator, StateOp, StateVarSpec


@dataclass
class TranslationContext:
    """Accumulates artifacts during translation.

    This is the single source of truth for all state accumulated during
    the translation process. It replaces scattered instance variables
    in the old IRTranslator.
    """

    symbol: str

    # Accumulated during translation
    indicators: dict[str, Indicator] = field(default_factory=dict)
    state_vars: dict[str, StateVarSpec] = field(default_factory=dict)
    on_bar_hooks: list[StateOp] = field(default_factory=list)
    on_bar_invested_ops: list[StateOp] = field(default_factory=list)
    on_fill_ops: list[StateOp] = field(default_factory=list)
    additional_symbols: list[str] = field(default_factory=list)

    def add_indicator(self, indicator: Indicator) -> None:
        """Add indicator if not already registered.

        Deduplication is by indicator ID. First registration wins.
        """
        if indicator.id not in self.indicators:
            self.indicators[indicator.id] = indicator

    def add_state_var(self, spec: StateVarSpec) -> None:
        """Add state variable if not already registered.

        Deduplication is by state var ID. First registration wins.
        """
        if spec.id not in self.state_vars:
            self.state_vars[spec.id] = spec
