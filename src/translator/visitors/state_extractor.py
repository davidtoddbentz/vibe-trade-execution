"""Visitor that extracts state variable declarations from Condition trees."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_trade_shared.models.ir import (
    AllOfCondition,
    AnyOfCondition,
    Condition,
    NotCondition,
    SequenceCondition,
    SequenceStep,
    StateCondition,
    StateVarSpec,
)

from .base import ConditionVisitor

if TYPE_CHECKING:
    from src.translator.context import TranslationContext


class StateExtractor(ConditionVisitor[None]):
    """Extracts state variable declarations from StateCondition nodes.

    Walks the condition tree and declares state variables for any
    StateCondition nodes found. StateConditions use boolean state variables
    to track outside/inside transitions.

    Usage:
        ctx = TranslationContext(symbol="BTC-USD")
        extractor = StateExtractor(ctx)
        extractor.visit(condition)
        # ctx.state_vars now contains all declared state variables
    """

    def __init__(self, ctx: TranslationContext) -> None:
        """Initialize with a TranslationContext.

        Args:
            ctx: The context to accumulate state variable declarations into
        """
        self.ctx = ctx

    def visit_default(self, condition: Condition) -> None:
        """Default handler: no state extraction needed.

        Most condition types don't declare state variables.
        """
        pass

    def visit_StateCondition(self, cond: StateCondition) -> None:
        """Declare state variable for this StateCondition.

        StateConditions always use boolean state variables to track
        whether the previous bar was outside/inside a band or condition.
        """
        self.ctx.add_state_var(
            StateVarSpec(
                id=cond.state_var,
                var_type="bool",
                default=False,
            )
        )

        # Also visit nested conditions to extract any state vars from them
        if cond.outside_condition:
            self.visit(cond.outside_condition)
        if cond.inside_condition:
            self.visit(cond.inside_condition)
        if cond.current_condition:
            self.visit(cond.current_condition)

    # =========================================================================
    # Combine methods - traverse children but return None
    # =========================================================================

    def combine_all_of(self, original: AllOfCondition, children: list[None]) -> None:
        """Combine AllOfCondition - state already extracted during traversal."""
        pass

    def combine_any_of(self, original: AnyOfCondition, children: list[None]) -> None:
        """Combine AnyOfCondition - state already extracted during traversal."""
        pass

    def combine_not(self, original: NotCondition, child: None) -> None:
        """Combine NotCondition - state already extracted during traversal."""
        pass

    def combine_sequence(
        self, original: SequenceCondition, steps: list[tuple[SequenceStep, None]]
    ) -> None:
        """Combine SequenceCondition - state already extracted during traversal."""
        pass
