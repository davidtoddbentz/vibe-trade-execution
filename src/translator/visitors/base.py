"""Base visitor class for typed Condition tree traversal."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from vibe_trade_shared.models.ir import (
        AllOfCondition,
        AnyOfCondition,
        Condition,
        NotCondition,
        SequenceCondition,
        SequenceStep,
    )

T = TypeVar("T")


class ConditionVisitor(ABC, Generic[T]):
    """Abstract visitor for Condition trees.

    Subclasses implement visit methods for specific condition types.
    The base class handles tree traversal for composite conditions
    (AllOf, AnyOf, Not, Sequence).

    Type parameter T is the return type of visit methods.

    Usage:
        class MyVisitor(ConditionVisitor[Condition]):
            def visit_default(self, cond):
                return cond  # pass through unchanged

            def visit_RegimeCondition(self, cond):
                return lower_regime(cond)  # transform
    """

    def visit(self, condition: Condition) -> T:
        """Dispatch to the appropriate visit method.

        Looks for visit_{ClassName} method, falls back to visit_default.
        """
        method_name = f"visit_{type(condition).__name__}"
        visitor = getattr(self, method_name, self.visit_default)
        return visitor(condition)

    @abstractmethod
    def visit_default(self, condition: Condition) -> T:
        """Default handler for condition types without specific visit methods.

        Subclasses must implement this to define default behavior.
        """
        ...

    # Composite conditions - traverse children and combine results

    def visit_AllOfCondition(self, cond: AllOfCondition) -> T:
        """Visit AllOfCondition: traverse all children, then combine."""
        visited_children = [self.visit(c) for c in cond.conditions]
        return self.combine_all_of(cond, visited_children)

    def visit_AnyOfCondition(self, cond: AnyOfCondition) -> T:
        """Visit AnyOfCondition: traverse all children, then combine."""
        visited_children = [self.visit(c) for c in cond.conditions]
        return self.combine_any_of(cond, visited_children)

    def visit_NotCondition(self, cond: NotCondition) -> T:
        """Visit NotCondition: traverse child, then combine."""
        visited_child = self.visit(cond.condition)
        return self.combine_not(cond, visited_child)

    def visit_SequenceCondition(self, cond: SequenceCondition) -> T:
        """Visit SequenceCondition: traverse each step's condition, then combine."""
        visited_steps: list[tuple[SequenceStep, T]] = []
        for step in cond.steps:
            visited_cond = self.visit(step.condition)
            visited_steps.append((step, visited_cond))
        return self.combine_sequence(cond, visited_steps)

    # Combine methods - subclasses override to customize combination logic

    @abstractmethod
    def combine_all_of(self, original: AllOfCondition, children: list[T]) -> T:
        """Combine results from AllOfCondition children."""
        ...

    @abstractmethod
    def combine_any_of(self, original: AnyOfCondition, children: list[T]) -> T:
        """Combine results from AnyOfCondition children."""
        ...

    @abstractmethod
    def combine_not(self, original: NotCondition, child: T) -> T:
        """Combine result from NotCondition child."""
        ...

    @abstractmethod
    def combine_sequence(
        self, original: SequenceCondition, steps: list[tuple[SequenceStep, T]]
    ) -> T:
        """Combine results from SequenceCondition steps."""
        ...
