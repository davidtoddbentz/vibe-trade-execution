"""Tests for ConditionVisitor base class."""

from vibe_trade_shared.models.ir import (
    AllOfCondition,
    AnyOfCondition,
    CompareCondition,
    CompareOp,
    LiteralRef,
    NotCondition,
    PriceField,
    PriceRef,
)

from src.translator.visitors.base import ConditionVisitor


class CountingVisitor(ConditionVisitor[int]):
    """Test visitor that counts condition nodes."""

    def visit_default(self, condition) -> int:
        return 1

    def combine_all_of(self, original, children: list[int]) -> int:
        return 1 + sum(children)

    def combine_any_of(self, original, children: list[int]) -> int:
        return 1 + sum(children)

    def combine_not(self, original, child: int) -> int:
        return 1 + child

    def combine_sequence(self, original, steps) -> int:
        return 1 + sum(c for _, c in steps)


def test_visitor_counts_single_condition():
    """Single condition returns 1."""
    visitor = CountingVisitor()
    cond = CompareCondition(
        left=PriceRef(field=PriceField.CLOSE),
        op=CompareOp.GT,
        right=LiteralRef(value=100.0),
    )
    assert visitor.visit(cond) == 1


def test_visitor_counts_all_of_children():
    """AllOf with 3 children returns 4 (1 + 3)."""
    visitor = CountingVisitor()
    child = CompareCondition(
        left=PriceRef(field=PriceField.CLOSE),
        op=CompareOp.GT,
        right=LiteralRef(value=100.0),
    )
    cond = AllOfCondition(conditions=[child, child, child])
    assert visitor.visit(cond) == 4


def test_visitor_counts_nested_structure():
    """Nested AllOf/AnyOf/Not structure counts all nodes."""
    visitor = CountingVisitor()
    leaf = CompareCondition(
        left=PriceRef(field=PriceField.CLOSE),
        op=CompareOp.GT,
        right=LiteralRef(value=100.0),
    )
    # Structure: AllOf(AnyOf(leaf, leaf), Not(leaf))
    # Count: 1 (AllOf) + 1 (AnyOf) + 1 + 1 + 1 (Not) + 1 = 6
    inner_any = AnyOfCondition(conditions=[leaf, leaf])
    inner_not = NotCondition(condition=leaf)
    outer = AllOfCondition(conditions=[inner_any, inner_not])

    assert visitor.visit(outer) == 6
