"""Tests for state_collector module.

Tests collect_state for gathering state variables, on_bar hooks, on_fill ops,
and on_bar_invested ops from archetypes, and bool state var declarations from
StateCondition nodes in the condition tree.
"""

from __future__ import annotations

from vibe_trade_shared.models.ir import (
    AllOfCondition,
    AnyOfCondition,
    CompareCondition,
    CompareOp,
    IncrementStateAction,
    IndicatorBandRef,
    LiteralRef,
    NotCondition,
    PriceRef,
    RegimeCondition,
    SequenceCondition,
    SequenceStep,
    SetStateAction,
    StateCondition,
    StateVarSpec,
)
from vibe_trade_shared.models.ir.enums import PriceField

from src.translator.compiler.context import CompilationContext
from src.translator.compiler.state_collector import collect_state

# =============================================================================
# Helpers
# =============================================================================


def _make_ctx() -> CompilationContext:
    return CompilationContext(symbol="BTC-USD")


class _MockArchetype:
    """Minimal archetype mock with configurable state methods."""

    def __init__(
        self,
        state_vars=None,
        on_fill_ops=None,
        on_bar_ops=None,
        on_bar_invested_ops=None,
    ):
        self._state_vars = state_vars or []
        self._on_fill_ops = on_fill_ops or []
        self._on_bar_ops = on_bar_ops or []
        self._on_bar_invested_ops = on_bar_invested_ops or []

    def get_state_vars(self):
        return self._state_vars

    def get_on_fill_ops(self):
        return self._on_fill_ops

    def get_on_bar_ops(self):
        return self._on_bar_ops

    def get_on_bar_invested_ops(self):
        return self._on_bar_invested_ops


# =============================================================================
# Archetype state collection tests
# =============================================================================


class TestArchetypeStateCollection:
    """Test collecting state from archetype methods."""

    def test_state_vars_collected(self):
        archetype = _MockArchetype(
            state_vars=[
                StateVarSpec(id="entry_price", var_type="float", default=0.0),
                StateVarSpec(id="bars_since_entry", var_type="int", default=0),
            ],
        )
        # Use a simple leaf condition (no state)
        condition = CompareCondition(
            left=PriceRef(field=PriceField.CLOSE),
            op=CompareOp.GT,
            right=LiteralRef(value=100.0),
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        assert "entry_price" in ctx.state_vars
        assert "bars_since_entry" in ctx.state_vars
        assert ctx.state_vars["entry_price"].var_type == "float"
        assert ctx.state_vars["bars_since_entry"].var_type == "int"

    def test_on_fill_ops_collected(self):
        on_fill = [
            SetStateAction(
                state_id="entry_price",
                value=PriceRef(field=PriceField.CLOSE),
            ),
            SetStateAction(
                state_id="bars_since_entry",
                value=LiteralRef(value=0),
            ),
        ]
        archetype = _MockArchetype(on_fill_ops=on_fill)
        condition = CompareCondition(
            left=PriceRef(field=PriceField.CLOSE),
            op=CompareOp.GT,
            right=LiteralRef(value=100.0),
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        assert len(ctx.on_fill_ops) == 2
        assert isinstance(ctx.on_fill_ops[0], SetStateAction)
        assert ctx.on_fill_ops[0].state_id == "entry_price"

    def test_on_bar_hooks_collected(self):
        on_bar = [
            IncrementStateAction(state_id="total_bars"),
        ]
        archetype = _MockArchetype(on_bar_ops=on_bar)
        condition = CompareCondition(
            left=PriceRef(field=PriceField.CLOSE),
            op=CompareOp.GT,
            right=LiteralRef(value=100.0),
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        assert len(ctx.on_bar_hooks) == 1
        assert isinstance(ctx.on_bar_hooks[0], IncrementStateAction)
        assert ctx.on_bar_hooks[0].state_id == "total_bars"

    def test_on_bar_invested_ops_collected(self):
        on_bar_invested = [
            IncrementStateAction(state_id="bars_since_entry"),
        ]
        archetype = _MockArchetype(on_bar_invested_ops=on_bar_invested)
        condition = CompareCondition(
            left=PriceRef(field=PriceField.CLOSE),
            op=CompareOp.GT,
            right=LiteralRef(value=100.0),
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        assert len(ctx.on_bar_invested_ops) == 1
        assert ctx.on_bar_invested_ops[0].state_id == "bars_since_entry"


# =============================================================================
# StateCondition bool var declaration tests
# =============================================================================


class TestStateConditionDeclaration:
    """Test that StateCondition nodes in the condition tree declare bool state vars."""

    def test_state_condition_declares_bool_var(self):
        archetype = _MockArchetype()
        condition = StateCondition(
            state_var="outside_bb_lower",
            trigger_on_transition=True,
            outside_condition=CompareCondition(
                left=PriceRef(field=PriceField.CLOSE),
                op=CompareOp.LT,
                right=IndicatorBandRef(indicator_id="bb_20", band="lower"),
            ),
            inside_condition=CompareCondition(
                left=PriceRef(field=PriceField.CLOSE),
                op=CompareOp.GTE,
                right=IndicatorBandRef(indicator_id="bb_20", band="lower"),
            ),
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        assert "outside_bb_lower" in ctx.state_vars
        var = ctx.state_vars["outside_bb_lower"]
        assert var.var_type == "bool"
        assert var.default is False

    def test_nested_state_condition_in_allOf(self):
        """StateCondition nested inside AllOf should still be found."""
        archetype = _MockArchetype()
        state_cond = StateCondition(
            state_var="outside_kc_upper",
            trigger_on_transition=True,
            outside_condition=CompareCondition(
                left=PriceRef(field=PriceField.CLOSE),
                op=CompareOp.GT,
                right=IndicatorBandRef(indicator_id="kc_20", band="upper"),
            ),
            inside_condition=CompareCondition(
                left=PriceRef(field=PriceField.CLOSE),
                op=CompareOp.LTE,
                right=IndicatorBandRef(indicator_id="kc_20", band="upper"),
            ),
        )
        condition = AllOfCondition(
            conditions=[
                RegimeCondition(
                    metric="trend_ma_relation",
                    op=CompareOp.GT,
                    value=1,
                ),
                state_cond,
            ],
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        assert "outside_kc_upper" in ctx.state_vars

    def test_nested_state_condition_in_anyOf(self):
        """StateCondition nested inside AnyOf should still be found."""
        archetype = _MockArchetype()
        condition = AnyOfCondition(
            conditions=[
                StateCondition(
                    state_var="state_a",
                    trigger_on_transition=True,
                    outside_condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=LiteralRef(value=100.0),
                    ),
                ),
                StateCondition(
                    state_var="state_b",
                    trigger_on_transition=True,
                    outside_condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=LiteralRef(value=200.0),
                    ),
                ),
            ],
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        assert "state_a" in ctx.state_vars
        assert "state_b" in ctx.state_vars

    def test_nested_state_condition_in_not(self):
        """StateCondition nested inside Not should still be found."""
        archetype = _MockArchetype()
        condition = NotCondition(
            condition=StateCondition(
                state_var="not_inside_band",
                trigger_on_transition=True,
                outside_condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=LiteralRef(value=100.0),
                ),
            ),
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        assert "not_inside_band" in ctx.state_vars

    def test_state_condition_in_sequence(self):
        """StateCondition inside a SequenceCondition step."""
        archetype = _MockArchetype()
        condition = SequenceCondition(
            steps=[
                SequenceStep(
                    condition=StateCondition(
                        state_var="step1_state",
                        trigger_on_transition=True,
                        outside_condition=CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.LT,
                            right=LiteralRef(value=100.0),
                        ),
                    ),
                    within_bars=5,
                ),
            ],
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        assert "step1_state" in ctx.state_vars


# =============================================================================
# Combined archetype + condition tree state
# =============================================================================


class TestCombinedState:
    """Test collecting state from both archetype and condition tree."""

    def test_both_sources_collected(self):
        """State vars from archetype AND StateCondition should both appear."""
        archetype = _MockArchetype(
            state_vars=[
                StateVarSpec(id="entry_price", var_type="float", default=0.0),
            ],
            on_fill_ops=[
                SetStateAction(
                    state_id="entry_price",
                    value=PriceRef(field=PriceField.CLOSE),
                ),
            ],
            on_bar_invested_ops=[
                IncrementStateAction(state_id="bars_since_entry"),
            ],
        )
        condition = AllOfCondition(
            conditions=[
                RegimeCondition(
                    metric="trend_ma_relation",
                    op=CompareOp.GT,
                    value=1,
                ),
                StateCondition(
                    state_var="outside_bb",
                    trigger_on_transition=True,
                    outside_condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=LiteralRef(value=100.0),
                    ),
                ),
            ],
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        # From archetype
        assert "entry_price" in ctx.state_vars
        assert len(ctx.on_fill_ops) == 1
        assert len(ctx.on_bar_invested_ops) == 1
        # From condition tree
        assert "outside_bb" in ctx.state_vars
        assert ctx.state_vars["outside_bb"].var_type == "bool"

    def test_leaf_condition_no_state(self):
        """Leaf condition with no StateCondition should not add state vars."""
        archetype = _MockArchetype()
        condition = CompareCondition(
            left=PriceRef(field=PriceField.CLOSE),
            op=CompareOp.GT,
            right=LiteralRef(value=100.0),
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        assert len(ctx.state_vars) == 0
        assert len(ctx.on_fill_ops) == 0
        assert len(ctx.on_bar_hooks) == 0
        assert len(ctx.on_bar_invested_ops) == 0

    def test_deduplication(self):
        """Same state var ID from archetype and condition tree should deduplicate."""
        archetype = _MockArchetype(
            state_vars=[
                StateVarSpec(id="my_state", var_type="float", default=0.0),
            ],
        )
        condition = StateCondition(
            state_var="my_state",
            trigger_on_transition=True,
            outside_condition=CompareCondition(
                left=PriceRef(field=PriceField.CLOSE),
                op=CompareOp.LT,
                right=LiteralRef(value=100.0),
            ),
        )
        ctx = _make_ctx()

        collect_state(archetype, condition, ctx)

        # Should only have one entry for "my_state"
        assert len(ctx.state_vars) == 1
        # First one wins (archetype) - float type, not bool
        assert ctx.state_vars["my_state"].var_type == "float"
