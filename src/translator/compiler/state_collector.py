"""State collector for extracting state declarations from archetypes and conditions.

Combines two sources of state:
1. Archetype-declared state (get_state_vars, get_on_fill_ops, etc.)
2. Condition-tree state (StateCondition nodes declare bool state vars)

This replaces the scattered state collection in the old translator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_trade_shared.models.archetypes.base import BaseArchetype
from vibe_trade_shared.models.ir import (
    AllOfCondition,
    AnyOfCondition,
    Condition,
    NotCondition,
    SequenceCondition,
    StateCondition,
    StateVarSpec,
)

if TYPE_CHECKING:
    from src.translator.compiler.context import CompilationContext


# =============================================================================
# Public API
# =============================================================================


def collect_state(
    archetype: BaseArchetype,
    condition: Condition,
    ctx: CompilationContext,
) -> None:
    """Collect all state declarations from archetype methods + condition tree.

    Gathers state from two sources:
    1. Archetype methods: get_state_vars(), get_on_fill_ops(), get_on_bar_ops(),
       get_on_bar_invested_ops()
    2. Condition tree: StateCondition nodes declare bool state vars for
       tracking outside/inside transitions

    All collected artifacts are registered into the CompilationContext.

    Args:
        archetype: Parsed archetype instance with state declaration methods.
        condition: The compiled IR condition tree to walk for StateCondition nodes.
        ctx: Compilation context to accumulate state into.
    """
    # From archetype
    for var in archetype.get_state_vars():
        ctx.add_state_var(var)

    ctx.on_fill_ops.extend(archetype.get_on_fill_ops())
    ctx.on_bar_hooks.extend(archetype.get_on_bar_ops())
    ctx.on_bar_invested_ops.extend(archetype.get_on_bar_invested_ops())

    # From condition tree: find StateCondition nodes, declare bool state vars
    _walk_for_state_conditions(condition, ctx)


# =============================================================================
# Internal: condition tree walk
# =============================================================================


def _walk_for_state_conditions(condition: Condition, ctx: CompilationContext) -> None:
    """Walk condition tree to find StateCondition nodes and declare their state vars.

    StateConditions use boolean state variables to track whether the previous bar
    was in an "outside" or "inside" state (e.g., price was outside Bollinger Band).
    These bool state vars must be declared in the strategy IR.

    Also recursively walks nested conditions inside StateCondition
    (outside_condition, inside_condition, current_condition) and composite
    conditions (AllOf, AnyOf, Not, Sequence).

    Args:
        condition: The condition tree to walk.
        ctx: Compilation context to register state vars into.
    """
    if isinstance(condition, StateCondition):
        # Declare the boolean state variable for this StateCondition
        ctx.add_state_var(
            StateVarSpec(
                id=condition.state_var,
                var_type="bool",
                default=False,
            )
        )
        # Recurse into nested conditions
        if condition.outside_condition:
            _walk_for_state_conditions(condition.outside_condition, ctx)
        if condition.inside_condition:
            _walk_for_state_conditions(condition.inside_condition, ctx)
        if condition.current_condition:
            _walk_for_state_conditions(condition.current_condition, ctx)
        return

    # Composite conditions - recurse into children
    if isinstance(condition, AllOfCondition):
        for child in condition.conditions:
            _walk_for_state_conditions(child, ctx)
        return

    if isinstance(condition, AnyOfCondition):
        for child in condition.conditions:
            _walk_for_state_conditions(child, ctx)
        return

    if isinstance(condition, NotCondition):
        _walk_for_state_conditions(condition.condition, ctx)
        return

    if isinstance(condition, SequenceCondition):
        for step in condition.steps:
            _walk_for_state_conditions(step.condition, ctx)
        return

    # Leaf conditions - no state to collect
