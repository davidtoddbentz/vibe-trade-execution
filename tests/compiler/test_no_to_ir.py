"""Guard tests: all archetypes use to_condition_spec(), no to_ir() fallback.

Phase 2 unified all archetypes onto a single compilation path:
  archetype.to_condition_spec() -> compile_condition_spec() -> typed IR Condition

These tests prevent regressions to the old to_ir() direct-IR path.
"""

import inspect

from vibe_trade_shared.models.archetypes.base import BaseArchetype

from src.translator.compiler import condition_builder


def test_no_to_ir_on_base_archetype():
    """BaseArchetype must not define to_ir() — all archetypes use to_condition_spec()."""
    assert "to_ir" not in BaseArchetype.__dict__, (
        "BaseArchetype must not define to_ir() — "
        "all archetypes compile via to_condition_spec() -> compile_condition_spec()"
    )


def test_no_dispatch_direct_builder():
    """_dispatch_direct_builder must be deleted — no fallback to to_ir()."""
    assert not hasattr(condition_builder, "_dispatch_direct_builder"), (
        "_dispatch_direct_builder still exists in condition_builder — "
        "all archetypes should compile via compile_condition_spec()"
    )


def test_all_condition_spec_types_have_handlers():
    """Every ConditionSpec type literal must have a dispatch case in compile_condition_spec()."""
    from typing import get_args

    from vibe_trade_shared.models.archetypes.primitives import ConditionSpec

    # Extract all valid type literals from the Literal annotation
    type_field = ConditionSpec.model_fields["type"]
    valid_types = get_args(type_field.annotation)
    assert len(valid_types) > 0, "Could not extract ConditionSpec type literals"

    # Check that compile_condition_spec source contains a dispatch for each type
    source = inspect.getsource(condition_builder.compile_condition_spec)
    missing = [t for t in valid_types if f'"{t}"' not in source]
    assert not missing, (
        f"compile_condition_spec() missing handlers for: {missing}\n"
        "Every ConditionSpec type must have a dispatch case."
    )
