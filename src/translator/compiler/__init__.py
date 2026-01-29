"""Compiler package for strategy translation.

Phases:
1. condition_builder  - Build typed IR Condition from archetype
2. indicator_resolver - Resolve inline IndicatorRefs to indicator_id refs
3. state_collector    - Collect state vars from archetype + condition tree

All phases accumulate artifacts into CompilationContext.
"""

from src.translator.compiler.condition_builder import build_condition, compile_condition_spec
from src.translator.compiler.context import CompilationContext
from src.translator.compiler.indicator_resolver import (
    resolve_condition,
    resolve_state_ops,
    resolve_value_ref,
)
from src.translator.compiler.state_collector import collect_state

__all__ = [
    "CompilationContext",
    "build_condition",
    "collect_state",
    "compile_condition_spec",
    "resolve_condition",
    "resolve_state_ops",
    "resolve_value_ref",
]
