"""Strategy to IR translator.

Converts vibe-trade strategy schemas to StrategyIR (structured data).
The StrategyIR is serialized to JSON and interpreted at runtime by LEAN's
StrategyRuntime algorithm.

The translation pipeline:
  1. Schema (JSON from MCP) → IRTranslator → StrategyIR (Python dataclasses)
  2. StrategyIR.to_json() → JSON file
  3. LEAN StrategyRuntime.py reads JSON → executes trades

All evaluation logic is duplicated in two places:
  - src/translator/evaluator.py (Python, used for simulation tests)
  - lean/Algorithms/StrategyRuntime.py (Python in LEAN, actual execution)
"""

# IR-based translator
from .evaluator import (
    ActionExecutor,
    ConditionEvaluator,
    EvalContext,
    ExecContext,
    StateOperator,
    ValueResolver,
)
from .ir import StrategyIR
from .ir_translator import IRTranslationResult, IRTranslator

__all__ = [
    "IRTranslator",
    "IRTranslationResult",
    "StrategyIR",
    "ConditionEvaluator",
    "ValueResolver",
    "ActionExecutor",
    "StateOperator",
    "EvalContext",
    "ExecContext",
]
