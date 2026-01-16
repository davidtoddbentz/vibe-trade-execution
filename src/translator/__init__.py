"""Strategy to IR translator.

Converts vibe-trade strategy schemas to StrategyIR (structured data).
The StrategyIR is serialized to JSON and interpreted at runtime by LEAN's
StrategyRuntime algorithm.

The translation pipeline:
  1. Schema (JSON from MCP) → IRTranslator → StrategyIR (Python dataclasses)
  2. StrategyIR.to_json() → JSON file
  3. LEAN StrategyRuntime.py reads JSON → executes trades
"""

from .ir import StrategyIR
from .ir_translator import IRTranslator, TranslationError

__all__ = [
    "IRTranslator",
    "TranslationError",
    "StrategyIR",
]
