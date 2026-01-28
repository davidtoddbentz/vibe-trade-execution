"""Condition processing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_trade_shared.models.ir import Condition

from .visitors import IndicatorCollector, RegimeLowerer, StateExtractor

if TYPE_CHECKING:
    from .context import TranslationContext


class ConditionPipeline:
    """Processes conditions through a series of transformations.

    Pipeline order:
    1. Lower regime conditions to primitives
    2. Collect indicators (works on already-lowered tree)
    3. Extract state variable declarations

    Each phase uses a visitor that traverses the typed Condition tree.
    """

    def __init__(self, ctx: TranslationContext):
        self.ctx = ctx

    def process(self, condition: Condition) -> Condition:
        """Run condition through all transformation phases.

        Args:
            condition: The condition to process

        Returns:
            The transformed condition (may be different type after lowering)
        """
        # Phase 1: Lower regime conditions to primitives
        lowerer = RegimeLowerer()
        condition = lowerer.visit(condition)

        # Phase 2: Collect indicators (works on already-lowered tree)
        collector = IndicatorCollector(self.ctx)
        condition = collector.visit(condition)

        # Phase 3: Extract state variable declarations
        extractor = StateExtractor(self.ctx)
        extractor.visit(condition)

        return condition
