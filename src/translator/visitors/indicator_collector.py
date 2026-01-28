"""IndicatorCollector visitor for extracting indicators from Condition trees.

Walks a Condition tree and collects all required indicators into the TranslationContext.
Returns the condition unchanged (for pipeline chaining).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vibe_trade_shared.models.ir import (
    AllOfCondition,
    AnyOfCondition,
    BreakoutCondition,
    CompareCondition,
    Condition,
    CrossCondition,
    IndicatorBandRef,
    IndicatorPropertyRef,
    IndicatorRef,
    IntermarketCondition,
    IRExpression,
    NotCondition,
    RegimeCondition,
    SequenceCondition,
    SequenceStep,
    SpreadCondition,
    SqueezeCondition,
    StateCondition,
    TrailingBreakoutCondition,
    ValueRef,
)

from src.translator.ir import (
    BollingerBands,
    DonchianChannel,
    KeltnerChannel,
    Maximum,
    Minimum,
)
from src.translator.registries.indicators import create_indicator_from_ref
from src.translator.registries.regimes import get_regime_indicators
from src.translator.visitors.base import ConditionVisitor

if TYPE_CHECKING:
    from src.translator.context import TranslationContext


class IndicatorCollector(ConditionVisitor[Condition]):
    """Visitor that collects indicators from Condition trees.

    Walks the condition tree and extracts:
    1. Indicators from IndicatorRef nodes in ValueRefs
    2. Indicators required by specific condition types (RegimeCondition, etc.)

    The visitor returns conditions unchanged, allowing it to be chained in a pipeline.

    Usage:
        ctx = TranslationContext(symbol="BTC-USD")
        collector = IndicatorCollector(ctx)
        collector.visit(condition)
        # ctx.indicators now contains all required indicators
    """

    def __init__(self, ctx: TranslationContext) -> None:
        """Initialize with a TranslationContext.

        Args:
            ctx: The context to accumulate indicators into
        """
        self.ctx = ctx

    def visit_default(self, condition: Condition) -> Condition:
        """Default handler: pass through unchanged.

        Unknown condition types are returned as-is without indicator extraction.
        """
        return condition

    # =========================================================================
    # ValueRef extraction helpers
    # =========================================================================

    def _collect_from_value_ref(self, ref: ValueRef) -> None:
        """Recursively extract indicators from a ValueRef.

        Handles:
        - IndicatorRef: Creates and registers the indicator
        - IndicatorBandRef: Extracts indicator_id and creates appropriate band indicator
        - IndicatorPropertyRef: Extracts indicator_id reference
        - IRExpression: Recursively processes left and right operands
        - Other ValueRef types: Ignored (no indicators needed)
        """
        if isinstance(ref, IndicatorRef):
            indicator = create_indicator_from_ref(ref)
            self.ctx.add_indicator(indicator)
        elif isinstance(ref, IndicatorBandRef):
            # IndicatorBandRef references a pre-declared indicator by ID
            # The indicator should already be in the context or will be added elsewhere
            pass
        elif isinstance(ref, IndicatorPropertyRef):
            # IndicatorPropertyRef references a pre-declared indicator by ID
            # The indicator should already be in the context or will be added elsewhere
            pass
        elif isinstance(ref, IRExpression):
            # Recursively process both sides of the expression
            self._collect_from_value_ref(ref.left)
            self._collect_from_value_ref(ref.right)
        # Other ValueRef types (PriceRef, LiteralRef, VolumeRef, etc.) don't need indicators

    # =========================================================================
    # Condition-specific visit methods
    # =========================================================================

    def visit_CompareCondition(self, cond: CompareCondition) -> Condition:
        """Extract indicators from CompareCondition left/right sides."""
        self._collect_from_value_ref(cond.left)
        self._collect_from_value_ref(cond.right)
        return cond

    def visit_CrossCondition(self, cond: CrossCondition) -> Condition:
        """Extract indicators from CrossCondition left/right sides."""
        self._collect_from_value_ref(cond.left)
        self._collect_from_value_ref(cond.right)
        return cond

    def visit_RegimeCondition(self, cond: RegimeCondition) -> Condition:
        """Extract indicators required by RegimeCondition.

        Uses the regime registry to determine which indicators are needed
        for this specific regime metric.
        """
        indicators = get_regime_indicators(cond)
        for indicator in indicators:
            self.ctx.add_indicator(indicator)
        return cond

    def visit_BreakoutCondition(self, cond: BreakoutCondition) -> Condition:
        """Extract MAX/MIN indicators from BreakoutCondition.

        BreakoutCondition needs rolling max and min to detect breakouts.
        """
        lookback = cond.lookback_bars
        self.ctx.add_indicator(Maximum(id=f"max_{lookback}", period=lookback))
        self.ctx.add_indicator(Minimum(id=f"min_{lookback}", period=lookback))
        return cond

    def visit_SqueezeCondition(self, cond: SqueezeCondition) -> Condition:
        """Extract BB/KC indicators from SqueezeCondition.

        SqueezeCondition typically needs Bollinger Bands and/or Keltner Channels
        for squeeze detection.
        """
        # Default period for band indicators
        period = 20

        # For squeeze detection we need both BB and KC
        self.ctx.add_indicator(BollingerBands(id=f"bb_{period}", period=period, multiplier=2.0))
        self.ctx.add_indicator(KeltnerChannel(id=f"kc_{period}", period=period, multiplier=2.0))
        return cond

    def visit_TrailingBreakoutCondition(self, cond: TrailingBreakoutCondition) -> Condition:
        """Extract band indicator from TrailingBreakoutCondition.

        Creates the appropriate band indicator (BB, KC, or DC) based on band_type.
        """
        period = cond.band_length
        band_type = cond.band_type

        if band_type == "bollinger":
            self.ctx.add_indicator(
                BollingerBands(
                    id=f"bb_{period}",
                    period=period,
                    multiplier=cond.band_mult,
                )
            )
        elif band_type == "keltner":
            self.ctx.add_indicator(
                KeltnerChannel(
                    id=f"kc_{period}",
                    period=period,
                    multiplier=cond.band_mult,
                )
            )
        elif band_type == "donchian":
            self.ctx.add_indicator(DonchianChannel(id=f"dc_{period}", period=period))
        return cond

    def visit_SpreadCondition(self, cond: SpreadCondition) -> Condition:
        """Add additional symbols from SpreadCondition.

        SpreadCondition requires both symbol_a and symbol_b to be subscribed.
        """
        # Add both symbols to additional_symbols if not the main symbol
        if cond.symbol_a != self.ctx.symbol and cond.symbol_a not in self.ctx.additional_symbols:
            self.ctx.additional_symbols.append(cond.symbol_a)
        if cond.symbol_b != self.ctx.symbol and cond.symbol_b not in self.ctx.additional_symbols:
            self.ctx.additional_symbols.append(cond.symbol_b)
        return cond

    def visit_IntermarketCondition(self, cond: IntermarketCondition) -> Condition:
        """Add leader symbol from IntermarketCondition.

        The leader symbol needs to be subscribed for intermarket analysis.
        """
        if (
            cond.leader_symbol != self.ctx.symbol
            and cond.leader_symbol not in self.ctx.additional_symbols
        ):
            self.ctx.additional_symbols.append(cond.leader_symbol)
        return cond

    def visit_StateCondition(self, cond: StateCondition) -> Condition:
        """Extract indicators from StateCondition's nested conditions.

        StateCondition can have outside_condition, inside_condition, and
        current_condition that may contain indicator references.
        """
        # Process nested conditions if they exist
        if cond.outside_condition:
            self.visit(cond.outside_condition)
        if cond.inside_condition:
            self.visit(cond.inside_condition)
        if cond.current_condition:
            self.visit(cond.current_condition)
        return cond

    # =========================================================================
    # Combine methods - preserve tree structure
    # =========================================================================

    def combine_all_of(self, original: AllOfCondition, children: list[Condition]) -> Condition:
        """Combine AllOfCondition - return original (indicators already collected)."""
        return original

    def combine_any_of(self, original: AnyOfCondition, children: list[Condition]) -> Condition:
        """Combine AnyOfCondition - return original (indicators already collected)."""
        return original

    def combine_not(self, original: NotCondition, child: Condition) -> Condition:
        """Combine NotCondition - return original (indicators already collected)."""
        return original

    def combine_sequence(
        self, original: SequenceCondition, steps: list[tuple[SequenceStep, Condition]]
    ) -> Condition:
        """Combine SequenceCondition - return original (indicators already collected)."""
        return original
