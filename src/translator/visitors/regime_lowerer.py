"""RegimeLowerer visitor for lowering RegimeCondition to primitive conditions.

This visitor transforms RegimeCondition nodes into primitive CompareCondition
nodes where possible, moving evaluation logic from runtime to translation time.

Metrics that can be lowered have a deterministic mapping to indicator comparisons:
- trend_ma_relation -> CompareCondition(EMA_fast - EMA_slow, op, value)
- trend_adx -> CompareCondition(ADX, op, value)
- vol_atr_pct -> CompareCondition(ATR / price * 100, op, value)
- dist_from_vwap_pct -> CompareCondition((price - VWAP) / VWAP * 100, op, value)
- ret_pct -> CompareCondition(ROC * 100, op, value)

Metrics NOT in the registry (liquidity_sweep, flag_pattern, etc.) pass through
unchanged as they require complex runtime logic.
"""

from __future__ import annotations

from collections.abc import Callable

from vibe_trade_shared.models.ir import (
    AllOfCondition,
    AnyOfCondition,
    CompareCondition,
    Condition,
    IndicatorRef,
    IRExpression,
    LiteralRef,
    NotCondition,
    PriceField,
    PriceRef,
    RegimeCondition,
    SequenceCondition,
    SequenceStep,
)

from src.translator.visitors.base import ConditionVisitor

# Type alias for regime lowering functions
RegimeLoweringFn = Callable[[RegimeCondition], CompareCondition]


# =============================================================================
# Lowering Functions
# =============================================================================


def _lower_trend_ma_relation(regime: RegimeCondition) -> CompareCondition:
    """Lower trend_ma_relation to CompareCondition.

    Formula: (EMA_fast - EMA_slow) op value

    When value=0 and op=GT, this checks if fast EMA is above slow EMA (uptrend).
    """
    fast = regime.ma_fast or 20
    slow = regime.ma_slow or 50

    # Create indicator refs for EMAs
    ema_fast = IndicatorRef(indicator_type="EMA", params={"period": fast})
    ema_slow = IndicatorRef(indicator_type="EMA", params={"period": slow})

    # Create expression: EMA_fast - EMA_slow
    diff_expr = IRExpression(op="-", left=ema_fast, right=ema_slow)

    return CompareCondition(
        left=diff_expr,
        op=regime.op,
        right=LiteralRef(value=float(regime.value)),
        semantic_label=f"trend_ma_relation: EMA({fast}) - EMA({slow}) {regime.op.value} {regime.value}",
    )


def _lower_trend_adx(regime: RegimeCondition) -> CompareCondition:
    """Lower trend_adx to CompareCondition.

    Formula: ADX op value

    ADX > 25 typically indicates a trending market.
    """
    lookback = regime.lookback_bars or 14

    adx_ref = IndicatorRef(indicator_type="ADX", params={"period": lookback})

    return CompareCondition(
        left=adx_ref,
        op=regime.op,
        right=LiteralRef(value=float(regime.value)),
        semantic_label=f"trend_adx: ADX({lookback}) {regime.op.value} {regime.value}",
    )


def _lower_vol_atr_pct(regime: RegimeCondition) -> CompareCondition:
    """Lower vol_atr_pct to CompareCondition.

    Formula: (ATR / close) * 100 op value

    This measures volatility as a percentage of price.
    """
    lookback = regime.lookback_bars or 14

    atr_ref = IndicatorRef(indicator_type="ATR", params={"period": lookback})
    close_ref = PriceRef(field=PriceField.CLOSE)

    # ATR / close
    ratio_expr = IRExpression(op="/", left=atr_ref, right=close_ref)

    # (ATR / close) * 100
    pct_expr = IRExpression(op="*", left=ratio_expr, right=LiteralRef(value=100.0))

    return CompareCondition(
        left=pct_expr,
        op=regime.op,
        right=LiteralRef(value=float(regime.value)),
        semantic_label=f"vol_atr_pct: ATR({lookback})/close*100 {regime.op.value} {regime.value}",
    )


def _lower_dist_from_vwap_pct(regime: RegimeCondition) -> CompareCondition:
    """Lower dist_from_vwap_pct to CompareCondition.

    Formula: ((close - VWAP) / VWAP) * 100 op value

    Positive value means price is above VWAP, negative means below.
    """
    close_ref = PriceRef(field=PriceField.CLOSE)
    vwap_ref = IndicatorRef(indicator_type="VWAP", params={})

    # close - VWAP
    diff_expr = IRExpression(op="-", left=close_ref, right=vwap_ref)

    # (close - VWAP) / VWAP
    ratio_expr = IRExpression(op="/", left=diff_expr, right=vwap_ref)

    # ((close - VWAP) / VWAP) * 100
    pct_expr = IRExpression(op="*", left=ratio_expr, right=LiteralRef(value=100.0))

    return CompareCondition(
        left=pct_expr,
        op=regime.op,
        right=LiteralRef(value=float(regime.value)),
        semantic_label=f"dist_from_vwap_pct: (close-VWAP)/VWAP*100 {regime.op.value} {regime.value}",
    )


def _lower_ret_pct(regime: RegimeCondition) -> CompareCondition:
    """Lower ret_pct to CompareCondition.

    Formula: ROC * 100 op value

    ROC (Rate of Change) already returns a percentage, but we multiply by 100
    to convert from decimal (0.01) to percent (1%).
    """
    lookback = regime.lookback_bars or 1

    roc_ref = IndicatorRef(indicator_type="ROC", params={"period": lookback})

    # ROC * 100
    pct_expr = IRExpression(op="*", left=roc_ref, right=LiteralRef(value=100.0))

    return CompareCondition(
        left=pct_expr,
        op=regime.op,
        right=LiteralRef(value=float(regime.value)),
        semantic_label=f"ret_pct: ROC({lookback})*100 {regime.op.value} {regime.value}",
    )


# =============================================================================
# Registry
# =============================================================================

REGIME_LOWERERS: dict[str, RegimeLoweringFn] = {
    "trend_ma_relation": _lower_trend_ma_relation,
    "trend_regime": _lower_trend_ma_relation,  # Alias
    "trend_adx": _lower_trend_adx,
    "vol_atr_pct": _lower_vol_atr_pct,
    "dist_from_vwap_pct": _lower_dist_from_vwap_pct,
    "ret_pct": _lower_ret_pct,
}


# =============================================================================
# RegimeLowerer Visitor
# =============================================================================


class RegimeLowerer(ConditionVisitor[Condition]):
    """Visitor that lowers RegimeCondition to primitive CompareCondition.

    Uses a registry pattern - metrics in REGIME_LOWERERS are transformed,
    metrics not in the registry pass through unchanged.

    This allows moving evaluation logic from runtime to translation time
    for deterministic regime conditions.

    Usage:
        lowerer = RegimeLowerer()
        lowered_condition = lowerer.visit(condition)
    """

    def visit_default(self, condition: Condition) -> Condition:
        """Default handler: pass through unchanged.

        Unknown condition types are returned as-is.
        """
        return condition

    def visit_RegimeCondition(self, cond: RegimeCondition) -> Condition:
        """Visit RegimeCondition: lower if possible, else pass through.

        Looks up the metric in REGIME_LOWERERS registry:
        - If found: applies the lowering function to transform to CompareCondition
        - If not found: returns the original RegimeCondition unchanged
        """
        lowerer = REGIME_LOWERERS.get(cond.metric)
        if lowerer is not None:
            return lowerer(cond)
        return cond

    # =========================================================================
    # Combine methods - preserve tree structure with transformed children
    # =========================================================================

    def combine_all_of(self, original: AllOfCondition, children: list[Condition]) -> Condition:
        """Combine AllOfCondition with transformed children."""
        return AllOfCondition(
            conditions=children,
            semantic_label=original.semantic_label,
        )

    def combine_any_of(self, original: AnyOfCondition, children: list[Condition]) -> Condition:
        """Combine AnyOfCondition with transformed children."""
        return AnyOfCondition(
            conditions=children,
            semantic_label=original.semantic_label,
        )

    def combine_not(self, original: NotCondition, child: Condition) -> Condition:
        """Combine NotCondition with transformed child."""
        return NotCondition(
            condition=child,
            semantic_label=original.semantic_label,
        )

    def combine_sequence(
        self, original: SequenceCondition, steps: list[tuple[SequenceStep, Condition]]
    ) -> Condition:
        """Combine SequenceCondition with transformed step conditions."""
        new_steps = [
            SequenceStep(
                condition=transformed_cond,
                hold_bars=step.hold_bars,
                within_bars=step.within_bars,
            )
            for step, transformed_cond in steps
        ]
        return SequenceCondition(
            steps=new_steps,
            semantic_label=original.semantic_label,
        )
