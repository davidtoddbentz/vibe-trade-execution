"""Lower RegimeCondition to primitive conditions where possible.

This module moves business logic from StrategyRuntime._evaluate_regime()
to the translator layer, ensuring the runtime only interprets data.

Metrics that are lowered to primitives:
- trend_ma_relation -> CompareCondition with IRExpression(EMA_fast - EMA_slow)
- trend_adx -> CompareCondition with IndicatorRef(ADX)
- vol_atr_pct -> CompareCondition with IRExpression(ATR / close * 100)
- vol_bb_width_pctile -> CompareCondition with IndicatorPropertyRef(BB.width_pctile)
- dist_from_vwap_pct -> CompareCondition with IRExpression((close - VWAP) / VWAP * 100)
- ret_pct -> CompareCondition with IndicatorRef(ROC) * 100

Metrics that stay as RegimeCondition (require runtime state):
- liquidity_sweep, flag_pattern, pennant_pattern (complex pattern detection)
- volume_pctile (requires rolling window)
- gap_pct (requires prev close tracking)
"""
from vibe_trade_shared.models.ir import (
    CompareCondition,
    Condition,
    IndicatorRef,
    IRExpression,
    LiteralRef,
    PriceField,
    PriceRef,
    RegimeCondition,
)


def lower_regime_condition(regime: RegimeCondition) -> Condition:
    """Lower a RegimeCondition to primitive conditions where possible.

    Args:
        regime: The RegimeCondition to lower

    Returns:
        Either a lowered primitive Condition or the original RegimeCondition
    """
    metric = regime.metric
    if isinstance(metric, str):
        metric_str = metric
    else:
        metric_str = metric.value

    # Trend metrics
    if metric_str == "trend_ma_relation":
        ma_fast = regime.ma_fast or 20
        ma_slow = regime.ma_slow or 50
        return CompareCondition(
            left=IRExpression(
                op="-",
                left=IndicatorRef(indicator_type="EMA", params={"period": ma_fast}),
                right=IndicatorRef(indicator_type="EMA", params={"period": ma_slow}),
            ),
            op=regime.op,
            right=LiteralRef(value=float(regime.value)),
            semantic_label=f"trend_ma_relation: EMA({ma_fast}) - EMA({ma_slow}) {regime.op.value} {regime.value}",
        )

    if metric_str == "trend_adx":
        return CompareCondition(
            left=IndicatorRef(indicator_type="ADX", params={"period": 14}),
            op=regime.op,
            right=LiteralRef(value=float(regime.value)),
            semantic_label=f"trend_adx: ADX {regime.op.value} {regime.value}",
        )

    # Volatility metrics
    if metric_str == "vol_atr_pct":
        return CompareCondition(
            left=IRExpression(
                op="*",
                left=IRExpression(
                    op="/",
                    left=IndicatorRef(indicator_type="ATR", params={"period": 14}),
                    right=PriceRef(field=PriceField.CLOSE),
                ),
                right=LiteralRef(value=100.0),
            ),
            op=regime.op,
            right=LiteralRef(value=float(regime.value)),
            semantic_label=f"vol_atr_pct: ATR% {regime.op.value} {regime.value}",
        )

    if metric_str == "dist_from_vwap_pct":
        return CompareCondition(
            left=IRExpression(
                op="*",
                left=IRExpression(
                    op="/",
                    left=IRExpression(
                        op="-",
                        left=PriceRef(field=PriceField.CLOSE),
                        right=IndicatorRef(indicator_type="VWAP", params={}),
                    ),
                    right=IndicatorRef(indicator_type="VWAP", params={}),
                ),
                right=LiteralRef(value=100.0),
            ),
            op=regime.op,
            right=LiteralRef(value=float(regime.value)),
            semantic_label=f"dist_from_vwap_pct: VWAP_dist% {regime.op.value} {regime.value}",
        )

    if metric_str == "ret_pct":
        lookback = regime.lookback_bars or 1
        return CompareCondition(
            left=IRExpression(
                op="*",
                left=IndicatorRef(indicator_type="ROC", params={"period": lookback}),
                right=LiteralRef(value=100.0),
            ),
            op=regime.op,
            right=LiteralRef(value=float(regime.value)),
            semantic_label=f"ret_pct: ROC({lookback})% {regime.op.value} {regime.value}",
        )

    # Pattern metrics - cannot be lowered, require runtime state
    if metric_str in ("liquidity_sweep", "flag_pattern", "pennant_pattern",
                      "volume_pctile", "gap_pct", "vol_bb_width_pctile", "bb_width_pctile",
                      "price_level_touch", "price_level_cross"):
        return regime

    # Unknown metric - return as-is for runtime to handle
    return regime
