"""Regime indicator registry.

Maps regime metrics to their required indicators declaratively.
This replaces the giant if/elif chain in _process_inline_indicators.
"""

from __future__ import annotations

from collections.abc import Callable

from vibe_trade_shared.models.ir import RegimeCondition

from src.translator.ir import (
    ADX,
    ATR,
    EMA,
    VWAP,
    BollingerBands,
    Indicator,
    Maximum,
    Minimum,
    Percentile,
    RateOfChange,
    RollingWindow,
    VolumeSMA,
)
from src.translator.errors import TranslationError

# Type alias for regime indicator handler functions
RegimeIndicatorHandler = Callable[[RegimeCondition], list[Indicator]]


# =============================================================================
# Handler Functions
# Each handler takes a RegimeCondition and returns a list of required indicators.
# =============================================================================


def _trend_ma_relation(regime: RegimeCondition) -> list[Indicator]:
    """trend_ma_relation needs fast and slow EMA."""
    fast = regime.ma_fast or 20
    slow = regime.ma_slow or 50
    return [
        EMA(id=f"ema_{fast}", period=fast),
        EMA(id=f"ema_{slow}", period=slow),
    ]


def _trend_adx(regime: RegimeCondition) -> list[Indicator]:
    """trend_adx needs ADX indicator."""
    lookback = regime.lookback_bars or 14
    return [ADX(id=f"adx_{lookback}", period=lookback)]


def _vol_atr_pct(regime: RegimeCondition) -> list[Indicator]:
    """vol_atr_pct needs ATR indicator."""
    lookback = regime.lookback_bars or 14
    return [ATR(id=f"atr_{lookback}", period=lookback)]


def _vol_bb_width_pctile(regime: RegimeCondition) -> list[Indicator]:
    """vol_bb_width_pctile needs Bollinger Bands indicator."""
    length = regime.lookback_bars or 20
    return [BollingerBands(id=f"bb_{length}", period=length, multiplier=2.0)]


def _dist_from_vwap_pct(regime: RegimeCondition) -> list[Indicator]:
    """dist_from_vwap_pct needs VWAP indicator."""
    return [VWAP(id="vwap")]


def _ret_pct(regime: RegimeCondition) -> list[Indicator]:
    """ret_pct needs RateOfChange indicator."""
    lookback = regime.lookback_bars or 1
    return [RateOfChange(id=f"roc_{lookback}", period=lookback)]


def _volume_pctile(regime: RegimeCondition) -> list[Indicator]:
    """volume_pctile needs PercentileRank on volume."""
    pctile_period = regime.pctile_period or 100
    return [
        Percentile(
            id=f"volume_pctile_{pctile_period}",
            period=pctile_period,
            source="volume",
        )
    ]


def _liquidity_sweep(regime: RegimeCondition) -> list[Indicator]:
    """liquidity_sweep needs MAX/MIN indicators for level detection."""
    lookback = regime.lookback_bars or 20
    return [
        Minimum(id=f"min_{lookback}", period=lookback),
        Maximum(id=f"max_{lookback}", period=lookback),
    ]


def _price_level_touch(regime: RegimeCondition) -> list[Indicator]:
    """price_level_touch needs MAX/MIN indicators for dynamic levels."""
    lookback = regime.lookback_bars or 20
    return [
        Minimum(id=f"min_{lookback}", period=lookback),
        Maximum(id=f"max_{lookback}", period=lookback),
    ]


def _price_level_cross(regime: RegimeCondition) -> list[Indicator]:
    """price_level_cross needs MAX/MIN indicators for dynamic levels."""
    lookback = regime.lookback_bars or 20
    return [
        Minimum(id=f"min_{lookback}", period=lookback),
        Maximum(id=f"max_{lookback}", period=lookback),
    ]


def _gap_pct(regime: RegimeCondition) -> list[Indicator]:
    """gap_pct needs rolling window to access previous close."""
    return [RollingWindow(id="prev_close", period=2)]


def _volume_spike(regime: RegimeCondition) -> list[Indicator]:
    """volume_spike needs volume SMA for comparison."""
    lookback = regime.lookback_bars or 20
    return [VolumeSMA(id=f"vol_sma_{lookback}", period=lookback)]


def _volume_dip(regime: RegimeCondition) -> list[Indicator]:
    """volume_dip needs volume SMA for comparison."""
    lookback = regime.lookback_bars or 20
    return [VolumeSMA(id=f"vol_sma_{lookback}", period=lookback)]


def _pennant_pattern(regime: RegimeCondition) -> list[Indicator]:
    """pennant_pattern needs momentum and consolidation indicators."""
    momentum_bars = regime.pennant_momentum_bars or 5
    consolidation_bars = regime.pennant_consolidation_bars or 10
    return [
        RateOfChange(id="momentum_roc", period=momentum_bars),
        ATR(id="pattern_atr", period=consolidation_bars),
        Maximum(id="pattern_max", period=consolidation_bars),
        Minimum(id="pattern_min", period=consolidation_bars),
    ]


def _flag_pattern(regime: RegimeCondition) -> list[Indicator]:
    """flag_pattern needs similar indicators to pennant."""
    momentum_bars = regime.flag_momentum_bars or 5
    consolidation_bars = regime.flag_consolidation_bars or 10
    return [
        RateOfChange(id="momentum_roc", period=momentum_bars),
        ATR(id="pattern_atr", period=consolidation_bars),
        Maximum(id="pattern_max", period=consolidation_bars),
        Minimum(id="pattern_min", period=consolidation_bars),
    ]


def _no_indicators(regime: RegimeCondition) -> list[Indicator]:
    """Metrics that don't require indicators."""
    return []


# =============================================================================
# Registry
# =============================================================================

REGIME_INDICATOR_HANDLERS: dict[str, RegimeIndicatorHandler] = {
    # Trend metrics
    "trend_ma_relation": _trend_ma_relation,
    "trend_regime": _trend_ma_relation,  # Alias
    "trend_adx": _trend_adx,
    # Volatility metrics
    "vol_atr_pct": _vol_atr_pct,
    "vol_bb_width_pctile": _vol_bb_width_pctile,
    "bb_width_pctile": _vol_bb_width_pctile,  # Alias
    "vol_regime": _vol_bb_width_pctile,  # Alias
    # Distance metrics
    "dist_from_vwap_pct": _dist_from_vwap_pct,
    # Return metrics
    "ret_pct": _ret_pct,
    # Volume metrics
    "volume_pctile": _volume_pctile,
    "volume_spike": _volume_spike,
    "volume_dip": _volume_dip,
    # Level metrics
    "liquidity_sweep": _liquidity_sweep,
    "price_level_touch": _price_level_touch,
    "price_level_cross": _price_level_cross,
    # Gap metrics
    "gap_pct": _gap_pct,
    # Pattern metrics
    "flag_pattern": _flag_pattern,
    "pennant_pattern": _pennant_pattern,
    # Metrics that don't require indicators
    "session_phase": _no_indicators,
    "risk_event_prob": _no_indicators,
}


def get_regime_indicators(regime: RegimeCondition) -> list[Indicator]:
    """Get the list of indicators required for a regime condition.

    Args:
        regime: The RegimeCondition to process

    Returns:
        List of typed Indicator objects required by this regime

    Raises:
        TranslationError: If the metric is unknown
    """
    handler = REGIME_INDICATOR_HANDLERS.get(regime.metric)
    if handler is None:
        raise TranslationError(f"Unknown regime metric: {regime.metric}")

    return handler(regime)
