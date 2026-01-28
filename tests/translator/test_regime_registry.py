"""Tests for regime indicator registry."""

import pytest
from vibe_trade_shared.models.ir import CompareOp, RegimeCondition

from src.translator.ir import ADX, ATR, EMA, VWAP, RateOfChange
from src.translator.registries.regimes import (
    REGIME_INDICATOR_HANDLERS,
    get_regime_indicators,
)


def test_trend_ma_relation_creates_two_emas():
    """trend_ma_relation needs fast and slow EMA."""
    regime = RegimeCondition(
        metric="trend_ma_relation",
        op=CompareOp.GT,
        value=0,
        ma_fast=10,
        ma_slow=30,
    )
    indicators = get_regime_indicators(regime)

    assert len(indicators) == 2
    assert any(isinstance(i, EMA) and i.period == 10 for i in indicators)
    assert any(isinstance(i, EMA) and i.period == 30 for i in indicators)


def test_trend_ma_relation_uses_defaults():
    """trend_ma_relation uses default periods when not specified."""
    regime = RegimeCondition(
        metric="trend_ma_relation",
        op=CompareOp.GT,
        value=0,
    )
    indicators = get_regime_indicators(regime)

    assert len(indicators) == 2
    assert any(isinstance(i, EMA) and i.period == 20 for i in indicators)
    assert any(isinstance(i, EMA) and i.period == 50 for i in indicators)


def test_trend_adx_creates_adx():
    """trend_adx needs ADX indicator."""
    regime = RegimeCondition(
        metric="trend_adx",
        op=CompareOp.GT,
        value=25,
        lookback_bars=14,
    )
    indicators = get_regime_indicators(regime)

    assert len(indicators) == 1
    assert isinstance(indicators[0], ADX)
    assert indicators[0].period == 14


def test_vol_atr_pct_creates_atr():
    """vol_atr_pct needs ATR indicator."""
    regime = RegimeCondition(
        metric="vol_atr_pct",
        op=CompareOp.LT,
        value=2.0,
        lookback_bars=14,
    )
    indicators = get_regime_indicators(regime)

    assert len(indicators) == 1
    assert isinstance(indicators[0], ATR)


def test_dist_from_vwap_pct_creates_vwap():
    """dist_from_vwap_pct needs VWAP indicator."""
    regime = RegimeCondition(
        metric="dist_from_vwap_pct",
        op=CompareOp.LT,
        value=1.0,
    )
    indicators = get_regime_indicators(regime)

    assert len(indicators) == 1
    assert isinstance(indicators[0], VWAP)


def test_ret_pct_creates_roc():
    """ret_pct needs RateOfChange indicator."""
    regime = RegimeCondition(
        metric="ret_pct",
        op=CompareOp.GT,
        value=5.0,
        lookback_bars=5,
    )
    indicators = get_regime_indicators(regime)

    assert len(indicators) == 1
    assert isinstance(indicators[0], RateOfChange)
    assert indicators[0].period == 5


def test_unknown_metric_raises():
    """Unknown metric raises TranslationError."""
    from src.translator.ir_translator import TranslationError

    regime = RegimeCondition(
        metric="unknown_metric",
        op=CompareOp.GT,
        value=0,
    )

    with pytest.raises(TranslationError, match="Unknown regime metric"):
        get_regime_indicators(regime)


def test_registry_has_all_metrics():
    """Registry contains handlers for all common metrics."""
    expected = {
        "trend_ma_relation",
        "trend_adx",
        "vol_atr_pct",
        "vol_bb_width_pctile",
        "dist_from_vwap_pct",
        "ret_pct",
        "volume_pctile",
    }
    assert expected.issubset(set(REGIME_INDICATOR_HANDLERS.keys()))
