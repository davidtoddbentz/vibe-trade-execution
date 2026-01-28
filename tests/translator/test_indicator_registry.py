"""Tests for indicator registry."""

import pytest
from vibe_trade_shared.models.ir import IndicatorRef

from src.translator.ir import ATR, EMA, SMA, BollingerBands
from src.translator.registries.indicators import (
    INDICATOR_FACTORIES,
    create_indicator_from_ref,
    generate_indicator_id,
)


def test_generate_id_with_period():
    """ID includes period when present."""
    assert generate_indicator_id("ema", {"period": 20}) == "ema_20"
    assert generate_indicator_id("atr", {"period": 14}) == "atr_14"


def test_generate_id_without_period():
    """ID is just prefix when no period."""
    assert generate_indicator_id("vwap", {}) == "vwap"
    assert generate_indicator_id("vwap", None) == "vwap"


def test_create_ema_from_ref():
    """IndicatorRef with EMA creates typed EMA."""
    ref = IndicatorRef(indicator_type="EMA", params={"period": 20})
    indicator = create_indicator_from_ref(ref)

    assert isinstance(indicator, EMA)
    assert indicator.id == "ema_20"
    assert indicator.period == 20


def test_create_sma_from_ref():
    """IndicatorRef with SMA creates typed SMA."""
    ref = IndicatorRef(indicator_type="SMA", params={"period": 50})
    indicator = create_indicator_from_ref(ref)

    assert isinstance(indicator, SMA)
    assert indicator.id == "sma_50"
    assert indicator.period == 50


def test_create_bb_from_ref():
    """IndicatorRef with BB creates typed BollingerBands."""
    ref = IndicatorRef(indicator_type="BB", params={"period": 20, "multiplier": 2.5})
    indicator = create_indicator_from_ref(ref)

    assert isinstance(indicator, BollingerBands)
    assert indicator.id == "bb_20"
    assert indicator.period == 20
    assert indicator.multiplier == 2.5


def test_create_atr_from_ref():
    """IndicatorRef with ATR creates typed ATR."""
    ref = IndicatorRef(indicator_type="ATR", params={"period": 14})
    indicator = create_indicator_from_ref(ref)

    assert isinstance(indicator, ATR)
    assert indicator.id == "atr_14"
    assert indicator.period == 14


def test_unknown_indicator_raises():
    """Unknown indicator type raises TranslationError."""
    from src.translator.ir_translator import TranslationError

    ref = IndicatorRef(indicator_type="UNKNOWN", params={})

    with pytest.raises(TranslationError, match="Unknown indicator type: UNKNOWN"):
        create_indicator_from_ref(ref)


def test_registry_has_all_indicator_types():
    """Registry contains all expected indicator types."""
    expected = {
        "EMA",
        "SMA",
        "BB",
        "KC",
        "DC",
        "ATR",
        "RSI",
        "MAX",
        "MIN",
        "ROC",
        "ADX",
        "VWAP",
        "AVWAP",
        "PCTILE",
        "GAP",
        "RW",
        "VOL_SMA",
    }
    assert expected.issubset(set(INDICATOR_FACTORIES.keys()))
