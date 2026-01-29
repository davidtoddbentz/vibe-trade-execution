"""Indicator factory registry.

Maps indicator type strings to factory functions that create typed Indicator objects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vibe_trade_shared.models.ir import IndicatorRef
from vibe_trade_shared.models.ir.indicators import (
    ADX,
    ATR,
    EMA,
    RSI,
    SMA,
    VWAP,
    AnchoredVWAP,
    BollingerBands,
    DonchianChannel,
    Gap,
    KeltnerChannel,
    Maximum,
    Minimum,
    Percentile,
    RateOfChange,
    RollingWindow,
    VolumeSMA,
    IndicatorUnion,
)

from src.translator.errors import TranslationError

# Type alias for indicator factory functions
IndicatorFactory = Callable[[str, dict[str, Any]], IndicatorUnion]


def generate_indicator_id(prefix: str, params: dict[str, Any] | None) -> str:
    """Generate deterministic indicator ID from type and params.

    Format: {prefix}_{period} if period exists, else just {prefix}.
    """
    if not params or "period" not in params:
        return prefix
    return f"{prefix}_{params['period']}"


# Registry mapping indicator type strings to factory functions
INDICATOR_FACTORIES: dict[str, IndicatorFactory] = {
    "EMA": lambda id, p: EMA(id=id, period=p.get("period", 20)),
    "SMA": lambda id, p: SMA(id=id, period=p.get("period", 20)),
    "BB": lambda id, p: BollingerBands(
        id=id,
        period=p.get("period", 20),
        multiplier=p.get("multiplier", 2.0),
    ),
    "KC": lambda id, p: KeltnerChannel(
        id=id,
        period=p.get("period", 20),
        multiplier=p.get("multiplier", 2.0),
    ),
    "DC": lambda id, p: DonchianChannel(id=id, period=p.get("period", 20)),
    "ATR": lambda id, p: ATR(id=id, period=p.get("period", 14)),
    "RSI": lambda id, p: RSI(id=id, period=p.get("period", 14)),
    "MAX": lambda id, p: Maximum(id=id, period=p.get("period", 50)),
    "MIN": lambda id, p: Minimum(id=id, period=p.get("period", 50)),
    "ROC": lambda id, p: RateOfChange(id=id, period=p.get("period", 14)),
    "ADX": lambda id, p: ADX(id=id, period=p.get("period", 14)),
    "VWAP": lambda id, p: VWAP(id=id, period=p.get("period", 0)),
    "AVWAP": lambda id, p: AnchoredVWAP(
        id=id,
        anchor=p.get("anchor", "session"),
        anchor_datetime=p.get("anchor_datetime"),
    ),
    "PCTILE": lambda id, p: Percentile(
        id=id,
        period=p.get("period", 100),
        percentile=p.get("percentile", 10.0),
        source=p.get("source", "close"),
    ),
    "GAP": lambda id, p: Gap(id=id, session=p.get("session", "us")),
    "RW": lambda id, p: RollingWindow(id=id, period=p.get("period", 2)),
    "VOL_SMA": lambda id, p: VolumeSMA(id=id, period=p.get("period", 20)),
}


def create_indicator_from_ref(ref: IndicatorRef) -> IndicatorUnion:
    """Create a typed Indicator from an IndicatorRef.

    Args:
        ref: IndicatorRef with indicator_type and params

    Returns:
        Typed Indicator instance

    Raises:
        TranslationError: If indicator_type is unknown
    """
    if ref.indicator_type is None:
        raise TranslationError("IndicatorRef missing indicator_type")

    ind_type = ref.indicator_type.upper()
    params = ref.params or {}

    factory = INDICATOR_FACTORIES.get(ind_type)
    if factory is None:
        raise TranslationError(f"Unknown indicator type: {ref.indicator_type}")

    ind_id = generate_indicator_id(ind_type.lower(), params)
    return factory(ind_id, params)
