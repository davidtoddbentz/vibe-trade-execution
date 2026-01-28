"""Registry modules for declarative mappings."""

from .indicators import (
    INDICATOR_FACTORIES,
    create_indicator_from_ref,
    generate_indicator_id,
)
from .regimes import (
    REGIME_INDICATOR_HANDLERS,
    get_regime_indicators,
)

__all__ = [
    "INDICATOR_FACTORIES",
    "create_indicator_from_ref",
    "generate_indicator_id",
    "REGIME_INDICATOR_HANDLERS",
    "get_regime_indicators",
]
