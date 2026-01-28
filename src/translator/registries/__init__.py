"""Registry modules for declarative mappings."""

from .indicators import (
    INDICATOR_FACTORIES,
    create_indicator_from_ref,
    generate_indicator_id,
)

__all__ = [
    "INDICATOR_FACTORIES",
    "create_indicator_from_ref",
    "generate_indicator_id",
]
