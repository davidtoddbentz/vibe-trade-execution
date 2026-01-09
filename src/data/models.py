"""Data models for market data."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class Resolution(str, Enum):
    """Supported candle resolutions."""

    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

    @property
    def minutes(self) -> int:
        """Return the resolution in minutes."""
        mapping = {
            "1m": 1,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "4h": 240,
            "1d": 1440,
            "1w": 10080,
        }
        return mapping[self.value]


class Candle(BaseModel):
    """OHLCV candle data."""

    model_config = {"frozen": True}

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    resolution: str = "1m"

    def __lt__(self, other: "Candle") -> bool:
        """Enable sorting by timestamp."""
        return self.timestamp < other.timestamp
