"""Data service protocol for market data fetching.

This module defines the interface for fetching OHLCV market data.
The protocol allows different implementations:
- BigQueryDataService: Production, fetches from BigQuery
- MockDataService: Testing, returns synthetic data
"""

from datetime import datetime
from typing import Protocol

from vibe_trade_shared.models.data import OHLCVBar


class DataService(Protocol):
    """Protocol for fetching OHLCV market data.

    Implementations should handle:
    - Connection to data source (BQ, mock, etc.)
    - Resolution mapping (1m, 1h, 1d → table/format)
    - Date range filtering
    - Returning data as Pydantic models
    """

    def get_ohlcv(
        self,
        symbol: str,
        resolution: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars for symbol and date range.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            resolution: Bar resolution ("1m", "1h", "1d")
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            List of OHLCVBar models, sorted by timestamp ascending

        Raises:
            DataFetchError: If data cannot be fetched
        """
        ...


class DataFetchError(Exception):
    """Raised when data fetching fails."""

    pass
