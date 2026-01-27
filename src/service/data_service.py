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


class MockDataService:
    """In-memory DataService for testing.

    Seed with test data, then pass to BacktestService like production
    uses BigQueryDataService.

    Usage:
        ds = MockDataService()
        ds.seed("BTC-USD", "1h", bars)
        service = BacktestService(data_service=ds, backtest_url=url)
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], list[OHLCVBar]] = {}

    def seed(self, symbol: str, resolution: str, bars: list[OHLCVBar]) -> None:
        """Seed bars for a symbol/resolution pair."""
        self._data[(symbol, resolution)] = bars

    def get_ohlcv(
        self,
        symbol: str,
        resolution: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCVBar]:
        """Return seeded bars. Raises DataFetchError if not seeded."""
        key = (symbol, resolution)
        if key not in self._data:
            raise DataFetchError(
                f"No test data seeded for {symbol}/{resolution}. "
                f"Call ds.seed('{symbol}', '{resolution}', bars) first."
            )
        return self._data[key]
