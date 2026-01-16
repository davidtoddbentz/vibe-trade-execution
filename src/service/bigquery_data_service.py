"""BigQuery implementation of DataService.

Fetches OHLCV market data from BigQuery for backtesting.
"""

import os
from datetime import datetime, timedelta

from google.cloud import bigquery
from vibe_trade_shared.models.data import OHLCVBar

from .data_service import DataFetchError, DataService


class BigQueryDataService:
    """Fetch OHLCV data from BigQuery.

    Implements the DataService protocol for production use.

    Queries the market_data.candles_parsed view in BigQuery.
    """

    # Map resolution strings to BQ granularity values
    RESOLUTION_MAP = {
        "1m": "1m",
        "1min": "1m",
        "minute": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "hour": "1h",
        "4h": "4h",
        "1d": "1d",
        "daily": "1d",
    }

    def __init__(self, project_id: str | None = None):
        """Initialize BigQuery data service.

        Args:
            project_id: GCP project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
        """
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise DataFetchError(
                "No project_id provided and GOOGLE_CLOUD_PROJECT env var not set"
            )
        self._client: bigquery.Client | None = None

    @property
    def client(self) -> bigquery.Client:
        """Lazy-initialize BQ client."""
        if self._client is None:
            self._client = bigquery.Client(project=self.project_id)
        return self._client

    def get_ohlcv(
        self,
        symbol: str,
        resolution: str,
        start: datetime,
        end: datetime,
    ) -> list[OHLCVBar]:
        """Fetch OHLCV bars from BigQuery.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            resolution: Bar resolution ("1m", "1h", "1d")
            start: Start datetime (inclusive)
            end: End datetime (inclusive)

        Returns:
            List of OHLCVBar models, sorted by timestamp ascending

        Raises:
            DataFetchError: If data cannot be fetched or resolution is invalid
        """
        granularity = self.RESOLUTION_MAP.get(resolution.lower())
        if not granularity:
            raise DataFetchError(
                f"Invalid resolution: {resolution}. "
                f"Valid values: {list(self.RESOLUTION_MAP.keys())}"
            )

        query = f"""
        SELECT
            timestamp,
            open,
            high,
            low,
            close,
            volume
        FROM `{self.project_id}.market_data.candles_parsed`
        WHERE symbol = @symbol
          AND granularity = @granularity
          AND timestamp >= @start_date
          AND timestamp < @end_date
        ORDER BY timestamp
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                bigquery.ScalarQueryParameter("granularity", "STRING", granularity),
                bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start),
                bigquery.ScalarQueryParameter(
                    "end_date", "TIMESTAMP", end + timedelta(days=1)
                ),
            ]
        )

        try:
            query_job = self.client.query(query, job_config=job_config)
            results = list(query_job.result())
        except Exception as e:
            raise DataFetchError(f"BigQuery query failed: {e}") from e

        return [
            OHLCVBar(
                t=int(row.timestamp.timestamp() * 1000),  # Convert to ms since epoch
                o=float(row.open),
                h=float(row.high),
                l=float(row.low),
                c=float(row.close),
                v=float(row.volume),
            )
            for row in results
        ]


# For testing: verify it conforms to the protocol
def _verify_protocol() -> None:
    """Type check that BigQueryDataService implements DataService."""
    _: DataService = BigQueryDataService(project_id="test")


_verify_protocol()
