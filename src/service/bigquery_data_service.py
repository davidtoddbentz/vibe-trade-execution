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

    # Map resolution strings to BQ view names
    # Each resolution has a dedicated view: candles_1m, candles_5m, candles_1h, candles_1d
    RESOLUTION_TO_VIEW = {
        "1m": "candles_parsed",  # Raw 1m data
        "1min": "candles_parsed",
        "minute": "candles_parsed",
        "5m": "candles_5m",
        "15m": "candles_5m",  # Use 5m and filter client-side for now
        "1h": "candles_parsed",  # Direct 1h candles from backfill
        "hour": "candles_parsed",
        "4h": "candles_parsed",  # Use candles_parsed with granularity filter
        "1d": "candles_parsed",  # Direct 1d candles from backfill
        "daily": "candles_parsed",
    }

    # Map resolution strings to BQ granularity values (for candles_parsed filtering)
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

    def __init__(self, project_id: str | None = None, emulator_host: str | None = None):
        """Initialize BigQuery data service.

        Args:
            project_id: GCP project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
            emulator_host: BigQuery emulator endpoint (e.g., "http://localhost:9050").
                          If None, uses BIGQUERY_EMULATOR_HOST env var.
                          If set, uses anonymous credentials for emulator.
        """
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise DataFetchError(
                "No project_id provided and GOOGLE_CLOUD_PROJECT env var not set"
            )
        self.emulator_host = emulator_host or os.environ.get("BIGQUERY_EMULATOR_HOST")
        self._client: bigquery.Client | None = None

    @property
    def client(self) -> bigquery.Client:
        """Lazy-initialize BQ client."""
        if self._client is None:
            if self.emulator_host:
                # Use emulator with anonymous credentials
                from google.api_core.client_options import ClientOptions
                from google.auth.credentials import AnonymousCredentials
                
                client_options = ClientOptions(api_endpoint=self.emulator_host)
                self._client = bigquery.Client(
                    project=self.project_id,
                    client_options=client_options,
                    credentials=AnonymousCredentials(),
                )
            else:
                # Use real BigQuery with default credentials
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
        view_name = self.RESOLUTION_TO_VIEW.get(resolution.lower())
        if not view_name:
            raise DataFetchError(
                f"Invalid resolution: {resolution}. "
                f"Valid values: {list(self.RESOLUTION_TO_VIEW.keys())}"
            )

        # For aggregated views (1h, 1d, 5m), no granularity filter needed
        # For candles_parsed, we need to filter by granularity
        if view_name == "candles_parsed":
            granularity = self.RESOLUTION_MAP.get(resolution.lower())
            query = f"""
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM `{self.project_id}.market_data.{view_name}`
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
        else:
            # Aggregated views don't have granularity column
            query = f"""
            SELECT
                timestamp,
                open,
                high,
                low,
                close,
                volume
            FROM `{self.project_id}.market_data.{view_name}`
            WHERE symbol = @symbol
              AND timestamp >= @start_date
              AND timestamp < @end_date
            ORDER BY timestamp
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("symbol", "STRING", symbol),
                    bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start),
                    bigquery.ScalarQueryParameter(
                        "end_date", "TIMESTAMP", end + timedelta(days=1)
                    ),
                ]
            )

        try:
            if self.emulator_host:
                # Emulator returns timestamps as float strings which breaks the
                # standard client row parser. Use raw HTTP request instead.
                return self._query_emulator_raw(query, symbol, start, end, view_name)
            else:
                # Production BQ - use standard client
                query_job = self.client.query(query, job_config=job_config)
                results = list(query_job.result())
                return [
                    OHLCVBar(
                        t=int(row.timestamp.timestamp() * 1000),
                        o=float(row.open),
                        h=float(row.high),
                        l=float(row.low),
                        c=float(row.close),
                        v=float(row.volume),
                    )
                    for row in results
                ]
        except Exception as e:
            raise DataFetchError(f"BigQuery query failed: {e}") from e

    def _query_emulator_raw(
        self,
        query: str,
        symbol: str,
        start: datetime,
        end: datetime,
        view_name: str,
    ) -> list[OHLCVBar]:
        """Query emulator using raw HTTP to avoid timestamp parsing issues."""
        import json
        import urllib.request
        import urllib.error

        # Build simple query without parameters (emulator may not support all features)
        start_ts = start.strftime("%Y-%m-%d %H:%M:%S")
        end_ts = (end + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        simple_query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM `{self.project_id}.market_data.{view_name}`
        WHERE symbol = '{symbol}'
          AND timestamp >= TIMESTAMP('{start_ts}')
          AND timestamp < TIMESTAMP('{end_ts}')
        ORDER BY timestamp
        """

        url = f"{self.emulator_host}/bigquery/v2/projects/{self.project_id}/queries"
        payload = json.dumps({"query": simple_query, "useLegacySql": False}).encode()
        headers = {"Content-Type": "application/json"}

        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise DataFetchError(f"Emulator HTTP request failed: {e}") from e

        # Parse raw JSON response from emulator
        bars = []
        rows = result.get("rows", [])
        for row in rows:
            values = row.get("f", [])
            if len(values) >= 6:
                # Parse timestamp (seconds since epoch as float string)
                ts_val = values[0].get("v", "0")
                ts_ms = int(float(ts_val) * 1000)

                bars.append(OHLCVBar(
                    t=ts_ms,
                    o=float(values[1].get("v", 0)),
                    h=float(values[2].get("v", 0)),
                    l=float(values[3].get("v", 0)),
                    c=float(values[4].get("v", 0)),
                    v=float(values[5].get("v", 0)),
                ))

        return bars


# For testing: verify it conforms to the protocol
def _verify_protocol() -> None:
    """Type check that BigQueryDataService implements DataService."""
    _: DataService = BigQueryDataService(project_id="test")


_verify_protocol()
