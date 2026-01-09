"""GCS data fetcher for market candles."""

import json
import logging
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO

import fastavro
from google.cloud import storage

from src.data.aggregator import aggregate_candles
from src.data.models import Candle, Resolution

logger = logging.getLogger(__name__)


# Pattern to parse blob names like: btc-1m2026-01-02T17:29:05+00:00_b738ab
BLOB_NAME_PATTERN = re.compile(
    r"^(?P<symbol>[a-z]+)-(?P<resolution>\d+[mhd])"
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+\d{2}:\d{2})_[a-f0-9]+$"
)


def parse_blob_timestamp(blob_name: str) -> datetime | None:
    """Extract timestamp from blob name.

    Args:
        blob_name: Blob name like 'btc-1m2026-01-02T17:29:05+00:00_b738ab'

    Returns:
        Parsed datetime or None if parsing fails
    """
    match = BLOB_NAME_PATTERN.match(blob_name)
    if not match:
        return None

    ts_str = match.group("timestamp")
    try:
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


def parse_avro_candle(blob_data: bytes) -> Candle | None:
    """Parse Avro-wrapped Pub/Sub message containing candle JSON.

    Args:
        blob_data: Raw bytes from GCS blob (Avro format)

    Returns:
        Parsed Candle or None if parsing fails
    """
    try:
        reader = fastavro.reader(BytesIO(blob_data))
        for record in reader:
            # Pub/Sub Avro format has 'data' field with bytes
            data_bytes = record.get("data")
            if data_bytes:
                candle_json = json.loads(data_bytes)
                return Candle(
                    symbol=candle_json["symbol"],
                    timestamp=datetime.fromisoformat(candle_json["timestamp"]),
                    open=candle_json["open"],
                    high=candle_json["high"],
                    low=candle_json["low"],
                    close=candle_json["close"],
                    volume=candle_json["volume"],
                    resolution=candle_json.get("granularity", "1m"),
                )
    except Exception as e:
        logger.warning(f"Failed to parse Avro candle: {e}")
    return None


class DataFetcher:
    """Fetches market data from GCS blob storage."""

    def __init__(
        self,
        bucket_name: str = "batch-save",
        project_id: str | None = None,
        max_workers: int = 10,
    ):
        """Initialize the data fetcher.

        Args:
            bucket_name: GCS bucket containing candle data
            project_id: GCP project ID (uses default if not specified)
            max_workers: Max parallel blob downloads
        """
        self.bucket_name = bucket_name
        self.max_workers = max_workers
        self._client = storage.Client(project=project_id)
        self._bucket = self._client.bucket(bucket_name)

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for blob prefix matching.

        Args:
            symbol: Symbol like 'BTC-USD' or 'BTC'

        Returns:
            Normalized symbol like 'btc'
        """
        return symbol.lower().replace("-usd", "").replace("-", "")

    def _list_blobs_in_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> Iterator[storage.Blob]:
        """List blobs within a time range.

        Args:
            symbol: Trading symbol
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)

        Yields:
            Matching blobs
        """
        prefix = f"{self._normalize_symbol(symbol)}-1m"
        logger.debug(f"Listing blobs with prefix: {prefix}")

        for blob in self._bucket.list_blobs(prefix=prefix):
            ts = parse_blob_timestamp(blob.name)
            if ts and start <= ts <= end:
                yield blob

    def _download_and_parse(self, blob: storage.Blob) -> Candle | None:
        """Download a blob and parse to Candle.

        Args:
            blob: GCS blob to download

        Returns:
            Parsed Candle or None
        """
        try:
            data = blob.download_as_bytes()
            return parse_avro_candle(data)
        except Exception as e:
            logger.warning(f"Failed to download/parse {blob.name}: {e}")
            return None

    def fetch_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        resolution: Resolution = Resolution.MINUTE_1,
    ) -> list[Candle]:
        """Fetch candles for a symbol within a time range.

        Args:
            symbol: Trading symbol (e.g., 'BTC-USD', 'BTC')
            start: Start timestamp (inclusive)
            end: End timestamp (inclusive)
            resolution: Desired resolution (aggregates if > 1m)

        Returns:
            List of candles sorted by timestamp
        """
        # Ensure UTC
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        logger.info(f"Fetching {symbol} candles from {start} to {end}")

        # List matching blobs
        blobs = list(self._list_blobs_in_range(symbol, start, end))
        logger.info(f"Found {len(blobs)} blobs to download")

        if not blobs:
            return []

        # Download in parallel
        candles: list[Candle] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._download_and_parse, blob): blob for blob in blobs}

            for future in as_completed(futures):
                candle = future.result()
                if candle:
                    candles.append(candle)

        logger.info(f"Downloaded {len(candles)} candles")

        # Sort by timestamp
        candles.sort()

        # Aggregate if needed
        if resolution != Resolution.MINUTE_1:
            candles = aggregate_candles(candles, resolution)
            logger.info(f"Aggregated to {len(candles)} {resolution.value} candles")

        return candles

    def get_available_range(self, symbol: str) -> tuple[datetime, datetime] | None:
        """Get the available data range for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (earliest, latest) timestamps, or None if no data
        """
        prefix = f"{self._normalize_symbol(symbol)}-1m"

        earliest: datetime | None = None
        latest: datetime | None = None

        for blob in self._bucket.list_blobs(prefix=prefix):
            ts = parse_blob_timestamp(blob.name)
            if ts:
                if earliest is None or ts < earliest:
                    earliest = ts
                if latest is None or ts > latest:
                    latest = ts

        if earliest and latest:
            return (earliest, latest)
        return None
