"""Candle aggregation utilities."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from src.data.models import Candle, Resolution


def truncate_to_resolution(timestamp: datetime, resolution: Resolution) -> datetime:
    """Truncate a timestamp to the start of its resolution bucket.

    Args:
        timestamp: The timestamp to truncate
        resolution: Target resolution

    Returns:
        Truncated timestamp at the start of the bucket
    """
    # Ensure UTC
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    minutes = resolution.minutes

    if minutes < 60:
        # Sub-hourly: truncate to minute boundary
        minute_bucket = (timestamp.minute // minutes) * minutes
        return timestamp.replace(minute=minute_bucket, second=0, microsecond=0)

    elif minutes < 1440:
        # Sub-daily: truncate to hour boundary
        hours = minutes // 60
        hour_bucket = (timestamp.hour // hours) * hours
        return timestamp.replace(hour=hour_bucket, minute=0, second=0, microsecond=0)

    elif minutes == 1440:
        # Daily: truncate to day
        return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)

    else:
        # Weekly: truncate to Monday
        days_since_monday = timestamp.weekday()
        monday = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        return monday - timedelta(days=days_since_monday)


def aggregate_candles(
    candles: list[Candle],
    resolution: Resolution,
) -> list[Candle]:
    """Aggregate candles to a higher timeframe.

    Args:
        candles: List of candles (typically 1m resolution)
        resolution: Target resolution to aggregate to

    Returns:
        List of aggregated candles sorted by timestamp
    """
    if not candles:
        return []

    # Group candles by time bucket
    buckets: dict[datetime, list[Candle]] = defaultdict(list)
    for candle in candles:
        bucket = truncate_to_resolution(candle.timestamp, resolution)
        buckets[bucket].append(candle)

    # Aggregate each bucket
    aggregated = []
    symbol = candles[0].symbol

    for bucket_time, bucket_candles in sorted(buckets.items()):
        # Sort by timestamp to ensure correct open/close
        bucket_candles.sort(key=lambda c: c.timestamp)

        aggregated.append(
            Candle(
                symbol=symbol,
                timestamp=bucket_time,
                open=bucket_candles[0].open,
                high=max(c.high for c in bucket_candles),
                low=min(c.low for c in bucket_candles),
                close=bucket_candles[-1].close,
                volume=sum(c.volume for c in bucket_candles),
                resolution=resolution.value,
            )
        )

    return aggregated
