"""Tests for data architecture: BigQuery rollups and data loading.

This module tests that:
1. BigQuery rollup views correctly aggregate 1m data to 1h and 1d
2. Data loading via BigQueryDataService works at all resolutions
3. The data architecture maintains single source of truth (1m)

These tests require a connection to BigQuery (production or emulator).
"""

import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from vibe_trade_shared.models.data import OHLCVBar

from src.service.bigquery_data_service import BigQueryDataService


# Skip all tests in this module if BQ is not available
def is_bigquery_available() -> bool:
    """Check if BigQuery is accessible."""
    try:
        service = BigQueryDataService(project_id="vibe-trade-475704")
        # Try a simple query
        service.client.query("SELECT 1").result()
        return True
    except Exception:
        return False


requires_bigquery = pytest.mark.skipif(
    not is_bigquery_available(),
    reason="BigQuery not available. Set GOOGLE_CLOUD_PROJECT and credentials.",
)


# =============================================================================
# Base Test Classes
# =============================================================================


class BaseRollupTest(ABC):
    """Base class for testing BigQuery rollup views.

    Subclasses test specific resolution rollups (1m→1h, 1m→1d).
    Provides common verification logic for OHLCV aggregation.
    """

    @property
    @abstractmethod
    def source_resolution(self) -> str:
        """Resolution of source data (e.g., '1m')."""
        ...

    @property
    @abstractmethod
    def target_resolution(self) -> str:
        """Resolution of rolled-up data (e.g., '1h')."""
        ...

    @property
    @abstractmethod
    def test_timestamp(self) -> datetime:
        """Timestamp to test rollup at (start of target period)."""
        ...

    @property
    @abstractmethod
    def source_count_expected(self) -> int:
        """Expected number of source bars in one target period."""
        ...

    @pytest.fixture
    def data_service(self) -> BigQueryDataService:
        """Create BigQuery data service."""
        return BigQueryDataService(project_id="vibe-trade-475704")

    def manual_aggregate(self, bars: list[OHLCVBar]) -> dict:
        """Manually aggregate OHLCV bars.

        Args:
            bars: Source bars to aggregate (must be sorted by timestamp)

        Returns:
            Dict with open, high, low, close, volume
        """
        if not bars:
            return {"open": None, "high": None, "low": None, "close": None, "volume": None}

        # Sort by timestamp to ensure correct open/close
        sorted_bars = sorted(bars, key=lambda b: b.t)

        return {
            "open": sorted_bars[0].o,
            "high": max(b.h for b in sorted_bars),
            "low": min(b.l for b in sorted_bars),
            "close": sorted_bars[-1].c,
            "volume": sum(Decimal(str(b.v)) for b in sorted_bars),
        }

    def test_rollup_ohlcv_accuracy(self, data_service: BigQueryDataService):
        """Verify rolled-up OHLCV matches manual calculation from source data.

        This is the core test that validates rollup correctness:
        1. Fetch source bars for one target period
        2. Manually calculate expected OHLCV
        3. Fetch rolled-up bar from target view
        4. Verify they match
        """
        # Get source bars for one target period
        from datetime import timedelta

        if self.target_resolution in ("1h", "hour"):
            period_end = self.test_timestamp + timedelta(hours=1)
        elif self.target_resolution in ("1d", "daily"):
            period_end = self.test_timestamp + timedelta(days=1)
        else:
            pytest.skip(f"Unknown target resolution: {self.target_resolution}")

        source_bars = data_service.get_ohlcv(
            symbol="BTC-USD",
            resolution=self.source_resolution,
            start=self.test_timestamp,
            end=period_end,
        )

        # Filter to just the bars within the period (get_ohlcv may return extra)
        source_bars = [
            b for b in source_bars
            if self.test_timestamp.timestamp() * 1000 <= b.t < period_end.timestamp() * 1000
        ]

        assert len(source_bars) > 0, f"No source bars found for {self.test_timestamp}"

        # Manual aggregation
        expected = self.manual_aggregate(source_bars)

        # Get rolled-up bar
        target_bars = data_service.get_ohlcv(
            symbol="BTC-USD",
            resolution=self.target_resolution,
            start=self.test_timestamp,
            end=period_end,
        )

        # Find the bar matching our test timestamp
        target_bar = None
        for bar in target_bars:
            bar_ts = datetime.fromtimestamp(bar.t / 1000, tz=timezone.utc)
            if bar_ts == self.test_timestamp:
                target_bar = bar
                break

        assert target_bar is not None, f"No target bar found for {self.test_timestamp}"

        # Verify OHLCV matches
        tolerance = Decimal("0.01")  # Allow small floating-point differences

        assert abs(Decimal(str(target_bar.o)) - Decimal(str(expected["open"]))) < tolerance, (
            f"Open mismatch: {target_bar.o} vs {expected['open']}"
        )
        assert abs(Decimal(str(target_bar.h)) - Decimal(str(expected["high"]))) < tolerance, (
            f"High mismatch: {target_bar.h} vs {expected['high']}"
        )
        assert abs(Decimal(str(target_bar.l)) - Decimal(str(expected["low"]))) < tolerance, (
            f"Low mismatch: {target_bar.l} vs {expected['low']}"
        )
        assert abs(Decimal(str(target_bar.c)) - Decimal(str(expected["close"]))) < tolerance, (
            f"Close mismatch: {target_bar.c} vs {expected['close']}"
        )
        assert abs(Decimal(str(target_bar.v)) - expected["volume"]) < Decimal("0.0001"), (
            f"Volume mismatch: {target_bar.v} vs {expected['volume']}"
        )


class BaseDataLoadingTest(ABC):
    """Base class for testing data loading at different resolutions.

    Subclasses test specific resolutions (1m, 1h, 1d).
    Provides common verification logic for data loading.
    """

    @property
    @abstractmethod
    def resolution(self) -> str:
        """Resolution to test (e.g., '1h')."""
        ...

    @property
    @abstractmethod
    def expected_bar_count_per_day(self) -> int:
        """Expected number of bars per day at this resolution."""
        ...

    @pytest.fixture
    def data_service(self) -> BigQueryDataService:
        """Create BigQuery data service."""
        return BigQueryDataService(project_id="vibe-trade-475704")

    def test_data_loads_successfully(self, data_service: BigQueryDataService):
        """Verify data can be loaded at this resolution."""
        start = datetime(2025, 6, 1, tzinfo=timezone.utc)
        end = datetime(2025, 6, 2, tzinfo=timezone.utc)

        bars = data_service.get_ohlcv(
            symbol="BTC-USD",
            resolution=self.resolution,
            start=start,
            end=end,
        )

        assert len(bars) > 0, f"No bars returned for resolution {self.resolution}"

    def test_bar_count_matches_resolution(self, data_service: BigQueryDataService):
        """Verify bar count matches expected for resolution."""
        start = datetime(2025, 6, 1, tzinfo=timezone.utc)
        end = datetime(2025, 6, 2, tzinfo=timezone.utc)

        bars = data_service.get_ohlcv(
            symbol="BTC-USD",
            resolution=self.resolution,
            start=start,
            end=end,
        )

        # Filter to just bars within the single day (API may return extra)
        bars = [
            b for b in bars
            if start.timestamp() * 1000 <= b.t < end.timestamp() * 1000
        ]

        # Allow some tolerance for incomplete days at boundaries
        expected_min = int(self.expected_bar_count_per_day * 0.9)
        expected_max = int(self.expected_bar_count_per_day * 1.1)

        assert expected_min <= len(bars) <= expected_max, (
            f"Expected ~{self.expected_bar_count_per_day} bars for {self.resolution}, got {len(bars)}"
        )

    def test_bars_are_sorted(self, data_service: BigQueryDataService):
        """Verify bars are returned in ascending timestamp order."""
        start = datetime(2025, 6, 1, tzinfo=timezone.utc)
        end = datetime(2025, 6, 2, tzinfo=timezone.utc)

        bars = data_service.get_ohlcv(
            symbol="BTC-USD",
            resolution=self.resolution,
            start=start,
            end=end,
        )

        timestamps = [b.t for b in bars]
        assert timestamps == sorted(timestamps), "Bars are not sorted by timestamp"

    def test_ohlcv_values_are_valid(self, data_service: BigQueryDataService):
        """Verify OHLCV values pass sanity checks."""
        start = datetime(2025, 6, 1, tzinfo=timezone.utc)
        end = datetime(2025, 6, 2, tzinfo=timezone.utc)

        bars = data_service.get_ohlcv(
            symbol="BTC-USD",
            resolution=self.resolution,
            start=start,
            end=end,
        )

        for bar in bars[:10]:  # Check first 10 bars
            # High >= Open, Close, Low
            assert bar.h >= bar.o, f"High ({bar.h}) < Open ({bar.o})"
            assert bar.h >= bar.c, f"High ({bar.h}) < Close ({bar.c})"
            assert bar.h >= bar.l, f"High ({bar.h}) < Low ({bar.l})"

            # Low <= Open, Close, High
            assert bar.l <= bar.o, f"Low ({bar.l}) > Open ({bar.o})"
            assert bar.l <= bar.c, f"Low ({bar.l}) > Close ({bar.c})"

            # Volume is positive
            assert bar.v >= 0, f"Volume is negative: {bar.v}"


# =============================================================================
# Concrete Rollup Tests
# =============================================================================


@requires_bigquery
class TestHourlyRollup(BaseRollupTest):
    """Test 1m → 1h rollup accuracy."""

    @property
    def source_resolution(self) -> str:
        return "1m"

    @property
    def target_resolution(self) -> str:
        return "1h"

    @property
    def test_timestamp(self) -> datetime:
        return datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    @property
    def source_count_expected(self) -> int:
        return 60  # 60 minutes in an hour


@requires_bigquery
class TestDailyRollup(BaseRollupTest):
    """Test 1m → 1d rollup accuracy."""

    @property
    def source_resolution(self) -> str:
        return "1m"

    @property
    def target_resolution(self) -> str:
        return "1d"

    @property
    def test_timestamp(self) -> datetime:
        return datetime(2025, 6, 1, 0, 0, 0, tzinfo=timezone.utc)

    @property
    def source_count_expected(self) -> int:
        return 1440  # 1440 minutes in a day


# =============================================================================
# Concrete Data Loading Tests
# =============================================================================


@requires_bigquery
class TestMinuteDataLoading(BaseDataLoadingTest):
    """Test 1m data loading."""

    @property
    def resolution(self) -> str:
        return "1m"

    @property
    def expected_bar_count_per_day(self) -> int:
        return 1440  # 1440 minutes in a day


@requires_bigquery
class TestHourlyDataLoading(BaseDataLoadingTest):
    """Test 1h data loading."""

    @property
    def resolution(self) -> str:
        return "1h"

    @property
    def expected_bar_count_per_day(self) -> int:
        return 24  # 24 hours in a day


@requires_bigquery
class TestDailyDataLoading(BaseDataLoadingTest):
    """Test 1d data loading."""

    @property
    def resolution(self) -> str:
        return "1d"

    @property
    def expected_bar_count_per_day(self) -> int:
        return 1  # 1 day bar per day


# =============================================================================
# Integration Tests
# =============================================================================


@requires_bigquery
class TestDataArchitectureIntegrity:
    """Test the overall data architecture integrity.

    These tests verify the single source of truth principle:
    - Only 1m data is stored
    - 1h and 1d are pure views over 1m
    """

    @pytest.fixture
    def data_service(self) -> BigQueryDataService:
        """Create BigQuery data service."""
        return BigQueryDataService(project_id="vibe-trade-475704")

    def test_resolution_consistency(self, data_service: BigQueryDataService):
        """Verify same time period at different resolutions shows consistent data.

        The close price of the 1d bar should equal the close of the last 1h bar
        which should equal the close of the last 1m bar.
        """
        test_date = datetime(2025, 6, 1, tzinfo=timezone.utc)
        next_date = datetime(2025, 6, 2, tzinfo=timezone.utc)

        # Get 1m bars
        bars_1m = data_service.get_ohlcv(
            symbol="BTC-USD",
            resolution="1m",
            start=test_date,
            end=next_date,
        )
        # Filter to just this day
        bars_1m = [
            b for b in bars_1m
            if test_date.timestamp() * 1000 <= b.t < next_date.timestamp() * 1000
        ]

        # Get 1h bars
        bars_1h = data_service.get_ohlcv(
            symbol="BTC-USD",
            resolution="1h",
            start=test_date,
            end=next_date,
        )
        bars_1h = [
            b for b in bars_1h
            if test_date.timestamp() * 1000 <= b.t < next_date.timestamp() * 1000
        ]

        # Get 1d bar
        bars_1d = data_service.get_ohlcv(
            symbol="BTC-USD",
            resolution="1d",
            start=test_date,
            end=next_date,
        )
        # Find the bar for test_date
        day_bar = None
        for bar in bars_1d:
            bar_ts = datetime.fromtimestamp(bar.t / 1000, tz=timezone.utc)
            if bar_ts.date() == test_date.date():
                day_bar = bar
                break

        assert len(bars_1m) > 0, "No 1m bars found"
        assert len(bars_1h) > 0, "No 1h bars found"
        assert day_bar is not None, "No 1d bar found"

        # Sort and get last bars
        bars_1m.sort(key=lambda b: b.t)
        bars_1h.sort(key=lambda b: b.t)
        last_1m = bars_1m[-1]
        last_1h = bars_1h[-1]

        # Day open should match first 1m open and first 1h open
        first_1m = bars_1m[0]
        first_1h = bars_1h[0]

        tolerance = 0.01
        assert abs(float(day_bar.o) - float(first_1m.o)) < tolerance, (
            f"1d open ({day_bar.o}) doesn't match first 1m open ({first_1m.o})"
        )
        assert abs(float(day_bar.o) - float(first_1h.o)) < tolerance, (
            f"1d open ({day_bar.o}) doesn't match first 1h open ({first_1h.o})"
        )

        # Day close should match last 1m close and last 1h close
        assert abs(float(day_bar.c) - float(last_1m.c)) < tolerance, (
            f"1d close ({day_bar.c}) doesn't match last 1m close ({last_1m.c})"
        )
        assert abs(float(day_bar.c) - float(last_1h.c)) < tolerance, (
            f"1d close ({day_bar.c}) doesn't match last 1h close ({last_1h.c})"
        )

    def test_no_duplicate_bars(self, data_service: BigQueryDataService):
        """Verify no duplicate timestamps in any resolution."""
        test_date = datetime(2025, 6, 1, tzinfo=timezone.utc)
        next_date = datetime(2025, 6, 2, tzinfo=timezone.utc)

        for resolution in ["1m", "1h", "1d"]:
            bars = data_service.get_ohlcv(
                symbol="BTC-USD",
                resolution=resolution,
                start=test_date,
                end=next_date,
            )

            timestamps = [b.t for b in bars]
            unique_timestamps = set(timestamps)

            assert len(timestamps) == len(unique_timestamps), (
                f"Duplicate timestamps found at {resolution} resolution: "
                f"{len(timestamps)} bars but only {len(unique_timestamps)} unique timestamps"
            )
