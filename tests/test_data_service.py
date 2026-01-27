"""Tests for MockDataService."""

from datetime import datetime, timezone

from vibe_trade_shared.models.data import OHLCVBar

from src.service.data_service import MockDataService


def test_mock_data_service_returns_seeded_bars():
    """MockDataService returns bars that were seeded."""
    bars = [
        OHLCVBar(t=1704067200000, o=100, h=101, l=99, c=100, v=1000),
        OHLCVBar(t=1704070800000, o=100, h=101, l=99, c=101, v=1000),
    ]
    ds = MockDataService()
    ds.seed("BTC-USD", "1h", bars)

    result = ds.get_ohlcv(
        symbol="BTC-USD",
        resolution="1h",
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert result == bars


def test_mock_data_service_raises_on_missing():
    """MockDataService raises DataFetchError for unseeded symbol."""
    ds = MockDataService()
    try:
        ds.get_ohlcv(
            symbol="ETH-USD",
            resolution="1h",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
        assert False, "Should have raised"
    except Exception as e:
        assert "ETH-USD" in str(e)
