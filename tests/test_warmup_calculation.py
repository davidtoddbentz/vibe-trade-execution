"""Tests for warmup period calculation.

This module tests the warmup calculation logic to ensure backtests
fetch sufficient historical data for indicator initialization.

Regression test for bug: backtests were fetching insufficient data
for indicator warmup, causing 0 trades when indicators couldn't initialize.
"""

from datetime import timedelta

from src.service.backtest_service import _calculate_warmup_bars, _resolution_to_timedelta


class TestCalculateWarmupBars:
    """Test warmup bar calculation from indicator periods."""

    def test_single_indicator_period(self):
        """Warmup should be at least indicator period + buffer."""
        strategy_ir = {
            "indicators": [
                {"id": "ema_20", "type": "EMA", "period": 20}
            ]
        }
        warmup = _calculate_warmup_bars(strategy_ir)
        # Should be max(20 + 10, 50) = 50
        assert warmup == 50

    def test_large_indicator_period(self):
        """Warmup should handle large indicator periods."""
        strategy_ir = {
            "indicators": [
                {"id": "ema_200", "type": "EMA", "period": 200}
            ]
        }
        warmup = _calculate_warmup_bars(strategy_ir)
        # Should be max(200 + 10, 50) = 210
        assert warmup == 210

    def test_multiple_indicators_uses_max(self):
        """Warmup should use maximum period across all indicators."""
        strategy_ir = {
            "indicators": [
                {"id": "ema_20", "type": "EMA", "period": 20},
                {"id": "ema_50", "type": "EMA", "period": 50},
                {"id": "bb_30", "type": "BollingerBands", "period": 30},
            ]
        }
        warmup = _calculate_warmup_bars(strategy_ir)
        # Should be max(50 + 10, 50) = 60
        assert warmup == 60

    def test_no_indicators_uses_minimum(self):
        """Warmup should use minimum even with no indicators."""
        strategy_ir = {"indicators": []}
        warmup = _calculate_warmup_bars(strategy_ir)
        # Should be max(0 + 10, 50) = 50
        assert warmup == 50

    def test_missing_indicators_key(self):
        """Warmup should handle missing indicators key gracefully."""
        strategy_ir = {}
        warmup = _calculate_warmup_bars(strategy_ir)
        # Should be max(0 + 10, 50) = 50
        assert warmup == 50

    def test_indicator_without_period(self):
        """Warmup should handle indicators without period field."""
        strategy_ir = {
            "indicators": [
                {"id": "custom_indicator", "type": "Custom"}
            ]
        }
        warmup = _calculate_warmup_bars(strategy_ir)
        # Should be max(0 + 10, 50) = 50
        assert warmup == 50

    def test_trend_pullback_typical_periods(self):
        """Test with typical trend_pullback archetype indicators.

        This is the exact scenario that caused the production bug:
        - EMA(30) for trend
        - BB(20) for band
        Backtest fetched 14 bars for a 14-day period, but needed 30+ for EMA.
        """
        strategy_ir = {
            "indicators": [
                {"id": "ema_30", "type": "EMA", "period": 30},
                {"id": "ema_50", "type": "EMA", "period": 50},
                {"id": "bollinger_20", "type": "BollingerBands", "period": 20},
            ]
        }
        warmup = _calculate_warmup_bars(strategy_ir)
        # Should be max(50 + 10, 50) = 60
        assert warmup == 60
        # This is way more than the 14 bars that were being fetched!


class TestResolutionToTimedelta:
    """Test resolution string to timedelta conversion."""

    def test_daily_resolution(self):
        """Daily resolution should return 1 day."""
        assert _resolution_to_timedelta("1d") == timedelta(days=1)
        assert _resolution_to_timedelta("daily") == timedelta(days=1)
        assert _resolution_to_timedelta("Daily") == timedelta(days=1)

    def test_4h_resolution(self):
        """4-hour resolution should return 4 hours."""
        assert _resolution_to_timedelta("4h") == timedelta(hours=4)

    def test_1h_resolution(self):
        """1-hour resolution should return 1 hour."""
        assert _resolution_to_timedelta("1h") == timedelta(hours=1)
        assert _resolution_to_timedelta("hour") == timedelta(hours=1)

    def test_15m_resolution(self):
        """15-minute resolution should return 15 minutes."""
        assert _resolution_to_timedelta("15m") == timedelta(minutes=15)

    def test_5m_resolution(self):
        """5-minute resolution should return 5 minutes."""
        assert _resolution_to_timedelta("5m") == timedelta(minutes=5)

    def test_default_to_1m(self):
        """Unknown resolution should default to 1 minute."""
        assert _resolution_to_timedelta("unknown") == timedelta(minutes=1)
        assert _resolution_to_timedelta("1m") == timedelta(minutes=1)

    def test_case_insensitive(self):
        """Resolution parsing should be case insensitive."""
        assert _resolution_to_timedelta("1D") == timedelta(days=1)
        assert _resolution_to_timedelta("1H") == timedelta(hours=1)
        assert _resolution_to_timedelta("DAILY") == timedelta(days=1)


class TestWarmupIntegration:
    """Integration tests for warmup calculation in backtest flow.

    These tests verify that warmup is correctly applied when
    fetching data for backtests.
    """

    def test_warmup_period_calculation_for_daily_backtest(self):
        """Verify warmup period is correct for daily resolution.

        Scenario: 14-day backtest (Jun 1-14) with EMA(30) indicator
        - Request: Jun 1 to Jun 14 (14 days)
        - Warmup needed: 30 + 10 + buffer = 50 bars minimum
        - With daily resolution, need 50 days of warmup
        - Data should be fetched from ~April 12 (50 days before Jun 1)
        """
        from datetime import datetime, timezone

        # This simulates what the backtest service should calculate
        request_start = datetime(2025, 6, 1, tzinfo=timezone.utc)
        strategy_ir = {
            "indicators": [
                {"id": "ema_30", "type": "EMA", "period": 30},
            ]
        }

        warmup_bars = _calculate_warmup_bars(strategy_ir)
        bar_duration = _resolution_to_timedelta("1d")
        warmup_start = request_start - (warmup_bars * bar_duration)

        # Warmup should extend back 50 days from Jun 1
        assert warmup_bars >= 40  # At least max(30+10, 50) = 50
        assert warmup_start < request_start
        # April 12 is 50 days before Jun 1
        assert warmup_start.month == 4
