"""Regression tests for backtest bugs discovered in January 2026.

These tests document specific bugs found in production and ensure they don't regress.
Each test is designed to FAIL with the current implementation, demonstrating the bug.

Bugs covered:
- Bug A: Equity curve loses cash/holdings data in transformation pipeline
- Bug B: DCA strategies with time_filter produce only 1 trade (accumulation not working)
- Bug C: Trades can occur during warmup period (before user's requested start_date)

To run these tests:
    1. Start LEAN: cd vibe-trade-lean && make run-api
    2. Run tests: uv run pytest tests/e2e/test_backtest_regression.py -v

Expected behavior:
    - These tests should FAIL until the bugs are fixed
    - After fixing, these tests serve as regression guards
"""

from datetime import datetime, timedelta, timezone

import httpx
import pytest
from vibe_trade_shared.models.data import OHLCVBar
from vibe_trade_shared.models.ir import (
    AllOfCondition,
    CompareCondition,
    CompareOp,
    EntryRule,
    IndicatorRef,
    IndicatorSpec,
    LiteralRef,
    PositionPolicy,
    PriceField,
    PriceRef,
    SetHoldingsAction,
    StrategyIR,
    TimeFilterCondition,
)

from src.service.backtest_service import BacktestRequest, BacktestService

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e


# =============================================================================
# Test Data Builders
# =============================================================================


def make_bars(
    prices: list[float],
    base_timestamp: int | None = None,
    interval_ms: int = 60000,
) -> list[OHLCVBar]:
    """Build OHLCV bars from close prices.

    Each bar: open=close, high=close+1, low=close-1, volume=1000
    Default timestamps: 1 minute apart (60000ms)

    Args:
        prices: List of close prices
        base_timestamp: Start timestamp in ms. Defaults to 2024-01-01 00:00:00 UTC
        interval_ms: Milliseconds between bars (default 60000 = 1 minute)
    """
    if base_timestamp is None:
        # 2024-01-01 00:00:00 UTC
        base_timestamp = 1704067200000

    return [
        OHLCVBar(
            t=base_timestamp + i * interval_ms,
            o=price,
            h=price + 1,
            l=price - 1,
            c=price,
            v=1000.0,
        )
        for i, price in enumerate(prices)
    ]


def make_hourly_bars(
    prices: list[float],
    start_date: datetime | None = None,
) -> list[OHLCVBar]:
    """Build hourly OHLCV bars from close prices.

    Args:
        prices: List of close prices (one per hour)
        start_date: Start datetime. Defaults to 2024-01-01 00:00:00 UTC
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    base_timestamp = int(start_date.timestamp() * 1000)
    return make_bars(prices, base_timestamp, interval_ms=3600000)  # 1 hour


def make_daily_bars(
    prices: list[float],
    start_date: datetime | None = None,
) -> list[OHLCVBar]:
    """Build daily OHLCV bars from close prices.

    Args:
        prices: List of close prices (one per day)
        start_date: Start datetime. Defaults to 2024-01-01 00:00:00 UTC
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    base_timestamp = int(start_date.timestamp() * 1000)
    return make_bars(prices, base_timestamp, interval_ms=86400000)  # 1 day


# =============================================================================
# Fixtures
# =============================================================================


def is_lean_available() -> bool:
    """Check if LEAN HTTP endpoint is running."""
    for port in [8083, 8081]:
        try:
            response = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            if response.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.TimeoutException):
            continue
    return False


requires_lean = pytest.mark.skipif(
    not is_lean_available(),
    reason="LEAN not running. Start with: cd vibe-trade-lean && make run-api",
)


@pytest.fixture
def backtest_service():
    """BacktestService configured for testing."""
    # Try ports in order
    for port in [8083, 8081]:
        try:
            response = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            if response.status_code == 200:
                return BacktestService(
                    data_service=None,
                    backtest_url=f"http://localhost:{port}/backtest",
                )
        except (httpx.ConnectError, httpx.TimeoutException):
            continue

    # Default to 8083
    return BacktestService(
        data_service=None,
        backtest_url="http://localhost:8083/backtest",
    )


# =============================================================================
# Bug A: Equity Curve Loses Cash/Holdings Data
# =============================================================================


@requires_lean
class TestBugA_EquityCurveDataLoss:
    """Bug A: Equity curve shows cash=0 and holdings_value=equity for all points.

    Root cause: Data transformation pipeline discards rich equity curve data.

    1. StrategyRuntime._track_equity() produces: {time, equity, cash, holdings, drawdown}
    2. serve_backtest.py extracts ONLY equity: [e.get("equity") for e in ...]
    3. backtest_service.py reconstructs with fake values: cash=0, holdings_value=equity

    Expected behavior: After a trade, cash should decrease and holdings should increase.
    Actual behavior: cash is always 0, holdings_value always equals total equity.

    Location: vibe-trade-lean/src/serve_backtest.py:303
              vibe-trade-execution/src/service/backtest_service.py:316-329
    """

    def test_equity_curve_has_accurate_cash_holdings(self, backtest_service):
        """Equity curve should reflect actual cash and holdings values.

        Strategy: Buy $1000 of BTC when price > 100
        Initial cash: $10,000
        Expected after trade:
            - cash ≈ $9,000 (minus the $1000 spent)
            - holdings_value ≈ $1,000 (the BTC position)

        Current bug: cash=0, holdings_value=10000 (equals total equity)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-equity-curve-bug",
            strategy_name="Equity Curve Bug Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(
                    sizing_mode="fixed_usd",
                    fixed_usd=1000.0,  # Buy exactly $1000
                ),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Data: price starts below 100, crosses above, stays there
        # Bar 0-2: below threshold (no trade)
        # Bar 3+: above threshold (entry)
        bars = make_bars([95, 97, 99, 101, 102, 103, 104, 105])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-equity-curve-bug",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
                initial_cash=10000.0,
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        assert result.response.trades, "Expected at least 1 trade"

        # Get the equity curve
        equity_curve = result.results.get("equity_curve", [])
        assert len(equity_curve) > 0, "Expected equity curve data"

        # Find a point AFTER the trade (should have holdings)
        # The trade happens at bar 3, so any point after that should show:
        # - cash < initial_cash (we spent some)
        # - holdings_value > 0 (we have a position)
        post_trade_points = equity_curve[1:]  # Skip first point (before trade possible)

        # BUG: Currently cash is always 0 and holdings_value equals equity
        # The bug manifests as:
        # 1. cash = 0 (should be ~9000 after buying $1000)
        # 2. holdings_value = equity (should be ~1000, not total equity)
        #
        # We check this by looking at ANY equity curve point after a trade occurs.
        # Since we bought $1000 of a position, we should have:
        # - cash ~ 9000
        # - holdings ~ 1000 (the position value)

        # Get the first equity curve point (after we have a position)
        assert len(equity_curve) >= 1, "Expected at least 1 equity curve point"
        point = equity_curve[0]

        cash = point.get("cash", 0)
        holdings = point.get("holdings_value", 0)
        equity = point.get("equity", 0)

        # BUG A manifests as: cash=0, holdings=equity
        # If cash is 0 AND holdings equals equity, the bug exists
        if cash == 0 and abs(holdings - equity) < 0.01:
            pytest.fail(
                f"BUG A CONFIRMED: Equity curve data loss detected!\n"
                f"  Point: {point}\n"
                f"  cash={cash} (should be ~9000 after buying $1000)\n"
                f"  holdings_value={holdings} (should be ~1000, not total equity)\n"
                f"  equity={equity}\n"
                f"  The transformation pipeline is discarding cash/holdings data."
            )


# =============================================================================
# Bug B: DCA Accumulation Not Working
# =============================================================================


@requires_lean
class TestBugB_DCAAccumulationNotWorking:
    """Bug B: DCA strategies produce only 1 trade instead of multiple.

    Root cause: Default position_policy.mode is "single", which prevents re-entry.

    In StrategyRuntime._can_accumulate():
        mode = policy.get("mode", "single")  # Default is "single"
        if mode == "single":
            return False  # No re-entry allowed

    When OnData() checks for entry while invested, it only calls _evaluate_entry()
    if _can_accumulate() returns True.

    Expected behavior: DCA with time_filter should enter on each qualifying bar.
    Actual behavior: Only 1 trade ever happens because accumulation is disabled.

    Location: vibe-trade-lean/src/Algorithms/StrategyRuntime.py:674-703
    """

    def test_dca_with_time_filter_produces_multiple_trades(self, backtest_service):
        """DCA strategy should enter multiple times on qualifying bars.

        Strategy: Buy $1000 every bar when price > 100, with accumulation enabled
        Data: 10 bars all above threshold
        Expected: Multiple trades (up to max_positions or all bars)

        Note: This test uses a simple condition (price > 100) with accumulation
        to isolate the accumulation bug from time_filter complexity.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-dca-accumulation",
            strategy_name="DCA Accumulation Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(
                    sizing_mode="fixed_usd",
                    fixed_usd=1000.0,
                    position_policy=PositionPolicy(
                        mode="accumulate",  # Enable accumulation
                        max_positions=5,  # Allow up to 5 entries
                        min_bars_between=1,  # Minimum 1 bar cooldown
                    ),
                ),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # All bars above threshold - should trigger entry on each bar after cooldown
        # With min_bars_between=1 and 10 bars, we should get 5 trades (max_positions)
        bars = make_bars([101, 102, 103, 104, 105, 106, 107, 108, 109, 110])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-dca-accumulation",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
                initial_cash=100000.0,
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"

        trades = result.response.trades
        trade_count = len(trades)

        # BUG: Currently only 1 trade happens because accumulation is broken
        # This test should FAIL until the bug is fixed
        assert trade_count > 1, (
            f"BUG B: Only {trade_count} trade(s) occurred, but expected multiple trades. "
            f"DCA accumulation is not working - check position_policy.mode handling. "
            f"Strategy should have entered on multiple bars with max_positions=5."
        )

        # With max_positions=5 and min_bars_between=1:
        # Bar 0: Entry #1
        # Bar 1: cooldown
        # Bar 2: Entry #2
        # Bar 3: cooldown
        # Bar 4: Entry #3
        # Bar 5: cooldown
        # Bar 6: Entry #4
        # Bar 7: cooldown
        # Bar 8: Entry #5 (max reached)
        # Bar 9: no entry (max reached)
        assert trade_count == 5, (
            f"Expected exactly 5 trades (max_positions=5), got {trade_count}. "
            f"Entry bars should be [0, 2, 4, 6, 8]."
        )

    def test_weekly_dca_with_time_filter(self, backtest_service):
        """Weekly DCA should enter every Monday.

        Strategy: Buy $1000 every Monday (day_of_week=0)
        Data: 4 weeks of daily bars (28 bars)
        Expected: 4 trades (one per Monday)

        This more closely matches the production DCA scenario.
        """
        # Start on a Monday: January 1, 2024 was a Monday
        start_date = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        strategy_ir = StrategyIR(
            strategy_id="test-weekly-dca",
            strategy_name="Weekly DCA Test",
            symbol="TESTUSD",
            resolution="Hour",  # Hourly bars
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(
                    conditions=[
                        # Always true condition (price > 0)
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=0.0),
                        ),
                        # Time filter: only on Mondays
                        TimeFilterCondition(
                            days_of_week=[0],  # Monday
                        ),
                    ]
                ),
                action=SetHoldingsAction(
                    sizing_mode="fixed_usd",
                    fixed_usd=1000.0,
                    position_policy=PositionPolicy(
                        mode="accumulate",
                        min_bars_between=24 * 7,  # 1 week in hours (168 hours)
                    ),
                ),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # 4 weeks of hourly data (24 * 28 = 672 bars)
        # Price just stays constant - we only care about time_filter
        prices = [100.0] * (24 * 28)
        bars = make_hourly_bars(prices, start_date)

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-weekly-dca",
                start_date=start_date,
                end_date=start_date + timedelta(days=28),
                symbol="TESTUSD",
                resolution="1h",
                initial_cash=100000.0,
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"

        trades = result.response.trades
        trade_count = len(trades)

        # BUG: Currently only 1 trade happens
        # This test should FAIL until the bug is fixed
        assert trade_count == 4, (
            f"BUG B: Expected 4 trades (one per Monday over 4 weeks), got {trade_count}. "
            f"Weekly DCA accumulation is not working."
        )


# =============================================================================
# Bug C: Trades During Warmup Period
# =============================================================================


@requires_lean
class TestBugC_TradesDuringWarmup:
    """Bug C: Trades can occur before user's requested start_date.

    Root cause: No mechanism prevents trades during the warmup period.

    The backtest service calculates warmup as:
        lean_start_date = request.start_date - (warmup_bars * bar_duration)

    LEAN receives lean_start_date and can execute trades from that point.
    For strategies without indicators, _indicators_ready() returns True immediately,
    so trades can happen during warmup (before user's requested start_date).

    IMPORTANT: This bug only manifests in PRODUCTION when data is fetched via
    DataService (which includes warmup bars). Tests using inline_bars bypass
    the warmup calculation. To properly test this, we need to:
    1. Simulate what LEAN receives in production (lean_start_date before user_start_date)
    2. OR mock the DataService

    For now, these tests document the expected behavior. The bug was observed in
    production where trade timestamps showed Dec 9 for a Jan 24 backtest.

    Location: vibe-trade-execution/src/service/backtest_service.py:216
              vibe-trade-lean/src/Algorithms/StrategyRuntime.py (OnData)
    """

    def test_no_trades_before_requested_start_date_simulated_warmup(self, backtest_service):
        """Verify trades only occur after user's requested start date, not during warmup.

        This test verifies the fix for Bug C:
        1. User requests backtest starting Jan 15
        2. Data includes warmup period (Jan 1-14) with triggering prices
        3. BacktestService passes trading_start_date=Jan 15 to LEAN
        4. LEAN should NOT trade during warmup (Jan 1-14), only after Jan 15

        The fix is in BacktestService: trading_start_date=request.start_date.date()
        And in StrategyRuntime: skip trading when Time < trading_start_date
        """
        # User wants to start Jan 15, but we include warmup data from Jan 1
        warmup_start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        user_start_date = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
        user_end_date = datetime(2024, 1, 20, 0, 0, 0, tzinfo=timezone.utc)

        strategy_ir = StrategyIR(
            strategy_id="test-warmup-trades",
            strategy_name="Warmup Trades Bug Test",
            symbol="TESTUSD",
            resolution="Hour",
            indicators=[],  # NO INDICATORS - warmup should still be respected
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Build data starting Jan 1 (warmup period)
        # All prices above threshold - would trigger immediately in warmup
        warmup_prices = [105.0] * (14 * 24)  # Jan 1-14 (warmup) - should NOT trade
        user_prices = [110.0] * (6 * 24)     # Jan 15-20 (user period) - CAN trade
        all_prices = warmup_prices + user_prices
        bars = make_hourly_bars(all_prices, warmup_start)

        # Pass user's actual start date (Jan 15) - the fix passes this as trading_start_date
        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-warmup-trades",
                start_date=user_start_date,  # User's requested start (Jan 15)
                end_date=user_end_date,
                symbol="TESTUSD",
                resolution="1h",
                initial_cash=100000.0,
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        assert result.response.trades, "Expected at least 1 trade"

        # Check trade timestamps
        trade = result.response.trades[0]
        entry_time = trade.entry_time
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)

        # Fix verified: Trade should occur on/after Jan 15, NOT during warmup
        if entry_time < user_start_date:
            pytest.fail(
                f"BUG C CONFIRMED: Trade during warmup period!\n"
                f"  Trade entry_time: {entry_time}\n"
                f"  User start_date:  {user_start_date}\n"
                f"  Warmup start:     {warmup_start}\n"
                f"  The system allows trades before user's requested backtest window.\n"
                f"  Fix: Pass 'trading_start_date' to LEAN to prevent warmup trades."
            )

    def test_warmup_with_indicators_respects_start_date(self, backtest_service):
        """Even with indicators requiring warmup, trades should not occur before start_date.

        Strategy: EMA crossover (requires warmup)
        Data: Includes crossover during warmup period
        Expected: First trade should be on/after start_date, not during warmup

        The fix passes trading_start_date to LEAN so indicators can warm up
        without executing trades during the warmup window.
        """
        warmup_start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        user_start_date = datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
        user_end_date = datetime(2024, 2, 15, 0, 0, 0, tzinfo=timezone.utc)

        strategy_ir = StrategyIR(
            strategy_id="test-warmup-indicator",
            strategy_name="Warmup Indicator Bug Test",
            symbol="TESTUSD",
            resolution="Hour",
            indicators=[
                IndicatorSpec(id="ema_10", type="EMA", period=10),
                IndicatorSpec(id="ema_30", type="EMA", period=30),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=IndicatorRef(indicator_id="ema_10"),
                    op=CompareOp.GT,
                    right=IndicatorRef(indicator_id="ema_30"),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Build data starting Jan 1 with strong uptrend
        # EMA10 should cross above EMA30 well before user_start_date
        warmup_prices = [100 + i * 0.5 for i in range(31 * 24)]  # Strong uptrend
        last_warmup_price = warmup_prices[-1]
        user_prices = [last_warmup_price + i * 0.1 for i in range(15 * 24)]
        all_prices = warmup_prices + user_prices
        bars = make_hourly_bars(all_prices, warmup_start)

        # Pass user's actual start date (Feb 1) - fix passes this as trading_start_date
        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-warmup-indicator",
                start_date=user_start_date,  # User's requested start (Feb 1)
                end_date=user_end_date,
                symbol="TESTUSD",
                resolution="1h",
                initial_cash=100000.0,
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"

        # Any trades that occur should be checked against user_start_date
        for trade in result.response.trades:
            entry_time = trade.entry_time
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)

            if entry_time < user_start_date:
                pytest.fail(
                    f"BUG C CONFIRMED: Trade during warmup period!\n"
                    f"  Trade entry_time: {entry_time}\n"
                    f"  User start_date:  {user_start_date}\n"
                    f"  The indicator strategy traded during warmup."
                )
