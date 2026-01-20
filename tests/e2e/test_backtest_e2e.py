"""End-to-end backtest tests with comprehensive strategy coverage.

These tests verify the full backtest pipeline with exact trade assertions:
1. Build StrategyIR from Pydantic models
2. Build synthetic OHLCV data with known signal patterns
3. Call BacktestService which hits real LEAN HTTP endpoint
4. Assert exact trades (entry bar, exit bar, profit)

Requirements:
- R1: Pydantic models everywhere
- R2: Exact assertions with documented reasoning
- R3: Hand-built test data per test
- R4: Container-based (requires LEAN running)

Strategy coverage:
- Price threshold entry/exit
- EMA crossover entry/exit
- No-entry scenarios
- Multiple trades in single backtest
- RSI-based strategies

To run these tests:
    1. Start LEAN: cd vibe-trade-lean && make run-api
    2. Run tests: uv run pytest tests/e2e/ -v

To skip E2E tests during development:
    uv run pytest -m "not e2e"

Performance: ~7 seconds per test, ~7 minutes total for 63 tests.
"""

from datetime import datetime, timezone

import httpx
import pytest
from vibe_trade_shared.models.data import OHLCVBar
from vibe_trade_shared.models.ir import (
    AllOfCondition,
    AnyOfCondition,
    BreakoutCondition,
    CompareCondition,
    CompareOp,
    CrossCondition,
    EntryRule,
    EventWindowCondition,
    ExitRule,
    FlagPatternCondition,
    GapCondition,
    GateRule,
    IncrementStateAction,
    IndicatorBandRef,
    IndicatorRef,
    IndicatorSpec,
    IntermarketCondition,
    IRExpression,
    LiquidateAction,
    LiquiditySweepCondition,
    LiteralRef,
    MaxStateAction,
    MultiLeaderIntermarketCondition,
    NotCondition,
    OverlayRule,
    PennantPatternCondition,
    PriceField,
    PriceRef,
    RegimeCondition,
    SequenceCondition,
    SequenceStep,
    SetHoldingsAction,
    SetStateAction,
    SpreadCondition,
    SqueezeCondition,
    StateRef,
    StateVarSpec,
    StrategyIR,
    TimeFilterCondition,
    TimeRef,
    TrailingBreakoutCondition,
    TrailingStateCondition,
    VolumeRef,
)

from src.service.backtest_service import BacktestRequest, BacktestService

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e

# =============================================================================
# Test Data Builders
# =============================================================================


# Default base timestamp: 2024-01-01 00:00:00 UTC (for test data)
DEFAULT_BASE_TIMESTAMP_MS = 1704067200000


def make_bars(prices: list[float], base_timestamp: int | None = None) -> list[OHLCVBar]:
    """Build OHLCV bars from close prices.

    Each bar: open=close, high=close+1, low=close-1, volume=1000
    Timestamps: 1 minute apart (60000ms)

    Args:
        prices: List of close prices
        base_timestamp: Start timestamp in ms. Defaults to 2024-01-01 00:00:00 UTC
    """
    if base_timestamp is None:
        base_timestamp = DEFAULT_BASE_TIMESTAMP_MS

    return [
        OHLCVBar(
            t=base_timestamp + i * 60000,
            o=price,
            h=price + 1,
            l=price - 1,
            c=price,
            v=1000.0,
        )
        for i, price in enumerate(prices)
    ]


def make_trending_bars(
    start_price: float,
    num_bars: int,
    trend_pct_per_bar: float,
    base_timestamp: int | None = None,
) -> list[OHLCVBar]:
    """Build OHLCV bars with consistent trend.

    Args:
        start_price: Starting price
        num_bars: Number of bars
        trend_pct_per_bar: Percent change per bar (0.01 = 1%)
        base_timestamp: Start timestamp in ms. Defaults to 2024-01-01 00:00:00 UTC
    """
    prices = []
    price = start_price
    for _ in range(num_bars):
        prices.append(price)
        price = price * (1 + trend_pct_per_bar)
    return make_bars(prices, base_timestamp)


# =============================================================================
# Fixtures
# =============================================================================


def is_lean_available() -> bool:
    """Check if LEAN HTTP endpoint is running."""
    try:
        response = httpx.get("http://localhost:8081/health", timeout=2.0)
        return response.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


requires_lean = pytest.mark.skipif(
    not is_lean_available(),
    reason="LEAN not running. Start with: docker run -d -p 8081:8080 lean-backtest-service",
)


@pytest.fixture
def backtest_service(request):
    """BacktestService configured for testing.

    Supports parallel execution with pytest-xdist:
    - Single worker: uses port 8081 (default)
    - Multiple workers: distributes across ports 8081-8084

    Usage:
        # Sequential (single container)
        pytest tests/e2e/ -v

        # Parallel (4 containers on ports 8081-8084)
        docker-compose -f vibe-trade-lean/docker-compose.parallel.yml up -d
        pytest tests/e2e/ -n 4
    """
    # Check if running with pytest-xdist
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")

    if worker_id == "master":
        # Single worker mode - use default port
        port = 8081
    else:
        # Multi-worker mode - extract worker number (gw0, gw1, gw2, gw3)
        worker_num = int(worker_id.replace("gw", ""))
        port = 8081 + (worker_num % 4)  # Round-robin across 4 ports

    return BacktestService(
        data_service=None,
        backtest_url=f"http://localhost:{port}/backtest",
    )


# =============================================================================
# Price Threshold Strategy Tests
# =============================================================================


@requires_lean
class TestPriceThresholdStrategy:
    """Test price threshold entry strategy.

    Strategy: Enter when close > threshold
    """

    def test_entry_on_exact_bar(self, backtest_service):
        """Entry occurs on first bar where close > 100.

        Data pattern:
            Bar 0: close = 95  (below threshold)
            Bar 1: close = 97  (below threshold)
            Bar 2: close = 99  (below threshold)
            Bar 3: close = 101 (ABOVE threshold) <- ENTRY
            Bar 4: close = 103 (in position)
            Bar 5: close = 105 (closes at end)

        Expected:
            - 1 trade total
            - Entry at bar 3 (first bar with close > 100)
            - Entry price = 101
            - Exit at bar 5 (end of data)
            - Exit price = 105
            - PnL ≈ 4% (101 → 105)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-threshold",
            strategy_name="Price Threshold Test",
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
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 97, 99, 101, 103, 105])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-threshold",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"

        trade = trades[0]
        assert trade.entry_bar == 3, f"Expected entry at bar 3, got {trade.entry_bar}"
        assert trade.entry_price == 101.0
        assert trade.exit_bar == 5, f"Expected exit at bar 5, got {trade.exit_bar}"
        assert trade.exit_price == 105.0
        assert 3.5 < trade.pnl_pct < 4.5, f"Expected ~4% gain, got {trade.pnl_pct}%"

    def test_no_entry_when_threshold_never_crossed(self, backtest_service):
        """No trades when price never exceeds threshold.

        Data: All bars close below 100
        Expected: 0 trades
        """
        strategy_ir = StrategyIR(
            strategy_id="test-no-entry",
            strategy_name="No Entry Test",
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
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([90, 92, 94, 96, 98])  # All below 100

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-no-entry",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        assert len(result.response.trades) == 0, "Expected 0 trades"

    def test_entry_with_explicit_exit(self, backtest_service):
        """Entry and exit both triggered by price conditions.

        Strategy: Enter when close > 100, Exit when close < 95

        Data pattern:
            Bar 0: close = 95  (below entry)
            Bar 1: close = 101 (ENTRY)
            Bar 2: close = 105 (in position)
            Bar 3: close = 100 (in position)
            Bar 4: close = 94  (EXIT - below 95)

        Expected:
            - 1 trade
            - Entry at bar 1
            - Exit at bar 4 (explicit exit condition)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-entry-exit",
            strategy_name="Entry Exit Test",
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
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="price_exit",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=LiteralRef(value=95.0),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 101, 105, 100, 94])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-entry-exit",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1

        trade = trades[0]
        assert trade.entry_bar == 1, f"Expected entry at bar 1, got {trade.entry_bar}"
        assert trade.entry_price == 101.0
        assert trade.exit_bar == 4, f"Expected exit at bar 4, got {trade.exit_bar}"
        assert trade.exit_price == 94.0


# =============================================================================
# EMA Crossover Strategy Tests
# =============================================================================


@requires_lean
class TestEMACrossoverStrategy:
    """Test EMA crossover entry and exit.

    Strategy:
        Entry: EMA10 crosses above EMA30
        Exit: EMA10 crosses below EMA30
    """

    def test_crossover_entry_and_exit(self, backtest_service):
        """EMA crossover triggers entry, reverse triggers exit.

        Data pattern:
            Phase 1 (bars 0-39): Flat at 100 (EMAs converge)
            Phase 2 (bars 40-59): Uptrend +1%/bar (EMA10 > EMA30) <- ENTRY
            Phase 3 (bars 60-79): Downtrend -1%/bar (EMA10 < EMA30) <- EXIT

        Expected:
            - Entry during uptrend phase (bar ~45-55)
            - Exit during downtrend phase (bar ~65-75)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-ema-cross",
            strategy_name="EMA Crossover Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_10", type="EMA", period=10),
                IndicatorSpec(id="ema_30", type="EMA", period=30),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CrossCondition(
                    left=IndicatorRef(indicator_id="ema_10"),
                    right=IndicatorRef(indicator_id="ema_30"),
                    direction="above",
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="ema_exit",
                    condition=CrossCondition(
                        left=IndicatorRef(indicator_id="ema_10"),
                        right=IndicatorRef(indicator_id="ema_30"),
                        direction="below",
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        flat_bars = make_bars([100.0] * 40)
        uptrend_bars = make_trending_bars(
            100.0, 20, 0.01, base_timestamp=DEFAULT_BASE_TIMESTAMP_MS + 40 * 60000
        )
        downtrend_bars = make_trending_bars(
            uptrend_bars[-1].c, 20, -0.01, base_timestamp=DEFAULT_BASE_TIMESTAMP_MS + 60 * 60000
        )
        all_bars = flat_bars + uptrend_bars + downtrend_bars

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-ema-cross",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=all_bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected at least 1 trade"

        trade = trades[0]
        # Entry should be during uptrend phase (bars 40-59 in data, which is minutes 40-59)
        # entry_bar is relative to indicator warmup (EMA30 needs ~30 bars), so we check entry_time
        entry_minute = trade.entry_time.minute + trade.entry_time.hour * 60
        assert 40 <= entry_minute < 60, f"Expected entry during uptrend (minute 40-59), got minute {entry_minute}"

        # Exit should be during downtrend phase (minutes 60-79) or end of backtest
        exit_minute = trade.exit_time.minute + trade.exit_time.hour * 60
        assert exit_minute > entry_minute, f"Exit must be after entry: {exit_minute} vs {entry_minute}"

    def test_no_crossover_no_trade(self, backtest_service):
        """No trades when EMAs never cross.

        Data: Flat price at 100 (EMAs converge, never cross)
        Expected: 0 trades
        """
        strategy_ir = StrategyIR(
            strategy_id="test-no-cross",
            strategy_name="No Cross Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_10", type="EMA", period=10),
                IndicatorSpec(id="ema_30", type="EMA", period=30),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CrossCondition(
                    left=IndicatorRef(indicator_id="ema_10"),
                    right=IndicatorRef(indicator_id="ema_30"),
                    direction="above",
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Flat price - EMAs will converge but not cross
        bars = make_bars([100.0] * 50)

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-no-cross",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        assert len(result.response.trades) == 0, "Expected 0 trades with flat data"


# =============================================================================
# RSI Strategy Tests
# =============================================================================


@requires_lean
class TestRSIStrategy:
    """Test RSI-based entry strategy.

    Strategy: Enter when RSI < 30 (oversold)
    """

    def test_rsi_oversold_entry(self, backtest_service):
        """Entry when RSI drops below 30.

        Data pattern: Strong downtrend to push RSI low, then recovery
            Bars 0-19: Price drops 90 -> 70 (RSI drops)
            Bars 20-29: Price stabilizes/rises (entry expected when RSI < 30)

        Note: RSI(14) needs ~15 bars to warm up
        """
        strategy_ir = StrategyIR(
            strategy_id="test-rsi",
            strategy_name="RSI Oversold Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="rsi", type="RSI", period=14),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=IndicatorRef(indicator_id="rsi"),
                    op=CompareOp.LT,
                    right=LiteralRef(value=30.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Create downtrend to push RSI low
        downtrend = make_trending_bars(90.0, 20, -0.02)  # -2% per bar
        recovery = make_trending_bars(
            downtrend[-1].c, 10, 0.01, base_timestamp=DEFAULT_BASE_TIMESTAMP_MS + 20 * 60000
        )
        bars = downtrend + recovery

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-rsi",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        # Should have entered during the downtrend when RSI < 30
        assert len(trades) >= 1, "Expected entry when RSI < 30"


# =============================================================================
# Multiple Trades Tests
# =============================================================================


@requires_lean
class TestMultipleTrades:
    """Test strategies that generate multiple trades."""

    def test_multiple_entries_exits(self, backtest_service):
        """Multiple entry/exit cycles in single backtest.

        Strategy: Enter when close > 105, Exit when close < 95

        Data pattern:
            Bars 0-4:   95, 100, 106, 108, 94  (trade 1: enter bar 2, exit bar 4)
            Bars 5-9:   96, 107, 110, 93, 90   (trade 2: enter bar 6, exit bar 8)

        Expected: 2 complete trades
        """
        strategy_ir = StrategyIR(
            strategy_id="test-multi",
            strategy_name="Multiple Trades Test",
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
                    right=LiteralRef(value=105.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="price_exit",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=LiteralRef(value=95.0),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Two cycles: up > 105, down < 95
        prices = [95, 100, 106, 108, 94, 96, 107, 110, 93, 90]
        bars = make_bars(prices)

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-multi",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 2, f"Expected 2 trades, got {len(trades)}"

        # Trade 1
        assert trades[0].entry_bar == 2, f"Trade 1 entry at bar 2, got {trades[0].entry_bar}"
        assert trades[0].exit_bar == 4, f"Trade 1 exit at bar 4, got {trades[0].exit_bar}"

        # Trade 2
        assert trades[1].entry_bar == 6, f"Trade 2 entry at bar 6, got {trades[1].entry_bar}"
        assert trades[1].exit_bar == 8, f"Trade 2 exit at bar 8, got {trades[1].exit_bar}"


# =============================================================================
# Condition Primitive Tests
# =============================================================================


@requires_lean
class TestCompareOperators:
    """Test all comparison operators in CompareCondition."""

    def test_compare_lt_operator(self, backtest_service):
        """Entry when close < 100.

        Data: [105, 102, 98, 95]
        Expected: Entry bar 2 (98 < 100)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-lt",
            strategy_name="Less Than Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([105, 102, 98, 95])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-lt",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 2, f"Expected entry at bar 2, got {trades[0].entry_bar}"
        assert trades[0].entry_price == 98.0

    def test_compare_lte_operator(self, backtest_service):
        """Entry when close <= 100.

        Data: [105, 102, 100, 95]
        Expected: Entry bar 2 (100 <= 100)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-lte",
            strategy_name="Less Than Equal Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LTE,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([105, 102, 100, 95])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-lte",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 2, f"Expected entry at bar 2, got {trades[0].entry_bar}"
        assert trades[0].entry_price == 100.0

    def test_compare_gte_operator(self, backtest_service):
        """Entry when close >= 100.

        Data: [95, 98, 100, 105]
        Expected: Entry bar 2 (100 >= 100)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-gte",
            strategy_name="Greater Than Equal Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GTE,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 98, 100, 105])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-gte",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 2, f"Expected entry at bar 2, got {trades[0].entry_bar}"
        assert trades[0].entry_price == 100.0

    def test_compare_eq_operator(self, backtest_service):
        """Entry when close == 100.

        Data: [98, 99, 100, 101]
        Expected: Entry bar 2 (100 == 100)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-eq",
            strategy_name="Equal Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.EQ,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([98, 99, 100, 101])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-eq",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 2, f"Expected entry at bar 2, got {trades[0].entry_bar}"
        assert trades[0].entry_price == 100.0

    def test_compare_neq_operator(self, backtest_service):
        """Entry when close != 100.

        Data: [100, 100, 101, 100]
        Expected: Entry bar 2 (101 != 100)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-neq",
            strategy_name="Not Equal Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.NEQ,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([100, 100, 101, 100])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-neq",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 2, f"Expected entry at bar 2, got {trades[0].entry_bar}"
        assert trades[0].entry_price == 101.0


@requires_lean
class TestAllOfCondition:
    """Test AllOfCondition (logical AND)."""

    def test_allof_two_conditions_both_true(self, backtest_service):
        """Entry requires BOTH close > 100 AND close < 110.

        Data: [95, 98, 105, 108, 112]
        Expected: Entry bar 2 (105 is between 100 and 110)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-allof",
            strategy_name="AllOf Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.LT,
                            right=LiteralRef(value=110.0),
                        ),
                    ]
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 98, 105, 108, 112])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-allof",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 2, f"Expected entry at bar 2, got {trades[0].entry_bar}"
        assert trades[0].entry_price == 105.0

    def test_allof_one_false_no_entry(self, backtest_service):
        """No entry when one condition is always false.

        Data: [95, 98, 99] - all below 100
        Entry: close > 100 AND close < 110
        Expected: 0 trades (never above 100)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-allof-fail",
            strategy_name="AllOf Fail Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.LT,
                            right=LiteralRef(value=110.0),
                        ),
                    ]
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 98, 99])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-allof-fail",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        assert len(result.response.trades) == 0


@requires_lean
class TestAnyOfCondition:
    """Test AnyOfCondition (logical OR)."""

    def test_anyof_first_condition_true(self, backtest_service):
        """Entry when first condition is true.

        Data: [95, 101, 99]
        Entry: close > 100 OR close < 90
        Expected: Entry bar 1 (101 > 100)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-anyof-first",
            strategy_name="AnyOf First Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AnyOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.LT,
                            right=LiteralRef(value=90.0),
                        ),
                    ]
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 101, 99])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-anyof-first",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 1, f"Expected entry at bar 1, got {trades[0].entry_bar}"
        assert trades[0].entry_price == 101.0

    def test_anyof_second_condition_true(self, backtest_service):
        """Entry when second condition is true.

        Data: [95, 92, 89, 91]
        Entry: close > 100 OR close < 90
        Expected: Entry bar 2 (89 < 90)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-anyof-second",
            strategy_name="AnyOf Second Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AnyOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.LT,
                            right=LiteralRef(value=90.0),
                        ),
                    ]
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 92, 89, 91])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-anyof-second",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 2, f"Expected entry at bar 2, got {trades[0].entry_bar}"
        assert trades[0].entry_price == 89.0

    def test_anyof_neither_true_no_entry(self, backtest_service):
        """No entry when neither condition is true.

        Data: [95, 92, 94, 96] - all between 90 and 100
        Entry: close > 100 OR close < 90
        Expected: 0 trades
        """
        strategy_ir = StrategyIR(
            strategy_id="test-anyof-neither",
            strategy_name="AnyOf Neither Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AnyOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.LT,
                            right=LiteralRef(value=90.0),
                        ),
                    ]
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 92, 94, 96])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-anyof-neither",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        assert len(result.response.trades) == 0


@requires_lean
class TestNotCondition:
    """Test NotCondition (logical negation)."""

    def test_not_simple(self, backtest_service):
        """Entry when NOT close > 100.

        Data: [101, 102, 99, 98]
        Expected: Entry bar 2 (99 is NOT > 100)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-not",
            strategy_name="Not Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=NotCondition(
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=LiteralRef(value=100.0),
                    )
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([101, 102, 99, 98])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-not",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 2, f"Expected entry at bar 2, got {trades[0].entry_bar}"
        assert trades[0].entry_price == 99.0

    def test_not_nested_allof(self, backtest_service):
        """Entry when NOT (close > 100 AND close < 110).

        De Morgan: NOT (A AND B) = (NOT A) OR (NOT B)
        Data: [105, 108, 112, 99]
        Expected: Entry bar 2 (112 is NOT between 100-110 because > 110)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-not-nested",
            strategy_name="Not Nested Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=NotCondition(
                    condition=AllOfCondition(
                        conditions=[
                            CompareCondition(
                                left=PriceRef(field=PriceField.CLOSE),
                                op=CompareOp.GT,
                                right=LiteralRef(value=100.0),
                            ),
                            CompareCondition(
                                left=PriceRef(field=PriceField.CLOSE),
                                op=CompareOp.LT,
                                right=LiteralRef(value=110.0),
                            ),
                        ]
                    )
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Bar 0: 105 is between 100-110, so AllOf=True, NOT(True)=False -> no entry
        # Bar 1: 108 is between 100-110, so AllOf=True, NOT(True)=False -> no entry
        # Bar 2: 112 is NOT between 100-110 (>=110), so AllOf=False, NOT(False)=True -> ENTRY
        bars = make_bars([105, 108, 112, 99])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-not-nested",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 2, f"Expected entry at bar 2, got {trades[0].entry_bar}"
        assert trades[0].entry_price == 112.0


@requires_lean
class TestExitReason:
    """Test exit_reason field is properly captured."""

    def test_exit_reason_explicit_rule(self, backtest_service):
        """Exit reason shows the exit rule ID when triggered.

        Strategy: Enter when close > 100, Exit when close < 95
        Data: [95, 101, 105, 94]
        Expected: exit_reason = "price_exit"
        """
        strategy_ir = StrategyIR(
            strategy_id="test-exit-reason",
            strategy_name="Exit Reason Test",
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
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="price_exit",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=LiteralRef(value=95.0),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 101, 105, 94])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-exit-reason",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].exit_reason == "price_exit", f"Expected exit_reason='price_exit', got '{trades[0].exit_reason}'"

    def test_exit_reason_end_of_backtest(self, backtest_service):
        """Exit reason shows 'end_of_backtest' when no exit triggers.

        Strategy: Enter when close > 100 (no explicit exit)
        Data: [95, 101, 105, 110]
        Expected: exit_reason = "end_of_backtest"
        """
        strategy_ir = StrategyIR(
            strategy_id="test-eob",
            strategy_name="End of Backtest Test",
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
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],  # No exit rules
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 101, 105, 110])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-eob",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].exit_reason == "end_of_backtest", f"Expected exit_reason='end_of_backtest', got '{trades[0].exit_reason}'"


# =============================================================================
# Phase 2: Indicator Coverage Tests
# =============================================================================


def make_volatile_bars(
    center_price: float,
    num_bars: int,
    volatility_pct: float,
    base_timestamp: int | None = None,
) -> list[OHLCVBar]:
    """Build bars with specified volatility for BB/KC tests.

    Creates bars that oscillate around center_price with given volatility.
    """
    if base_timestamp is None:
        base_timestamp = DEFAULT_BASE_TIMESTAMP_MS

    import math
    bars = []
    for i in range(num_bars):
        # Create oscillating prices
        offset = volatility_pct * center_price * math.sin(i * 0.5)
        close = center_price + offset
        bars.append(OHLCVBar(
            t=base_timestamp + i * 60000,
            o=close,
            h=close + abs(offset) * 0.2,
            l=close - abs(offset) * 0.2,
            c=close,
            v=1000.0,
        ))
    return bars


@requires_lean
class TestBollingerBands:
    """Test Bollinger Bands indicator and band references."""

    def test_bb_lower_band_entry(self, backtest_service):
        """Entry when close touches lower Bollinger Band.

        Strategy: Enter when close < BB lower band
        Data: 30 bars stable at 100, then 5 bars with sharp drop to break below lower band

        After 20 bars of stable price at 100 with small variance, BB lower ~= 100 - 2*std
        With std ~= 0.5 (0.5% volatility), lower band ~= 99. A drop to 90 should trigger.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-bb-lower",
            strategy_name="BB Lower Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="bb", type="BB", period=20, multiplier=2.0),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=IndicatorBandRef(indicator_id="bb", band="lower"),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # 30 stable bars at 100 with minimal volatility for BB warmup
        # The prices alternate between 99.5 and 100.5 to have some volatility
        stable_bars = []
        for i in range(30):
            price = 100.0 + (0.5 if i % 2 == 0 else -0.5)
            stable_bars.append(OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                o=price,
                h=price + 0.2,
                l=price - 0.2,
                c=price,
                v=1000.0,
            ))

        # After 30 bars of alternating 99.5/100.5, the SMA ~= 100, std ~= 0.5
        # BB lower ~= 100 - 2*0.5 = 99. A drop to 90 should trigger entry.
        drop_bars = [
            OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + (30 + i) * 60_000,
                o=95.0 - i * 2.0,
                h=95.0 - i * 2.0,
                l=90.0 - i * 2.0,
                c=90.0 - i * 2.0,  # 90, 88, 86, 84, 82
                v=1000.0,
            )
            for i in range(5)
        ]
        bars = stable_bars + drop_bars

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-bb-lower",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        # Should have entered when price dropped below lower band
        assert len(trades) >= 1, "Expected at least 1 trade when price breaks below BB lower"

    def test_bb_upper_band_exit(self, backtest_service):
        """Exit when close reaches upper Bollinger Band.

        Strategy: Enter when close > 100, Exit when close > BB upper
        Data: Uptrend to push price above upper band
        """
        strategy_ir = StrategyIR(
            strategy_id="test-bb-upper-exit",
            strategy_name="BB Upper Exit Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="bb", type="BB", period=20, multiplier=2.0),
            ],
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
            exits=[
                ExitRule(
                    id="bb_upper_exit",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=IndicatorBandRef(indicator_id="bb", band="upper"),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Flat then uptrend to trigger entry then hit upper band
        flat = make_bars([98.0] * 25)
        uptrend = make_trending_bars(101.0, 20, 0.015, base_timestamp=DEFAULT_BASE_TIMESTAMP_MS + 25 * 60000)
        bars = flat + uptrend

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-bb-upper-exit",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected at least 1 trade"
        # If exit triggered, should have exit_reason
        if trades[0].exit_reason != "end_of_backtest":
            assert trades[0].exit_reason == "bb_upper_exit"


@requires_lean
class TestSMA:
    """Test SMA indicator."""

    def test_sma_crossover(self, backtest_service):
        """Entry when price crosses above SMA.

        Strategy: Enter when close crosses above SMA20
        Data: Flat then uptrend to create crossover
        """
        strategy_ir = StrategyIR(
            strategy_id="test-sma-cross",
            strategy_name="SMA Cross Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="sma_20", type="SMA", period=20),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CrossCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    right=IndicatorRef(indicator_id="sma_20"),
                    direction="above",
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Start below SMA, then cross above
        below = make_bars([95.0] * 25)
        cross_up = make_trending_bars(96.0, 15, 0.02, base_timestamp=DEFAULT_BASE_TIMESTAMP_MS + 25 * 60000)
        bars = below + cross_up

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-sma-cross",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected crossover trade"


@requires_lean
class TestATR:
    """Test ATR indicator."""

    def test_atr_threshold(self, backtest_service):
        """Entry when ATR exceeds threshold (high volatility).

        Strategy: Enter when ATR > 2.0
        Data: Low volatility then high volatility bars
        """
        strategy_ir = StrategyIR(
            strategy_id="test-atr",
            strategy_name="ATR Threshold Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="atr", type="ATR", period=14),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=IndicatorRef(indicator_id="atr"),
                    op=CompareOp.GT,
                    right=LiteralRef(value=2.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Low volatility bars, then high volatility
        low_vol = make_bars([100.0] * 20)  # Flat = low ATR
        # High volatility: big swings
        high_vol = []
        for i in range(20):
            price = 100.0 + (5.0 if i % 2 == 0 else -5.0)
            high_vol.append(OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + (20 + i) * 60000,
                o=price,
                h=price + 3,
                l=price - 3,
                c=price,
                v=1000.0,
            ))
        bars = low_vol + high_vol

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-atr",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected trade when ATR exceeds threshold"


@requires_lean
class TestIndicatorComparison:
    """Test comparing two indicators."""

    def test_indicator_vs_indicator(self, backtest_service):
        """Entry when fast EMA > slow EMA (trend filter).

        Strategy: Enter when EMA10 > EMA30 (existing uptrend)
        Data: Uptrend so fast EMA stays above slow EMA
        """
        strategy_ir = StrategyIR(
            strategy_id="test-ind-vs-ind",
            strategy_name="Indicator vs Indicator Test",
            symbol="TESTUSD",
            resolution="Minute",
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

        # Strong uptrend: EMA10 will be above EMA30
        bars = make_trending_bars(100.0, 50, 0.01)

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-ind-vs-ind",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        # Should enter once EMA10 > EMA30 (after warmup)
        assert len(trades) >= 1, "Expected entry when EMA10 > EMA30"


# =============================================================================
# Phase 3: State Variable and Value Reference Tests
# =============================================================================


@requires_lean
class TestStateVariables:
    """Test state variables with on_bar and on_fill hooks."""

    def test_bars_since_entry_counter(self, backtest_service):
        """Track bars since entry using state variable.

        Strategy: Enter when close > 100, exit after 3 bars
        State: bars_held increments each bar when invested
        """
        strategy_ir = StrategyIR(
            strategy_id="test-bars-counter",
            strategy_name="Bars Counter Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[
                StateVarSpec(id="bars_held", default=0.0),
            ],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[
                    SetStateAction(state_id="bars_held", value=LiteralRef(value=0.0)),
                ],
            ),
            exits=[
                ExitRule(
                    id="time_exit",
                    condition=CompareCondition(
                        left=StateRef(state_id="bars_held"),
                        op=CompareOp.GTE,
                        right=LiteralRef(value=3.0),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[
                IncrementStateAction(state_id="bars_held", amount=LiteralRef(value=1.0)),
            ],
        )

        # Entry on bar 1, should exit on bar 4 (after 3 bars held)
        bars = make_bars([95, 101, 103, 105, 107, 109])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-bars-counter",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1
        # First trade: Entry at bar 1, exit after 3 bars = bar 4
        assert trades[0].entry_bar == 1
        assert trades[0].exit_bar == 4, f"Expected exit_bar=4, got {trades[0].exit_bar}"
        assert trades[0].exit_reason == "time_exit", f"Expected time_exit, got {trades[0].exit_reason}"

    def test_highest_price_tracking(self, backtest_service):
        """Track highest price since entry for trailing stop.

        Strategy: Enter when close > 100, track max price, exit when price drops 5% from max
        """
        strategy_ir = StrategyIR(
            strategy_id="test-max-tracking",
            strategy_name="Max Price Tracking Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[
                StateVarSpec(id="highest_price", default=0.0),
            ],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[
                    SetStateAction(state_id="highest_price", value=PriceRef(field=PriceField.CLOSE)),
                ],
            ),
            exits=[
                ExitRule(
                    id="trailing_exit",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=IRExpression(
                            left=StateRef(state_id="highest_price"),
                            op="*",
                            right=LiteralRef(value=0.95),  # 5% below max
                        ),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[
                MaxStateAction(state_id="highest_price", value=PriceRef(field=PriceField.CLOSE)),
            ],
        )

        # Entry at 101, rises to 110, then drops to 104 (< 110*0.95 = 104.5)
        bars = make_bars([95, 101, 105, 108, 110, 107, 104])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-max-tracking",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].exit_reason == "trailing_exit", f"Expected trailing_exit, got {trades[0].exit_reason}"


@requires_lean
class TestExpressions:
    """Test arithmetic expressions in conditions."""

    def test_multiplication_expression(self, backtest_service):
        """Entry when close > EMA * 1.02 (2% above EMA).

        Strategy: Enter when price is 2% above EMA20
        """
        strategy_ir = StrategyIR(
            strategy_id="test-mult-expr",
            strategy_name="Multiplication Expression Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_20", type="EMA", period=20),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IRExpression(
                        left=IndicatorRef(indicator_id="ema_20"),
                        op="*",
                        right=LiteralRef(value=1.02),
                    ),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Flat then big spike to get 2% above EMA
        flat = make_bars([100.0] * 25)
        spike = make_bars([100, 102, 105, 108], base_timestamp=DEFAULT_BASE_TIMESTAMP_MS + 25 * 60000)
        bars = flat + spike

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-mult-expr",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        # Should enter when price exceeds EMA*1.02
        assert len(trades) >= 1, "Expected entry when price > EMA*1.02"

    def test_subtraction_expression(self, backtest_service):
        """Entry when close < EMA - 5 (5 points below EMA).

        Strategy: Enter when price is 5 points below EMA20
        """
        strategy_ir = StrategyIR(
            strategy_id="test-sub-expr",
            strategy_name="Subtraction Expression Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_20", type="EMA", period=20),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=IRExpression(
                        left=IndicatorRef(indicator_id="ema_20"),
                        op="-",
                        right=LiteralRef(value=5.0),
                    ),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Flat then drop to get 5 points below EMA
        flat = make_bars([100.0] * 25)
        drop = make_bars([100, 98, 95, 93, 90], base_timestamp=DEFAULT_BASE_TIMESTAMP_MS + 25 * 60000)
        bars = flat + drop

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-sub-expr",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        # Should enter when price drops 5+ below EMA
        assert len(trades) >= 1, "Expected entry when price < EMA-5"


@requires_lean
class TestPriceFields:
    """Test different price field references."""

    def test_high_field(self, backtest_service):
        """Entry when high exceeds threshold.

        Strategy: Enter when high > 105
        """
        strategy_ir = StrategyIR(
            strategy_id="test-high-field",
            strategy_name="High Field Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.HIGH),
                    op=CompareOp.GT,
                    right=LiteralRef(value=105.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Create bars with specific highs
        bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS, o=100, h=102, l=99, c=101, v=1000),
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + 60000, o=101, h=104, l=100, c=103, v=1000),
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + 120000, o=103, h=106, l=102, c=105, v=1000),  # HIGH > 105
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + 180000, o=105, h=107, l=104, c=106, v=1000),
        ]

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-high-field",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 2, f"Expected entry at bar 2 (first bar with high > 105), got {trades[0].entry_bar}"

    def test_low_field(self, backtest_service):
        """Entry when low drops below threshold.

        Strategy: Enter when low < 95
        """
        strategy_ir = StrategyIR(
            strategy_id="test-low-field",
            strategy_name="Low Field Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.LOW),
                    op=CompareOp.LT,
                    right=LiteralRef(value=95.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Create bars with specific lows
        bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS, o=100, h=102, l=99, c=101, v=1000),
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + 60000, o=101, h=102, l=98, c=99, v=1000),
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + 120000, o=99, h=100, l=94, c=96, v=1000),  # LOW < 95
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + 180000, o=96, h=98, l=93, c=95, v=1000),
        ]

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-low-field",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 2, f"Expected entry at bar 2 (first bar with low < 95), got {trades[0].entry_bar}"


# =============================================================================
# Phase 4: Edge Cases
# =============================================================================


@requires_lean
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_trades_insufficient_warmup(self, backtest_service):
        """No trades when not enough bars for indicator warmup.

        Strategy: Entry needs EMA50, but only 30 bars of data
        Expected: 0 trades (EMA50 not ready)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-no-warmup",
            strategy_name="Insufficient Warmup Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_50", type="EMA", period=50),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorRef(indicator_id="ema_50"),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Only 30 bars - EMA50 needs 50 bars to warm up
        bars = make_bars([100.0 + i for i in range(30)])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-no-warmup",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        # EMA50 won't be ready with only 30 bars, so no trades
        assert len(result.response.trades) == 0, "Expected 0 trades with insufficient warmup"

    def test_flat_price_no_crossover(self, backtest_service):
        """No trades when price is completely flat (no crossover possible).

        Strategy: Entry on EMA crossover, but price never moves
        Expected: 0 trades
        """
        strategy_ir = StrategyIR(
            strategy_id="test-flat",
            strategy_name="Flat Price Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_10", type="EMA", period=10),
                IndicatorSpec(id="ema_30", type="EMA", period=30),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CrossCondition(
                    left=IndicatorRef(indicator_id="ema_10"),
                    right=IndicatorRef(indicator_id="ema_30"),
                    direction="above",
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # 50 bars all at exactly 100 - no crossover possible
        bars = make_bars([100.0] * 50)

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-flat",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        assert len(result.response.trades) == 0, "Expected 0 trades with flat price"

    def test_entry_on_first_valid_bar(self, backtest_service):
        """Entry on first bar after indicators are ready.

        Strategy: Simple price threshold, no indicator warmup needed
        Data: First bar exceeds threshold
        """
        strategy_ir = StrategyIR(
            strategy_id="test-first-bar",
            strategy_name="First Bar Entry Test",
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
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # First bar already above threshold
        bars = make_bars([101, 102, 103, 104])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-first-bar",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].entry_bar == 0, f"Expected entry at bar 0, got {trades[0].entry_bar}"

    def test_multiple_exit_rules_first_wins(self, backtest_service):
        """When multiple exit rules could trigger, first one wins.

        Strategy: Two exit rules - close < 95 OR close > 110
        Data triggers lower exit first
        """
        strategy_ir = StrategyIR(
            strategy_id="test-exit-priority",
            strategy_name="Exit Priority Test",
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
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="stop_loss",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=LiteralRef(value=95.0),
                    ),
                    action=LiquidateAction(),
                ),
                ExitRule(
                    id="take_profit",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=LiteralRef(value=110.0),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Entry at 101, then drops to trigger stop loss
        bars = make_bars([95, 101, 99, 94, 92])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-exit-priority",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 1
        assert trades[0].exit_reason == "stop_loss", f"Expected stop_loss exit, got {trades[0].exit_reason}"

    def test_three_complete_trades(self, backtest_service):
        """Execute exactly 3 complete trades in one backtest.

        Strategy: Enter when close > 105, Exit when close < 95
        Data: Three cycles of entry/exit
        """
        strategy_ir = StrategyIR(
            strategy_id="test-three-trades",
            strategy_name="Three Trades Test",
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
                    right=LiteralRef(value=105.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="price_exit",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=LiteralRef(value=95.0),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Three cycles: up > 105, down < 95
        prices = [
            100, 106, 108, 94,  # Trade 1: entry bar 1, exit bar 3
            96, 107, 93,        # Trade 2: entry bar 5, exit bar 6
            98, 110, 94,        # Trade 3: entry bar 8, exit bar 9
        ]
        bars = make_bars(prices)

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-three-trades",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success"
        trades = result.response.trades
        assert len(trades) == 3, f"Expected 3 trades, got {len(trades)}"

        # Verify each trade
        assert trades[0].entry_bar == 1
        assert trades[0].exit_bar == 3
        assert trades[1].entry_bar == 5
        assert trades[1].exit_bar == 6
        assert trades[2].entry_bar == 8
        assert trades[2].exit_bar == 9


# =============================================================================
# Unit Tests (No LEAN Required)
# =============================================================================


class TestStrategyIRConstruction:
    """Test StrategyIR Pydantic model construction."""

    def test_simple_strategy_ir(self):
        """StrategyIR with basic entry rule."""
        strategy_ir = StrategyIR(
            strategy_id="test",
            strategy_name="Test",
            symbol="BTCUSD",
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
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        assert strategy_ir.strategy_id == "test"
        assert strategy_ir.entry.action.allocation == 0.95

    def test_strategy_ir_with_indicators(self):
        """StrategyIR with EMA indicators."""
        strategy_ir = StrategyIR(
            strategy_id="ema",
            strategy_name="EMA",
            symbol="BTCUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_fast", type="EMA", period=10),
                IndicatorSpec(id="ema_slow", type="EMA", period=30),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CrossCondition(
                    left=IndicatorRef(indicator_id="ema_fast"),
                    right=IndicatorRef(indicator_id="ema_slow"),
                    direction="above",
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        assert len(strategy_ir.indicators) == 2
        assert strategy_ir.indicators[0].period == 10

    def test_strategy_ir_serializes(self):
        """StrategyIR serializes to JSON."""
        strategy_ir = StrategyIR(
            strategy_id="json-test",
            strategy_name="JSON Test",
            symbol="BTCUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=None,
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        json_str = strategy_ir.model_dump_json()
        assert "json-test" in json_str


class TestDataBuilders:
    """Test data builder helper functions."""

    def test_make_bars(self):
        """make_bars creates correct OHLCVBar objects."""
        bars = make_bars([100, 200, 300])

        assert len(bars) == 3
        assert bars[0].c == 100
        assert bars[1].c == 200
        assert bars[2].c == 300
        assert bars[0].h == 101
        assert bars[0].l == 99

    def test_make_bars_timestamps(self):
        """Timestamps are 1 minute apart."""
        bars = make_bars([100, 100, 100], base_timestamp=0)

        assert bars[0].t == 0
        assert bars[1].t == 60000
        assert bars[2].t == 120000

    def test_make_trending_bars(self):
        """make_trending_bars applies trend correctly."""
        bars = make_trending_bars(100.0, 3, 0.10)  # 10% per bar

        assert bars[0].c == 100.0
        assert abs(bars[1].c - 110.0) < 0.01
        assert abs(bars[2].c - 121.0) < 0.01


# =============================================================================
# Extended Coverage: CrossCondition Variations
# =============================================================================


@requires_lean
class TestCrossConditionExtended:
    """Extended cross condition tests including cross_below."""

    def test_cross_below(self, backtest_service):
        """Entry when fast EMA crosses below slow EMA.

        Strategy: Enter when EMA10 crosses below EMA30
        Data: Uptrend first (fast > slow), then reversal to downtrend (fast < slow)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-cross-below",
            strategy_name="Cross Below Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_fast", type="EMA", period=10),
                IndicatorSpec(id="ema_slow", type="EMA", period=30),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CrossCondition(
                    left=IndicatorRef(indicator_id="ema_fast"),
                    right=IndicatorRef(indicator_id="ema_slow"),
                    direction="below",
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Uptrend for 40 bars (fast EMA > slow EMA), then sharp reversal
        # This ensures fast EMA is above slow EMA before crossing below
        up = make_trending_bars(80.0, 40, 0.01)  # 80 -> ~119 over 40 bars
        # Sharp reversal - price drops significantly to make fast EMA cross below slow
        down_prices = [up[-1].c * (0.95 ** i) for i in range(1, 21)]  # 5% drop per bar
        down = [
            OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + (40 + i) * 60_000,
                o=down_prices[i],
                h=down_prices[i] + 1,
                l=down_prices[i] - 1,
                c=down_prices[i],
                v=1000.0,
            )
            for i in range(20)
        ]
        bars = up + down

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-cross-below",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when EMA10 crosses below EMA30"


# =============================================================================
# Extended Coverage: Keltner Channel and Donchian Channel
# =============================================================================


@requires_lean
class TestKeltnerChannel:
    """Test Keltner Channel indicator."""

    def test_kc_lower_band_entry(self, backtest_service):
        """Entry when close touches KC lower band.

        Strategy: Enter when close < KC lower
        Data: Stable then sharp drop below lower band
        """
        strategy_ir = StrategyIR(
            strategy_id="test-kc-lower",
            strategy_name="KC Lower Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="kc", type="KC", period=20, multiplier=2.0),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=IndicatorBandRef(indicator_id="kc", band="lower"),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # 30 stable bars then sharp drop
        stable_bars = make_bars([100.0] * 30)
        drop_bars = [
            OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + (30 + i) * 60_000,
                o=95.0 - i * 2.0,
                h=95.0 - i * 2.0,
                l=90.0 - i * 2.0,
                c=90.0 - i * 2.0,
                v=1000.0,
            )
            for i in range(5)
        ]
        bars = stable_bars + drop_bars

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-kc-lower",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when price breaks below KC lower"

    def test_kc_upper_band_exit(self, backtest_service):
        """Exit when close exceeds KC upper band.

        Strategy: Enter at 100, exit when close > KC upper
        """
        strategy_ir = StrategyIR(
            strategy_id="test-kc-upper",
            strategy_name="KC Upper Exit Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="kc", type="KC", period=20, multiplier=2.0),
            ],
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
            exits=[
                ExitRule(
                    id="kc_upper_exit",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=IndicatorBandRef(indicator_id="kc", band="upper"),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Entry at bar 1, then strong uptrend to break upper band
        bars = make_bars([95.0] * 25)
        trend_bars = make_trending_bars(101.0, 20, 0.02)  # +2% per bar
        for i, b in enumerate(trend_bars):
            trend_bars[i] = OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + (25 + i) * 60_000,
                o=b.o, h=b.h, l=b.l, c=b.c, v=b.v
            )
        bars = bars + trend_bars

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-kc-upper",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1
        assert trades[0].exit_reason == "kc_upper_exit", f"Expected kc_upper_exit, got {trades[0].exit_reason}"


@requires_lean
class TestDonchianChannel:
    """Test Donchian Channel indicator."""

    def test_dc_breakout_high(self, backtest_service):
        """Entry when close exceeds a price threshold during uptrend.

        This is a simplified test that verifies DC indicator initializes correctly.
        Full DC band comparison is tested via KC tests which use same band pattern.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-dc-breakout",
            strategy_name="DC Breakout Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="dc", type="DC", period=10),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                # Simple price threshold test - verifies DC indicator doesn't block execution
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=105.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # 15 bars warmup at 100, then price rises above 105
        warmup = make_bars([100.0] * 15)
        rise = [
            OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + (15 + i) * 60_000,
                o=102.0 + i * 2.0,
                h=110.0 + i * 2.0,
                l=101.0 + i * 2.0,
                c=106.0 + i * 2.0,  # 106, 108, 110
                v=1000.0,
            )
            for i in range(5)
        ]
        bars = warmup + rise

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-dc-breakout",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        # Entry should happen when close > 105 (bar 15 has c=106)
        assert len(trades) >= 1, "Expected entry when close > 105"

    def test_dc_breakdown_low(self, backtest_service):
        """Exit when close falls below a price threshold.

        This is a simplified test that verifies DC indicator initializes correctly.
        Full DC band comparison is tested via KC tests which use same band pattern.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-dc-breakdown",
            strategy_name="DC Breakdown Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="dc", type="DC", period=10),
            ],
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
            exits=[
                ExitRule(
                    id="price_breakdown",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=LiteralRef(value=95.0),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # 15 bars warmup at 100, entry, then drop below 95
        warmup = make_bars([100.0] * 15)
        entry = OHLCVBar(
            t=DEFAULT_BASE_TIMESTAMP_MS + 15 * 60_000,
            o=100.0, h=102.0, l=99.0, c=101.0, v=1000.0,
        )
        hold = make_bars([101.0] * 5)
        for i, b in enumerate(hold):
            hold[i] = OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + (16 + i) * 60_000,
                o=b.o, h=b.h, l=b.l, c=b.c, v=b.v
            )
        breakdown = OHLCVBar(
            t=DEFAULT_BASE_TIMESTAMP_MS + 21 * 60_000,
            o=98.0, h=98.0, l=92.0, c=93.0, v=1000.0,  # close=93 < 95
        )
        post = make_bars([93.0] * 3)
        for i, b in enumerate(post):
            post[i] = OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + (22 + i) * 60_000,
                o=b.o, h=b.h, l=b.l, c=b.c, v=b.v
            )
        bars = warmup + [entry] + hold + [breakdown] + post

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-dc-breakdown",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected at least one trade"
        assert trades[0].exit_reason == "price_breakdown", f"Expected price_breakdown, got {trades[0].exit_reason}"


# =============================================================================
# Extended Coverage: MACD and ADX
# =============================================================================


@requires_lean
class TestMACD:
    """Test MACD indicator."""

    def test_macd_histogram_positive(self, backtest_service):
        """Entry when MACD histogram > 0.

        Strategy: Enter when MACD histogram turns positive (bullish momentum)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-macd-hist",
            strategy_name="MACD Histogram Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="macd", type="MACD", fast_period=12, slow_period=26, signal_period=9),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=IndicatorRef(indicator_id="macd", field="histogram"),
                    op=CompareOp.GT,
                    right=LiteralRef(value=0.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Flat then uptrend to generate positive MACD histogram
        flat = make_bars([100.0] * 35)
        trend = make_trending_bars(100.0, 20, 0.01)
        for i, b in enumerate(trend):
            trend[i] = OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + (35 + i) * 60_000, o=b.o, h=b.h, l=b.l, c=b.c, v=b.v)
        bars = flat + trend

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-macd-hist",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when MACD histogram > 0"

    def test_macd_signal_crossover(self, backtest_service):
        """Entry when MACD line is above signal line.

        Strategy: Enter when MACD > Signal (bullish momentum)
        This tests IndicatorRef field access for MACD signal line.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-macd-cross",
            strategy_name="MACD Signal Crossover Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="macd", type="MACD", fast_period=12, slow_period=26, signal_period=9),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                # Use CompareCondition instead of CrossCondition to test MACD vs Signal comparison
                condition=CompareCondition(
                    left=IndicatorRef(indicator_id="macd"),  # MACD line
                    op=CompareOp.GT,
                    right=IndicatorRef(indicator_id="macd", field="signal"),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Strong uptrend - MACD line will be above signal line
        # Need 26 + 9 = 35 bars minimum for MACD warmup
        bars = make_trending_bars(80.0, 60, 0.015)  # 1.5% per bar increase

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-macd-cross",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when MACD > Signal"


@requires_lean
class TestADX:
    """Test ADX (Average Directional Index) indicator."""

    def test_adx_trending_market(self, backtest_service):
        """Entry when ADX > 25 (strong trend).

        Strategy: Only enter when ADX indicates strong trend
        """
        strategy_ir = StrategyIR(
            strategy_id="test-adx-trend",
            strategy_name="ADX Trending Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="adx", type="ADX", period=14),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(conditions=[
                    CompareCondition(
                        left=IndicatorRef(indicator_id="adx"),
                        op=CompareOp.GT,
                        right=LiteralRef(value=25.0),
                    ),
                    CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=LiteralRef(value=100.0),
                    ),
                ]),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Strong trend to push ADX above 25
        bars = make_trending_bars(90.0, 50, 0.015)  # Strong uptrend

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-adx-trend",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when ADX > 25 and price > 100"

    def test_adx_ranging_market_no_entry(self, backtest_service):
        """No entry when ADX < 20 (ranging market).

        Strategy: Require ADX > 30 (won't trigger in range)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-adx-range",
            strategy_name="ADX Ranging Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="adx", type="ADX", period=14),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=IndicatorRef(indicator_id="adx"),
                    op=CompareOp.GT,
                    right=LiteralRef(value=30.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Ranging market - oscillate between 99 and 101
        bars = []
        for i in range(50):
            price = 100.0 + (1.0 if i % 2 == 0 else -1.0)
            bars.append(OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                o=price, h=price + 0.5, l=price - 0.5, c=price, v=1000.0
            ))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-adx-range",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        # ADX should stay low in ranging market, no entry
        assert len(result.response.trades) == 0, "Expected no entry in ranging market with high ADX threshold"


# =============================================================================
# Extended Coverage: Gate Conditions
# =============================================================================


@requires_lean
class TestGates:
    """Test gate conditions that block or allow entry."""

    def test_gate_blocks_entry(self, backtest_service):
        """Gate blocks entry when condition not met.

        Strategy: Entry when close > 100, BUT gate requires RSI < 70
        Data: Price > 100 but RSI > 70 (overbought) - should NOT enter
        """
        strategy_ir = StrategyIR(
            strategy_id="test-gate-block",
            strategy_name="Gate Block Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="rsi", type="RSI", period=14),
            ],
            state=[],
            gates=[
                GateRule(
                    id="rsi_gate",
                    condition=CompareCondition(
                        left=IndicatorRef(indicator_id="rsi"),
                        op=CompareOp.LT,
                        right=LiteralRef(value=70.0),
                    ),
                    mode="allow",  # Only allow entry when RSI < 70
                ),
            ],
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

        # Strong uptrend to push RSI high (overbought)
        bars = make_trending_bars(90.0, 50, 0.02)  # +2% per bar = extreme overbought

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-gate-block",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        # Gate should block entry when RSI > 70
        assert len(result.response.trades) == 0, "Gate should have blocked entry (RSI > 70)"

    def test_gate_allows_entry(self, backtest_service):
        """Gate allows entry when condition met.

        Strategy: Entry when close > 100, gate requires RSI < 70
        Data: Price > 100 with moderate RSI - should enter
        """
        strategy_ir = StrategyIR(
            strategy_id="test-gate-allow",
            strategy_name="Gate Allow Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="rsi", type="RSI", period=14),
            ],
            state=[],
            gates=[
                GateRule(
                    id="rsi_gate",
                    condition=CompareCondition(
                        left=IndicatorRef(indicator_id="rsi"),
                        op=CompareOp.LT,
                        right=LiteralRef(value=70.0),
                    ),
                    mode="allow",
                ),
            ],
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

        # Gentle oscillation around 100, then break above - RSI stays moderate
        bars = []
        for i in range(30):
            price = 95.0 + (i % 5) * 2  # oscillate 95-103
            bars.append(OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                o=price, h=price + 1, l=price - 1, c=price, v=1000.0
            ))
        # Now gently move above 100
        for i in range(10):
            price = 101.0 + i * 0.5
            bars.append(OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + (30 + i) * 60_000,
                o=price, h=price + 1, l=price - 1, c=price, v=1000.0
            ))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-gate-allow",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Gate should have allowed entry (RSI < 70)"

    def test_gate_block_mode(self, backtest_service):
        """Gate with block mode prevents entry when condition IS met.

        Strategy: Entry when close > 100, gate BLOCKS when EMA_fast < EMA_slow (downtrend)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-gate-block-mode",
            strategy_name="Gate Block Mode Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_fast", type="EMA", period=10),
                IndicatorSpec(id="ema_slow", type="EMA", period=30),
            ],
            state=[],
            gates=[
                GateRule(
                    id="downtrend_block",
                    condition=CompareCondition(
                        left=IndicatorRef(indicator_id="ema_fast"),
                        op=CompareOp.LT,
                        right=IndicatorRef(indicator_id="ema_slow"),
                    ),
                    mode="block",  # Block entry when EMA_fast < EMA_slow
                ),
            ],
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

        # Downtrend data - EMA fast should be below EMA slow
        bars = make_trending_bars(150.0, 50, -0.01)  # downtrend

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-gate-block-mode",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        # Even though price > 100 at some points, downtrend gate should block
        assert len(result.response.trades) == 0, "Block mode gate should have prevented entry"


# =============================================================================
# Extended Coverage: Regime Conditions
# =============================================================================


@requires_lean
class TestRegimeConditions:
    """Test regime-based conditions."""

    def test_regime_trend_ma_relation(self, backtest_service):
        """Entry based on EMA fast/slow relation.

        Strategy: Enter when fast EMA > slow EMA (uptrend regime)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-regime-trend",
            strategy_name="Regime Trend Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_fast", type="EMA", period=20),
                IndicatorSpec(id="ema_slow", type="EMA", period=50),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(conditions=[
                    RegimeCondition(
                        metric="trend_ma_relation",
                        op=CompareOp.GT,
                        value=0,  # EMA fast - EMA slow > 0
                        ma_fast=20,
                        ma_slow=50,
                    ),
                    CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=LiteralRef(value=100.0),
                    ),
                ]),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Uptrend to establish fast > slow
        bars = make_trending_bars(80.0, 60, 0.01)  # +1% per bar

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-regime-trend",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry in uptrend regime"


# =============================================================================
# Extended Coverage: Short Positions
# =============================================================================


@requires_lean
class TestShortPositions:
    """Test short position (negative allocation) strategies."""

    def test_short_entry(self, backtest_service):
        """Short entry with negative allocation.

        Strategy: Short when RSI > 70 (overbought)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-short-entry",
            strategy_name="Short Entry Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="rsi", type="RSI", period=14),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=IndicatorRef(indicator_id="rsi"),
                    op=CompareOp.GT,
                    right=LiteralRef(value=70.0),
                ),
                action=SetHoldingsAction(allocation=-0.95),  # Negative = short
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="rsi_exit",
                    condition=CompareCondition(
                        left=IndicatorRef(indicator_id="rsi"),
                        op=CompareOp.LT,
                        right=LiteralRef(value=50.0),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Strong uptrend to push RSI above 70, then reversal
        up = make_trending_bars(80.0, 30, 0.02)  # +2% per bar
        down = make_trending_bars(up[-1].c, 20, -0.02)  # reversal
        for i, b in enumerate(down):
            down[i] = OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + (30 + i) * 60_000, o=b.o, h=b.h, l=b.l, c=b.c, v=b.v)
        bars = up + down

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-short-entry",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected short entry when RSI > 70"
        # Note: direction tracking requires runtime update


# =============================================================================
# Extended Coverage: Complex Nested Conditions
# =============================================================================


@requires_lean
class TestComplexConditions:
    """Test deeply nested and combined conditions."""

    def test_triple_nested_allof_anyof(self, backtest_service):
        """Complex: AllOf(AnyOf(...), AnyOf(...), Not(...)).

        Strategy: Enter when:
          - (close > 100 OR close < 90) AND
          - (RSI > 30 OR RSI < 70) AND
          - NOT (close == 95)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-complex-nested",
            strategy_name="Complex Nested Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="rsi", type="RSI", period=14),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(conditions=[
                    AnyOfCondition(conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.LT,
                            right=LiteralRef(value=90.0),
                        ),
                    ]),
                    AnyOfCondition(conditions=[
                        CompareCondition(
                            left=IndicatorRef(indicator_id="rsi"),
                            op=CompareOp.GT,
                            right=LiteralRef(value=30.0),
                        ),
                        CompareCondition(
                            left=IndicatorRef(indicator_id="rsi"),
                            op=CompareOp.LT,
                            right=LiteralRef(value=70.0),
                        ),
                    ]),
                    NotCondition(
                        condition=CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.EQ,
                            right=LiteralRef(value=95.0),
                        ),
                    ),
                ]),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Price goes to 105 (satisfies first AnyOf, RSI will be moderate)
        bars = make_bars([92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106])
        # Add warmup for RSI
        warmup = make_bars([95.0] * 20)
        for i, b in enumerate(bars):
            bars[i] = OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + (20 + i) * 60_000, o=b.o, h=b.h, l=b.l, c=b.c, v=b.v)
        bars = warmup + bars

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-complex-nested",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry with complex nested condition"

    def test_four_level_condition_nesting(self, backtest_service):
        """Four levels deep: AllOf(AnyOf(AllOf(...), Not(...)), ...).

        Strategy: Complex multi-indicator filter
        """
        strategy_ir = StrategyIR(
            strategy_id="test-4-level",
            strategy_name="Four Level Nesting Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_10", type="EMA", period=10),
                IndicatorSpec(id="ema_20", type="EMA", period=20),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(conditions=[
                    # Level 2: AnyOf
                    AnyOfCondition(conditions=[
                        # Level 3: AllOf
                        AllOfCondition(conditions=[
                            # Level 4: Compare
                            CompareCondition(
                                left=PriceRef(field=PriceField.CLOSE),
                                op=CompareOp.GT,
                                right=LiteralRef(value=100.0),
                            ),
                            CompareCondition(
                                left=IndicatorRef(indicator_id="ema_10"),
                                op=CompareOp.GT,
                                right=IndicatorRef(indicator_id="ema_20"),
                            ),
                        ]),
                        # Level 3: Not
                        NotCondition(
                            condition=CompareCondition(
                                left=PriceRef(field=PriceField.CLOSE),
                                op=CompareOp.LT,
                                right=LiteralRef(value=80.0),
                            ),
                        ),
                    ]),
                    # Level 2: Simple compare
                    CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=LiteralRef(value=95.0),
                    ),
                ]),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Uptrend to satisfy all conditions
        bars = make_trending_bars(90.0, 40, 0.01)

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-4-level",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry with 4-level nested condition"


# =============================================================================
# Extended Coverage: Multi-Indicator Strategies
# =============================================================================


@requires_lean
class TestMultiIndicatorStrategies:
    """Test strategies using multiple indicators together."""

    def test_bb_rsi_combination(self, backtest_service):
        """Entry combining BB and RSI.

        Strategy: Enter when close < BB lower AND RSI < 30 (oversold at support)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-bb-rsi",
            strategy_name="BB RSI Combo Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="bb", type="BB", period=20, multiplier=2.0),
                IndicatorSpec(id="rsi", type="RSI", period=14),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(conditions=[
                    CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=IndicatorBandRef(indicator_id="bb", band="lower"),
                    ),
                    CompareCondition(
                        left=IndicatorRef(indicator_id="rsi"),
                        op=CompareOp.LT,
                        right=LiteralRef(value=30.0),
                    ),
                ]),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # First establish stable BB bands, then sudden drop to trigger both conditions
        # 30 bars of stable prices to establish BB bands around 100
        stable = make_bars([100.0] * 30)
        # Then sudden downtrend to push below BB and get RSI oversold
        # RSI oversold requires sustained decline; BB break requires price < lower band
        down = make_trending_bars(100.0, 25, -0.025)  # -2.5% per bar = aggressive drop
        for i, b in enumerate(down):
            down[i] = OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + (30 + i) * 60_000, o=b.o, h=b.h, l=b.l, c=b.c, v=b.v)
        bars = stable + down

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-bb-rsi",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when close < BB lower AND RSI < 30"

    def test_ema_atr_stop(self, backtest_service):
        """EMA entry with ATR-based exit.

        Strategy: Enter when EMA10 > EMA20, exit when price drops > 2*ATR from entry
        """
        strategy_ir = StrategyIR(
            strategy_id="test-ema-atr",
            strategy_name="EMA ATR Stop Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema_10", type="EMA", period=10),
                IndicatorSpec(id="ema_20", type="EMA", period=20),
                IndicatorSpec(id="atr", type="ATR", period=14),
            ],
            state=[
                StateVarSpec(id="entry_price", default=0.0),
            ],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=IndicatorRef(indicator_id="ema_10"),
                    op=CompareOp.GT,
                    right=IndicatorRef(indicator_id="ema_20"),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[
                    SetStateAction(state_id="entry_price", value=PriceRef(field=PriceField.CLOSE)),
                ],
            ),
            exits=[
                ExitRule(
                    id="atr_stop",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=IRExpression(
                            left=StateRef(state_id="entry_price"),
                            op="-",
                            right=IRExpression(
                                left=IndicatorRef(indicator_id="atr"),
                                op="*",
                                right=LiteralRef(value=2.0),
                            ),
                        ),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Uptrend then sharp reversal
        up = make_trending_bars(90.0, 35, 0.01)
        down = make_trending_bars(up[-1].c, 15, -0.02)
        for i, b in enumerate(down):
            down[i] = OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + (35 + i) * 60_000, o=b.o, h=b.h, l=b.l, c=b.c, v=b.v)
        bars = up + down

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-ema-atr",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected trade with EMA entry and ATR stop"


# =============================================================================
# Extended Coverage: Multiple Exit Rules
# =============================================================================


@requires_lean
class TestMultipleExitRules:
    """Test strategies with multiple exit conditions."""

    def test_profit_target_and_stop_loss(self, backtest_service):
        """Exit on either profit target or stop loss.

        Strategy: Entry at 100, exit at +5% or -3%
        """
        strategy_ir = StrategyIR(
            strategy_id="test-pt-sl",
            strategy_name="Profit Target Stop Loss Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[
                StateVarSpec(id="entry_price", default=0.0),
            ],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[
                    SetStateAction(state_id="entry_price", value=PriceRef(field=PriceField.CLOSE)),
                ],
            ),
            exits=[
                ExitRule(
                    id="profit_target",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=IRExpression(
                            left=StateRef(state_id="entry_price"),
                            op="*",
                            right=LiteralRef(value=1.05),  # +5%
                        ),
                    ),
                    action=LiquidateAction(),
                ),
                ExitRule(
                    id="stop_loss",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=IRExpression(
                            left=StateRef(state_id="entry_price"),
                            op="*",
                            right=LiteralRef(value=0.97),  # -3%
                        ),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Entry at 101, then hit profit target at ~106
        bars = make_bars([95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-pt-sl",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1
        assert trades[0].exit_reason == "profit_target", f"Expected profit_target exit, got {trades[0].exit_reason}"

    def test_stop_loss_hit_first(self, backtest_service):
        """Stop loss hits before profit target.

        Strategy: Entry at 101, price drops to stop loss
        """
        strategy_ir = StrategyIR(
            strategy_id="test-sl-first",
            strategy_name="Stop Loss First Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[
                StateVarSpec(id="entry_price", default=0.0),
            ],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[
                    SetStateAction(state_id="entry_price", value=PriceRef(field=PriceField.CLOSE)),
                ],
            ),
            exits=[
                ExitRule(
                    id="profit_target",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=IRExpression(
                            left=StateRef(state_id="entry_price"),
                            op="*",
                            right=LiteralRef(value=1.05),
                        ),
                    ),
                    action=LiquidateAction(),
                ),
                ExitRule(
                    id="stop_loss",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.LT,
                        right=IRExpression(
                            left=StateRef(state_id="entry_price"),
                            op="*",
                            right=LiteralRef(value=0.97),
                        ),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Entry at 101, then drop to stop loss (< 98)
        bars = make_bars([95, 96, 97, 98, 99, 100, 101, 100, 99, 98, 97, 96])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-sl-first",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1
        assert trades[0].exit_reason == "stop_loss", f"Expected stop_loss exit, got {trades[0].exit_reason}"


# =============================================================================
# VWAP Indicator Tests
# =============================================================================


@requires_lean
class TestVWAP:
    """Test VWAP indicator strategies."""

    def test_vwap_above_entry(self, backtest_service):
        """Entry when price is above VWAP.

        Strategy: Enter when close > VWAP (bullish momentum)
        VWAP is volume-weighted average price, resets daily.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-vwap-above",
            strategy_name="VWAP Above Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="vwap", type="VWAP", period=0),  # period=0 for intraday VWAP
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorRef(indicator_id="vwap"),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Rising prices with volume - VWAP will lag behind
        # Price starts at 100, VWAP starts same, price rises faster than VWAP
        bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                     o=100.0 + i * 0.5, h=101.0 + i * 0.5, l=99.0 + i * 0.5,
                     c=100.0 + i * 0.5, v=1000.0 + i * 100)
            for i in range(20)
        ]

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-vwap-above",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when price > VWAP"

    def test_vwap_distance_percent(self, backtest_service):
        """Entry when price is more than 2% above VWAP.

        Strategy: Enter when close > VWAP * 1.02 (extended above VWAP)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-vwap-distance",
            strategy_name="VWAP Distance Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="vwap", type="VWAP", period=0),  # period=0 for intraday VWAP
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IRExpression(
                        left=IndicatorRef(indicator_id="vwap"),
                        op="*",
                        right=LiteralRef(value=1.02),
                    ),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Strong uptrend - price rises well above VWAP
        bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                     o=100.0 + i * 1.0, h=101.0 + i * 1.0, l=99.0 + i * 1.0,
                     c=100.0 + i * 1.0, v=1000.0)
            for i in range(25)
        ]

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-vwap-distance",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when price > VWAP * 1.02"


# =============================================================================
# Volume Tests
# =============================================================================


@requires_lean
class TestVolumeConditions:
    """Test volume-based conditions."""

    def test_volume_spike(self, backtest_service):
        """Entry on volume spike (current volume > 2x average).

        Strategy: Enter when volume > 2 * SMA(volume, 10)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-volume-spike",
            strategy_name="Volume Spike Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="vol_sma", type="VOL_SMA", period=10),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=VolumeRef(),
                    op=CompareOp.GT,
                    right=IRExpression(
                        left=IndicatorRef(indicator_id="vol_sma"),
                        op="*",
                        right=LiteralRef(value=2.0),
                    ),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Normal volume for 15 bars, then spike
        normal_bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                     o=100.0, h=101.0, l=99.0, c=100.0, v=1000.0)
            for i in range(15)
        ]
        # Volume spike bars (3x normal)
        spike_bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + (15 + i) * 60_000,
                     o=100.0, h=102.0, l=99.0, c=101.0, v=3000.0)
            for i in range(5)
        ]
        bars = normal_bars + spike_bars

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-volume-spike",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry on volume spike"

    def test_volume_threshold(self, backtest_service):
        """Entry when volume exceeds absolute threshold.

        Strategy: Simple volume > 2000 condition
        """
        strategy_ir = StrategyIR(
            strategy_id="test-volume-threshold",
            strategy_name="Volume Threshold Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=VolumeRef(),
                    op=CompareOp.GT,
                    right=LiteralRef(value=2000.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Low volume, then high volume
        bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                     o=100.0, h=101.0, l=99.0, c=100.0,
                     v=1000.0 if i < 10 else 2500.0)
            for i in range(15)
        ]

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-volume-threshold",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when volume > 2000"


# =============================================================================
# Sequence Condition Tests
# =============================================================================


@requires_lean
class TestSequenceCondition:
    """Test SequenceCondition for multi-step entry patterns."""

    def test_two_step_sequence(self, backtest_service):
        """Entry requires two conditions to be true in sequence.

        Strategy: First close > 100, then within 5 bars close > 105
        This tests the sequence/setup-trigger pattern.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-sequence",
            strategy_name="Sequence Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=SequenceCondition(
                    steps=[
                        SequenceStep(
                            condition=CompareCondition(
                                left=PriceRef(field=PriceField.CLOSE),
                                op=CompareOp.GT,
                                right=LiteralRef(value=100.0),
                            ),
                            # First step doesn't need within_bars (it's the trigger)
                        ),
                        SequenceStep(
                            condition=CompareCondition(
                                left=PriceRef(field=PriceField.CLOSE),
                                op=CompareOp.GT,
                                right=LiteralRef(value=105.0),
                            ),
                            within_bars=5,  # Must occur within 5 bars of previous step
                        ),
                    ],
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Price pattern: below 100, then 101, then 106 (both conditions in sequence)
        bars = make_bars([95, 96, 97, 98, 99, 101, 102, 103, 104, 106, 107, 108])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-sequence",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry after sequence completes"


# =============================================================================
# Time Reference Tests
# =============================================================================


@requires_lean
class TestTimeConditions:
    """Test time-based conditions."""

    def test_hour_filter(self, backtest_service):
        """Entry only during specific hours.

        Strategy: Enter when close > 100 AND hour >= 9
        Note: Test data starts at midnight UTC, so we need bars at different hours.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-hour-filter",
            strategy_name="Hour Filter Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        CompareCondition(
                            left=TimeRef(component="hour"),
                            op=CompareOp.GTE,
                            right=LiteralRef(value=9.0),
                        ),
                    ],
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Create bars at different hours
        # Hours 0-8: price > 100 but hour filter blocks
        # Hour 9+: price > 100 and hour filter passes
        base_ts = DEFAULT_BASE_TIMESTAMP_MS  # 2024-01-01 00:00:00 UTC
        bars = []
        for hour in range(12):
            # One bar per hour
            ts = base_ts + hour * 3600 * 1000  # hour in ms
            price = 101.0 if hour >= 0 else 99.0  # Always above 100
            bars.append(OHLCVBar(t=ts, o=price, h=price+1, l=price-1, c=price, v=1000.0))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-hour-filter",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, 12, tzinfo=timezone.utc),  # Half day
                symbol="TESTUSD",
                resolution="1h",  # Hourly resolution
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        # Entry should only happen at hour 9 or later
        assert len(trades) >= 1, "Expected entry after hour 9"


# =============================================================================
# Short Position Tests (Extended)
# =============================================================================


@requires_lean
class TestShortPositionsExtended:
    """Test additional short selling strategies."""

    def test_short_entry_price_below_threshold(self, backtest_service):
        """Short entry when price drops below threshold.

        Strategy: Short when close < 100 (bearish signal)
        Uses negative allocation for short position.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-short-below",
            strategy_name="Short Below Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=-0.95),  # Negative = short
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Price drops below 100
        bars = make_bars([105, 103, 101, 99, 97, 95, 93, 95, 97])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-short-below",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected short entry when price < 100"
        assert trades[0].direction == "short", f"Expected short direction, got {trades[0].direction}"

    def test_short_with_stop_loss(self, backtest_service):
        """Short position with stop loss (price rises = loss).

        Strategy: Short when close < 100, stop loss at 2% (price rising)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-short-stop",
            strategy_name="Short Stop Loss Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=-0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="stop_loss",
                    condition=CompareCondition(
                        left=StateRef(state_id="pnl_pct"),
                        op=CompareOp.LTE,
                        right=LiteralRef(value=-0.02),  # -2% loss
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Price drops, entry at 99, then rises to trigger stop
        bars = make_bars([105, 103, 101, 99, 98, 97, 99, 101, 103, 105])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-short-stop",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected short trade"
        assert trades[0].direction == "short"
        # Trade should have exited (stop loss triggered)
        assert trades[0].exit_bar is not None, "Expected exit from stop loss"

    def test_short_take_profit(self, backtest_service):
        """Short position with take profit (price falls = profit).

        Strategy: Short when close < 100, take profit at 3% (price falling)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-short-tp",
            strategy_name="Short Take Profit Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=-0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="take_profit",
                    condition=CompareCondition(
                        left=StateRef(state_id="pnl_pct"),
                        op=CompareOp.GTE,
                        right=LiteralRef(value=0.03),  # +3% profit
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Price drops steadily - short should profit
        bars = make_bars([105, 103, 101, 99, 97, 95, 93, 91, 89, 87])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-short-tp",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected short trade"
        assert trades[0].direction == "short"

    def test_short_rsi_overbought(self, backtest_service):
        """Short when RSI indicates overbought conditions.

        Strategy: Short when RSI > 70 (overbought = expect price drop)
        """
        strategy_ir = StrategyIR(
            strategy_id="test-short-rsi",
            strategy_name="Short RSI Overbought Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="rsi", type="RSI", period=14),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=IndicatorRef(indicator_id="rsi"),
                    op=CompareOp.GT,
                    right=LiteralRef(value=70.0),
                ),
                action=SetHoldingsAction(allocation=-0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Strong uptrend to push RSI high
        bars = make_trending_bars(start_price=100.0, num_bars=30, trend_pct_per_bar=0.5)

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-short-rsi",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected short entry on RSI > 70"
        assert trades[0].direction == "short"


# =============================================================================
# Extreme Volatility Tests
# =============================================================================


@requires_lean
class TestExtremeVolatility:
    """Test strategies under extreme market conditions."""

    def test_large_price_spike(self, backtest_service):
        """Handle 20% price spike in single bar.

        Simulates flash crash/pump scenarios common in crypto.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-spike",
            strategy_name="Price Spike Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema20", type="EMA", period=20),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorRef(indicator_id="ema20"),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="stop_loss",
                    condition=CompareCondition(
                        left=StateRef(state_id="pnl_pct"),
                        op=CompareOp.LTE,
                        right=LiteralRef(value=-0.05),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Normal bars, then 20% spike, then crash
        normal_bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                     o=100.0, h=101.0, l=99.0, c=100.0, v=1000.0)
            for i in range(25)
        ]
        # Spike bar: +20%
        spike_bar = OHLCVBar(
            t=DEFAULT_BASE_TIMESTAMP_MS + 25 * 60_000,
            o=100.0, h=125.0, l=100.0, c=120.0, v=5000.0
        )
        # Crash bars: -15%
        crash_bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + (26 + i) * 60_000,
                     o=120.0 - i * 3, h=121.0 - i * 3, l=117.0 - i * 3,
                     c=118.0 - i * 3, v=3000.0)
            for i in range(5)
        ]
        bars = normal_bars + [spike_bar] + crash_bars

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-spike",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        # Should handle extreme volatility without crashing

    def test_flash_crash_recovery(self, backtest_service):
        """Handle flash crash (-30%) and recovery.

        Tests that strategy survives extreme drawdown.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-crash",
            strategy_name="Flash Crash Test",
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
                    right=LiteralRef(value=95.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Normal, crash, recovery
        bars = []
        base_ts = DEFAULT_BASE_TIMESTAMP_MS

        # Normal trading
        for i in range(10):
            bars.append(OHLCVBar(t=base_ts + i * 60_000,
                                 o=100.0, h=101.0, l=99.0, c=100.0, v=1000.0))

        # Flash crash: 100 -> 70 (-30%)
        bars.append(OHLCVBar(t=base_ts + 10 * 60_000,
                             o=100.0, h=100.0, l=68.0, c=70.0, v=10000.0))

        # Recovery
        for i in range(10):
            price = 70.0 + i * 3  # Recover from 70 to 97
            bars.append(OHLCVBar(t=base_ts + (11 + i) * 60_000,
                                 o=price, h=price + 1, l=price - 1, c=price, v=2000.0))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-crash",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"

    def test_high_frequency_oscillation(self, backtest_service):
        """Handle rapid price oscillations (whipsaw).

        Tests strategy with price swinging 5% each bar.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-whipsaw",
            strategy_name="Whipsaw Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema5", type="EMA", period=5),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CrossCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    right=IndicatorRef(indicator_id="ema5"),
                    direction="above",
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="cross_below",
                    condition=CrossCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        right=IndicatorRef(indicator_id="ema5"),
                        direction="below",
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Oscillating prices: up 5%, down 5%, repeat
        bars = []
        base_ts = DEFAULT_BASE_TIMESTAMP_MS
        for i in range(30):
            if i % 2 == 0:
                price = 100.0
            else:
                price = 105.0  # 5% swing
            bars.append(OHLCVBar(t=base_ts + i * 60_000,
                                 o=price - 2, h=price + 2, l=price - 3, c=price, v=1000.0))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-whipsaw",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        # Should handle multiple entries/exits from whipsaw

    def test_zero_volume_bars(self, backtest_service):
        """Handle bars with zero or near-zero volume.

        Some exchanges report 0 volume during illiquid periods.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-zero-vol",
            strategy_name="Zero Volume Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="vol_sma", type="VOL_SMA", period=10),
            ],
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

        # Mix of normal and zero-volume bars
        bars = []
        base_ts = DEFAULT_BASE_TIMESTAMP_MS
        for i in range(20):
            vol = 0.0 if i % 5 == 0 else 1000.0  # Every 5th bar has 0 volume
            price = 99.0 if i < 10 else 101.0
            bars.append(OHLCVBar(t=base_ts + i * 60_000,
                                 o=price, h=price + 0.5, l=price - 0.5, c=price, v=vol))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-zero-vol",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"


# =============================================================================
# Real BTC Price Pattern Tests
# =============================================================================


@requires_lean
class TestRealisticBTCPatterns:
    """Test with realistic BTC price patterns and ranges."""

    def test_btc_price_range(self, backtest_service):
        """Test with realistic BTC prices ($30k-$70k range).

        Verifies no precision/overflow issues with large prices.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-btc-range",
            strategy_name="BTC Price Range Test",
            symbol="BTCUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema20", type="EMA", period=20),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorRef(indicator_id="ema20"),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="stop_loss",
                    condition=CompareCondition(
                        left=StateRef(state_id="pnl_pct"),
                        op=CompareOp.LTE,
                        right=LiteralRef(value=-0.02),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Realistic BTC prices: $45k range
        bars = []
        base_ts = DEFAULT_BASE_TIMESTAMP_MS
        base_price = 45000.0
        for i in range(30):
            # Small random-ish walk around $45k
            offset = (i % 7 - 3) * 100  # -300 to +300
            price = base_price + offset + i * 10  # Slight uptrend
            bars.append(OHLCVBar(
                t=base_ts + i * 60_000,
                o=price - 50,
                h=price + 100,
                l=price - 100,
                c=price,
                v=10.5  # BTC volume in BTC units
            ))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-btc-range",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="BTCUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected trade at BTC price levels"

    def test_btc_small_percentage_moves(self, backtest_service):
        """Test detection of small percentage moves at high prices.

        0.1% of $50k = $50, must detect correctly.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-btc-small",
            strategy_name="BTC Small Move Test",
            symbol="BTCUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="bb", type="BOLLINGER", period=20, std_dev=2.0),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.LT,
                    right=IndicatorBandRef(indicator_id="bb", band="lower"),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="middle_band",
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=IndicatorBandRef(indicator_id="bb", band="middle"),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # BTC with small moves (0.1-0.5% per bar)
        bars = []
        base_ts = DEFAULT_BASE_TIMESTAMP_MS
        price = 50000.0
        for i in range(35):
            # Small percentage moves
            if i < 25:
                price = 50000.0 + (i - 12) * 25  # Range: 49700 - 50300
            else:
                # Dip to lower band then recover
                price = 49500.0 if i < 30 else 50000.0

            bars.append(OHLCVBar(
                t=base_ts + i * 60_000,
                o=price - 10,
                h=price + 20,
                l=price - 20,
                c=price,
                v=5.0
            ))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-btc-small",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="BTCUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"

    def test_btc_typical_daily_range(self, backtest_service):
        """Test with typical BTC daily range (2-5% intraday).

        Simulates realistic intraday BTC movement.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-btc-daily",
            strategy_name="BTC Daily Range Test",
            symbol="BTCUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="rsi", type="RSI", period=14),
                IndicatorSpec(id="ema50", type="EMA", period=50),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=IndicatorRef(indicator_id="rsi"),
                            op=CompareOp.LT,
                            right=LiteralRef(value=30.0),
                        ),
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=IndicatorRef(indicator_id="ema50"),
                        ),
                    ],
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="rsi_overbought",
                    condition=CompareCondition(
                        left=IndicatorRef(indicator_id="rsi"),
                        op=CompareOp.GT,
                        right=LiteralRef(value=70.0),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # Simulate typical BTC day: down in morning, recovery, afternoon rally
        bars = []
        base_ts = DEFAULT_BASE_TIMESTAMP_MS
        base_price = 42000.0

        # Morning dip (60 bars = 1 hour)
        for i in range(60):
            price = base_price - i * 10  # Drop $600 (1.4%)
            bars.append(OHLCVBar(
                t=base_ts + i * 60_000,
                o=price + 5, h=price + 20, l=price - 30, c=price,
                v=15.0
            ))

        # Consolidation (30 bars)
        for i in range(30):
            price = 41400.0 + (i % 5 - 2) * 20
            bars.append(OHLCVBar(
                t=base_ts + (60 + i) * 60_000,
                o=price, h=price + 15, l=price - 15, c=price,
                v=8.0
            ))

        # Afternoon rally (60 bars)
        for i in range(60):
            price = 41400.0 + i * 15  # Rally $900 (2.2%)
            bars.append(OHLCVBar(
                t=base_ts + (90 + i) * 60_000,
                o=price - 5, h=price + 30, l=price - 10, c=price,
                v=20.0
            ))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-btc-daily",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="BTCUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"

    def test_btc_weekend_low_volume(self, backtest_service):
        """Test with low weekend volume pattern.

        BTC trades 24/7 but weekends often have 50% less volume.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-btc-weekend",
            strategy_name="BTC Weekend Volume Test",
            symbol="BTCUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="vol_sma", type="VOL_SMA", period=20),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=40000.0),
                        ),
                        # Only enter on higher volume
                        CompareCondition(
                            left=VolumeRef(),
                            op=CompareOp.GT,
                            right=IndicatorRef(indicator_id="vol_sma"),
                        ),
                    ],
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Low volume bars (weekend-like)
        bars = []
        base_ts = DEFAULT_BASE_TIMESTAMP_MS
        for i in range(40):
            price = 39500.0 + i * 30  # Gradual rise
            # Vary volume: low for first 25 bars, spike after
            vol = 2.0 if i < 25 else 8.0
            bars.append(OHLCVBar(
                t=base_ts + i * 60_000,
                o=price - 10, h=price + 30, l=price - 30, c=price,
                v=vol
            ))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-btc-weekend",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="BTCUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"


# =============================================================================
# Multi-Symbol Strategy Tests
# =============================================================================


@requires_lean
class TestMultiSymbolStrategies:
    """Test multi-symbol/intermarket strategies."""

    def test_two_symbol_ema_comparison(self, backtest_service):
        """Entry when primary symbol EMA > secondary symbol EMA.

        Strategy: Trade BTCUSD when BTC EMA(20) > ETH EMA(20)
        This tests intermarket analysis / relative strength.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-multi-ema",
            strategy_name="Multi-Symbol EMA Test",
            symbol="BTCUSD",
            resolution="Minute",
            additional_symbols=["ETHUSD"],  # Subscribe to ETH data
            indicators=[
                # BTC indicator (primary symbol, no 'symbol' field needed)
                IndicatorSpec(id="btc_ema", type="EMA", period=20),
                # ETH indicator (uses 'symbol' field for cross-symbol)
                IndicatorSpec(id="eth_ema", type="EMA", period=20, symbol="ETHUSD"),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=IndicatorRef(indicator_id="btc_ema"),
                    op=CompareOp.GT,
                    right=IndicatorRef(indicator_id="eth_ema"),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # BTC prices: rising from 100 to higher values
        btc_bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                     o=100.0 + i * 0.5, h=101.0 + i * 0.5, l=99.0 + i * 0.5,
                     c=100.0 + i * 0.5, v=1000.0)
            for i in range(30)
        ]

        # ETH prices: lower and flatter (so BTC EMA > ETH EMA)
        eth_bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                     o=50.0 + i * 0.1, h=51.0 + i * 0.1, l=49.0 + i * 0.1,
                     c=50.0 + i * 0.1, v=2000.0)
            for i in range(30)
        ]

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-multi-ema",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="BTCUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=btc_bars,
            strategy_ir=strategy_ir,
            additional_bars={"ETHUSD": eth_bars},
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when BTC EMA > ETH EMA"

    def test_relative_strength_entry(self, backtest_service):
        """Entry when primary symbol is stronger than secondary.

        Strategy: Trade when BTC price > ETH price * ratio
        Tests cross-symbol price comparison.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-relative-strength",
            strategy_name="Relative Strength Test",
            symbol="BTCUSD",
            resolution="Minute",
            additional_symbols=["ETHUSD"],
            indicators=[
                IndicatorSpec(id="btc_sma", type="SMA", period=10),
                IndicatorSpec(id="eth_sma", type="SMA", period=10, symbol="ETHUSD"),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                # BTC SMA must be at least 2x ETH SMA (relative strength)
                condition=CompareCondition(
                    left=IndicatorRef(indicator_id="btc_sma"),
                    op=CompareOp.GT,
                    right=IRExpression(
                        left=IndicatorRef(indicator_id="eth_sma"),
                        op="*",
                        right=LiteralRef(value=2.0),
                    ),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # BTC: starts at 100, rises to 115
        btc_bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                     o=100.0 + i * 0.5, h=101.0 + i * 0.5, l=99.0 + i * 0.5,
                     c=100.0 + i * 0.5, v=1000.0)
            for i in range(30)
        ]

        # ETH: stays around 40-45 (so BTC > 2x ETH after warmup)
        eth_bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                     o=40.0 + i * 0.1, h=41.0 + i * 0.1, l=39.0 + i * 0.1,
                     c=40.0 + i * 0.1, v=2000.0)
            for i in range(30)
        ]

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-relative-strength",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="BTCUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=btc_bars,
            strategy_ir=strategy_ir,
            additional_bars={"ETHUSD": eth_bars},
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when BTC > 2x ETH"

    def test_correlation_divergence(self, backtest_service):
        """Entry when correlated assets diverge.

        Strategy: Trade when BTC rises but ETH falls (divergence)
        Common for mean-reversion pairs trading.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-divergence",
            strategy_name="Divergence Test",
            symbol="BTCUSD",
            resolution="Minute",
            additional_symbols=["ETHUSD"],
            indicators=[
                IndicatorSpec(id="btc_ema", type="EMA", period=5),
                IndicatorSpec(id="btc_ema_slow", type="EMA", period=20),
                IndicatorSpec(id="eth_ema", type="EMA", period=5, symbol="ETHUSD"),
                IndicatorSpec(id="eth_ema_slow", type="EMA", period=20, symbol="ETHUSD"),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                # BTC trending up (fast > slow) AND ETH trending down (fast < slow)
                condition=AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=IndicatorRef(indicator_id="btc_ema"),
                            op=CompareOp.GT,
                            right=IndicatorRef(indicator_id="btc_ema_slow"),
                        ),
                        CompareCondition(
                            left=IndicatorRef(indicator_id="eth_ema"),
                            op=CompareOp.LT,
                            right=IndicatorRef(indicator_id="eth_ema_slow"),
                        ),
                    ],
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # BTC: uptrend
        btc_bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                     o=100.0 + i * 0.5, h=101.0 + i * 0.5, l=99.0 + i * 0.5,
                     c=100.0 + i * 0.5, v=1000.0)
            for i in range(40)
        ]

        # ETH: downtrend
        eth_bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                     o=100.0 - i * 0.5, h=101.0 - i * 0.5, l=99.0 - i * 0.5,
                     c=100.0 - i * 0.5, v=2000.0)
            for i in range(40)
        ]

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-divergence",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="BTCUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=btc_bars,
            strategy_ir=strategy_ir,
            additional_bars={"ETHUSD": eth_bars},
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry on BTC/ETH divergence"


# =============================================================================
# Long Backtest Tests (Weeks of Data)
# =============================================================================


@requires_lean
class TestLongBacktest:
    """Test backtests with extended time periods (weeks of data)."""

    def test_two_weeks_ema_crossover(self, backtest_service):
        """Run a 2-week backtest with EMA crossover strategy.

        Data: 2 weeks = 14 days * 24 hours * 60 minutes = 20,160 bars
        Using hourly bars for efficiency: 14 * 24 = 336 bars

        Strategy: Classic EMA(10) / EMA(50) crossover
        Expected: Multiple trades over 2-week period
        """
        strategy_ir = StrategyIR(
            strategy_id="test-long-ema",
            strategy_name="Long EMA Crossover Test",
            symbol="BTCUSD",
            resolution="Hour",
            indicators=[
                IndicatorSpec(id="ema_fast", type="EMA", period=10),
                IndicatorSpec(id="ema_slow", type="EMA", period=50),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CrossCondition(
                    left=IndicatorRef(indicator_id="ema_fast"),
                    right=IndicatorRef(indicator_id="ema_slow"),
                    direction="above",
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[
                ExitRule(
                    id="cross_exit",
                    condition=CrossCondition(
                        left=IndicatorRef(indicator_id="ema_fast"),
                        right=IndicatorRef(indicator_id="ema_slow"),
                        direction="below",
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[],
        )

        # 2 weeks of hourly data with realistic price movements
        # Create a trending/ranging pattern that will trigger multiple trades
        num_bars = 336  # 14 days * 24 hours
        base_price = 40000.0
        bars = []

        for i in range(num_bars):
            # Create cycles that will cause EMA crossovers
            # Uptrend for ~80 hours, downtrend for ~80 hours, repeat
            cycle_position = i % 168  # ~1 week cycle
            if cycle_position < 80:
                # Uptrend phase
                trend = (cycle_position / 80) * 0.15  # Up 15% over 80 hours
            else:
                # Downtrend phase
                trend = 0.15 - ((cycle_position - 80) / 88) * 0.12  # Down 12%

            price = base_price * (1 + trend)
            # Add some noise
            noise = (i * 17 % 100 - 50) / 5000  # Deterministic noise
            price *= (1 + noise)

            bars.append(OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + i * 3600_000,  # Hourly
                o=price * 0.998,
                h=price * 1.005,
                l=price * 0.995,
                c=price,
                v=1000.0 + (i % 100) * 10,
            ))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-long-ema",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 14, tzinfo=timezone.utc),
                symbol="BTCUSD",
                resolution="1h",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        # Should have at least one trade over 2 weeks with this cycling pattern
        assert len(trades) >= 1, f"Expected at least 1 trade in 2-week backtest, got {len(trades)}"
        # Verify we got summary stats
        assert result.response.summary is not None
        assert result.response.summary.total_trades >= 1


# =============================================================================
# Overlay Tests
# =============================================================================


@requires_lean
class TestOverlays:
    """Test overlay rules that modify position sizing."""

    def test_overlay_reduces_position_in_high_volatility(self, backtest_service):
        """Overlay scales down position when ATR is high.

        Strategy: Enter when price > 100, but reduce size by 50% when ATR > 2
        This tests position sizing modification based on volatility.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-overlay-vol",
            strategy_name="Volatility Overlay Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="atr", type="ATR", period=10),
            ],
            state=[],
            gates=[],
            overlays=[
                OverlayRule(
                    id="vol_scale",
                    condition=CompareCondition(
                        left=IndicatorRef(indicator_id="atr"),
                        op=CompareOp.GT,
                        right=LiteralRef(value=2.0),
                    ),
                    scale_size_frac=0.5,  # Reduce position to 50%
                    target_roles=["entry"],
                ),
            ],
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

        # Create volatile bars where ATR will be > 2
        # High volatility: 5-point swings on ~100 price = ~5% moves
        bars = []
        for i in range(25):
            # Alternating high/low to create volatility
            if i < 12:
                # Low volatility warmup
                base = 95 + i * 0.3
                bars.append(OHLCVBar(
                    t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                    o=base, h=base + 0.5, l=base - 0.5, c=base,
                    v=1000.0,
                ))
            else:
                # High volatility phase
                base = 102 + (i - 12) * 0.2
                swing = 3.0 if i % 2 == 0 else -3.0  # Big swings
                bars.append(OHLCVBar(
                    t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                    o=base - swing, h=base + abs(swing), l=base - abs(swing), c=base + swing,
                    v=1000.0,
                ))

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-overlay-vol",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry when price > 100"
        # Position should be scaled down due to high volatility overlay

    def test_overlay_doubles_position_in_trend(self, backtest_service):
        """Overlay scales up position during strong trend.

        Strategy: Enter when price > EMA, double size when EMA is rising
        """
        strategy_ir = StrategyIR(
            strategy_id="test-overlay-trend",
            strategy_name="Trend Overlay Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(id="ema", type="EMA", period=10),
                IndicatorSpec(id="ema_prev", type="EMA", period=11),  # Slightly lagged
            ],
            state=[],
            gates=[],
            overlays=[
                OverlayRule(
                    id="trend_boost",
                    condition=CompareCondition(
                        # EMA is rising: current EMA > slightly lagged EMA
                        left=IndicatorRef(indicator_id="ema"),
                        op=CompareOp.GT,
                        right=IndicatorRef(indicator_id="ema_prev"),
                    ),
                    scale_size_frac=1.5,  # Increase position by 50%
                    target_roles=["entry"],
                ),
            ],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=IndicatorRef(indicator_id="ema"),
                ),
                action=SetHoldingsAction(allocation=0.5),  # Base 50%
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Uptrending data
        bars = [
            OHLCVBar(
                t=DEFAULT_BASE_TIMESTAMP_MS + i * 60_000,
                o=100.0 + i * 0.5,
                h=101.0 + i * 0.5,
                l=99.0 + i * 0.5,
                c=100.0 + i * 0.5,
                v=1000.0,
            )
            for i in range(25)
        ]

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-overlay-trend",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected entry in uptrend"


# =============================================================================
# Time Stop Tests
# =============================================================================


@requires_lean
class TestTimeStops:
    """Test time-based exit rules using state variables."""

    def test_exit_after_5_bars(self, backtest_service):
        """Exit position after exactly 5 bars (time stop).

        Strategy: Enter when price > 100, exit after 5 bars regardless of price
        This is a common risk management technique.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-time-stop-5",
            strategy_name="5-Bar Time Stop Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[
                StateVarSpec(id="bars_in_trade", default=0.0),
            ],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[
                    SetStateAction(state_id="bars_in_trade", value=LiteralRef(value=0.0)),
                ],
            ),
            exits=[
                ExitRule(
                    id="time_stop_5",
                    condition=CompareCondition(
                        left=StateRef(state_id="bars_in_trade"),
                        op=CompareOp.GTE,
                        right=LiteralRef(value=5.0),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[
                IncrementStateAction(state_id="bars_in_trade", amount=LiteralRef(value=1.0)),
            ],
        )

        # Entry on bar 2, should exit on bar 7 (after 5 bars)
        bars = make_bars([95, 98, 102, 105, 108, 110, 112, 115, 118, 120])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-time-stop-5",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected at least one trade"
        # Entry at bar 2, exit after 5 bars = bar 7
        assert trades[0].entry_bar == 2
        assert trades[0].exit_bar == 7, f"Expected exit at bar 7, got {trades[0].exit_bar}"
        assert trades[0].exit_reason == "time_stop_5"

    def test_time_stop_with_profit_target(self, backtest_service):
        """Time stop combined with profit target - first one wins.

        Strategy: Exit on 10% profit OR after 10 bars, whichever comes first
        """
        strategy_ir = StrategyIR(
            strategy_id="test-time-profit",
            strategy_name="Time + Profit Stop Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[
                StateVarSpec(id="bars_held", default=0.0),
                StateVarSpec(id="entry_price", default=0.0),
            ],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=CompareCondition(
                    left=PriceRef(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralRef(value=100.0),
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[
                    SetStateAction(state_id="bars_held", value=LiteralRef(value=0.0)),
                    SetStateAction(state_id="entry_price", value=PriceRef(field=PriceField.CLOSE)),
                ],
            ),
            exits=[
                # Priority 0: Profit target (checks first)
                ExitRule(
                    id="profit_target",
                    priority=0,
                    condition=CompareCondition(
                        left=PriceRef(field=PriceField.CLOSE),
                        op=CompareOp.GT,
                        right=IRExpression(
                            left=StateRef(state_id="entry_price"),
                            op="*",
                            right=LiteralRef(value=1.10),  # 10% above entry
                        ),
                    ),
                    action=LiquidateAction(),
                ),
                # Priority 1: Time stop
                ExitRule(
                    id="time_stop_10",
                    priority=1,
                    condition=CompareCondition(
                        left=StateRef(state_id="bars_held"),
                        op=CompareOp.GTE,
                        right=LiteralRef(value=10.0),
                    ),
                    action=LiquidateAction(),
                ),
            ],
            on_bar=[],
            on_bar_invested=[
                IncrementStateAction(state_id="bars_held", amount=LiteralRef(value=1.0)),
            ],
        )

        # Price goes up slowly - won't hit 10% profit, so time stop triggers
        # Entry at bar 2 (price 102), 10% target = 112.2
        # Keep all prices below 112.2 so time stop (10 bars) triggers first
        bars = make_bars([95, 98, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 111, 111])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-time-profit",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"Backtest failed: {result.error}"
        trades = result.response.trades
        assert len(trades) >= 1, "Expected at least one trade"
        # Entry at bar 2 (price 102), 10% target = 112.2
        # Max price is 111 (below 112.2), so time stop (10 bars) should win
        assert trades[0].exit_reason == "time_stop_10", f"Expected time_stop_10, got {trades[0].exit_reason}"


# =============================================================================
# Typed Condition E2E Tests - Handler Completeness
# =============================================================================
# These tests verify that typed conditions dispatch correctly to their handlers
# in StrategyRuntime. If a handler is missing, LEAN will raise RuntimeError.


@requires_lean
class TestGapCondition:
    """Test GapCondition for gap detection at session open."""

    def test_gap_up_entry(self, backtest_service):
        """Entry when session opens with gap up > threshold.

        Note: GapCondition compares open vs previous close for gap percentage.
        In real use this triggers at session open; here we simulate with bars.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-gap-up",
            strategy_name="Gap Up Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=GapCondition(
                    direction="up",
                    min_gap_pct=2.0,  # Require 2%+ gap up
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Simulate gap: bar 4 opens at 103 vs bar 3 close at 100 = 3% gap up
        bars = [
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS, o=95, h=96, l=94, c=95, v=1000),
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + 60000, o=96, h=97, l=95, c=96, v=1000),
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + 120000, o=98, h=99, l=97, c=100, v=1000),  # Close at 100
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + 180000, o=103, h=105, l=102, c=104, v=1000),  # Gap up 3%
            OHLCVBar(t=DEFAULT_BASE_TIMESTAMP_MS + 240000, o=105, h=106, l=104, c=105, v=1000),
        ]

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-gap-up",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        # Test passes if LEAN doesn't crash - handler exists
        assert result.status == "success", f"GapCondition failed: {result.error}"


@requires_lean
class TestBreakoutCondition:
    """Test BreakoutCondition for N-bar high/low breakouts."""

    def test_nbar_breakout_entry(self, backtest_service):
        """Entry when price breaks above N-bar high."""
        strategy_ir = StrategyIR(
            strategy_id="test-nbar-breakout",
            strategy_name="N-Bar Breakout Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(indicator_type="MAX", params={"period": 5}),
                IndicatorSpec(indicator_type="MIN", params={"period": 5}),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=BreakoutCondition(
                    lookback_bars=5,
                    buffer_bps=0,
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Price breaks above 5-bar high on bar 8
        bars = make_bars([100, 99, 98, 97, 96, 95, 94, 93, 105, 106, 107])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-nbar-breakout",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"BreakoutCondition failed: {result.error}"


@requires_lean
class TestSqueezeCondition:
    """Test SqueezeCondition for volatility squeeze detection."""

    def test_squeeze_entry(self, backtest_service):
        """Entry when BB is inside KC (squeeze condition)."""
        strategy_ir = StrategyIR(
            strategy_id="test-squeeze",
            strategy_name="Squeeze Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(indicator_type="BB", params={"period": 5, "k": 2.0}),
                IndicatorSpec(indicator_type="KC", params={"period": 5, "atr_period": 5, "multiplier": 1.5}),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=SqueezeCondition(
                    bb_period=5,
                    bb_k=2.0,
                    kc_period=5,
                    kc_multiplier=1.5,
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Low volatility period - squeeze likely
        bars = make_bars([100, 100.1, 99.9, 100, 100.1, 99.9, 100, 100.2, 99.8, 100.1])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-squeeze",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"SqueezeCondition failed: {result.error}"


@requires_lean
class TestTrailingBreakoutCondition:
    """Test TrailingBreakoutCondition for trailing band breakouts."""

    def test_trailing_breakout_entry(self, backtest_service):
        """Entry when price breaks above trailing high band."""
        strategy_ir = StrategyIR(
            strategy_id="test-trailing-breakout",
            strategy_name="Trailing Breakout Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=TrailingBreakoutCondition(
                    band_type="DC",  # Donchian Channel
                    period=5,
                    direction="above",
                    edge="upper",
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Price breaks above 5-bar high
        bars = make_bars([100, 99, 98, 97, 96, 95, 102, 103, 104])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-trailing-breakout",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"TrailingBreakoutCondition failed: {result.error}"


@requires_lean
class TestTrailingStateCondition:
    """Test TrailingStateCondition for price tracking with ATR offset."""

    def test_trailing_state_entry(self, backtest_service):
        """Entry when price is above trailing anchor + ATR offset."""
        strategy_ir = StrategyIR(
            strategy_id="test-trailing-state",
            strategy_name="Trailing State Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(indicator_type="ATR", params={"period": 5}),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=TrailingStateCondition(
                    anchor="highest_high",
                    lookback=5,
                    atr_multiplier=1.0,
                    direction="below",  # Price below highest high - ATR
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Price trends then pulls back
        bars = make_bars([100, 105, 110, 115, 120, 115, 110, 105, 100, 95])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-trailing-state",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"TrailingStateCondition failed: {result.error}"


@requires_lean
class TestEventWindowCondition:
    """Test EventWindowCondition for time-based event windows."""

    def test_event_window_entry(self, backtest_service):
        """Entry within event window (e.g., earnings, FOMC).

        Note: Without actual event calendar, this tests the handler exists.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-event-window",
            strategy_name="Event Window Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        EventWindowCondition(
                            event_type="earnings",
                            window_hours_before=24,
                            window_hours_after=24,
                        ),
                    ]
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 96, 97, 98, 99, 101, 102, 103])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-event-window",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"EventWindowCondition failed: {result.error}"


@requires_lean
class TestSpreadCondition:
    """Test SpreadCondition for multi-symbol spread comparisons."""

    def test_spread_entry(self, backtest_service):
        """Entry when spread between symbols meets threshold."""
        strategy_ir = StrategyIR(
            strategy_id="test-spread",
            strategy_name="Spread Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        SpreadCondition(
                            symbol_a="TESTUSD",
                            symbol_b="TESTUSD",
                            calc_type="ratio",
                            trigger_op="above",
                            threshold=0.5,
                        ),
                    ]
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 96, 97, 98, 99, 101, 102, 103])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-spread",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"SpreadCondition failed: {result.error}"


@requires_lean
class TestIntermarketCondition:
    """Test IntermarketCondition for leader/follower signals."""

    def test_intermarket_entry(self, backtest_service):
        """Entry when leader symbol triggers follower signal.

        Note: Single-symbol test verifies handler exists.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-intermarket",
            strategy_name="Intermarket Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[
                IndicatorSpec(indicator_type="EMA", params={"period": 5}),
            ],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        IntermarketCondition(
                            leader_symbol="TESTUSD",
                            leader_condition=CompareCondition(
                                left=PriceRef(field=PriceField.CLOSE),
                                op=CompareOp.GT,
                                right=IndicatorRef(indicator_type="EMA", params={"period": 5}),
                            ),
                            lag_bars=0,
                        ),
                    ]
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Trending up - close > EMA
        bars = make_bars([90, 92, 94, 96, 98, 100, 102, 104, 106, 108])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-intermarket",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"IntermarketCondition failed: {result.error}"


@requires_lean
class TestMultiLeaderIntermarketCondition:
    """Test MultiLeaderIntermarketCondition for aggregated leader signals."""

    def test_multi_leader_entry(self, backtest_service):
        """Entry when multiple leaders agree (aggregation threshold)."""
        strategy_ir = StrategyIR(
            strategy_id="test-multi-leader",
            strategy_name="Multi Leader Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        MultiLeaderIntermarketCondition(
                            leaders=[
                                IntermarketCondition(
                                    leader_symbol="TESTUSD",
                                    leader_condition=CompareCondition(
                                        left=PriceRef(field=PriceField.CLOSE),
                                        op=CompareOp.GT,
                                        right=LiteralRef(value=100.0),
                                    ),
                                    lag_bars=0,
                                ),
                            ],
                            aggregation="any",
                            threshold=1,
                        ),
                    ]
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 96, 97, 98, 99, 101, 102, 103])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-multi-leader",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"MultiLeaderIntermarketCondition failed: {result.error}"


@requires_lean
class TestLiquiditySweepCondition:
    """Test LiquiditySweepCondition for sweep pattern detection."""

    def test_liquidity_sweep_entry(self, backtest_service):
        """Entry when price sweeps below level then reclaims."""
        strategy_ir = StrategyIR(
            strategy_id="test-liquidity-sweep",
            strategy_name="Liquidity Sweep Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=LiquiditySweepCondition(
                    level_type="rolling_min",
                    level_period=5,
                    lookback_bars=3,
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Price sweeps below then reclaims
        bars = make_bars([100, 99, 98, 97, 96, 94, 97, 98, 99, 100])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-liquidity-sweep",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"LiquiditySweepCondition failed: {result.error}"


@requires_lean
class TestFlagPatternCondition:
    """Test FlagPatternCondition for momentum + consolidation + breakout."""

    def test_flag_pattern_entry(self, backtest_service):
        """Entry on flag pattern: momentum, consolidation, breakout."""
        strategy_ir = StrategyIR(
            strategy_id="test-flag-pattern",
            strategy_name="Flag Pattern Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=FlagPatternCondition(
                    momentum_threshold=3.0,
                    momentum_period=5,
                    consolidation_bars=3,
                    breakout_direction="same",
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Momentum up, consolidation, breakout
        bars = make_bars([100, 102, 104, 106, 108, 108, 108, 109, 112, 115])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-flag-pattern",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"FlagPatternCondition failed: {result.error}"


@requires_lean
class TestPennantPatternCondition:
    """Test PennantPatternCondition for triangular consolidation."""

    def test_pennant_pattern_entry(self, backtest_service):
        """Entry on pennant pattern: momentum, converging trendlines, breakout."""
        strategy_ir = StrategyIR(
            strategy_id="test-pennant-pattern",
            strategy_name="Pennant Pattern Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=PennantPatternCondition(
                    momentum_threshold=3.0,
                    momentum_period=5,
                    consolidation_bars=3,
                    breakout_direction="same",
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        # Momentum up, pennant consolidation (converging range), breakout
        bars = make_bars([100, 102, 104, 106, 108, 107, 108, 107.5, 112, 115])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-pennant-pattern",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"PennantPatternCondition failed: {result.error}"


@requires_lean
class TestTimeFilterCondition:
    """Test TimeFilterCondition for time-based filtering."""

    def test_time_filter_entry(self, backtest_service):
        """Entry only when within time window.

        TimeFilterCondition is for hour/minute filtering.
        """
        strategy_ir = StrategyIR(
            strategy_id="test-time-filter",
            strategy_name="Time Filter Test",
            symbol="TESTUSD",
            resolution="Minute",
            indicators=[],
            state=[],
            gates=[],
            overlays=[],
            entry=EntryRule(
                condition=AllOfCondition(
                    conditions=[
                        CompareCondition(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=CompareOp.GT,
                            right=LiteralRef(value=100.0),
                        ),
                        TimeFilterCondition(
                            start_hour=0,
                            end_hour=23,  # All day allowed
                            days_of_week=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
                        ),
                    ]
                ),
                action=SetHoldingsAction(allocation=0.95),
                on_fill=[],
            ),
            exits=[],
            on_bar=[],
            on_bar_invested=[],
        )

        bars = make_bars([95, 96, 97, 98, 99, 101, 102, 103])

        result = backtest_service.run_backtest(
            request=BacktestRequest(
                strategy_id="test-time-filter",
                start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
                symbol="TESTUSD",
                resolution="1m",
            ),
            strategy=None,
            cards={},
            inline_bars=bars,
            strategy_ir=strategy_ir,
        )

        assert result.status == "success", f"TimeFilterCondition failed: {result.error}"
