"""Integration tests for LEAN backtest execution.

These tests verify that translated strategies produce correct trading signals
when run through the actual LEAN engine with deterministic data.
"""

import json
import pytest
from pathlib import Path

from src.lean_runner.engine import LeanEngine


@pytest.fixture
def lean_engine():
    """Create LEAN engine for testing."""
    engine = LeanEngine()
    if not engine.check_docker_available():
        pytest.skip("Docker/LEAN not available")
    return engine


class TestDeterministicBacktest:
    """Test backtests with deterministic data where we know expected outcomes."""

    def test_ema_crossover_deterministic(self, lean_engine):
        """Verify EMA crossover trades correctly on deterministic data.

        Data pattern:
        - Hours 0-49: flat at $100
        - Hours 50-59: ramp up $100→$200 (+$10/hour)
        - Hours 60-97: flat at $200
        - Hours 98-107: ramp down $200→$100 (-$10/hour)
        - Hours 108-120: flat at $100

        Expected trades with EMA(5)/EMA(10):
        - BUY at Jan 3 02:00 when price jumps to $110 (fast EMA crosses above slow)
        - SELL at Jan 5 02:00 when price drops to $190 (fast EMA crosses below slow)

        Exact calculation:
        - Starting cash: $100,000
        - SetHoldings(0.95) → $95,000 to invest
        - Buy @ $110 → 861 shares (LEAN rounds down)
        - Cost: 861 × $110 = $94,710, remaining cash: $5,290
        - Sell @ $190 → 861 × $190 = $163,590
        - Final: $163,590 + $5,290 = $168,880
        - Return: 68.88%
        """
        result = lean_engine.run_backtest(
            algorithm_name="EmaDeterministicTest",
            start_date="20240101",
            end_date="20240106",
            cash=100000.0,
        )

        assert result["status"] == "success", f"Backtest failed: {result.get('error', result.get('stderr'))}"

        # Check results file
        results_dir = Path(lean_engine.project_root) / "lean" / "Results"
        summary_file = results_dir / "EmaDeterministicTest-summary.json"

        assert summary_file.exists(), "Summary file not found"

        with open(summary_file) as f:
            summary = json.load(f)

        stats = summary["statistics"]

        # Exact expected values
        end_equity = float(summary["runtimeStatistics"]["Equity"].replace("$", "").replace(",", ""))
        assert end_equity == 168880.0, f"Expected $168,880, got ${end_equity:,.2f}"

        net_profit_str = stats["Net Profit"]  # "68.880%"
        net_profit = float(net_profit_str.replace("%", ""))
        assert net_profit == 68.88, f"Expected 68.88% profit, got {net_profit}%"

        # Exactly 2 orders (1 buy, 1 sell)
        total_orders = int(stats["Total Orders"])
        assert total_orders == 2, f"Expected 2 orders, got {total_orders}"

        # 100% win rate (single winning trade)
        win_rate_str = stats["Win Rate"]
        win_rate = int(win_rate_str.replace("%", ""))
        assert win_rate == 100, f"Expected 100% win rate, got {win_rate}%"

    def test_ema_crossover_flat_market_no_trades(self, lean_engine):
        """Verify no trades when EMAs never cross (flat market)."""
        # This would require a separate flat-only CSV file
        # For now, skip this test
        pytest.skip("Requires flat market test data")
