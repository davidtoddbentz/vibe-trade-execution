"""Validate LEAN statistics against independent calculations.

These tests verify that LEAN's native statistics (Sharpe, Sortino, etc.)
match expected values calculated independently using standard formulas.
"""

import math
from datetime import date, datetime, timedelta, timezone

import pytest
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.archetypes.entry.rule_trigger import EntryRuleTrigger, EventSlot
from vibe_trade_shared.models.archetypes.exit.fixed_targets import FixedTargets, FixedTargetsEvent
from vibe_trade_shared.models.archetypes.primitives import (
    CompareSpec,
    ConditionSpec,
    LiteralRef,
    Operator,
    PriceField,
    PriceRef,
)
from vibe_trade_shared.models.data import OHLCVBar

from src.models.lean_backtest import BacktestConfig
from src.service.backtest_service import BacktestService
from src.service.data_service import MockDataService


def calculate_sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from a list of returns.
    
    Sharpe = (mean_return - risk_free_rate) / std_dev_returns
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance)
    
    if std_dev == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / std_dev


def calculate_sortino_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio (only considers downside volatility).
    
    Sortino = (mean_return - risk_free_rate) / downside_deviation
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = sum(returns) / len(returns)
    
    # Only consider negative returns for downside deviation
    downside_returns = [r for r in returns if r < 0]
    if not downside_returns:
        return float('inf')  # No downside risk
    
    downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
    downside_deviation = math.sqrt(downside_variance)
    
    if downside_deviation == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / downside_deviation


def calculate_win_rate(trades: list) -> float:
    """Calculate win rate as percentage of winning trades."""
    if not trades:
        return 0.0
    
    winning = sum(1 for t in trades if (t.pnl or 0) > 0)
    return (winning / len(trades)) * 100.0


class TestStatisticsValidation:
    """Validate LEAN statistics match independent calculations."""

    @pytest.mark.skip(reason="Independent calculation validation - implement when needed for production")
    def test_sharpe_ratio_matches_calculation(self):
        """Verify LEAN's Sharpe ratio matches independent calculation."""
        # Create strategy with known behavior
        entry = Card.for_archetype(
            EntryRuleTrigger(
                label="always_enter",
                event=EventSlot(
                    condition=ConditionSpec(
                        compare=CompareSpec(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=Operator.GT,
                            right=LiteralRef(value=1000.0),  # Always true for our data
                        )
                    )
                ),
            )
        )

        exit_card = Card.for_archetype(
            FixedTargets(
                label="fixed_targets",
                event=FixedTargetsEvent(
                    stop_loss_pct=2.0,
                    take_profit_pct=5.0,
                ),
            )
        )

        strategy = Strategy(
            name="Sharpe Test",
            symbol="BTC-USD",
            resolution="1d",
            cards=[entry, exit_card],
        )

        # Generate controlled price data with known returns
        bars = []
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        prices = [45000, 45500, 45200, 46000, 45800, 47000, 46500, 48000]  # Known price path
        
        for i, price in enumerate(prices):
            bars.append(
                OHLCVBar(
                    time=base_time + timedelta(days=i),
                    open=price,
                    high=price * 1.01,
                    low=price * 0.99,
                    close=price,
                    volume=1000000.0,
                )
            )

        data_service = MockDataService()
        data_service.seed_bars("BTC-USD", "1d", bars)

        service = BacktestService(data_service=data_service, initial_cash=100000.0)

        result = service.run_backtest(
            strategy=strategy,
            config=BacktestConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 8),
                initial_cash=100000.0,
                symbol="BTC-USD",
                resolution="1d",
            ),
        )

        assert result.status == "success"
        assert result.summary is not None
        
        # Calculate daily returns from equity curve
        if result.equity_curve and len(result.equity_curve) > 1:
            returns = []
            for i in range(1, len(result.equity_curve)):
                prev_equity = result.equity_curve[i-1].equity
                curr_equity = result.equity_curve[i].equity
                if prev_equity > 0:
                    daily_return = (curr_equity - prev_equity) / prev_equity
                    returns.append(daily_return)
            
            if returns:
                expected_sharpe = calculate_sharpe_ratio(returns)
                actual_sharpe = result.summary.sharpe_ratio or 0.0
                
                # Allow 10% tolerance for differences in calculation method
                tolerance = abs(expected_sharpe * 0.1) if expected_sharpe != 0 else 0.1
                assert abs(actual_sharpe - expected_sharpe) <= tolerance, (
                    f"Sharpe ratio mismatch: LEAN={actual_sharpe:.4f}, "
                    f"Expected={expected_sharpe:.4f}, Tolerance={tolerance:.4f}"
                )

    @pytest.mark.skip(reason="Independent calculation validation - implement when needed for production")  
    def test_win_rate_matches_calculation(self):
        """Verify LEAN's win rate matches count of winning trades."""
        # This is simpler - just count winning vs losing trades
        entry = Card.for_archetype(
            EntryRuleTrigger(
                label="enter",
                event=EventSlot(
                    condition=ConditionSpec(
                        compare=CompareSpec(
                            left=PriceRef(field=PriceField.CLOSE),
                            op=Operator.GT,
                            right=LiteralRef(value=1000.0),
                        )
                    )
                ),
            )
        )

        exit_card = Card.for_archetype(
            FixedTargets(
                label="targets",
                event=FixedTargetsEvent(
                    stop_loss_pct=5.0,
                    take_profit_pct=3.0,
                ),
            )
        )

        strategy = Strategy(
            name="Win Rate Test",
            symbol="BTC-USD",
            resolution="1h",
            cards=[entry, exit_card],
        )

        # Create data that will generate multiple complete trades
        bars = []
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        base_price = 45000.0
        
        for i in range(50):
            # Oscillate to trigger entries and exits
            price = base_price + (i % 10 - 5) * 200
            bars.append(
                OHLCVBar(
                    time=base_time + timedelta(hours=i),
                    open=price,
                    high=price + 100,
                    low=price - 100,
                    close=price,
                    volume=1000000.0,
                )
            )

        data_service = MockDataService()
        data_service.seed_bars("BTC-USD", "1h", bars)

        service = BacktestService(data_service=data_service, initial_cash=100000.0)

        result = service.run_backtest(
            strategy=strategy,
            config=BacktestConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 3),
                initial_cash=100000.0,
                symbol="BTC-USD",
                resolution="1h",
            ),
        )

        assert result.status == "success"
        assert result.summary is not None
        
        # Calculate expected win rate from trades
        if result.trades:
            expected_win_rate = calculate_win_rate(result.trades)
            actual_win_rate = result.summary.win_rate or 0.0
            
            # Win rate should match exactly (it's just counting)
            assert abs(actual_win_rate - expected_win_rate) < 0.1, (
                f"Win rate mismatch: LEAN={actual_win_rate:.2f}%, "
                f"Expected={expected_win_rate:.2f}%"
            )
