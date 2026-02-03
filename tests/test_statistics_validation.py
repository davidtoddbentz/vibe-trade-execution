"""Validate LEAN statistics against independent calculations.

These tests verify that LEAN's native statistics (Sharpe, Sortino, etc.)
match expected values calculated independently using standard formulas.
"""

import math
from statistics import NormalDist
from datetime import date, datetime, timezone

import pytest
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.archetypes.entry.rule_trigger import (
    EntryRuleTrigger,
    EventSlot,
)
from vibe_trade_shared.models.archetypes.exit.fixed_targets import (
    FixedTargets,
    FixedTargetsEvent,
)
from vibe_trade_shared.models.archetypes.primitives import (
    CompareSpec,
    ConditionSpec,
    ContextSpec,
    EntryActionSpec,
    ExitActionSpec,
    SignalRef,
)
from vibe_trade_shared.models.data import OHLCVBar
from vibe_trade_shared.models.ir import PositionPolicy
from vibe_trade_shared.models.strategy import Attachment

from src.models.lean_backtest import BacktestConfig
from src.service.backtest_service import BacktestService
from src.service.data_service import MockDataService


pytestmark = pytest.mark.e2e

TRADING_DAYS_PER_YEAR = 252
TRADE_AVG_TOLERANCE = 0.01  # ±1% for trade averages (expectancy, PL ratio)
RISK_RATIO_TOLERANCE = 0.15  # ±15% for Sharpe/Sortino/Info ratios (annualization differences)
VOLATILITY_TOLERANCE = 0.12  # ±12% for volatility and VaR (calculation method differences)


def _ms(dt: datetime) -> int:
    """Convert datetime to milliseconds since epoch."""
    return int(dt.timestamp() * 1000)


def make_bars(
    prices: list[float],
    start: datetime | None = None,
    interval_ms: int = 86_400_000,
) -> list[OHLCVBar]:
    """Build OHLCV bars from close prices with deterministic highs/lows."""
    if start is None:
        start = datetime(2024, 1, 1, 6, 0, tzinfo=timezone.utc)
    base_ms = _ms(start)
    bars = []
    for i, price in enumerate(prices):
        bars.append(
            OHLCVBar(
                t=base_ms + i * interval_ms,
                o=price,
                h=price * 1.01,
                l=price * 0.99,
                c=price,
                v=1_000_000.0,
            )
        )
    return bars


def card_from_archetype(card_id: str, archetype: EntryRuleTrigger | FixedTargets) -> Card:
    """Build a Card from a typed archetype instance."""
    return Card(
        id=card_id,
        type=archetype.TYPE_ID,
        name=card_id,
        schema_etag="test",
        slots=archetype.model_dump(exclude_none=True, by_alias=True),
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )


def make_strategy(
    cards: list[Card],
    name: str,
    symbol: str,
) -> tuple[Strategy, dict[str, Card]]:
    """Build Strategy from a list of Cards."""
    attachments = [
        Attachment(card_id=card.id, role=card.type.split(".")[0], enabled=True, overrides={})
        for card in cards
    ]
    strategy = Strategy(
        id=f"test-{name.lower().replace(' ', '-')}",
        name=name,
        universe=[symbol],
        attachments=attachments,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )
    cards_dict = {card.id: card for card in cards}
    return strategy, cards_dict


def price_above(threshold: float) -> ConditionSpec:
    """Close price > threshold."""
    return ConditionSpec(
        type="compare",
        compare=CompareSpec(
            lhs=SignalRef(type="price", field="close"),
            op=">",
            rhs=threshold,
        ),
    )


def price_below(threshold: float) -> ConditionSpec:
    """Close price < threshold."""
    return ConditionSpec(
        type="compare",
        compare=CompareSpec(
            lhs=SignalRef(type="price", field="close"),
            op="<",
            rhs=threshold,
        ),
    )


def _equity_values(equity_curve: list) -> list[float]:
    values = []
    for point in equity_curve or []:
        if isinstance(point, dict):
            values.append(float(point.get("equity", 0.0)))
        else:
            values.append(float(getattr(point, "equity", point)))
    return values


def calculate_returns(values: list[float]) -> list[float]:
    """Calculate period-over-period returns from a list of values."""
    returns = []
    for i in range(1, len(values)):
        prev = values[i - 1]
        curr = values[i]
        if prev != 0:
            returns.append((curr - prev) / prev)
    return returns


def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
) -> float:
    """Calculate Sharpe ratio from a list of returns.

    Standard formula:
      - sharpe = (mean_return - risk_free_rate) / std_dev
      - If periods_per_year provided, annualize: sharpe * sqrt(periods_per_year)

    This is the correct textbook formula. Do NOT compound returns for short periods.
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)
    # Sample variance (N-1)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    if variance <= 0:
        return 0.0
    std_dev = math.sqrt(variance)

    if std_dev == 0:
        return 0.0

    sharpe = (mean_return - risk_free_rate) / std_dev

    # Annualize if requested
    if periods_per_year:
        sharpe *= math.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int | None = None,
) -> float:
    """Calculate Sortino ratio (only considers downside volatility).

    Standard formula:
      - sortino = (mean_return - risk_free_rate) / downside_deviation
      - Downside deviation uses only negative returns
      - If periods_per_year provided, annualize: sortino * sqrt(periods_per_year)
    """
    if len(returns) < 2:
        return 0.0

    mean_return = sum(returns) / len(returns)

    # Only consider negative returns for downside deviation
    downside_returns = [r for r in returns if r < 0]
    if not downside_returns:
        return float('inf')  # No downside risk

    # Calculate downside deviation (standard deviation of negative returns)
    if len(downside_returns) < 2:
        return 0.0
    downside_mean = sum(downside_returns) / len(downside_returns)
    downside_variance = sum((r - downside_mean) ** 2 for r in downside_returns) / (
        len(downside_returns) - 1
    )
    if downside_variance <= 0:
        return 0.0
    downside_deviation = math.sqrt(downside_variance)

    sortino = (mean_return - risk_free_rate) / downside_deviation

    # Annualize if requested
    if periods_per_year:
        sortino *= math.sqrt(periods_per_year)

    return sortino


def calculate_win_rate(trades: list) -> float:
    """Calculate win rate as percentage of winning trades."""
    if not trades:
        return 0.0
    
    winning = sum(1 for t in trades if (t.pnl or 0) > 0)
    return (winning / len(trades)) * 100.0


def _trade_return_pct(trade) -> float:
    if trade.pnl_pct is not None:
        return float(trade.pnl_pct)
    if trade.entry_price and trade.exit_price:
        raw_return = (trade.exit_price - trade.entry_price) / trade.entry_price * 100.0
        if trade.direction == "short":
            return -raw_return
        return raw_return
    if trade.pnl is not None and trade.entry_price and trade.quantity:
        position_value = abs(trade.entry_price * trade.quantity)
        if position_value != 0:
            return float(trade.pnl) / position_value * 100.0
    return 0.0


def calculate_expectancy(trades: list) -> float:
    """Calculate expectancy using LEAN's formula.

    LEAN uses expectancy as:
        (win_rate * (avg_win / avg_loss)) - loss_rate
    where win/loss rates are fractions and avg_win/avg_loss is the P/L ratio.
    """
    if not trades:
        return 0.0
    returns = [_trade_return_pct(t) for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    if not wins or not losses:
        return 0.0
    win_rate = len(wins) / len(returns)
    loss_rate = len(losses) / len(returns)
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    if avg_loss == 0:
        return 0.0
    profit_loss_ratio = avg_win / avg_loss
    return (win_rate * profit_loss_ratio) - loss_rate


def calculate_profit_loss_ratio(trades: list) -> float:
    """Calculate P/L ratio: avg_win / abs(avg_loss)."""
    if not trades:
        return 0.0
    returns = [_trade_return_pct(t) for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    if not wins or not losses:
        return 0.0
    avg_win = sum(wins) / len(wins)
    avg_loss = abs(sum(losses) / len(losses))
    if avg_loss == 0:
        return 0.0
    return avg_win / avg_loss


def calculate_annual_volatility(returns: list[float], periods_per_year: int) -> float:
    """Calculate annualized standard deviation as a percentage."""
    if len(returns) < 2:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance)
    return std_dev * math.sqrt(periods_per_year) * 100.0


def calculate_var_95(returns: list[float], lookback_period_days: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calculate Value at Risk at 95% confidence (LEAN variance-covariance method)."""
    if not returns:
        return 0.0
    period_returns = returns[-lookback_period_days:]
    if len(period_returns) < 2:
        return 0.0
    mean_return = sum(period_returns) / len(period_returns)
    variance = sum((r - mean_return) ** 2 for r in period_returns) / (len(period_returns) - 1)
    if variance <= 0:
        return 0.0
    std_dev = math.sqrt(variance)
    value_at_risk = NormalDist(mean_return, std_dev).inv_cdf(1 - 0.95)
    return round(value_at_risk, 3)


def calculate_information_ratio(
    returns: list[float],
    benchmark_returns: list[float],
    periods_per_year: int,
) -> float:
    """Calculate Information ratio using standard formula.

    Information Ratio = mean(active_return) / std_dev(active_return) * sqrt(periods_per_year)
    where active_return = strategy_return - benchmark_return
    """
    if len(returns) < 2 or len(benchmark_returns) < 2:
        return 0.0
    paired_length = min(len(returns), len(benchmark_returns))
    algo_returns = returns[:paired_length]
    bench_returns = benchmark_returns[:paired_length]
    active_returns = [r - b for r, b in zip(algo_returns, bench_returns)]
    if len(active_returns) < 2:
        return 0.0

    mean_active = sum(active_returns) / len(active_returns)
    variance = sum((r - mean_active) ** 2 for r in active_returns) / (len(active_returns) - 1)
    if variance <= 0:
        return 0.0
    tracking_error = math.sqrt(variance)

    if tracking_error == 0:
        return 0.0

    info_ratio = mean_active / tracking_error
    # Annualize
    return info_ratio * math.sqrt(periods_per_year)


def _sharpe_components(returns: list[float], periods_per_year: int) -> dict[str, float]:
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    sharpe_base = (mean_return / std_dev) if std_dev > 0 else 0.0
    sharpe_annualized = sharpe_base * math.sqrt(periods_per_year)
    return {
        "mean_return": mean_return,
        "variance": variance,
        "std_dev": std_dev,
        "sharpe_base": sharpe_base,
        "sharpe_annualized": sharpe_annualized,
    }


def _var_components(returns: list[float], lookback_period_days: int) -> dict[str, float]:
    period_returns = returns[-lookback_period_days:]
    mean_return = sum(period_returns) / len(period_returns)
    variance = sum((r - mean_return) ** 2 for r in period_returns) / (len(period_returns) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0
    value_at_risk = (
        NormalDist(mean_return, std_dev).inv_cdf(1 - 0.95) if std_dev > 0 else 0.0
    )
    return {
        "mean_return": mean_return,
        "variance": variance,
        "std_dev": std_dev,
        "value_at_risk": round(value_at_risk, 3),
    }


def assert_within_tolerance(actual: float, expected: float, rel_tolerance: float, label: str) -> None:
    if expected == 0:
        assert abs(actual) <= rel_tolerance, f"{label} mismatch: {actual:.6f} vs {expected:.6f}"
    else:
        diff = abs(actual - expected)
        tolerance = abs(expected) * rel_tolerance
        assert diff <= tolerance, (
            f"{label} mismatch: LEAN={actual:.6f}, Expected={expected:.6f}, "
            f"Tolerance={tolerance:.6f}"
        )


class TestStatisticsValidation:
    """Validate LEAN statistics match independent calculations."""

    def test_sharpe_ratio_matches_calculation(self, lean_url: str):
        """Verify LEAN's Sharpe ratio matches independent calculation."""
        # Create strategy with known behavior
        entry = EntryRuleTrigger(
            context=ContextSpec(symbol="TESTUSD", tf="1d"),
            event=EventSlot(condition=price_above(1.0)),
            action=EntryActionSpec(direction="long", position_policy=PositionPolicy(mode="single")),
        )
        entry_card = card_from_archetype("entry_1", entry)
        strategy, cards = make_strategy([entry_card], name="Sharpe Test", symbol="TESTUSD")

        # Generate controlled price data with known returns
        prices = [450, 455, 452, 460, 458, 470, 465, 480]  # Known price path
        bars = make_bars(prices)

        data_service = MockDataService()
        data_service.seed("TESTUSD", "1d", bars)
        service = BacktestService(data_service=data_service, backtest_url=lean_url)

        result = service.run_backtest(
            strategy=strategy,
            cards=cards,
            config=BacktestConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 8),
                initial_cash=100000.0,
                symbol="TESTUSD",
                resolution="1d",
            ),
        )

        assert result.status == "success", result.error
        assert result.response is not None
        assert result.response.summary is not None
        
        # Calculate daily returns from equity curve
        equity_values = _equity_values(result.response.equity_curve)
        returns = calculate_returns(equity_values)

        expected_sharpe = calculate_sharpe_ratio(returns, periods_per_year=TRADING_DAYS_PER_YEAR)
        actual_sharpe = result.response.summary.sharpe_ratio or 0.0
        if returns:
            components = _sharpe_components(returns, TRADING_DAYS_PER_YEAR)
            print(
                "Sharpe Debug (test_sharpe_ratio_matches_calculation): "
                f"equity_len={len(equity_values)}, "
                f"returns_len={len(returns)}, "
                f"mean={components['mean_return']:.6f}, "
                f"std_dev={components['std_dev']:.6f}, "
                f"sharpe_base={components['sharpe_base']:.6f}, "
                f"sharpe_annualized={components['sharpe_annualized']:.6f}, "
                f"expected_sharpe={expected_sharpe:.6f}, "
                f"lean_sharpe={actual_sharpe:.6f}"
            )

        assert_within_tolerance(
            actual_sharpe,
            expected_sharpe,
            RISK_RATIO_TOLERANCE,
            "Sharpe ratio",
        )

    def test_win_rate_matches_calculation(self, lean_url: str):
        """Verify LEAN's win rate matches count of winning trades."""
        # This is simpler - just count winning vs losing trades
        entry = EntryRuleTrigger(
            context=ContextSpec(symbol="TESTUSD", tf="1d"),
            event=EventSlot(condition=price_below(100.0)),
            action=EntryActionSpec(direction="long", position_policy=PositionPolicy(mode="single")),
        )
        exit_ = FixedTargets(
            context=ContextSpec(symbol="TESTUSD", tf="1d"),
            event=FixedTargetsEvent(sl_pct=2.0, tp_pct=2.0),
            action=ExitActionSpec(mode="close"),
        )
        entry_card = card_from_archetype("entry_1", entry)
        exit_card = card_from_archetype("exit_1", exit_)
        strategy, cards = make_strategy([entry_card, exit_card], name="Win Rate Test", symbol="TESTUSD")

        # Create data that will generate multiple complete trades
        warmup = [105.0] * 5
        trade_pattern = [
            98.0, 101.0, 105.0,  # win
            98.0, 95.0, 105.0,   # loss
            98.0, 101.0, 105.0,  # win
            98.0, 101.0, 105.0,  # win
        ]
        prices = warmup + trade_pattern
        bars = make_bars(prices)

        data_service = MockDataService()
        data_service.seed("TESTUSD", "1d", bars)

        service = BacktestService(data_service=data_service, backtest_url=lean_url)

        result = service.run_backtest(
            strategy=strategy,
            cards=cards,
            config=BacktestConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 20),
                initial_cash=100000.0,
                symbol="TESTUSD",
                resolution="1d",
            ),
        )

        assert result.status == "success", result.error
        assert result.response is not None
        assert result.response.summary is not None
        
        # Calculate expected win rate from trades
        trades = result.response.trades
        expected_win_rate = calculate_win_rate(trades)
        actual_win_rate = result.response.summary.win_rate or 0.0

        # Win rate should match exactly (it's just counting)
        assert abs(actual_win_rate - expected_win_rate) < 0.1, (
            f"Win rate mismatch: LEAN={actual_win_rate:.2f}%, "
            f"Expected={expected_win_rate:.2f}%"
        )


class TestComprehensiveStatistics:
    """Validate ALL LEAN statistics with deterministic price data."""

    def test_all_risk_adjusted_metrics(self, lean_url: str):
        """Test Sharpe, Sortino, Information ratios with known returns."""
        entry = EntryRuleTrigger(
            context=ContextSpec(symbol="TESTUSD", tf="1d"),
            event=EventSlot(condition=price_above(100.0)),
            action=EntryActionSpec(direction="long", position_policy=PositionPolicy(mode="single")),
        )
        exit_ = FixedTargets(
            context=ContextSpec(symbol="TESTUSD", tf="1d"),
            event=FixedTargetsEvent(sl_pct=3.0, tp_pct=4.0),
            action=ExitActionSpec(mode="close"),
        )
        entry_card = card_from_archetype("entry_1", entry)
        exit_card = card_from_archetype("exit_1", exit_)
        strategy, cards = make_strategy(
            [entry_card, exit_card], name="Risk Adjusted Metrics", symbol="TESTUSD"
        )

        prices = [450, 460, 455, 470, 468, 480, 475, 490, 485, 500]
        bars = make_bars(prices)
        data_service = MockDataService()
        data_service.seed("TESTUSD", "1d", bars)
        service = BacktestService(data_service=data_service, backtest_url=lean_url)

        result = service.run_backtest(
            strategy=strategy,
            cards=cards,
            config=BacktestConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 20),
                initial_cash=100000.0,
                symbol="TESTUSD",
                resolution="1d",
            ),
        )

        assert result.status == "success", result.error
        assert result.response is not None
        summary = result.response.summary
        assert summary is not None

        equity_values = _equity_values(result.response.equity_curve)
        returns = calculate_returns(equity_values)
        benchmark_returns = calculate_returns(prices)

        expected_sharpe = calculate_sharpe_ratio(
            returns, periods_per_year=TRADING_DAYS_PER_YEAR
        )
        expected_sortino = calculate_sortino_ratio(
            returns, periods_per_year=TRADING_DAYS_PER_YEAR
        )
        expected_information = calculate_information_ratio(
            returns, benchmark_returns, TRADING_DAYS_PER_YEAR
        )
        if returns:
            components = _sharpe_components(returns, TRADING_DAYS_PER_YEAR)
            print(
                "Sharpe Debug (test_all_risk_adjusted_metrics): "
                f"equity_len={len(equity_values)}, "
                f"returns_len={len(returns)}, "
                f"mean={components['mean_return']:.6f}, "
                f"std_dev={components['std_dev']:.6f}, "
                f"sharpe_base={components['sharpe_base']:.6f}, "
                f"sharpe_annualized={components['sharpe_annualized']:.6f}, "
                f"expected_sharpe={expected_sharpe:.6f}, "
                f"lean_sharpe={(summary.sharpe_ratio or 0.0):.6f}"
            )

        assert_within_tolerance(
            summary.sharpe_ratio or 0.0,
            expected_sharpe,
            RISK_RATIO_TOLERANCE,
            "Sharpe ratio",
        )
        assert_within_tolerance(
            summary.sortino_ratio or 0.0,
            expected_sortino,
            RISK_RATIO_TOLERANCE,
            "Sortino ratio",
        )
        assert_within_tolerance(
            summary.information_ratio or 0.0,
            expected_information,
            RISK_RATIO_TOLERANCE,
            "Information ratio",
        )

    def test_all_trade_quality_metrics(self, lean_url: str):
        """Test expectancy, PL ratio, win/loss rates with known trades."""
        entry = EntryRuleTrigger(
            context=ContextSpec(symbol="TESTUSD", tf="1d"),
            event=EventSlot(condition=price_below(100.0)),
            action=EntryActionSpec(direction="long", position_policy=PositionPolicy(mode="single")),
        )
        exit_ = FixedTargets(
            context=ContextSpec(symbol="TESTUSD", tf="1d"),
            event=FixedTargetsEvent(sl_pct=2.0, tp_pct=2.0),
            action=ExitActionSpec(mode="close"),
        )
        entry_card = card_from_archetype("entry_1", entry)
        exit_card = card_from_archetype("exit_1", exit_)
        strategy, cards = make_strategy([entry_card, exit_card], name="Trade Quality Metrics", symbol="TESTUSD")

        warmup = [105.0] * 5
        trade_pattern = [
            98.0, 101.0, 105.0,  # win
            98.0, 101.0, 105.0,  # win
            98.0, 95.0, 105.0,   # loss
            98.0, 101.0, 105.0,  # win
            98.0, 95.0, 105.0,   # loss
            98.0, 101.0, 105.0,  # win
            98.0, 95.0, 105.0,   # loss
            98.0, 101.0, 105.0,  # win
        ]
        prices = warmup + trade_pattern
        bars = make_bars(prices)

        data_service = MockDataService()
        data_service.seed("TESTUSD", "1d", bars)
        service = BacktestService(data_service=data_service, backtest_url=lean_url)

        result = service.run_backtest(
            strategy=strategy,
            cards=cards,
            config=BacktestConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 2, 15),
                initial_cash=100000.0,
                symbol="TESTUSD",
                resolution="1d",
            ),
        )

        assert result.status == "success", result.error
        assert result.response is not None
        summary = result.response.summary
        trades = result.response.trades
        assert summary is not None

        expected_win_rate = calculate_win_rate(trades)
        expected_loss_rate = 100.0 - expected_win_rate
        expected_expectancy = calculate_expectancy(trades)
        expected_pl_ratio = calculate_profit_loss_ratio(trades)

        assert summary.total_trades == len(trades)
        assert summary.winning_trades == sum(1 for t in trades if (t.pnl or 0) > 0)
        assert summary.losing_trades == sum(1 for t in trades if (t.pnl or 0) <= 0)

        assert abs((summary.win_rate or 0.0) - expected_win_rate) < 0.1
        assert abs((summary.loss_rate or 0.0) - expected_loss_rate) < 0.1

        assert_within_tolerance(
            summary.expectancy or 0.0,
            expected_expectancy,
            TRADE_AVG_TOLERANCE,
            "Expectancy",
        )
        assert_within_tolerance(
            summary.profit_loss_ratio or 0.0,
            expected_pl_ratio,
            TRADE_AVG_TOLERANCE,
            "Profit/Loss ratio",
        )

    def test_all_risk_metrics(self, lean_url: str):
        """Test annual volatility and VaR 95%."""
        entry = EntryRuleTrigger(
            context=ContextSpec(symbol="TESTUSD", tf="1d"),
            event=EventSlot(condition=price_above(1.0)),
            action=EntryActionSpec(direction="long", position_policy=PositionPolicy(mode="single")),
        )
        entry_card = card_from_archetype("entry_1", entry)
        strategy, cards = make_strategy([entry_card], name="Risk Metrics", symbol="TESTUSD")

        prices = [100, 102, 99, 103, 97, 105, 101, 106, 100, 108]
        bars = make_bars(prices)

        data_service = MockDataService()
        data_service.seed("TESTUSD", "1d", bars)
        service = BacktestService(data_service=data_service, backtest_url=lean_url)

        result = service.run_backtest(
            strategy=strategy,
            cards=cards,
            config=BacktestConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 20),
                initial_cash=100000.0,
                symbol="TESTUSD",
                resolution="1d",
            ),
        )

        assert result.status == "success", result.error
        assert result.response is not None
        summary = result.response.summary
        assert summary is not None

        equity_values = _equity_values(result.response.equity_curve)
        returns = calculate_returns(equity_values)

        expected_volatility = calculate_annual_volatility(
            returns, TRADING_DAYS_PER_YEAR
        )
        expected_var_95 = calculate_var_95(returns)
        if returns:
            var_components = _var_components(returns, TRADING_DAYS_PER_YEAR)
            print(
                "VaR Debug (test_all_risk_metrics): "
                f"equity_len={len(equity_values)}, "
                f"equity_sample={equity_values[:3]}...{equity_values[-3:]}, "
                f"returns_len={len(returns)}, "
                f"returns_sample={returns[:5]}, "
                f"mean={var_components['mean_return']:.6f}, "
                f"variance={var_components['variance']:.6f}, "
                f"std_dev={var_components['std_dev']:.6f}, "
                f"expected_var_95={expected_var_95:.6f}, "
                f"lean_var_95={(summary.value_at_risk_95 or 0.0):.6f}"
            )

        assert_within_tolerance(
            summary.annual_standard_deviation or 0.0,
            expected_volatility,
            VOLATILITY_TOLERANCE,
            "Annual volatility",
        )
        assert_within_tolerance(
            summary.value_at_risk_95 or 0.0,
            expected_var_95,
            VOLATILITY_TOLERANCE,
            "VaR 95%",
        )
