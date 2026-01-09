"""Backtest result models.

Comprehensive data structures for capturing backtest output,
similar to what QuantConnect/LEAN provides.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OrderDirection(str, Enum):
    """Order direction."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order status."""

    FILLED = "filled"
    CANCELLED = "cancelled"
    INVALID = "invalid"


@dataclass
class Trade:
    """A single completed trade (entry + exit)."""

    trade_id: str
    symbol: str
    direction: OrderDirection  # long or short
    entry_time: datetime
    entry_price: float
    entry_quantity: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_quantity: float | None = None
    pnl: float | None = None  # Realized P&L
    pnl_percent: float | None = None  # Return %
    duration_bars: int | None = None
    exit_reason: str | None = None  # Which exit rule triggered
    fees: float = 0.0


@dataclass
class Order:
    """A single order event."""

    order_id: str
    symbol: str
    direction: OrderDirection
    quantity: float
    order_type: str  # market, limit, stop
    time: datetime
    status: OrderStatus
    fill_price: float | None = None
    fill_quantity: float | None = None
    fees: float = 0.0
    message: str | None = None


@dataclass
class EquityPoint:
    """Single point on equity curve."""

    time: datetime
    equity: float  # Total portfolio value
    cash: float
    holdings_value: float
    drawdown: float  # Current drawdown %
    drawdown_duration: int  # Bars in drawdown


@dataclass
class PerformanceStatistics:
    """Performance statistics from backtest."""

    # Returns
    total_return: float  # Total return %
    annual_return: float  # Annualized return %
    benchmark_return: float | None = None

    # Risk metrics
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    max_drawdown: float = 0.0  # Max drawdown %
    max_drawdown_duration: int = 0  # Bars in max drawdown
    volatility: float | None = None  # Annualized volatility

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float | None = None  # Gross profit / gross loss
    average_win: float = 0.0
    average_loss: float = 0.0
    average_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_trade_duration: float = 0.0  # In bars

    # Portfolio
    net_profit: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_fees: float = 0.0

    # Exposure
    average_exposure: float = 0.0  # % of time invested
    max_exposure: float = 0.0


@dataclass
class ChartSeries:
    """A data series for charting."""

    name: str
    series_type: str  # line, bar, scatter, candle
    data: list[tuple[datetime, float]] = field(default_factory=list)
    color: str | None = None


@dataclass
class BacktestResult:
    """Complete backtest result."""

    # Identification
    backtest_id: str
    strategy_id: str
    strategy_name: str
    symbol: str

    # Time range
    start_date: datetime
    end_date: datetime
    resolution: str  # Hour, Minute, Daily

    # Configuration
    initial_cash: float
    parameters: dict = field(default_factory=dict)

    # Results
    final_equity: float = 0.0
    statistics: PerformanceStatistics = field(default_factory=PerformanceStatistics)

    # Detailed data
    trades: list[Trade] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)
    equity_curve: list[EquityPoint] = field(default_factory=list)

    # Charts
    charts: dict[str, list[ChartSeries]] = field(default_factory=dict)

    # Logs
    logs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Metadata
    runtime_seconds: float = 0.0
    completed_at: datetime | None = None
    status: str = "pending"  # pending, running, completed, failed

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        from dataclasses import asdict

        def convert(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            return obj

        result = asdict(self)
        # Handle datetime serialization
        return _convert_datetimes(result)


def _convert_datetimes(obj):
    """Recursively convert datetimes to ISO strings."""
    if isinstance(obj, dict):
        return {k: _convert_datetimes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_datetimes(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Enum):
        return obj.value
    return obj
