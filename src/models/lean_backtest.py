"""LEAN backtest request and response models.

These models define the interface between Execution and LEAN services.
Execution sends LEANBacktestRequest, LEAN returns LEANBacktestResponse.
"""

from datetime import date, datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator
from vibe_trade_shared.models.data import OHLCVBar


class BacktestDataInput(BaseModel):
    """Data input for backtest - either inline or GCS reference.

    For tests and small backtests, use inline bars.
    For large production backtests, use GCS reference.
    """

    symbol: str = Field(..., description="Trading symbol (e.g., 'BTC-USD')")
    resolution: str = Field(..., description="Bar resolution ('1m', '1h', '1d')")

    # Option 1: Inline data (tests, small backtests)
    bars: list[OHLCVBar] | None = Field(
        default=None, description="Inline OHLCV bars (for tests and small backtests)"
    )

    # Option 2: GCS reference (large production backtests)
    gcs_uri: str | None = Field(
        default=None, description="GCS URI for parquet data (gs://bucket/path.parquet)"
    )

    @model_validator(mode="after")
    def validate_data_source(self) -> Self:
        if self.bars is None and self.gcs_uri is None:
            raise ValueError("Must provide either 'bars' or 'gcs_uri'")
        if self.bars is not None and self.gcs_uri is not None:
            raise ValueError("Provide either 'bars' or 'gcs_uri', not both")
        return self


class BacktestConfig(BaseModel):
    """Configuration for backtest execution."""

    start_date: date  # LEAN data processing start (may include warmup)
    end_date: date
    initial_cash: float = Field(default=100000.0, ge=0)
    trading_start_date: date | None = Field(
        default=None,
        description="When trading is allowed to begin (user's actual start date). "
        "If None, uses start_date. Used to prevent trades during warmup period.",
    )
    # Execution params
    symbol: str = Field(default="BTC-USD", description="Trading symbol")
    resolution: str = Field(default="1h", description="Bar resolution ('1m', '1h', '1d')")
    fee_pct: float = Field(default=0.0, description="Fee as percentage (0.1 = 0.1%)")
    slippage_pct: float = Field(default=0.0, description="Slippage as percentage (0.05 = 0.05%)")


class LEANBacktestRequest(BaseModel):
    """Request body for LEAN backtest endpoint.

    Sent from Execution to LEAN HTTP endpoint.
    """

    strategy_ir: dict = Field(..., description="StrategyIR.model_dump()")
    data: BacktestDataInput
    config: BacktestConfig
    additional_data: list[BacktestDataInput] = Field(
        default_factory=list,
        description="Additional symbol data for multi-symbol strategies",
    )


class Trade(BaseModel):
    """Single trade from backtest with exact bar info."""

    entry_bar: int = Field(..., description="Bar index of entry")
    entry_price: float = Field(..., description="Entry fill price")
    entry_time: datetime
    exit_bar: int | None = Field(default=None, description="Bar index of exit")
    exit_price: float | None = Field(default=None, description="Exit fill price")
    exit_time: datetime | None = None
    exit_reason: str | None = Field(default=None, description="Why trade exited (exit rule ID or 'end_of_backtest')")
    direction: Literal["long", "short"]
    quantity: float
    pnl: float | None = Field(default=None, description="Profit/loss in quote currency")
    pnl_pct: float | None = Field(default=None, description="Profit/loss as percentage")



class EquityPoint(BaseModel):
    """Single point on equity curve with full portfolio breakdown.

    Matches the structure from StrategyRuntime._track_equity().
    """

    time: str  # ISO format timestamp
    equity: float  # Total portfolio value
    cash: float  # Cash balance
    holdings: float  # Holdings value
    drawdown: float  # Current drawdown percentage


class BacktestSummary(BaseModel):
    """Summary metrics from backtest.

    Includes both custom calculations and LEAN's native PortfolioStatistics.
    All LEAN statistics are optional for backward compatibility.
    """

    # Basic metrics (required, from custom calculations)
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float = Field(..., description="Total P&L in quote currency")
    total_pnl_pct: float = Field(..., description="Total P&L as percentage")
    max_drawdown_pct: float = Field(default=0.0, description="Maximum drawdown percentage")

    # LEAN PortfolioStatistics (all optional)

    # Risk-Adjusted Returns
    sharpe_ratio: float | None = Field(
        default=None,
        description="Sharpe ratio based on the strategy's excess returns",
    )
    sortino_ratio: float | None = Field(
        default=None,
        description="Sortino ratio based on downside deviation",
    )
    probabilistic_sharpe_ratio: float | None = Field(
        default=None,
        description="Probabilistic Sharpe ratio",
    )
    information_ratio: float | None = Field(
        default=None,
        description="Information ratio relative to the benchmark",
    )
    treynor_ratio: float | None = Field(
        default=None,
        description="Treynor ratio based on systematic risk",
    )

    # Performance Metrics
    compounding_annual_return: float | None = Field(
        default=None,
        description="Compounded annual return (CAGR)",
    )
    total_net_profit: float | None = Field(
        default=None,
        description="Total net profit over the backtest period",
    )
    start_equity: float | None = Field(default=None, description="Starting equity value")
    end_equity: float | None = Field(default=None, description="Ending equity value")

    # Risk Metrics
    drawdown: float | None = Field(
        default=None,
        description="Maximum drawdown as a decimal fraction",
    )
    annual_standard_deviation: float | None = Field(
        default=None,
        description="Annualized standard deviation of returns",
    )
    annual_variance: float | None = Field(
        default=None,
        description="Annualized variance of returns",
    )
    tracking_error: float | None = Field(
        default=None,
        description="Tracking error versus benchmark",
    )
    value_at_risk_99: float | None = Field(
        default=None,
        description="Value at risk at 99% confidence",
    )
    value_at_risk_95: float | None = Field(
        default=None,
        description="Value at risk at 95% confidence",
    )

    # Market Correlation
    alpha: float | None = Field(
        default=None,
        description="Alpha relative to the benchmark",
    )
    beta: float | None = Field(
        default=None,
        description="Beta relative to the benchmark",
    )

    # Trade Statistics (LEAN native)
    win_rate: float | None = Field(default=None, description="Percentage of winning trades")
    loss_rate: float | None = Field(default=None, description="Percentage of losing trades")
    average_win_rate: float | None = Field(
        default=None,
        description="Average return of winning trades",
    )
    average_loss_rate: float | None = Field(
        default=None,
        description="Average return of losing trades",
    )
    profit_loss_ratio: float | None = Field(
        default=None,
        description="Ratio of average winning trade return to average losing trade return",
    )
    expectancy: float | None = Field(
        default=None,
        description="Expected return per trade based on win/loss rates and averages",
    )

    # Activity Metrics
    portfolio_turnover: float | None = Field(
        default=None,
        description="Portfolio turnover over the backtest period",
    )


class LEANBacktestResponse(BaseModel):
    """Response from LEAN backtest endpoint.

    Returned from LEAN HTTP endpoint to Execution.
    """

    status: Literal["success", "error"]
    trades: list[Trade] = Field(default_factory=list)
    summary: BacktestSummary | None = None
    # Support both formats: list[EquityPoint] (full data) or list[float] (legacy)
    equity_curve: list[EquityPoint] | list[float] | None = None
    ohlcv_bars: list[OHLCVBar] | list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "OHLCV bars used in the backtest, for candlestick chart visualization. "
            "Supports canonical OHLCVBar or time/open/high/low/close/volume dicts."
        ),
    )
    indicators: dict[str, list[dict[str, Any]]] | None = Field(
        default=None,
        description=(
            "Indicator time-series data keyed by indicator name. Each value is a list of points "
            "with at least a time and value field, plus optional additional fields (e.g., "
            "Bollinger Bands upper/middle/lower)."
        ),
    )
    error: str | None = None
