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
    """Summary metrics from backtest."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float = Field(..., description="Total P&L in quote currency")
    total_pnl_pct: float = Field(..., description="Total P&L as percentage")
    max_drawdown_pct: float = Field(default=0.0, description="Maximum drawdown percentage")
    sharpe_ratio: float | None = None


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
