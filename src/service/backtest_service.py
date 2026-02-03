"""Backtest service - orchestrates strategy backtesting.

This service:
1. Fetches market data via DataService
2. Translates strategy to IR
3. Calls the backtest container endpoint (local Docker or Cloud Run) with data
4. Returns structured results

Data Flow:
    Execution: DataService.get_ohlcv() → fetches from BigQuery
    Execution: IRTranslator(strategy, cards).translate() → StrategyIR
    Execution: HTTP POST to LEAN container with strategy_ir + data (inline or GCS)
    LEAN: Deserialize, execute, return LEANBacktestResponse
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from src.models.lean_backtest import (
    BacktestConfig,
    BacktestDataInput,
    EquityPoint,
    LEANBacktestRequest,
    LEANBacktestResponse,
)
from src.service.data_service import DataService
from src.translator.ir_translator import IRTranslator

logger = logging.getLogger(__name__)

# Default endpoints
LOCAL_BACKTEST_URL = os.environ.get("LOCAL_BACKTEST_URL", "http://localhost:8083/backtest")
CLOUD_RUN_BACKTEST_URL = os.environ.get(
    "BACKTEST_SERVICE_URL",
    "https://vibe-trade-backtest-833596808881.us-central1.run.app/backtest",
)

# Threshold for using inline vs GCS data
INLINE_DATA_THRESHOLD = 10000  # Use inline if fewer bars


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    status: str  # "success", "error"
    strategy_id: str
    start_date: datetime
    end_date: datetime
    message: str | None = None
    results: dict[str, Any] | None = None
    algorithm_code: str | None = None
    error: str | None = None
    response: LEANBacktestResponse | None = None


def _calculate_warmup_bars(strategy_ir: dict) -> int:
    """Calculate the number of warmup bars needed for indicator initialization.

    Examines all indicators in the IR and returns the maximum period,
    plus a safety buffer. This ensures indicators are ready before trading starts.

    Args:
        strategy_ir: The strategy IR dict with indicators list

    Returns:
        Number of warmup bars needed (minimum 50 for safety)
    """
    max_period = 0
    for indicator in strategy_ir.get("indicators", []):
        period = indicator.get("period", 0)
        if period > max_period:
            max_period = period

    # Add safety buffer: max period + 10, minimum 50 bars
    warmup_bars = max(max_period + 10, 50)
    return warmup_bars


def _resolution_to_timedelta(resolution: str) -> timedelta:
    """Convert resolution string to timedelta for warmup calculation.

    Args:
        resolution: Resolution string like "1h", "1d", "1m"

    Returns:
        timedelta representing one bar's duration
    """
    resolution_lower = resolution.lower()
    if resolution_lower in ("1d", "daily"):
        return timedelta(days=1)
    elif resolution_lower in ("4h",):
        return timedelta(hours=4)
    elif resolution_lower in ("1h", "hour"):
        return timedelta(hours=1)
    elif resolution_lower in ("15m",):
        return timedelta(minutes=15)
    elif resolution_lower in ("5m",):
        return timedelta(minutes=5)
    else:  # Default to 1 minute
        return timedelta(minutes=1)


def _compute_benchmark(
    bars: list, start_timestamp_ms: int
) -> tuple[float | None, float | None, float | None]:
    """Compute buy-and-hold return and max drawdown from OHLCV bars.

    Only considers bars at or after start_timestamp_ms (the user's trading start date,
    excluding warmup data).

    Args:
        bars: List of OHLCVBar objects (with .t timestamp in ms, .c close price)
        start_timestamp_ms: Trading period start as milliseconds since epoch

    Returns:
        Tuple of (benchmark_return, benchmark_max_drawdown, alpha_placeholder)
        where alpha_placeholder is None (caller computes alpha from strategy return).
        Returns (None, None, None) if insufficient data.
    """
    # Filter to trading period only (exclude warmup bars)
    trading_bars = [b for b in bars if b.t >= start_timestamp_ms]

    if not trading_bars or len(trading_bars) < 2:
        return None, None, None

    first_close = trading_bars[0].c
    if first_close == 0:
        return None, None, None

    last_close = trading_bars[-1].c
    benchmark_return = (last_close - first_close) / first_close

    # Compute max drawdown
    peak = first_close
    max_dd = 0.0
    for bar in trading_bars:
        if bar.c > peak:
            peak = bar.c
        dd = (bar.c - peak) / peak  # Negative value
        if dd < max_dd:
            max_dd = dd

    return benchmark_return, max_dd, None


class BacktestService:
    """Service for running strategy backtests via HTTP endpoint.

    Orchestrates the full backtest flow:
    1. Fetch data via DataService (BQ or mock)
    2. Translate strategy to IR
    3. Call LEAN container with data
    4. Return structured results
    """

    def __init__(
        self,
        data_service: DataService | None = None,
        backtest_url: str | None = None,
        use_local: bool = False,
        auth_token: str | None = None,
    ):
        """Initialize backtest service.

        Args:
            data_service: Service for fetching market data. Required for fetching data.
            backtest_url: URL of the backtest service endpoint
            use_local: If True, use local Docker endpoint (localhost:8081)
            auth_token: Bearer token for Cloud Run authentication
        """
        self.data_service = data_service

        if backtest_url:
            self.backtest_url = backtest_url
        elif use_local:
            self.backtest_url = LOCAL_BACKTEST_URL
        else:
            self.backtest_url = CLOUD_RUN_BACKTEST_URL

        self.auth_token = auth_token

    def run_backtest(
        self,
        strategy: Any,
        cards: dict[str, Any],
        config: BacktestConfig,
    ) -> BacktestResult:
        """Run a backtest for a strategy using the clean 3-param interface.

        Always translates strategy+cards to IR, always fetches data via data_service.
        Uses BacktestConfig for all execution parameters.

        Args:
            strategy: Strategy model from vibe-trade-shared (has .id attribute)
            cards: Dict mapping card_id to Card objects
            config: BacktestConfig with symbol, resolution, dates, fees, etc.

        Returns:
            BacktestResult with status and results
        """
        strategy_id = getattr(strategy, "id", "unknown")
        # Convert date to datetime for date math and result fields
        start_datetime = datetime.combine(
            config.start_date, datetime.min.time(), tzinfo=timezone.utc
        )
        end_datetime = datetime.combine(
            config.end_date, datetime.min.time(), tzinfo=timezone.utc
        )

        try:
            logger.info(f"Starting backtest for strategy {strategy_id}")

            # Step 1: Translate strategy to IR
            logger.info("Translating strategy to IR...")
            translator = IRTranslator(strategy, cards)
            strategy_ir = translator.translate()

            if not strategy_ir:
                return BacktestResult(
                    status="error",
                    strategy_id=strategy_id,
                    start_date=start_datetime,
                    end_date=end_datetime,
                    error="Failed to translate strategy to IR",
                )

            # Step 2: Fetch market data via data_service (with warmup)
            if self.data_service is None:
                return BacktestResult(
                    status="error",
                    strategy_id=strategy_id,
                    start_date=start_datetime,
                    end_date=end_datetime,
                    error="No data_service configured",
                )

            ir_dict = strategy_ir.model_dump()
            warmup_bars = _calculate_warmup_bars(ir_dict)
            bar_duration = _resolution_to_timedelta(config.resolution)
            warmup_start = start_datetime - (warmup_bars * bar_duration)

            logger.info(
                f"Fetching data from DataService for {config.symbol}... "
                f"(with {warmup_bars} warmup bars starting {warmup_start.date()})"
            )
            bars = self.data_service.get_ohlcv(
                symbol=config.symbol,
                resolution=config.resolution,
                start=warmup_start,
                end=end_datetime,
            )
            logger.info(f"Fetched {len(bars)} bars (including warmup)")

            # Step 3: Build LEAN request
            # Apply backtest-level trading costs to the IR
            strategy_ir = strategy_ir.model_copy(
                update={
                    "fee_pct": config.fee_pct,
                    "slippage_pct": config.slippage_pct,
                }
            )

            ir_dict = strategy_ir.model_dump()
            ir_json = strategy_ir.model_dump_json(indent=2)

            # LEAN start_date must match earliest available data bar.
            # For minute resolution, LEAN reads per-day ZIP files and won't
            # find data for dates before our first bar. trading_start_date
            # already prevents trades during warmup.
            if bars:
                lean_start = datetime.fromtimestamp(
                    bars[0].t / 1000, tz=timezone.utc
                ).date()
            else:
                lean_start = warmup_start.date()

            lean_request = LEANBacktestRequest(
                strategy_ir=ir_dict,
                data=BacktestDataInput(
                    symbol=config.symbol,
                    resolution=config.resolution,
                    bars=bars,
                ),
                config=BacktestConfig(
                    start_date=lean_start,
                    end_date=config.end_date,
                    initial_cash=config.initial_cash,
                    # User's original start date prevents trades during warmup
                    trading_start_date=config.start_date,
                    symbol=config.symbol,
                    resolution=config.resolution,
                    fee_pct=config.fee_pct,
                    slippage_pct=config.slippage_pct,
                ),
            )

            # Step 4: Call LEAN container
            logger.info(f"Calling LEAN at {self.backtest_url}")
            response = self._call_lean_endpoint(lean_request)

            if response.status == "error" or response.error:
                return BacktestResult(
                    status="error",
                    strategy_id=strategy_id,
                    start_date=start_datetime,
                    end_date=end_datetime,
                    error=response.error,
                    algorithm_code=ir_json,
                    response=response,
                )

            # Build results from response in format expected by UI
            summary = response.summary

            # Transform trades to UI format
            ui_trades = []
            for i, t in enumerate(response.trades):
                ui_trades.append({
                    "trade_id": f"trade_{i}",
                    "symbol": config.symbol,
                    "direction": "buy" if t.direction == "long" else "sell",
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "entry_price": t.entry_price,
                    "entry_quantity": t.quantity,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                    "pnl_percent": t.pnl_pct,
                    "exit_reason": t.exit_reason,
                })

            # Transform statistics to UI format
            # Map all statistics from BacktestSummary to match BacktestStatistics schema
            win_rate = (summary.winning_trades / summary.total_trades) if summary and summary.total_trades > 0 else 0
            statistics = {
                # Core performance metrics
                "total_return": summary.total_pnl_pct / 100 if summary else 0,  # UI expects decimal
                "annual_return": summary.compounding_annual_return / 100 if summary and summary.compounding_annual_return is not None else 0,
                "net_profit": summary.total_pnl if summary else 0,

                # Risk-adjusted returns
                "sharpe_ratio": summary.sharpe_ratio if summary else None,
                "sortino_ratio": summary.sortino_ratio if summary and summary.sortino_ratio is not None else None,
                "information_ratio": summary.information_ratio if summary and summary.information_ratio is not None else None,
                "treynor_ratio": summary.treynor_ratio if summary and summary.treynor_ratio is not None else None,

                # Risk metrics
                "max_drawdown": summary.max_drawdown_pct / 100 if summary else 0,  # UI expects decimal
                "annual_standard_deviation": summary.annual_standard_deviation / 100 if summary and summary.annual_standard_deviation is not None else None,
                "value_at_risk_95": summary.value_at_risk_95 if summary and summary.value_at_risk_95 is not None else None,

                # Trade statistics
                "total_trades": summary.total_trades if summary else 0,
                "winning_trades": summary.winning_trades if summary else 0,
                "losing_trades": summary.losing_trades if summary else 0,
                "win_rate": win_rate,
                "loss_rate": summary.loss_rate / 100 if summary and summary.loss_rate is not None else None,
                "average_win": summary.average_win_rate / 100 if summary and summary.average_win_rate is not None else 0,
                "average_loss": summary.average_loss_rate / 100 if summary and summary.average_loss_rate is not None else 0,
                "profit_loss_ratio": summary.profit_loss_ratio if summary and summary.profit_loss_ratio is not None else None,
                "expectancy": summary.expectancy if summary and summary.expectancy is not None else None,

                # Market correlation (will be overwritten by benchmark calculation below if available)
                "alpha": summary.alpha if summary and summary.alpha is not None else None,
                "beta": summary.beta if summary and summary.beta is not None else None,
            } if summary else None

            # Compute buy-and-hold benchmark from OHLCV data
            # Only overwrite LEAN's alpha if we successfully calculate benchmark
            if statistics is not None and bars:
                start_ts_ms = int(start_datetime.timestamp() * 1000)
                bench_return, bench_dd, _ = _compute_benchmark(bars, start_ts_ms)
                statistics["benchmark_return"] = bench_return
                statistics["benchmark_max_drawdown"] = bench_dd
                if bench_return is not None:
                    strategy_return = statistics.get("total_return", 0) or 0
                    statistics["alpha"] = strategy_return - bench_return

            # Transform equity curve to EquityPoint format expected by UI
            equity_curve_points = []
            if response.equity_curve and len(response.equity_curve) > 0:
                first_point = response.equity_curve[0]

                # Check if we have structured data (EquityPoint) or legacy flat list
                if isinstance(first_point, EquityPoint):
                    # New format: EquityPoint objects - use actual data
                    for point in response.equity_curve:
                        equity_curve_points.append({
                            "time": point.time,
                            "equity": point.equity,
                            "cash": point.cash,
                            "holdings_value": point.holdings,
                            "drawdown": point.drawdown,
                        })
                elif isinstance(first_point, dict):
                    # Dict format from LEAN (raw JSON) - use actual data
                    for point in response.equity_curve:
                        equity_curve_points.append({
                            "time": point.get("time", ""),
                            "equity": point.get("equity", 0),
                            "cash": point.get("cash", 0),
                            "holdings_value": point.get("holdings", 0),
                            "drawdown": point.get("drawdown", 0),
                        })
                else:
                    # Legacy format: flat list of equity floats - reconstruct
                    num_points = len(response.equity_curve)
                    start_ts = start_datetime.timestamp()
                    end_ts = end_datetime.timestamp()
                    interval = (end_ts - start_ts) / max(num_points - 1, 1)

                    peak_equity = config.initial_cash
                    for i, equity in enumerate(response.equity_curve):
                        timestamp = start_ts + (i * interval)
                        time_str = datetime.fromtimestamp(timestamp).isoformat()

                        if equity > peak_equity:
                            peak_equity = equity
                        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0

                        equity_curve_points.append({
                            "time": time_str,
                            "equity": equity,
                            "cash": 0,  # Not available in legacy format
                            "holdings_value": equity,  # Approximate
                            "drawdown": drawdown,
                        })

            results = {
                "trades": ui_trades,
                "statistics": statistics,
                "equity_curve": equity_curve_points,
            }
            if response.ohlcv_bars:
                results["ohlcv_bars"] = [
                    bar.model_dump() if hasattr(bar, "model_dump") else bar
                    for bar in response.ohlcv_bars
                ]
            if response.indicators:
                results["indicators"] = response.indicators

            return BacktestResult(
                status="success",
                strategy_id=strategy_id,
                start_date=start_datetime,
                end_date=end_datetime,
                results=results,
                algorithm_code=ir_json,
                response=response,
                message=f"Backtest completed: {summary.total_trades if summary else 0} trades, "
                f"{summary.total_pnl_pct if summary else 0:.2f}% return",
            )

        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return BacktestResult(
                status="error",
                strategy_id=strategy_id,
                start_date=start_datetime,
                end_date=end_datetime,
                error=str(e),
            )

    def _call_lean_endpoint(
        self, request: LEANBacktestRequest, _max_retries: int = 2
    ) -> LEANBacktestResponse:
        """Call LEAN HTTP endpoint with new request format.

        Retries on RemoteProtocolError ("Server disconnected") which occurs when
        uvicorn closes a keep-alive connection between sequential requests.

        Args:
            request: LEANBacktestRequest with strategy_ir, data, and config

        Returns:
            LEANBacktestResponse with trades and summary
        """
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        last_error: Exception | None = None
        for attempt in range(_max_retries + 1):
            try:
                with httpx.Client(timeout=600.0) as client:  # 10 minute timeout
                    response = client.post(
                        self.backtest_url,
                        content=request.model_dump_json(),
                        headers=headers,
                    )

                    if response.status_code != 200:
                        return LEANBacktestResponse(
                            status="error",
                            error=f"HTTP {response.status_code}: {response.text}",
                        )

                    return LEANBacktestResponse.model_validate(response.json())

            except httpx.TimeoutException:
                return LEANBacktestResponse(
                    status="error",
                    error="Backtest request timed out after 10 minutes",
                )
            except httpx.RemoteProtocolError as e:
                last_error = e
                if attempt < _max_retries:
                    logger.warning(
                        f"LEAN connection reset (attempt {attempt + 1}/{_max_retries + 1}), retrying: {e}"
                    )
                    continue
            except Exception as e:
                return LEANBacktestResponse(
                    status="error",
                    error=f"Failed to call LEAN service: {e}",
                )

        return LEANBacktestResponse(
            status="error",
            error=f"Failed to call LEAN service after {_max_retries + 1} attempts: {last_error}",
        )
