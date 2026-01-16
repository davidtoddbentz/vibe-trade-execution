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
from datetime import datetime
from typing import Any

import httpx

from src.models.lean_backtest import (
    BacktestConfig,
    BacktestDataInput,
    LEANBacktestRequest,
    LEANBacktestResponse,
)
from src.service.data_service import DataService
from src.translator.ir_translator import IRTranslator
from vibe_trade_shared.models.data import OHLCVBar

logger = logging.getLogger(__name__)

# Default endpoints
LOCAL_BACKTEST_URL = "http://localhost:8081/backtest"
CLOUD_RUN_BACKTEST_URL = os.environ.get(
    "BACKTEST_SERVICE_URL",
    "https://vibe-trade-backtest-833596808881.us-central1.run.app/backtest",
)

# Threshold for using inline vs GCS data
INLINE_DATA_THRESHOLD = 10000  # Use inline if fewer bars


@dataclass
class BacktestRequest:
    """Request to run a backtest."""

    strategy_id: str
    start_date: datetime
    end_date: datetime
    symbol: str = "BTC-USD"
    resolution: str = "1h"
    initial_cash: float = 100000.0


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
        request: BacktestRequest,
        strategy: Any,  # Strategy from vibe-trade-shared
        cards: dict[str, Any],  # Cards from vibe-trade-shared (card_id -> Card)
        inline_bars: list[OHLCVBar] | None = None,
        strategy_ir: Any | None = None,  # Pre-built StrategyIR (for testing)
    ) -> BacktestResult:
        """Run a backtest for a strategy.

        Args:
            request: Backtest request parameters
            strategy: Strategy model (ignored if strategy_ir provided)
            cards: Dict mapping card_id to Card objects (ignored if strategy_ir provided)
            inline_bars: Optional pre-fetched bars (for testing). If None, fetches via DataService.
            strategy_ir: Optional pre-built StrategyIR (for testing). If None, translates from strategy.

        Returns:
            BacktestResult with status and results
        """
        try:
            logger.info(f"Starting backtest for strategy {request.strategy_id}")

            # Step 1: Get or translate strategy IR
            if strategy_ir is not None:
                logger.info("Using provided strategy IR")
            else:
                logger.info("Translating strategy to IR...")
                translator = IRTranslator(strategy, cards)
                strategy_ir = translator.translate()

            if not strategy_ir:
                return BacktestResult(
                    status="error",
                    strategy_id=request.strategy_id,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    error="Failed to translate strategy to IR",
                )

            # Step 2: Get market data
            if inline_bars is not None:
                # Use provided bars (tests)
                bars = inline_bars
                logger.info(f"Using {len(bars)} provided inline bars")
            elif self.data_service is not None:
                # Fetch from DataService (production)
                logger.info(f"Fetching data from DataService for {request.symbol}...")
                bars = self.data_service.get_ohlcv(
                    symbol=request.symbol,
                    resolution=request.resolution,
                    start=request.start_date,
                    end=request.end_date,
                )
                logger.info(f"Fetched {len(bars)} bars")
            else:
                return BacktestResult(
                    status="error",
                    strategy_id=request.strategy_id,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    error="No data source: provide inline_bars or configure data_service",
                )

            # Step 3: Build LEAN request with inline data
            # TODO: For large datasets, upload to GCS and use gcs_uri instead
            ir_dict = strategy_ir.model_dump()
            ir_json = strategy_ir.model_dump_json(indent=2)

            lean_request = LEANBacktestRequest(
                strategy_ir=ir_dict,
                data=BacktestDataInput(
                    symbol=request.symbol,
                    resolution=request.resolution,
                    bars=bars,
                ),
                config=BacktestConfig(
                    start_date=request.start_date.date(),
                    end_date=request.end_date.date(),
                    initial_cash=request.initial_cash,
                ),
            )

            # Step 4: Call LEAN container
            logger.info(f"Calling LEAN at {self.backtest_url}")
            response = self._call_lean_endpoint(lean_request)

            if response.status == "error" or response.error:
                return BacktestResult(
                    status="error",
                    strategy_id=request.strategy_id,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    error=response.error,
                    algorithm_code=ir_json,
                    response=response,
                )

            # Build results from response
            summary = response.summary
            results = {
                "trades": [t.model_dump() for t in response.trades],
                "summary": summary.model_dump() if summary else {},
                "equity_curve": response.equity_curve,
            }

            return BacktestResult(
                status="success",
                strategy_id=request.strategy_id,
                start_date=request.start_date,
                end_date=request.end_date,
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
                strategy_id=request.strategy_id,
                start_date=request.start_date,
                end_date=request.end_date,
                error=str(e),
            )

    def _call_lean_endpoint(self, request: LEANBacktestRequest) -> LEANBacktestResponse:
        """Call LEAN HTTP endpoint with new request format.

        Args:
            request: LEANBacktestRequest with strategy_ir, data, and config

        Returns:
            LEANBacktestResponse with trades and summary
        """
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

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
        except Exception as e:
            return LEANBacktestResponse(
                status="error",
                error=f"Failed to call LEAN service: {e}",
            )

