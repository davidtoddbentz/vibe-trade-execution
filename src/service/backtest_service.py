"""Backtest service - orchestrates strategy backtesting.

This service:
1. Translates strategy to IR
2. Calls the backtest container endpoint (local Docker or Cloud Run)
3. Returns results
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from src.translator.ir_translator import IRTranslator

logger = logging.getLogger(__name__)

# Default endpoints
LOCAL_BACKTEST_URL = "http://localhost:8081/backtest"
CLOUD_RUN_BACKTEST_URL = os.environ.get(
    "BACKTEST_SERVICE_URL",
    "https://vibe-trade-backtest-833596808881.us-central1.run.app/backtest",
)


@dataclass
class BacktestRequest:
    """Request to run a backtest."""

    strategy_id: str
    start_date: datetime
    end_date: datetime
    symbol: str = "BTC-USD"
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


class BacktestService:
    """Service for running strategy backtests via HTTP endpoint."""

    def __init__(
        self,
        backtest_url: str | None = None,
        use_local: bool = False,
        auth_token: str | None = None,
    ):
        """Initialize backtest service.

        Args:
            backtest_url: URL of the backtest service endpoint
            use_local: If True, use local Docker endpoint (localhost:8081)
            auth_token: Bearer token for Cloud Run authentication
        """
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
    ) -> BacktestResult:
        """Run a backtest for a strategy.

        Args:
            request: Backtest request parameters
            strategy: Strategy model
            cards: Dict mapping card_id to Card objects

        Returns:
            BacktestResult with status and results
        """
        try:
            logger.info(f"Starting backtest for strategy {request.strategy_id}")

            # Step 1: Translate strategy to IR
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

            # Step 2: Serialize IR to dict for HTTP request
            ir_dict = strategy_ir.model_dump()
            ir_json = strategy_ir.model_dump_json(indent=2)

            # Step 3: Call backtest service endpoint
            logger.info(f"Calling backtest service at {self.backtest_url}")
            result = self._call_backtest_endpoint(
                strategy_id=request.strategy_id,
                strategy_ir=ir_dict,
                symbol=request.symbol,
                start_date=request.start_date,
                end_date=request.end_date,
                initial_cash=request.initial_cash,
            )

            if result.get("status") == "error" or result.get("error"):
                return BacktestResult(
                    status="error",
                    strategy_id=request.strategy_id,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    error=result.get("error"),
                    algorithm_code=ir_json,
                )

            # Extract summary from response
            summary = result.get("summary", {})

            return BacktestResult(
                status="success",
                strategy_id=request.strategy_id,
                start_date=request.start_date,
                end_date=request.end_date,
                results={
                    "backtest_id": result.get("backtest_id"),
                    "results_path": result.get("results_path"),
                    "summary": summary,
                    "duration_seconds": result.get("duration_seconds"),
                },
                algorithm_code=ir_json,
                message=f"Backtest completed: {summary.get('total_trades', 0)} trades, "
                f"{summary.get('total_return_pct', 0):.2f}% return",
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

    def _call_backtest_endpoint(
        self,
        strategy_id: str,
        strategy_ir: dict[str, Any],
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_cash: float,
    ) -> dict[str, Any]:
        """Call the backtest service HTTP endpoint.

        Args:
            strategy_id: Strategy identifier
            strategy_ir: Strategy IR as dict
            symbol: Trading symbol
            start_date: Backtest start date
            end_date: Backtest end date
            initial_cash: Initial capital

        Returns:
            Response from backtest service
        """
        payload = {
            "strategy_id": strategy_id,
            "strategy_ir": strategy_ir,
            "symbol": symbol,
            "start_date": start_date.strftime("%Y%m%d"),
            "end_date": end_date.strftime("%Y%m%d"),
            "initial_cash": initial_cash,
        }

        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        try:
            with httpx.Client(timeout=600.0) as client:  # 10 minute timeout
                response = client.post(
                    self.backtest_url,
                    json=payload,
                    headers=headers,
                )

                if response.status_code != 200:
                    return {
                        "status": "error",
                        "error": f"HTTP {response.status_code}: {response.text}",
                    }

                return response.json()

        except httpx.TimeoutException:
            return {
                "status": "error",
                "error": "Backtest request timed out after 10 minutes",
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to call backtest service: {e}",
            }
