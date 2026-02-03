"""Backtest routes for strategy execution."""

import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _get_id_token_for_service(target_audience: str) -> str | None:
    """Get an ID token for authenticating to another Cloud Run service.

    Uses the metadata server when running on GCP, returns None locally.

    Args:
        target_audience: The URL of the target service

    Returns:
        ID token string, or None if not running on GCP
    """
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        request = google.auth.transport.requests.Request()
        token = google.oauth2.id_token.fetch_id_token(request, target_audience)
        return token
    except Exception as e:
        logger.debug(f"Could not get ID token (expected locally): {e}")
        return None

router = APIRouter(prefix="/backtest", tags=["backtest"])


def _get_firestore_client():
    """Get Firestore client (cached)."""
    from vibe_trade_shared import FirestoreClient

    return FirestoreClient.get_client(
        project=os.getenv("GOOGLE_CLOUD_PROJECT", "vibe-trade-475704"),
        database=os.getenv("FIRESTORE_DATABASE", "strategy"),
    )


def _extract_symbol_from_cards(cards: dict[str, Any]) -> str:
    """Extract symbol from entry card's context.

    Looks for an entry card and extracts the symbol from its context slot.
    Raises ValueError if no symbol found - symbol is required.
    """
    for card in cards.values():
        card_type = card.type if hasattr(card, "type") else card.get("type", "")
        if card_type.startswith("entry."):
            slots = card.slots if hasattr(card, "slots") else card.get("slots", {})
            context = slots.get("context", {})
            if symbol := context.get("symbol"):
                return symbol
    raise ValueError("No symbol found in entry card context - symbol is required")


def _extract_timeframe_from_cards(cards: dict[str, Any]) -> str:
    """Extract timeframe from entry card's context.

    Looks for an entry card and extracts the timeframe (tf) from its context slot.
    Falls back to 1h if no timeframe found.
    """
    for card in cards.values():
        card_type = card.type if hasattr(card, "type") else card.get("type", "")
        if card_type.startswith("entry."):
            slots = card.slots if hasattr(card, "slots") else card.get("slots", {})
            context = slots.get("context", {})
            if tf := context.get("tf"):
                return tf
    return "1h"  # Default fallback


class BacktestRequestModel(BaseModel):
    """Request to run a backtest.

    Symbol is extracted from the strategy's entry card context.
    Execution mode is determined by the environment (local vs Cloud Run).
    """

    strategy_id: str
    start_date: datetime
    end_date: datetime
    initial_cash: float = 100000.0
    # Trading costs
    fee_pct: float = 0.0  # Fee as percentage of trade value (0.1 = 0.1%)
    slippage_pct: float = 0.0  # Slippage as percentage of price (0.05 = 0.05%)


class BacktestStatus(str, Enum):
    """Status of a backtest."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class BacktestResponseModel(BaseModel):
    """Response from backtest submission."""

    backtest_id: str
    status: BacktestStatus
    strategy_id: str
    start_date: datetime
    end_date: datetime
    symbol: str
    message: str | None = None
    results: dict[str, Any] | None = None
    error: str | None = None


@router.post("", response_model=BacktestResponseModel)
async def run_backtest(request: BacktestRequestModel) -> BacktestResponseModel:
    """Run a backtest for a strategy.

    Execution mode is determined by environment:
    - LOCAL: runs synchronously via Docker, returns results immediately
    - CLOUD_RUN_JOB: creates async Cloud Run Job, returns pending status

    Symbol is extracted from the strategy's entry card context.
    """
    backtest_id = str(uuid.uuid4())
    logger.info(f"Backtest {backtest_id}: Starting for strategy {request.strategy_id}")

    try:
        # Fetch strategy and cards from Firestore
        from vibe_trade_shared import CardRepository, FirestoreClient, StrategyRepository

        client = FirestoreClient.get_client(
            project=os.getenv("GOOGLE_CLOUD_PROJECT", "vibe-trade-475704"),
            database=os.getenv("FIRESTORE_DATABASE", "strategy"),
        )
        strategy_repo = StrategyRepository(client)
        card_repo = CardRepository(client)

        strategy = strategy_repo.get_by_id(request.strategy_id)
        if not strategy:
            raise HTTPException(
                status_code=404,
                detail=f"Strategy not found: {request.strategy_id}",
            )

        # Get cards linked to this strategy (as dict mapping card_id -> Card)
        card_ids = [att.card_id for att in (strategy.attachments or [])]
        cards_list = [card_repo.get_by_id(cid) for cid in card_ids]
        cards = {c.id: c for c in cards_list if c is not None}

        # Extract symbol and timeframe from entry card context
        symbol = _extract_symbol_from_cards(cards)
        timeframe = _extract_timeframe_from_cards(cards)
        logger.info(f"Backtest {backtest_id}: Using symbol {symbol}, timeframe {timeframe} from strategy cards")

        return await _run_backtest(backtest_id, request, strategy, cards, symbol, timeframe)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest {backtest_id}: Failed - {e}", exc_info=True)
        return BacktestResponseModel(
            backtest_id=backtest_id,
            status=BacktestStatus.FAILED,
            strategy_id=request.strategy_id,
            start_date=request.start_date,
            end_date=request.end_date,
            symbol="unknown",
            error=str(e),
        )


async def _run_backtest(
    backtest_id: str,
    request: BacktestRequestModel,
    strategy: Any,
    cards: dict[str, Any],
    symbol: str,
    resolution: str = "1h",
) -> BacktestResponseModel:
    """Run backtest via HTTP endpoint.

    Works for both local development (Docker at localhost:8083) and production
    (Cloud Run Service). Configuration is entirely environment-based:
    - BACKTEST_SERVICE_URL: URL of LEAN backtest service (default: localhost:8083)
    - BIGQUERY_EMULATOR_HOST: Optional BigQuery emulator for local dev
    - GOOGLE_CLOUD_PROJECT: GCP project for BigQuery
    """
    from vibe_trade_shared import Backtest, BacktestRepository

    from src.models.lean_backtest import BacktestConfig
    from src.service.backtest_service import BacktestService
    from src.service.bigquery_data_service import BigQueryDataService

    # Determine backtest service URL from environment
    # In Cloud Run (K_SERVICE set), use the deployed backtest service
    # Locally, use localhost:8083 (Docker container)
    backtest_url = os.getenv("BACKTEST_SERVICE_URL")
    if not backtest_url:
        backtest_url = "http://localhost:8083/backtest"

    logger.info(f"Backtest {backtest_id}: Running via {backtest_url}")

    # Configure BigQuery data service
    emulator_host = os.getenv("BIGQUERY_EMULATOR_HOST")
    if emulator_host:
        logger.info(f"Using BigQuery emulator at {emulator_host}")

    data_service = BigQueryDataService(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT", "test-project"),
        emulator_host=emulator_host,
    )

    # Get auth token for Cloud Run service-to-service calls
    auth_token = None
    if backtest_url and backtest_url.startswith("https://"):
        # Extract the base URL (without path) for the audience
        from urllib.parse import urlparse

        parsed = urlparse(backtest_url)
        audience = f"{parsed.scheme}://{parsed.netloc}"
        auth_token = _get_id_token_for_service(audience)
        if auth_token:
            logger.info(f"Backtest {backtest_id}: Got auth token for {audience}")

    service = BacktestService(
        data_service=data_service,
        backtest_url=backtest_url,
        auth_token=auth_token,
    )

    result = service.run_backtest(
        strategy=strategy,
        cards=cards,
        config=BacktestConfig(
            start_date=request.start_date.date(),
            end_date=request.end_date.date(),
            symbol=symbol,
            resolution=resolution,
            initial_cash=request.initial_cash,
            fee_pct=request.fee_pct,
            slippage_pct=request.slippage_pct,
        ),
    )

    status = BacktestStatus.COMPLETED if result.status == "success" else BacktestStatus.FAILED

    # Save backtest result to Firestore
    persistence_error = None
    try:
        client = _get_firestore_client()
        backtest_repo = BacktestRepository(client)

        backtest = Backtest(
            id=backtest_id,
            strategy_id=request.strategy_id,
            owner_id=strategy.owner_id,
            status=status.value,
            symbol=symbol,
            start_date=request.start_date.isoformat(),
            end_date=request.end_date.isoformat(),
            initial_cash=request.initial_cash,
            results=result.results,
            message=result.message,
            error=result.error,
            created_at=Backtest.now_iso(),
            completed_at=Backtest.now_iso() if status == BacktestStatus.COMPLETED else None,
        )
        backtest_repo.create(backtest)
        logger.info(f"Backtest {backtest_id}: Saved to Firestore")
    except Exception as e:
        persistence_error = str(e)
        logger.error(f"Backtest {backtest_id}: Failed to save to Firestore - {e}", exc_info=True)
        # Don't fail the request if Firestore save fails, but track the error

    # Include persistence error in message if present
    message = result.message
    if persistence_error:
        warning = f"Warning: Failed to save to history - {persistence_error}"
        message = f"{message}. {warning}" if message else warning

    return BacktestResponseModel(
        backtest_id=backtest_id,
        status=status,
        strategy_id=request.strategy_id,
        start_date=request.start_date,
        end_date=request.end_date,
        symbol=symbol,
        message=message,
        results=result.results,
        error=result.error,
    )


class BacktestListItem(BaseModel):
    """Summary of a backtest for listing."""

    backtest_id: str
    status: str
    strategy_id: str
    symbol: str
    start_date: str
    end_date: str
    initial_cash: float
    total_return: float | None = None
    total_trades: int | None = None
    message: str | None = None
    error: str | None = None
    created_at: str


class BacktestListResponse(BaseModel):
    """Response for listing backtests."""

    backtests: list[BacktestListItem]
    total: int


@router.get("/strategy/{strategy_id}", response_model=BacktestListResponse)
async def list_backtests_for_strategy(
    strategy_id: str,
    limit: int = Query(default=20, ge=1, le=100),
) -> BacktestListResponse:
    """List backtests for a specific strategy.

    Returns backtests ordered by creation time (most recent first).
    """
    from vibe_trade_shared import BacktestRepository

    try:
        client = _get_firestore_client()
        backtest_repo = BacktestRepository(client)

        backtests = backtest_repo.get_by_strategy_id(strategy_id, limit=limit)

        items = []
        for bt in backtests:
            # Extract summary statistics if available
            total_return = None
            total_trades = None
            if bt.results:
                # Handle both dict and BacktestResults object
                if isinstance(bt.results, dict):
                    stats = bt.results.get("statistics", {})
                    if isinstance(stats, dict):
                        total_return = stats.get("total_return")
                        total_trades = stats.get("total_trades")
                elif hasattr(bt.results, "statistics"):
                    # It's a BacktestResults object
                    stats = bt.results.statistics
                    if stats:
                        total_return = stats.total_return
                        total_trades = stats.total_trades

            items.append(
                BacktestListItem(
                    backtest_id=bt.id,
                    status=bt.status,
                    strategy_id=bt.strategy_id,
                    symbol=bt.symbol,
                    start_date=bt.start_date,
                    end_date=bt.end_date,
                    initial_cash=bt.initial_cash,
                    total_return=total_return,
                    total_trades=total_trades,
                    message=bt.message,
                    error=bt.error,
                    created_at=bt.created_at,
                )
            )

        return BacktestListResponse(backtests=items, total=len(items))

    except Exception as e:
        logger.error(f"Failed to list backtests for strategy {strategy_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list backtests: {str(e)}",
        )


@router.get("/{backtest_id}", response_model=BacktestResponseModel)
async def get_backtest_status(backtest_id: str) -> BacktestResponseModel:
    """Get the status and results of a backtest."""
    from vibe_trade_shared import BacktestRepository

    try:
        client = _get_firestore_client()
        backtest_repo = BacktestRepository(client)

        backtest = backtest_repo.get_by_id(backtest_id)
        if not backtest:
            raise HTTPException(
                status_code=404,
                detail=f"Backtest not found: {backtest_id}",
            )

        # Convert results dict back if it's stored as dict
        results = None
        if backtest.results:
            if isinstance(backtest.results, dict):
                results = backtest.results
            else:
                # It's a BacktestResults object, convert to dict
                results = {
                    "trades": [t.model_dump() for t in backtest.results.trades],
                    "statistics": backtest.results.statistics.model_dump(),
                    "equity_curve": backtest.results.equity_curve,
                }
                if getattr(backtest.results, "ohlcv_bars", None):
                    results["ohlcv_bars"] = backtest.results.ohlcv_bars
                if getattr(backtest.results, "indicators", None):
                    results["indicators"] = backtest.results.indicators

        return BacktestResponseModel(
            backtest_id=backtest.id,
            status=BacktestStatus(backtest.status),
            strategy_id=backtest.strategy_id,
            start_date=datetime.fromisoformat(backtest.start_date.replace("Z", "+00:00")),
            end_date=datetime.fromisoformat(backtest.end_date.replace("Z", "+00:00")),
            symbol=backtest.symbol,
            message=backtest.message,
            results=results,
            error=backtest.error,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get backtest {backtest_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get backtest: {str(e)}",
        )
