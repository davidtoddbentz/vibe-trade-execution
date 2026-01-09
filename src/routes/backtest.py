"""Backtest routes for strategy execution."""

import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])


class BacktestMode(str, Enum):
    """Mode for running backtests."""

    LOCAL = "local"  # Run directly via Docker (for development)
    CLOUD_RUN_JOB = "cloud_run_job"  # Run as Cloud Run Job (for production)


def _default_backtest_mode() -> BacktestMode:
    """Get default backtest mode based on environment.

    When running in Cloud Run (K_SERVICE is set), use Cloud Run Jobs.
    For local development, use Docker.
    """
    if os.getenv("K_SERVICE"):
        return BacktestMode.CLOUD_RUN_JOB
    return BacktestMode.LOCAL


class BacktestRequestModel(BaseModel):
    """Request to run a backtest."""

    strategy_id: str
    start_date: datetime
    end_date: datetime
    symbol: str = "BTC-USD"
    initial_cash: float = 100000.0
    mode: BacktestMode = Field(
        default_factory=_default_backtest_mode,
        description="Execution mode: 'local' for Docker, 'cloud_run_job' for Cloud Run Jobs",
    )


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

    In LOCAL mode, runs the backtest synchronously via Docker and returns results.
    In CLOUD_RUN_JOB mode, creates a Cloud Run Job and returns immediately with pending status.
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

        if request.mode == BacktestMode.LOCAL:
            return await _run_local_backtest(backtest_id, request, strategy, cards)
        else:
            return await _run_cloud_job_backtest(backtest_id, request, strategy, cards)

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
            symbol=request.symbol,
            error=str(e),
        )


async def _run_local_backtest(
    backtest_id: str,
    request: BacktestRequestModel,
    strategy: Any,
    cards: dict[str, Any],
) -> BacktestResponseModel:
    """Run backtest locally via Docker container HTTP endpoint."""
    from src.service.backtest_service import BacktestRequest, BacktestService

    logger.info(f"Backtest {backtest_id}: Running via local backtest container")

    service = BacktestService(
        use_local=True,  # Uses localhost:8081
    )

    result = service.run_backtest(
        BacktestRequest(
            strategy_id=request.strategy_id,
            start_date=request.start_date,
            end_date=request.end_date,
            symbol=request.symbol,
            initial_cash=request.initial_cash,
        ),
        strategy,
        cards,
    )

    status = BacktestStatus.COMPLETED if result.status == "success" else BacktestStatus.FAILED

    return BacktestResponseModel(
        backtest_id=backtest_id,
        status=status,
        strategy_id=request.strategy_id,
        start_date=request.start_date,
        end_date=request.end_date,
        symbol=request.symbol,
        message=result.message,
        results=result.results,
        error=result.error,
    )


async def _run_cloud_job_backtest(
    backtest_id: str,
    request: BacktestRequestModel,
    strategy: Any,
    cards: dict[str, Any],
) -> BacktestResponseModel:
    """Run backtest as Cloud Run Job."""
    from src.service.cloud_run_job_service import CloudRunJobService

    logger.info(f"Backtest {backtest_id}: Creating Cloud Run Job")

    job_service = CloudRunJobService(
        project_id=os.getenv("GOOGLE_CLOUD_PROJECT", "vibe-trade-475704"),
        region=os.getenv("CLOUD_RUN_REGION", "us-central1"),
        lean_image=os.getenv("LEAN_IMAGE_URL"),  # Full Artifact Registry URL
        results_bucket=os.getenv("RESULTS_BUCKET", "vibe-trade-backtest-results"),
        data_bucket=os.getenv("GCS_BUCKET", "batch-save"),
    )

    # Create and submit the job
    job_result = await job_service.submit_backtest_job(
        backtest_id=backtest_id,
        strategy=strategy,
        cards=cards,
        start_date=request.start_date,
        end_date=request.end_date,
        symbol=request.symbol,
        initial_cash=request.initial_cash,
    )

    if job_result.get("status") == "error":
        return BacktestResponseModel(
            backtest_id=backtest_id,
            status=BacktestStatus.FAILED,
            strategy_id=request.strategy_id,
            start_date=request.start_date,
            end_date=request.end_date,
            symbol=request.symbol,
            error=job_result.get("error"),
        )

    return BacktestResponseModel(
        backtest_id=backtest_id,
        status=BacktestStatus.PENDING,
        strategy_id=request.strategy_id,
        start_date=request.start_date,
        end_date=request.end_date,
        symbol=request.symbol,
        message=f"Backtest job submitted: {job_result.get('job_name')}",
    )


@router.get("/{backtest_id}", response_model=BacktestResponseModel)
async def get_backtest_status(backtest_id: str) -> BacktestResponseModel:
    """Get the status of a backtest.

    For Cloud Run Jobs, checks job status and retrieves results if complete.
    """
    # TODO: Implement status checking
    # For now, return not found - we'll implement this with GCS result storage
    raise HTTPException(
        status_code=501,
        detail="Backtest status checking not yet implemented",
    )
