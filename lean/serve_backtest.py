#!/usr/bin/env python3
"""HTTP service for running backtests on Cloud Run.

This wraps the backtest logic in a FastAPI endpoint for use with
Cloud Run Services (vs Jobs) to benefit from warm containers.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from google.cloud import storage
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vibe Trade Backtest Service")

# Configuration from environment
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "vibe-trade-475704")
RESULTS_BUCKET = os.environ.get("RESULTS_BUCKET", "vibe-trade-backtest-results")
DATA_BUCKET = os.environ.get("DATA_BUCKET", "batch-save")


def generate_backtest_id() -> str:
    """Generate a structured backtest ID: {timestamp}_{short_uuid}."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_id}"


class BacktestRequest(BaseModel):
    """Request to run a backtest."""
    strategy_id: str  # Required - links backtest to strategy
    strategy_ir: dict[str, Any]
    symbol: str = "BTC-USD"
    start_date: str  # YYYYMMDD
    end_date: str  # YYYYMMDD
    initial_cash: float = 100000.0


class BacktestSummary(BaseModel):
    """Summary statistics from backtest."""
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    total_trades: int
    win_rate: float
    profit_factor: float


class BacktestResponse(BaseModel):
    """Response from a backtest."""
    backtest_id: str
    strategy_id: str
    status: str
    created_at: str
    results_path: str | None = None
    summary: BacktestSummary | None = None
    error: str | None = None
    duration_seconds: float | None = None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """Run a backtest and return results."""
    backtest_id = generate_backtest_id()
    created_at = datetime.now(timezone.utc)
    start_time = datetime.now()

    logger.info(f"Starting backtest {backtest_id} for strategy {request.strategy_id}")

    try:
        result = await _run_backtest_internal(request, backtest_id, created_at)
        duration = (datetime.now() - start_time).total_seconds()
        result.duration_seconds = duration
        logger.info(f"Backtest {backtest_id} completed in {duration:.1f}s")
        return result
    except Exception as e:
        logger.error(f"Backtest {backtest_id} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _run_backtest_internal(
    request: BacktestRequest,
    backtest_id: str,
    created_at: datetime,
) -> BacktestResponse:
    """Internal backtest execution."""
    storage_client = storage.Client()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Setup directories
        data_dir = temp_path / "Data"
        algo_dir = temp_path / "Algorithms"
        results_dir = temp_path / "Results"

        data_dir.mkdir()
        algo_dir.mkdir()
        results_dir.mkdir()

        # Write strategy IR to file
        ir_path = data_dir / "strategy_ir.json"
        with open(ir_path, "w") as f:
            json.dump(request.strategy_ir, f, indent=2)

        # Fetch market data
        start_date = datetime.strptime(request.start_date, "%Y%m%d")
        end_date = datetime.strptime(request.end_date, "%Y%m%d")

        candle_count = _fetch_market_data(
            DATA_BUCKET, request.symbol, start_date, end_date, data_dir
        )

        if candle_count == 0:
            return BacktestResponse(
                backtest_id=backtest_id,
                strategy_id=request.strategy_id,
                status="error",
                created_at=created_at.isoformat(),
                error=f"No market data found for {request.symbol}",
            )

        # Copy LEAN data files
        _copy_lean_data_files(data_dir)

        # Copy StrategyRuntime
        runtime_src = Path("/Lean/Algorithm.Python/StrategyRuntime.py")
        if not runtime_src.exists():
            runtime_src = Path(__file__).parent / "Algorithms" / "StrategyRuntime.py"
        shutil.copy(runtime_src, algo_dir / "StrategyRuntime.py")

        # Run LEAN
        lean_result = _run_lean(
            algo_dir=algo_dir,
            data_dir=data_dir,
            results_dir=results_dir,
            ir_path=str(ir_path),
            start_date=request.start_date,
            end_date=request.end_date,
            initial_cash=request.initial_cash,
        )

        # Capture strategy output
        strategy_output = None
        strategy_output_path = data_dir / "strategy_output.json"
        if strategy_output_path.exists():
            with open(strategy_output_path, "r") as f:
                strategy_output = json.load(f)

        # Build summary from strategy output
        summary = None
        if strategy_output:
            stats = strategy_output.get("statistics", {})
            initial_cash = strategy_output.get("initial_cash", request.initial_cash)
            final_equity = strategy_output.get("final_equity", initial_cash)
            total_return = ((final_equity - initial_cash) / initial_cash) * 100 if initial_cash else 0

            summary = BacktestSummary(
                final_equity=final_equity,
                total_return_pct=round(total_return, 2),
                max_drawdown_pct=round(strategy_output.get("max_drawdown_pct", 0), 2),
                total_trades=stats.get("total_trades", 0),
                win_rate=round(stats.get("win_rate", 0), 1),
                profit_factor=round(stats.get("profit_factor", 0), 2),
            )

        # GCS path organized by strategy_id
        gcs_folder = f"backtests/{request.strategy_id}/{backtest_id}"
        results_gcs_path = f"{gcs_folder}/results.json"

        # Full result for GCS storage
        full_result = {
            "backtest_id": backtest_id,
            "strategy_id": request.strategy_id,
            "created_at": created_at.isoformat(),
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_cash": request.initial_cash,
            "candle_count": candle_count,
            "status": lean_result.get("status", "unknown"),
            "strategy_output": strategy_output,
        }

        bucket = storage_client.bucket(RESULTS_BUCKET)

        # Upload results.json
        blob = bucket.blob(results_gcs_path)
        blob.upload_from_string(json.dumps(full_result, indent=2))

        # Also upload strategy_ir.json for reference
        ir_blob = bucket.blob(f"{gcs_folder}/strategy_ir.json")
        ir_blob.upload_from_string(json.dumps(request.strategy_ir, indent=2))

        return BacktestResponse(
            backtest_id=backtest_id,
            strategy_id=request.strategy_id,
            status=lean_result.get("status", "unknown"),
            created_at=created_at.isoformat(),
            results_path=f"gs://{RESULTS_BUCKET}/{gcs_folder}/",
            summary=summary,
            error=lean_result.get("stderr") if lean_result.get("status") == "error" else None,
        )


def _fetch_market_data(
    bucket: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> int:
    """Fetch market data from GCS."""
    # Import here to avoid slow startup
    from src.data import DataFetcher, LeanDataExporter

    fetcher = DataFetcher(bucket_name=bucket)
    candles = fetcher.fetch_candles(symbol, start_date, end_date)

    if not candles:
        logger.warning(f"No candles found for {symbol}")
        return 0

    exporter = LeanDataExporter(output_dir)
    exporter.export_candles(candles, symbol)

    logger.info(f"Fetched and exported {len(candles)} candles for {symbol}")
    return len(candles)


def _copy_lean_data_files(dest_dir: Path) -> None:
    """Copy required LEAN data files."""
    lean_data = Path("/Lean/Data")

    for subdir in ["symbol-properties", "market-hours"]:
        src = lean_data / subdir
        if src.exists():
            shutil.copytree(src, dest_dir / subdir)
            logger.info(f"Copied {subdir} to {dest_dir}")


def _run_lean(
    algo_dir: Path,
    data_dir: Path,
    results_dir: Path,
    ir_path: str,
    start_date: str,
    end_date: str,
    initial_cash: float,
) -> dict:
    """Run LEAN backtest."""
    config = {
        "environment": "backtesting",
        "algorithm-type-name": "StrategyRuntime",
        "algorithm-language": "Python",
        "algorithm-location": str(algo_dir / "StrategyRuntime.py"),
        "data-folder": str(data_dir),
        "results-destination-folder": str(results_dir),
        "parameters": {
            "strategy_ir_path": ir_path,
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": str(initial_cash),
            "data_folder": str(data_dir),
        },
        "log-handler": "QuantConnect.Logging.CompositeLogHandler",
        "messaging-handler": "QuantConnect.Messaging.Messaging",
        "job-queue-handler": "QuantConnect.Queues.JobQueue",
        "api-handler": "QuantConnect.Api.Api",
        "map-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskMapFileProvider",
        "factor-file-provider": "QuantConnect.Data.Auxiliary.LocalDiskFactorFileProvider",
        "data-provider": "QuantConnect.Lean.Engine.DataFeeds.DefaultDataProvider",
        "object-store": "QuantConnect.Lean.Engine.Storage.LocalObjectStore",
        "data-aggregator": "QuantConnect.Lean.Engine.DataFeeds.AggregationManager",
    }

    config_path = algo_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Running LEAN backtest")

    cmd = [
        "dotnet",
        "/Lean/Launcher/bin/Debug/QuantConnect.Lean.Launcher.dll",
        "--config", str(config_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )

    if result.returncode != 0:
        return {
            "status": "error",
            "exit_code": result.returncode,
            "stdout": result.stdout[-5000:] if result.stdout else "",  # Truncate
            "stderr": result.stderr[-5000:] if result.stderr else "",
        }

    return {"status": "success"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
