#!/usr/bin/env python3
"""Backtest runner for Cloud Run Jobs.

This script is the entrypoint for Cloud Run Jobs that run backtests.
It:
1. Downloads strategy IR from GCS
2. Fetches market data from GCS using vibe-trade-data
3. Runs LEAN with StrategyRuntime
4. Uploads results to GCS

Environment Variables:
- BACKTEST_ID: Unique backtest identifier
- STRATEGY_IR_GCS_PATH: GCS path to strategy IR JSON (gs://bucket/path)
- RESULTS_GCS_PATH: GCS path to write results (gs://bucket/path)
- DATA_BUCKET: GCS bucket with market data
- SYMBOL: Trading symbol (e.g., BTC-USD)
- START_DATE: Start date (YYYYMMDD)
- END_DATE: End date (YYYYMMDD)
- INITIAL_CASH: Initial capital
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from google.cloud import storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_gcs_path(gcs_path: str) -> tuple[str, str]:
    """Parse gs://bucket/path into (bucket, path)."""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}")
    parts = gcs_path[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def download_from_gcs(gcs_path: str, local_path: Path) -> None:
    """Download file from GCS."""
    bucket_name, blob_path = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(str(local_path))
    logger.info(f"Downloaded {gcs_path} to {local_path}")


def upload_to_gcs(local_path: Path, gcs_path: str) -> None:
    """Upload file to GCS."""
    bucket_name, blob_path = parse_gcs_path(gcs_path)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(str(local_path))
    logger.info(f"Uploaded {local_path} to {gcs_path}")


def fetch_market_data(
    bucket: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
) -> int:
    """Fetch market data using vibe-trade-data.

    Returns the number of candles fetched.
    """
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


def copy_lean_data_files(dest_dir: Path) -> None:
    """Copy required LEAN data files (symbol-properties, market-hours)."""
    # These files should be in the container at /Lean/Data
    lean_data = Path("/Lean/Data")

    for subdir in ["symbol-properties", "market-hours"]:
        src = lean_data / subdir
        if src.exists():
            shutil.copytree(src, dest_dir / subdir)
            logger.info(f"Copied {subdir} to {dest_dir}")


def run_lean(
    algo_dir: Path,
    data_dir: Path,
    results_dir: Path,
    ir_path: str,
    start_date: str,
    end_date: str,
    initial_cash: float,
) -> dict:
    """Run LEAN backtest."""
    # Create LEAN config file with parameters
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
        # Standard handlers
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

    logger.info(f"Created LEAN config at {config_path}")

    cmd = [
        "dotnet",
        "/Lean/Launcher/bin/Debug/QuantConnect.Lean.Launcher.dll",
        "--config", str(config_path),
    ]

    logger.info(f"Running LEAN: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,  # 1 hour timeout
    )

    if result.returncode != 0:
        return {
            "status": "error",
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    # Try to parse results
    results_file = results_dir / "backtest-results.json"
    if results_file.exists():
        with open(results_file) as f:
            return {
                "status": "success",
                "results": json.load(f),
            }

    return {
        "status": "success",
        "stdout": result.stdout,
    }


def main():
    """Main entry point."""
    # Read environment variables
    backtest_id = os.environ.get("BACKTEST_ID")
    ir_gcs_path = os.environ.get("STRATEGY_IR_GCS_PATH")
    results_gcs_path = os.environ.get("RESULTS_GCS_PATH")
    data_bucket = os.environ.get("DATA_BUCKET", "batch-save")
    symbol = os.environ.get("SYMBOL", "BTC-USD")
    start_date_str = os.environ.get("START_DATE", "20240101")
    end_date_str = os.environ.get("END_DATE", "20241231")
    initial_cash = float(os.environ.get("INITIAL_CASH", "100000"))

    if not all([backtest_id, ir_gcs_path, results_gcs_path]):
        logger.error("Missing required environment variables")
        return 1

    logger.info(f"Starting backtest {backtest_id}")
    logger.info(f"Symbol: {symbol}, Date range: {start_date_str} - {end_date_str}")

    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Setup directories
        data_dir = temp_path / "Data"
        algo_dir = temp_path / "Algorithms"
        results_dir = temp_path / "Results"

        data_dir.mkdir()
        algo_dir.mkdir()
        results_dir.mkdir()

        # Download strategy IR
        ir_local = data_dir / "strategy_ir.json"
        download_from_gcs(ir_gcs_path, ir_local)

        # Fetch market data
        candle_count = fetch_market_data(
            data_bucket, symbol, start_date, end_date, data_dir
        )

        if candle_count == 0:
            result = {
                "status": "error",
                "error": f"No market data found for {symbol}",
                "backtest_id": backtest_id,
            }
        else:
            # Copy LEAN data files
            copy_lean_data_files(data_dir)

            # Copy StrategyRuntime
            runtime_src = Path("/Lean/Algorithm.Python/StrategyRuntime.py")
            if not runtime_src.exists():
                # Fallback to local path in container
                runtime_src = Path(__file__).parent / "Algorithms" / "StrategyRuntime.py"
            shutil.copy(runtime_src, algo_dir / "StrategyRuntime.py")

            # Run LEAN
            lean_result = run_lean(
                algo_dir=algo_dir,
                data_dir=data_dir,
                results_dir=results_dir,
                ir_path=str(ir_local),
                start_date=start_date_str,
                end_date=end_date_str,
                initial_cash=initial_cash,
            )

            result = {
                "backtest_id": backtest_id,
                "symbol": symbol,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "candle_count": candle_count,
                **lean_result,
            }

            # Capture debug log if it exists
            debug_log_path = data_dir / "debug.log"
            if debug_log_path.exists():
                with open(debug_log_path, "r") as f:
                    result["debug_log"] = f.read()
                logger.info(f"Captured debug log from {debug_log_path}")

            # Capture strategy output from StrategyRuntime
            strategy_output_path = data_dir / "strategy_output.json"
            if strategy_output_path.exists():
                with open(strategy_output_path, "r") as f:
                    result["strategy_output"] = json.load(f)
                logger.info(f"Captured strategy output from {strategy_output_path}")

        # Write and upload results
        results_local = temp_path / "results.json"
        with open(results_local, "w") as f:
            json.dump(result, f, indent=2)

        upload_to_gcs(results_local, results_gcs_path)

        logger.info(f"Backtest {backtest_id} completed with status: {result['status']}")

        return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    exit(main())
