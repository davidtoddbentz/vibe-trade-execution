"""Backtest service - orchestrates strategy backtesting.

This service:
1. Fetches strategy from Firestore
2. Translates to LEAN algorithm
3. Fetches market data from GCS
4. Exports to LEAN format
5. Runs LEAN backtest
6. Returns results
"""

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.lean_runner.engine import LeanEngine
from src.translator.ir_translator import IRTranslator

logger = logging.getLogger(__name__)


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
    """Service for running strategy backtests."""

    def __init__(
        self,
        gcs_bucket: str = "batch-save",
        lean_image: str = "quantconnect/lean:latest",
    ):
        """Initialize backtest service.

        Args:
            gcs_bucket: GCS bucket containing market data
            lean_image: Docker image for LEAN
        """
        self.gcs_bucket = gcs_bucket
        self.lean_engine = LeanEngine(lean_image=lean_image)

    def run_backtest(
        self,
        request: BacktestRequest,
        strategy: Any,  # Strategy from vibe-trade-shared
        cards: list[Any],  # Cards from vibe-trade-shared
    ) -> BacktestResult:
        """Run a backtest for a strategy.

        Args:
            request: Backtest request parameters
            strategy: Strategy model
            cards: List of cards associated with the strategy

        Returns:
            BacktestResult with status and results
        """
        try:
            # Import here to avoid circular imports and allow graceful degradation
            from vibe_trade_data import DataFetcher, LeanDataExporter

            logger.info(f"Starting backtest for strategy {request.strategy_id}")

            # Step 1: Translate strategy to LEAN algorithm
            logger.info("Translating strategy to IR...")
            translator = IRTranslator(strategy, cards)
            ir_result = translator.translate()

            if not ir_result or not ir_result.ir:
                return BacktestResult(
                    status="error",
                    strategy_id=request.strategy_id,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    error="Failed to translate strategy to IR",
                )

            # Step 2: Serialize IR to JSON
            logger.info("Serializing strategy IR...")
            ir_json = ir_result.ir.model_dump_json(indent=2)

            # Step 3: Fetch market data from GCS
            logger.info(f"Fetching market data for {request.symbol}...")
            fetcher = DataFetcher(bucket_name=self.gcs_bucket)
            candles = fetcher.fetch_candles(
                request.symbol,
                request.start_date,
                request.end_date,
            )

            if not candles:
                return BacktestResult(
                    status="error",
                    strategy_id=request.strategy_id,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    error=f"No market data found for {request.symbol} in date range",
                    algorithm_code=ir_json,
                )

            logger.info(f"Fetched {len(candles)} candles")

            # Step 4: Export to LEAN format and run backtest
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Export data
                data_dir = temp_path / "Data"
                exporter = LeanDataExporter(data_dir)
                exporter.export_candles(candles, request.symbol)

                # Copy required LEAN data files (symbol-properties, market-hours)
                self._copy_lean_data_files(data_dir)

                # Copy StrategyRuntime.py and write IR JSON
                algo_dir = temp_path / "Algorithms"
                algo_dir.mkdir(parents=True, exist_ok=True)

                # Copy the StrategyRuntime from lean/Algorithms
                runtime_src = Path(__file__).parent.parent.parent / "lean" / "Algorithms" / "StrategyRuntime.py"
                runtime_dst = algo_dir / "StrategyRuntime.py"
                shutil.copy(runtime_src, runtime_dst)

                # Write IR JSON to data directory
                ir_file = data_dir / "strategy_ir.json"
                ir_file.write_text(ir_json)

                # Results directory
                results_dir = temp_path / "Results"
                results_dir.mkdir(parents=True, exist_ok=True)

                # Step 5: Run LEAN with StrategyRuntime
                logger.info("Running LEAN backtest with StrategyRuntime...")
                result = self._run_lean(
                    algo_dir,
                    data_dir,
                    results_dir,
                    "StrategyRuntime",
                    ir_path="/Data/strategy_ir.json",
                    start_date=request.start_date,
                    end_date=request.end_date,
                    initial_cash=request.initial_cash,
                )

                if result.get("status") == "error":
                    return BacktestResult(
                        status="error",
                        strategy_id=request.strategy_id,
                        start_date=request.start_date,
                        end_date=request.end_date,
                        error=result.get("error") or result.get("stderr"),
                        algorithm_code=ir_json,
                    )

                return BacktestResult(
                    status="success",
                    strategy_id=request.strategy_id,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    results=result.get("results"),
                    algorithm_code=ir_json,
                    message=f"Backtest completed with {len(candles)} candles",
                )

        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return BacktestResult(
                status="error",
                strategy_id=request.strategy_id,
                start_date=request.start_date,
                end_date=request.end_date,
                error=f"Missing dependency: {e}. Install vibe-trade-data.",
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

    def _generate_lean_algorithm(
        self,
        ir_strategy: Any,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_cash: float,
    ) -> str:
        """Generate LEAN algorithm code from IR.

        This is a simplified version - the full implementation would use
        the IR to generate complete trading logic.
        """
        # For now, generate a basic algorithm structure
        # TODO: Use ir_strategy to generate actual trading logic
        symbol_var = symbol.replace("-", "").upper()
        csv_filename = f"{symbol.lower().replace('-', '_')}_data.csv"

        return f'''"""Auto-generated LEAN algorithm from strategy IR."""
from AlgorithmImports import *


class {symbol_var}Data(PythonData):
    """Custom data reader for {symbol}."""

    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(
            "/Data/{csv_filename}",
            SubscriptionTransportMedium.LocalFile,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line or line.startswith("datetime"):
            return None

        data = {symbol_var}Data()
        try:
            parts = line.split(",")
            data.Time = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            data.Symbol = config.Symbol
            data.Value = float(parts[4])  # close
            data["Open"] = float(parts[1])
            data["High"] = float(parts[2])
            data["Low"] = float(parts[3])
            data["Close"] = float(parts[4])
            data["Volume"] = float(parts[5])
        except Exception:
            return None
        return data


class BacktestAlgorithm(QCAlgorithm):
    """Strategy backtest algorithm."""

    def Initialize(self):
        self.SetStartDate({start_date.year}, {start_date.month}, {start_date.day})
        self.SetEndDate({end_date.year}, {end_date.month}, {end_date.day})
        self.SetCash({initial_cash})

        self.symbol = self.AddData({symbol_var}Data, "{symbol}", Resolution.Hour).Symbol
        self.SetBenchmark(lambda dt: {initial_cash})

        # TODO: Initialize indicators from IR strategy
        self.Debug("Algorithm initialized")

    def OnData(self, data):
        if not data.ContainsKey(self.symbol):
            return

        bar = data[self.symbol]
        # TODO: Implement trading logic from IR strategy
        self.Debug(f"Data: {{bar.Time}} Close={{bar.Close}}")
'''

    def _run_lean(
        self,
        algo_dir: Path,
        data_dir: Path,
        results_dir: Path,
        algorithm_name: str,
        ir_path: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        initial_cash: float | None = None,
    ) -> dict[str, Any]:
        """Run LEAN Docker container."""
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{algo_dir}:/Lean/Algorithm.Python",
            "-v",
            f"{data_dir}:/Data",
            "-v",
            f"{results_dir}:/Results",
            self.lean_engine.lean_image,
            "--algorithm-type-name",
            algorithm_name,
            "--algorithm-language",
            "Python",
            "--algorithm-location",
            f"/Lean/Algorithm.Python/{algorithm_name}.py",
            "--data-folder",
            "/Data",
            "--results-destination-folder",
            "/Results",
        ]

        # Add parameters for StrategyRuntime (LEAN expects comma-separated key=value pairs)
        params = []
        if ir_path:
            params.append(f"strategy_ir_path={ir_path}")
        if start_date:
            params.append(f"start_date={start_date.strftime('%Y%m%d')}")
        if end_date:
            params.append(f"end_date={end_date.strftime('%Y%m%d')}")
        if initial_cash:
            params.append(f"initial_cash={initial_cash}")

        if params:
            cmd.extend(["--parameters", ",".join(params)])

        logger.debug(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                return {
                    "status": "error",
                    "exit_code": result.returncode,
                    "stderr": result.stderr,
                    "stdout": result.stdout,
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

        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "Backtest timed out after 5 minutes",
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    def _copy_lean_data_files(self, data_dir: Path) -> None:
        """Copy required LEAN data files to the data directory.

        LEAN requires symbol-properties and market-hours databases
        to be present in the Data folder.
        """
        # Find the lean data directory relative to this file
        service_dir = Path(__file__).parent
        lean_data_dir = service_dir.parent.parent / "lean" / "data"

        # Copy symbol-properties
        src_symbol_props = lean_data_dir / "symbol-properties"
        if src_symbol_props.exists():
            dst_symbol_props = data_dir / "symbol-properties"
            shutil.copytree(src_symbol_props, dst_symbol_props)
            logger.debug(f"Copied symbol-properties to {dst_symbol_props}")

        # Copy market-hours
        src_market_hours = lean_data_dir / "market-hours"
        if src_market_hours.exists():
            dst_market_hours = data_dir / "market-hours"
            shutil.copytree(src_market_hours, dst_market_hours)
            logger.debug(f"Copied market-hours to {dst_market_hours}")
