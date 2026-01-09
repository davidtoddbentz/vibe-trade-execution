"""Export candles to LEAN-compatible format."""

import csv
import logging
from datetime import datetime
from pathlib import Path

from src.data.models import Candle

logger = logging.getLogger(__name__)


def export_to_lean_csv(
    candles: list[Candle],
    output_path: Path | str,
) -> Path:
    """Export candles to LEAN-compatible CSV format.

    LEAN expects CSV with columns: datetime,open,high,low,close,volume
    DateTime format: YYYY-MM-DD HH:MM:SS

    Args:
        candles: List of candles to export
        output_path: Path to write the CSV file

    Returns:
        Path to the written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by timestamp
    sorted_candles = sorted(candles, key=lambda c: c.timestamp)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "open", "high", "low", "close", "volume"])

        for candle in sorted_candles:
            # Format datetime without timezone for LEAN
            dt_str = candle.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow(
                [
                    dt_str,
                    f"{candle.open:.2f}",
                    f"{candle.high:.2f}",
                    f"{candle.low:.2f}",
                    f"{candle.close:.2f}",
                    f"{candle.volume:.2f}",
                ]
            )

    logger.info(f"Exported {len(sorted_candles)} candles to {output_path}")
    return output_path


def export_for_backtest(
    candles: list[Candle],
    output_dir: Path | str,
    symbol: str = "BTC-USD",
) -> dict[str, Path]:
    """Export candles organized for LEAN backtest.

    Creates the file structure LEAN expects:
    {output_dir}/{symbol}_data.csv

    Args:
        candles: List of candles to export
        output_dir: Base directory for data files
        symbol: Trading symbol

    Returns:
        Dict mapping symbol to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize symbol for filename
    symbol_normalized = symbol.lower().replace("-", "_")
    filename = f"{symbol_normalized}_data.csv"
    file_path = output_dir / filename

    export_to_lean_csv(candles, file_path)

    return {symbol: file_path}


class LeanDataExporter:
    """Exports market data for LEAN backtests."""

    def __init__(self, output_dir: Path | str):
        """Initialize exporter.

        Args:
            output_dir: Base directory for exported data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_candles(
        self,
        candles: list[Candle],
        symbol: str,
    ) -> Path:
        """Export candles for a single symbol.

        Args:
            candles: List of candles
            symbol: Trading symbol

        Returns:
            Path to exported file
        """
        result = export_for_backtest(candles, self.output_dir, symbol)
        return result[symbol]

    def get_data_path(self, symbol: str) -> Path:
        """Get the expected path for a symbol's data file.

        Args:
            symbol: Trading symbol

        Returns:
            Path where the data file would be
        """
        symbol_normalized = symbol.lower().replace("-", "_")
        return self.output_dir / f"{symbol_normalized}_data.csv"

    def get_date_range(self, candles: list[Candle]) -> tuple[datetime, datetime] | None:
        """Get the date range covered by candles.

        Args:
            candles: List of candles

        Returns:
            Tuple of (start, end) datetimes, or None if empty
        """
        if not candles:
            return None

        sorted_candles = sorted(candles, key=lambda c: c.timestamp)
        return (sorted_candles[0].timestamp, sorted_candles[-1].timestamp)
