"""Vibe Trade Execution Service."""

# Re-export from vibe-trade-data for convenience
try:
    from vibe_trade_data import (
        Candle,
        DataFetcher,
        LeanDataExporter,
        Resolution,
        aggregate_candles,
    )

    __all__ = [
        "Candle",
        "DataFetcher",
        "LeanDataExporter",
        "Resolution",
        "aggregate_candles",
    ]
except ImportError:
    # vibe-trade-data not installed
    pass
