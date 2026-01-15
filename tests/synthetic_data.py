"""Synthetic data generator for LEAN backtesting.

Generates deterministic OHLCV data with known signal patterns,
writes in LEAN-compatible format for end-to-end testing.

Usage:
    from tests.synthetic_data import SyntheticDataWriter, generate_uptrend_with_pullback

    writer = SyntheticDataWriter("/Data")
    candles, expected = generate_uptrend_with_pullback(pullback_bar=60)
    writer.write(candles, symbol="BTCUSD", start_date=datetime(2024, 1, 1))

    # Run LEAN backtest...
    # Verify trades at expected bars
"""

import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class ExpectedSignal:
    """Expected signal from synthetic data."""
    bar: int
    signal_type: str  # "entry" or "exit"
    direction: str  # "long" or "short"
    reason: str  # Human-readable reason


@dataclass
class SyntheticScenario:
    """Complete synthetic test scenario."""
    name: str
    candles: list[dict]  # OHLCV candles with timestamps
    expected_signals: list[ExpectedSignal]
    description: str


class SyntheticDataWriter:
    """Write synthetic data in LEAN-compatible format."""

    def __init__(self, data_dir: str = "/Data"):
        self.data_dir = Path(data_dir)

    def write(
        self,
        candles: list[dict],
        symbol: str,
        start_date: datetime,
        market: str = "coinbase",
        resolution: str = "minute",
    ) -> Path:
        """Write candles to LEAN-compatible ZIP files.

        Args:
            candles: List of dicts with open, high, low, close, volume
            symbol: Trading symbol (e.g., "BTCUSD")
            start_date: Start date for timestamps
            market: Exchange name
            resolution: Data resolution

        Returns:
            Path to the data directory
        """
        # Add timestamps to candles (1 minute intervals)
        timestamped = []
        current_time = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        for candle in candles:
            timestamped.append({
                "timestamp": current_time,
                "open": candle["open"],
                "high": candle["high"],
                "low": candle["low"],
                "close": candle["close"],
                "volume": candle.get("volume", 1000.0),
            })
            current_time += timedelta(minutes=1)

        # Group by date and write
        normalized_symbol = symbol.lower().replace("-", "").replace("/", "")
        base_path = self.data_dir / "crypto" / market / resolution / normalized_symbol
        base_path.mkdir(parents=True, exist_ok=True)

        # Group candles by date
        grouped: dict[str, list[dict]] = {}
        for candle in timestamped:
            date_str = candle["timestamp"].strftime("%Y%m%d")
            if date_str not in grouped:
                grouped[date_str] = []
            grouped[date_str].append(candle)

        # Write each day's data
        for date_str, day_candles in grouped.items():
            day_candles.sort(key=lambda c: c["timestamp"])
            csv_lines = [self._candle_to_csv(c) for c in day_candles]
            csv_content = "\n".join(csv_lines)

            zip_path = base_path / f"{date_str}_trade.zip"
            csv_filename = f"{date_str}_trade.csv"

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(csv_filename, csv_content)

        return base_path

    def _candle_to_csv(self, candle: dict) -> str:
        """Convert candle to LEAN CSV format."""
        ts = candle["timestamp"]
        midnight = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        ms_since_midnight = int((ts - midnight).total_seconds() * 1000)

        return (
            f"{ms_since_midnight},"
            f"{candle['open']},"
            f"{candle['high']},"
            f"{candle['low']},"
            f"{candle['close']},"
            f"{candle['volume']}"
        )

    def cleanup(self, symbol: str, market: str = "coinbase", resolution: str = "minute"):
        """Remove synthetic data files."""
        normalized_symbol = symbol.lower().replace("-", "").replace("/", "")
        path = self.data_dir / "crypto" / market / resolution / normalized_symbol
        if path.exists():
            for f in path.glob("*.zip"):
                f.unlink()
            path.rmdir()


# =============================================================================
# Synthetic Data Generators
# =============================================================================


def make_ohlcv(closes: list[float], spread_pct: float = 0.005) -> list[dict]:
    """Create OHLCV candles from explicit close prices.

    Args:
        closes: List of close prices
        spread_pct: Spread around close for high/low

    Returns:
        List of OHLCV dicts
    """
    candles = []
    prev_close = closes[0]

    for close in closes:
        spread = close * spread_pct
        open_price = prev_close
        high = max(open_price, close) + spread
        low = min(open_price, close) - spread

        candles.append({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0,
        })
        prev_close = close

    return candles


def generate_uptrend_with_pullback(
    n_bars: int = 200,
    pullback_bar: int = 100,
    base_price: float = 50000.0,
) -> SyntheticScenario:
    """Generate uptrend with pullback for TrendPullback testing.

    Creates:
    - Steady uptrend (EMA20 > EMA50)
    - Sharp pullback at pullback_bar that touches BB lower
    - Recovery and continued uptrend

    The pullback is designed to trigger TrendPullback entry signals.

    Returns:
        SyntheticScenario with candles and expected signals
    """
    closes = []
    price = base_price

    for i in range(n_bars):
        if i < pullback_bar:
            # Steady uptrend: +0.3% per bar
            price = base_price * (1.003 ** i)
        elif i < pullback_bar + 5:
            # Sharp pullback: -2% per bar for 5 bars
            bars_into_pullback = i - pullback_bar
            peak = base_price * (1.003 ** pullback_bar)
            price = peak * (0.98 ** (bars_into_pullback + 1))
        else:
            # Resume uptrend from pullback low
            pullback_low = base_price * (1.003 ** pullback_bar) * (0.98 ** 5)
            bars_since_low = i - (pullback_bar + 5)
            price = pullback_low * (1.003 ** bars_since_low)

        closes.append(price)

    candles = make_ohlcv(closes)

    # Expected signals: during the pullback recovery
    # TrendPullback triggers when EMA20 > EMA50 AND close touches BB lower
    # This happens during bars pullback_bar to pullback_bar+4 approximately
    expected = [
        ExpectedSignal(
            bar=pullback_bar + 2,  # Middle of pullback
            signal_type="entry",
            direction="long",
            reason="EMA20 > EMA50 AND close touches BB lower",
        )
    ]

    return SyntheticScenario(
        name="uptrend_with_pullback",
        candles=candles,
        expected_signals=expected,
        description=f"Uptrend with pullback at bar {pullback_bar}. "
                    f"Expect TrendPullback entry during pullback recovery.",
    )


def generate_breakout(
    n_bars: int = 200,
    consolidation_bars: int = 100,
    base_price: float = 50000.0,
) -> SyntheticScenario:
    """Generate consolidation followed by breakout.

    Creates:
    - Sideways consolidation (price oscillates in range)
    - Sharp breakout above Donchian channel upper
    - Continued trend after breakout

    Returns:
        SyntheticScenario with candles and expected signals
    """
    closes = []

    for i in range(n_bars):
        if i < consolidation_bars:
            # Oscillate in 2% range around base price
            import math
            price = base_price + base_price * 0.01 * math.sin(i * 0.3)
        else:
            # Breakout: jump 5% and trend up
            bars_since_breakout = i - consolidation_bars
            price = base_price * 1.05 + bars_since_breakout * base_price * 0.002

        closes.append(price)

    candles = make_ohlcv(closes)

    expected = [
        ExpectedSignal(
            bar=consolidation_bars,
            signal_type="entry",
            direction="long",
            reason="Close breaks above Donchian channel upper",
        )
    ]

    return SyntheticScenario(
        name="breakout",
        candles=candles,
        expected_signals=expected,
        description=f"Consolidation for {consolidation_bars} bars then breakout. "
                    f"Expect BreakoutTrendfollow entry at bar {consolidation_bars}.",
    )


def generate_mean_reversion(
    n_bars: int = 200,
    oversold_bar: int = 100,
    base_price: float = 50000.0,
) -> SyntheticScenario:
    """Generate mean reversion setup.

    Creates:
    - Normal price action
    - Sharp drop to oversold (below BB lower)
    - Reversion back to mean

    Returns:
        SyntheticScenario with candles and expected signals
    """
    closes = []
    price = base_price

    for i in range(n_bars):
        if i < oversold_bar:
            # Normal fluctuation
            import math
            price = base_price + base_price * 0.005 * math.sin(i * 0.2)
        elif i < oversold_bar + 5:
            # Sharp drop
            bars_into_drop = i - oversold_bar
            price = base_price * (0.97 ** (bars_into_drop + 1))
        else:
            # Reversion to mean
            bars_since_bottom = i - (oversold_bar + 5)
            bottom = base_price * (0.97 ** 5)
            price = bottom + (base_price - bottom) * min(1.0, bars_since_bottom / 20)

        closes.append(price)

    candles = make_ohlcv(closes)

    expected = [
        ExpectedSignal(
            bar=oversold_bar + 3,
            signal_type="entry",
            direction="long",
            reason="Price below BB lower, expecting reversion",
        )
    ]

    return SyntheticScenario(
        name="mean_reversion",
        candles=candles,
        expected_signals=expected,
        description=f"Drop to oversold at bar {oversold_bar}, then reversion. "
                    f"Expect RangeMeanReversion entry during oversold.",
    )
