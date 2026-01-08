"""
EMA Crossover Strategy with Custom CSV Data.

Uses LEAN's custom data capability to load our CSV file directly.
"""

from AlgorithmImports import *
from datetime import datetime
import os


class BtcCsvData(PythonData):
    """Custom data type for reading our BTC-USD CSV file."""

    def GetSource(self, config, date, isLiveMode):
        """Return the data source (our CSV file)."""
        # In backtest mode, read from local file
        # The path should be relative to the Data folder mounted in Docker
        return SubscriptionDataSource(
            "/Data/btc_usd_hourly.csv",
            SubscriptionTransportMedium.LocalFile,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        """Parse each line of CSV."""
        if not line.strip():
            return None

        # Skip header
        if line.startswith("datetime"):
            return None

        data = BtcCsvData()
        data.Symbol = config.Symbol

        try:
            parts = line.split(',')
            # datetime,open,high,low,close,volume
            dt = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            data.Time = dt
            data.Value = float(parts[4])  # close price
            data["Open"] = float(parts[1])
            data["High"] = float(parts[2])
            data["Low"] = float(parts[3])
            data["Close"] = float(parts[4])
            data["Volume"] = float(parts[5])
        except Exception as e:
            return None

        return data


class EmaCrossoverCsvBacktest(QCAlgorithm):
    """EMA Crossover strategy using custom CSV data."""

    def Initialize(self):
        """Initialize the algorithm."""
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 6, 30)
        self.SetCash(100000)

        # Add custom data
        self.symbol = self.AddData(BtcCsvData, "BTCUSD", Resolution.Hour).Symbol

        # Manual indicator tracking (custom data doesn't auto-register)
        self.ema_fast_period = 20
        self.ema_slow_period = 50
        self.ema_fast = ExponentialMovingAverage(self.ema_fast_period)
        self.ema_slow = ExponentialMovingAverage(self.ema_slow_period)

        # State
        self.entry_triggered = False
        self.bars_since_entry = 0

        self.Log(f"Strategy initialized: EMA Crossover CSV Backtest")
        self.Log(f"Symbol: {self.symbol}")

    def OnData(self, data):
        """Process new data."""
        if self.symbol not in data:
            return

        bar = data[self.symbol]
        if bar is None:
            return

        # Update indicators manually
        self.ema_fast.Update(bar.Time, bar.Close)
        self.ema_slow.Update(bar.Time, bar.Close)

        # Wait for indicators to be ready
        if not self.ema_fast.IsReady or not self.ema_slow.IsReady:
            return

        fast_val = self.ema_fast.Current.Value
        slow_val = self.ema_slow.Current.Value

        # Check exits first (if in position)
        if self.Portfolio[self.symbol].Invested:
            self.bars_since_entry += 1
            # Exit: fast EMA below slow
            if fast_val < slow_val:
                self.Liquidate(self.symbol)
                self.Log(f"EXIT: Fast EMA ({fast_val:.2f}) < Slow EMA ({slow_val:.2f})")

        # Check entries (if not in position)
        if not self.Portfolio[self.symbol].Invested:
            # Entry: fast EMA above slow
            if fast_val > slow_val:
                self.SetHoldings(self.symbol, 0.95)
                self.Log(f"ENTRY: Fast EMA ({fast_val:.2f}) > Slow EMA ({slow_val:.2f})")
                self.entry_triggered = True
                self.bars_since_entry = 0

    def OnOrderEvent(self, orderEvent):
        """Handle order events."""
        if orderEvent.Status == OrderStatus.Filled:
            direction = "BUY" if orderEvent.FillQuantity > 0 else "SELL"
            self.Log(f"{direction} @ ${orderEvent.FillPrice:.2f}")

    def OnEndOfAlgorithm(self):
        """Algorithm ended."""
        self.Log(f"Final portfolio value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
