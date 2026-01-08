"""
Deterministic EMA Crossover Test.

Uses short EMA periods (5/10) on predictable data to verify trading logic.

Data pattern:
- Hours 0-49: flat at $100 (EMAs converge to 100)
- Hours 50-59: ramp up $100→$200 (+$10/hour) → EXPECT BUY (fast EMA > slow)
- Hours 60-97: flat at $200 (EMAs converge to 200)
- Hours 98-107: ramp down $200→$100 (-$10/hour) → EXPECT SELL (fast EMA < slow)
- Hours 108-120: flat at $100

Expected trades:
1. BUY during Jan 3 uptrend (around hour 52-55 when fast crosses above slow)
2. SELL during Jan 5 downtrend (around hour 100-103 when fast crosses below slow)

Final P&L should be ~+100% (bought ~$100, sold ~$200)
"""

from AlgorithmImports import *
from datetime import datetime


class BtcDeterministicData(PythonData):
    """Custom data for deterministic test."""

    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(
            "/Data/btc_deterministic.csv",
            SubscriptionTransportMedium.LocalFile,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line.strip() or line.startswith("datetime"):
            return None

        data = BtcDeterministicData()
        data.Symbol = config.Symbol

        try:
            parts = line.split(',')
            dt = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            data.Time = dt
            data.Value = float(parts[4])
            data["Open"] = float(parts[1])
            data["High"] = float(parts[2])
            data["Low"] = float(parts[3])
            data["Close"] = float(parts[4])
            data["Volume"] = float(parts[5])
        except:
            return None

        return data


class EmaDeterministicTest(QCAlgorithm):
    """Test EMA crossover with deterministic data."""

    def Initialize(self):
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 1, 6)
        self.SetCash(100000)

        self.symbol = self.AddData(BtcDeterministicData, "BTCTEST", Resolution.Hour).Symbol

        # Short EMAs for quick response
        self.ema_fast = ExponentialMovingAverage(5)
        self.ema_slow = ExponentialMovingAverage(10)

        self.bar_count = 0
        self.Log("=== DETERMINISTIC EMA TEST ===")
        self.Log("EMA periods: fast=5, slow=10")
        self.Log("Expected: BUY during uptrend (Jan 3), SELL during downtrend (Jan 5)")

    def OnData(self, data):
        if self.symbol not in data:
            return

        bar = data[self.symbol]
        if bar is None:
            return

        self.bar_count += 1
        price = bar.Close

        # Update EMAs
        self.ema_fast.Update(bar.Time, price)
        self.ema_slow.Update(bar.Time, price)

        if not self.ema_fast.IsReady or not self.ema_slow.IsReady:
            return

        fast = self.ema_fast.Current.Value
        slow = self.ema_slow.Current.Value

        # Log every bar for debugging
        invested = "LONG" if self.Portfolio[self.symbol].Invested else "FLAT"
        self.Log(f"Bar {self.bar_count} | {bar.Time} | Price: ${price:.2f} | Fast: ${fast:.2f} | Slow: ${slow:.2f} | {invested}")

        # Trading logic
        if self.Portfolio[self.symbol].Invested:
            if fast < slow:
                self.Liquidate(self.symbol)
                self.Log(f">>> SELL SIGNAL: Fast ({fast:.2f}) < Slow ({slow:.2f})")
        else:
            if fast > slow:
                self.SetHoldings(self.symbol, 0.95)
                self.Log(f">>> BUY SIGNAL: Fast ({fast:.2f}) > Slow ({slow:.2f})")

    def OnOrderEvent(self, orderEvent):
        if orderEvent.Status == OrderStatus.Filled:
            direction = "BUY" if orderEvent.FillQuantity > 0 else "SELL"
            self.Log(f"*** ORDER FILLED: {direction} {abs(orderEvent.FillQuantity):.2f} @ ${orderEvent.FillPrice:.2f}")

    def OnEndOfAlgorithm(self):
        self.Log("=== FINAL RESULTS ===")
        self.Log(f"Total bars processed: {self.bar_count}")
        self.Log(f"Final portfolio value: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Log(f"Net profit: ${self.Portfolio.TotalPortfolioValue - 100000:,.2f}")
        pct = (self.Portfolio.TotalPortfolioValue / 100000 - 1) * 100
        self.Log(f"Return: {pct:.1f}%")
