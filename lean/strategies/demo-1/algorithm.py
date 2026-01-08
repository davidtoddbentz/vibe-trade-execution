"""
Demo Strategy: Simple Moving Average Crossover with Pub/Sub Data.

This strategy demonstrates:
- Connecting to Pub/Sub data feed
- Processing real-time candles
- Making trading decisions
- Logging activity
"""

from AlgorithmImports import *
import sys
import os

# Add data feed to path (use main DataFeeds directory)
sys.path.insert(0, "/Lean/DataFeeds")
from PubSubCandle import PubSubCandle


class DemoStrategy(QCAlgorithm):
    """Demo strategy using Pub/Sub data feed."""
    
    def Initialize(self):
        """Initialize the algorithm."""
        # For live mode, don't set dates - it runs continuously
        self.SetCash(100000)
        
        # Log that we're in live mode
        if self.LiveMode:
            self.Log("🌐 Running in LIVE MODE - waiting for Pub/Sub data...")
        else:
            self.Log("📊 Running in BACKTEST MODE")
        
        # Subscribe to Pub/Sub data feed
        self.symbol = self.AddData(
            PubSubCandle,
            "BTC-USD",
            Resolution.Minute
        ).Symbol
        
        # Create indicators
        self.fast_ma = self.EMA(self.symbol, 10, Resolution.Minute)
        self.slow_ma = self.EMA(self.symbol, 50, Resolution.Minute)
        
        # Track state
        self.previous_fast = None
        self.previous_slow = None
        self.data_count = 0
        
        # Logging
        self.Log("=" * 60)
        self.Log("🚀 Demo Strategy Initialized")
        self.Log(f"   Symbol: {self.symbol}")
        self.Log(f"   Data Source: GCP Pub/Sub (vibe-trade-candles-btc-usd-1m)")
        self.Log(f"   Fast MA: 10 periods")
        self.Log(f"   Slow MA: 50 periods")
        self.Log("=" * 60)
        
        # Schedule status updates every 30 seconds
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.Every(TimeSpan.FromSeconds(30)),
            self.LogStatus
        )
    
    def OnData(self, data):
        """Called when new data arrives from Pub/Sub."""
        self.data_count += 1
        
        if self.symbol not in data:
            return
        
        # Wait for indicators
        if not self.fast_ma.IsReady or not self.slow_ma.IsReady:
            return
        
        # Get indicator values
        fast_value = self.fast_ma.Current.Value
        slow_value = self.slow_ma.Current.Value
        
        # Get previous values
        prev_fast = self.previous_fast if self.previous_fast is not None else fast_value
        prev_slow = self.previous_slow if self.previous_slow is not None else slow_value
        
        # Get price
        if isinstance(data[self.symbol], TradeBar):
            current_price = data[self.symbol].Close
        else:
            current_price = self.Securities[self.symbol].Price
        
        # Detect crossover
        bullish_cross = prev_fast <= prev_slow and fast_value > slow_value
        bearish_cross = prev_fast >= prev_slow and fast_value < slow_value
        
        # Get position
        holdings = self.Portfolio[self.symbol]
        
        # Trading logic
        if bullish_cross and not holdings.Invested:
            quantity = self.CalculateOrderQuantity(self.symbol, 0.95)
            self.MarketOrder(self.symbol, quantity)
            self.Log(f"🟢 BUY: {quantity:.6f} @ ${current_price:.2f} | Fast: ${fast_value:.2f}, Slow: ${slow_value:.2f}")
        
        elif bearish_cross and holdings.Invested:
            self.Liquidate(self.symbol)
            self.Log(f"🔴 SELL: @ ${current_price:.2f} | Fast: ${fast_value:.2f}, Slow: ${slow_value:.2f}")
        
        # Update previous values
        self.previous_fast = fast_value
        self.previous_slow = slow_value
        
        # Log data reception
        if self.data_count % 10 == 0:
            self.Log(f"📊 Processed {self.data_count} data points | Price: ${current_price:.2f}")
    
    def LogStatus(self):
        """Log current status."""
        portfolio_value = self.Portfolio.TotalPortfolioValue
        holdings = self.Portfolio[self.symbol]
        position = "LONG" if holdings.Invested else "FLAT"
        
        self.Log(f"💼 Status | Portfolio: ${portfolio_value:,.2f} | Position: {position} | Data Points: {self.data_count}")

