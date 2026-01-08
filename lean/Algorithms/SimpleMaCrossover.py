"""
Simple Moving Average Crossover Strategy for LEAN.

This demonstrates:
- How to subscribe to crypto data
- How to use indicators (EMA)
- How to detect crossovers
- How to place orders
- How to manage positions
"""

from AlgorithmImports import *


class SimpleMaCrossover(QCAlgorithm):
    """Simple moving average crossover strategy."""
    
    def Initialize(self):
        """Initialize the algorithm."""
        # Set backtest period
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        
        # Set starting capital
        self.SetCash(100000)  # $100k
        
        # Add crypto symbol
        # Note: LEAN uses different symbol formats
        # For BTC, we can use GDAX (Coinbase) or custom data
        # We'll use GDAX for now, later we'll add custom data feed
        self.symbol = self.AddCrypto("BTCUSD", Resolution.Minute, Market.GDAX).Symbol
        
        # Create indicators
        self.fast_ma = self.EMA(self.symbol, 10, Resolution.Minute)
        self.slow_ma = self.EMA(self.symbol, 50, Resolution.Minute)
        
        # Track previous state for crossover detection
        self.previous_fast = None
        self.previous_slow = None
        
        # Logging
        self.Log(f"✅ Initialized MA Crossover Strategy")
        self.Log(f"   Symbol: {self.symbol}")
        self.Log(f"   Fast MA: 10 periods")
        self.Log(f"   Slow MA: 50 periods")
    
    def OnData(self, data):
        """Called when new market data arrives."""
        # Wait for indicators to be ready
        if not self.fast_ma.IsReady or not self.slow_ma.IsReady:
            return
        
        # Skip if we don't have data for our symbol
        if self.symbol not in data:
            return
        
        # Get current indicator values
        fast_value = self.fast_ma.Current.Value
        slow_value = self.slow_ma.Current.Value
        
        # Get previous values (for crossover detection)
        prev_fast = self.previous_fast if self.previous_fast is not None else fast_value
        prev_slow = self.previous_slow if self.previous_slow is not None else slow_value
        
        # Detect crossover
        bullish_cross = prev_fast <= prev_slow and fast_value > slow_value
        bearish_cross = prev_fast >= prev_slow and fast_value < slow_value
        
        # Get current position
        holdings = self.Portfolio[self.symbol]
        current_price = data[self.symbol].Close
        
        # Entry: Bullish crossover
        if bullish_cross and not holdings.Invested:
            # Calculate order quantity (use 95% of available cash)
            quantity = self.CalculateOrderQuantity(self.symbol, 0.95)
            self.MarketOrder(self.symbol, quantity)
            self.Log(f"🟢 BUY: {quantity:.6f} @ ${current_price:.2f} (Fast: ${fast_value:.2f}, Slow: ${slow_value:.2f})")
        
        # Exit: Bearish crossover
        elif bearish_cross and holdings.Invested:
            self.Liquidate(self.symbol)
            self.Log(f"🔴 SELL: @ ${current_price:.2f} (Fast: ${fast_value:.2f}, Slow: ${slow_value:.2f})")
        
        # Update previous values
        self.previous_fast = fast_value
        self.previous_slow = slow_value
    
    def OnEndOfAlgorithm(self):
        """Called when algorithm ends."""
        portfolio_value = self.Portfolio.TotalPortfolioValue
        self.Log(f"📊 Final Portfolio Value: ${portfolio_value:,.2f}")


