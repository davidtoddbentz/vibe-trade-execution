"""
Simple Momentum Strategy for LEAN.

Buys when price momentum is strong, sells when momentum weakens.
"""

from AlgorithmImports import *


class SimpleMomentum(QCAlgorithm):
    """Simple momentum-based strategy."""
    
    def Initialize(self):
        """Initialize the algorithm."""
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Add symbol
        self.symbol = self.AddCrypto("BTCUSD", Resolution.Minute, Market.GDAX).Symbol
        
        # Momentum indicator (rate of change over 20 periods)
        self.momentum = self.ROC(self.symbol, 20, Resolution.Minute)
        
        # Price change threshold
        self.momentum_threshold = 0.02  # 2% momentum threshold
        
        self.Log(f"✅ Initialized Momentum Strategy")
        self.Log(f"   Symbol: {self.symbol}")
        self.Log(f"   Momentum Period: 20")
        self.Log(f"   Threshold: {self.momentum_threshold * 100}%")
    
    def OnData(self, data):
        """Called when new data arrives."""
        if not self.momentum.IsReady:
            return
        
        if self.symbol not in data:
            return
        
        holdings = self.Portfolio[self.symbol]
        momentum_value = self.momentum.Current.Value
        current_price = data[self.symbol].Close
        
        # Buy signal: strong positive momentum
        if momentum_value > self.momentum_threshold and not holdings.Invested:
            quantity = self.CalculateOrderQuantity(self.symbol, 0.9)
            self.MarketOrder(self.symbol, quantity)
            self.Log(f"🟢 BUY (momentum={momentum_value:.4f}): {quantity:.6f} @ ${current_price:.2f}")
        
        # Sell signal: momentum turns negative
        elif momentum_value < -self.momentum_threshold and holdings.Invested:
            self.Liquidate(self.symbol)
            self.Log(f"🔴 SELL (momentum={momentum_value:.4f}): @ ${current_price:.2f}")


