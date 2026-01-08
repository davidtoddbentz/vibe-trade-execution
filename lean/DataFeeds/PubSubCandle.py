"""
LEAN Custom Data Type for Pub/Sub Candles.

This defines a custom data type that LEAN can use to consume
candles from GCP Pub/Sub topic: vibe-trade-candles-btc-usd-1m

For now, this is a simplified version that can be extended
to actually connect to Pub/Sub in live mode.
"""

from AlgorithmImports import *
import json
from datetime import datetime


class PubSubCandle(PythonData):
    """Custom data type for Pub/Sub candle messages.
    
    Message format from Pub/Sub:
    {
        "symbol": "BTC-USD",
        "timestamp": "2024-01-01T00:00:00Z",
        "open": 42000.0,
        "high": 42100.0,
        "low": 41900.0,
        "close": 42050.0,
        "volume": 123.45,
        "granularity": "1m"
    }
    
    Note: In live mode, this would need to subscribe to Pub/Sub.
    For now, this provides the structure for integration.
    """
    
    def GetSource(self, config, date, isLiveMode):
        """Return the SubscriptionDataSource for this data type.
        
        In live mode, this should subscribe to Pub/Sub.
        For backtesting, we can read from files or use a data queue handler.
        """
        if isLiveMode:
            # Live mode: Use custom data queue handler
            # The actual Pub/Sub subscription happens in the data queue handler
            topic = self._get_topic_from_symbol(config.Symbol.Value)
            return SubscriptionDataSource(
                f"pubsub://{topic}",
                SubscriptionTransportMedium.Rest,
                FileFormat.Csv
            )
        else:
            # Backtest mode: Can read from files
            # For now, return a file-based source (can be extended)
            symbol_normalized = config.Symbol.Value.replace("-", "").lower()
            return SubscriptionDataSource(
                f"/Data/crypto/gdax/minute/{symbol_normalized}/{date:yyyyMMdd}.zip",
                SubscriptionTransportMedium.LocalFile,
                FileFormat.Csv
            )
    
    def Reader(self, config, line, date, isLiveMode):
        """Parse a line of data into a PubSubCandle object.
        
        This is called for each message received.
        In live mode, 'line' will be JSON from Pub/Sub.
        In backtest mode, 'line' will be CSV from files.
        """
        if not line or not line.strip():
            return None
        
        try:
            candle = PubSubCandle()
            candle.Symbol = config.Symbol
            
            if isLiveMode:
                # Live mode: Parse JSON from Pub/Sub
                data = json.loads(line)
                
                # Parse timestamp
                timestamp_str = data.get("timestamp", "")
                if "T" in timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
                
                candle.Time = timestamp
                candle.EndTime = timestamp
                
                # Create TradeBar from OHLCV
                candle.Data = TradeBar(
                    timestamp,
                    config.Symbol,
                    float(data.get("open", 0)),
                    float(data.get("high", 0)),
                    float(data.get("low", 0)),
                    float(data.get("close", 0)),
                    float(data.get("volume", 0))
                )
                
                candle.Value = candle.Data.Close
                
            else:
                # Backtest mode: Parse CSV (standard LEAN format)
                # Format: time,open,high,low,close,volume
                parts = line.split(",")
                if len(parts) < 6:
                    return None
                
                timestamp = datetime.strptime(parts[0], "%Y%m%d %H%M%S")
                candle.Time = timestamp
                candle.EndTime = timestamp
                
                candle.Data = TradeBar(
                    timestamp,
                    config.Symbol,
                    float(parts[1]),  # open
                    float(parts[2]),  # high
                    float(parts[3]),  # low
                    float(parts[4]),  # close
                    float(parts[5])   # volume
                )
                
                candle.Value = candle.Data.Close
            
            return candle
            
        except Exception as e:
            # Log error but don't crash
            return None
    
    @staticmethod
    def _get_topic_from_symbol(symbol: str) -> str:
        """Convert LEAN symbol to Pub/Sub topic name.
        
        Args:
            symbol: LEAN symbol (e.g., "BTC-USD" or "BTCUSD")
            
        Returns:
            Pub/Sub topic name (e.g., "vibe-trade-candles-btc-usd-1m")
        """
        # Normalize symbol: BTCUSD -> BTC-USD -> btc-usd
        if "-" not in symbol:
            # Assume format like BTCUSD -> BTC-USD
            if len(symbol) >= 6:
                base = symbol[:3]
                quote = symbol[3:]
                symbol = f"{base}-{quote}"
        
        normalized = symbol.lower()
        return f"vibe-trade-candles-{normalized}-1m"
