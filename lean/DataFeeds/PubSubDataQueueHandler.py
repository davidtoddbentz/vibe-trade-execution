"""
LEAN Data Queue Handler for GCP Pub/Sub.

This implements LEAN's data queue handler interface to subscribe to
Pub/Sub topics and feed real-time data to LEAN's engine.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime
from queue import Queue, Empty
from typing import Dict, List, Optional

try:
    from google.cloud import pubsub_v1
    from google.cloud.pubsub_v1.types import FlowControl
except ImportError:
    pubsub_v1 = None
    FlowControl = None

# LEAN imports
try:
    from AlgorithmImports import *
except ImportError:
    # For testing outside LEAN
    BaseData = object
    Symbol = object
    TradeBar = object
    Resolution = object

logger = logging.getLogger(__name__)


class PubSubDataQueueHandler:
    """
    LEAN Data Queue Handler for GCP Pub/Sub.
    
    Subscribes to Pub/Sub topics and feeds data to LEAN's engine.
    """
    
    def __init__(self):
        """Initialize the data queue handler."""
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
        self.subscriber = None
        self.subscriptions: Dict[str, pubsub_v1.subscriber.futures.StreamingPullFuture] = {}
        self.data_queue: Queue = Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.symbol_to_topic: Dict[str, str] = {}
        
        if not pubsub_v1:
            logger.error("google-cloud-pubsub not installed. Install with: pip install google-cloud-pubsub")
            return
            
        if not self.project_id:
            logger.warning("GOOGLE_CLOUD_PROJECT not set. Pub/Sub subscription will not work.")
            return
        
        try:
            self.subscriber = pubsub_v1.SubscriberClient()
            logger.info(f"✅ PubSubDataQueueHandler initialized (project: {self.project_id})")
        except Exception as e:
            logger.error(f"Failed to initialize Pub/Sub subscriber: {e}")
    
    def _get_topic_name(self, symbol: str) -> str:
        """Get Pub/Sub topic name for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "BTC-USD")
            
        Returns:
            Topic name (e.g., "vibe-trade-candles-btc-usd-1m")
        """
        # Normalize symbol
        if "-" not in symbol:
            # Assume format like BTCUSD -> BTC-USD
            if len(symbol) >= 6:
                base = symbol[:3]
                quote = symbol[3:]
                symbol = f"{base}-{quote}"
        
        normalized = symbol.lower()
        return f"vibe-trade-candles-{normalized}-1m"
    
    def _get_subscription_name(self, topic_name: str) -> str:
        """Get subscription name for a topic.
        
        Args:
            topic_name: Pub/Sub topic name
            
        Returns:
            Subscription name (e.g., "vibe-trade-lean-{topic}")
        """
        # Create a subscription name based on topic
        # Format: vibe-trade-lean-{symbol}-1m
        topic_parts = topic_name.split("-")
        if len(topic_parts) >= 4:
            symbol_part = "-".join(topic_parts[2:-1])  # Extract symbol part
            return f"vibe-trade-lean-{symbol_part}-1m"
        return f"vibe-trade-lean-{topic_name}"
    
    def _ensure_subscription(self, topic_name: str) -> str:
        """Ensure a subscription exists for a topic.
        
        Args:
            topic_name: Pub/Sub topic name
            
        Returns:
            Subscription path
        """
        if not self.subscriber or not self.project_id:
            raise RuntimeError("Pub/Sub client not initialized")
        
        subscription_name = self._get_subscription_name(topic_name)
        topic_path = self.subscriber.topic_path(self.project_id, topic_name)
        subscription_path = self.subscriber.subscription_path(self.project_id, subscription_name)
        
        try:
            # Try to get existing subscription
            self.subscriber.get_subscription(request={"name": subscription_path})
            logger.info(f"Using existing subscription: {subscription_name}")
        except Exception:
            # Create new subscription
            try:
                self.subscriber.create_subscription(
                    request={
                        "name": subscription_path,
                        "topic": topic_path,
                    }
                )
                logger.info(f"Created subscription: {subscription_name}")
            except Exception as e:
                logger.error(f"Failed to create subscription {subscription_name}: {e}")
                raise
        
        return subscription_path
    
    def _parse_message(self, message_data: bytes, symbol: Symbol) -> Optional[BaseData]:
        """Parse a Pub/Sub message into LEAN BaseData.
        
        Args:
            message_data: Raw message bytes from Pub/Sub
            symbol: LEAN symbol object
            
        Returns:
            BaseData object (TradeBar) or None if parsing fails
        """
        try:
            data = json.loads(message_data.decode("utf-8"))
            
            # Parse timestamp
            timestamp_str = data.get("timestamp", "")
            if "T" in timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                timestamp = datetime.fromisoformat(timestamp_str)
            
            # Create TradeBar
            trade_bar = TradeBar(
                timestamp,
                symbol,
                float(data.get("open", 0)),
                float(data.get("high", 0)),
                float(data.get("low", 0)),
                float(data.get("close", 0)),
                float(data.get("volume", 0))
            )
            
            return trade_bar
            
        except Exception as e:
            logger.error(f"Failed to parse Pub/Sub message: {e}")
            return None
    
    def _message_callback(self, message: pubsub_v1.subscriber.message.Message, symbol: Symbol):
        """Callback for received Pub/Sub messages.
        
        Args:
            message: Pub/Sub message
            symbol: LEAN symbol object
        """
        try:
            # Parse message
            base_data = self._parse_message(message.data, symbol)
            
            if base_data:
                # Add to queue for LEAN to consume
                self.data_queue.put(base_data)
                logger.debug(f"📥 Received data for {symbol}: {base_data.Time} @ ${base_data.Close}")
            
            # Acknowledge message
            message.ack()
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            message.nack()
    
    def Subscribe(self, symbols: List[Symbol], resolution: Resolution) -> None:
        """
        Subscribe to data feeds for the given symbols.
        
        This is called by LEAN when the algorithm requests data.
        
        Args:
            symbols: List of LEAN symbols to subscribe to
            resolution: Data resolution (e.g., Resolution.Minute)
        """
        if not self.subscriber or not self.project_id:
            logger.error("Cannot subscribe: Pub/Sub client not initialized")
            return
        
        for symbol in symbols:
            try:
                symbol_str = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
                topic_name = self._get_topic_name(symbol_str)
                self.symbol_to_topic[symbol_str] = topic_name
                
                # Ensure subscription exists
                subscription_path = self._ensure_subscription(topic_name)
                
                # Create callback with symbol
                callback = lambda msg, s=symbol: self._message_callback(msg, s)
                
                # Start streaming pull
                flow_control = FlowControl(max_messages=100)
                future = self.subscriber.subscribe(
                    subscription_path,
                    callback=callback,
                    flow_control=flow_control
                )
                
                self.subscriptions[symbol_str] = future
                logger.info(f"✅ Subscribed to {topic_name} for {symbol_str}")
                
            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol_str}: {e}")
    
    def Unsubscribe(self, symbols: List[Symbol]) -> None:
        """
        Unsubscribe from data feeds for the given symbols.
        
        Args:
            symbols: List of LEAN symbols to unsubscribe from
        """
        for symbol in symbols:
            symbol_str = symbol.Value if hasattr(symbol, 'Value') else str(symbol)
            
            if symbol_str in self.subscriptions:
                try:
                    future = self.subscriptions[symbol_str]
                    future.cancel()
                    del self.subscriptions[symbol_str]
                    logger.info(f"Unsubscribed from {symbol_str}")
                except Exception as e:
                    logger.error(f"Failed to unsubscribe from {symbol_str}: {e}")
    
    def GetNextTicks(self) -> List[BaseData]:
        """
        Get the next batch of data points.
        
        This is called by LEAN to get new data.
        
        Returns:
            List of BaseData objects
        """
        ticks = []
        
        # Get all available messages from queue (non-blocking)
        while True:
            try:
                tick = self.data_queue.get_nowait()
                ticks.append(tick)
            except Empty:
                break
        
        return ticks
    
    def SetJob(self, job: Dict) -> None:
        """
        Set the job configuration.
        
        Called by LEAN to provide job context.
        
        Args:
            job: Job configuration dictionary
        """
        logger.info(f"Job set: {job.get('AlgorithmId', 'unknown')}")
    
    def Dispose(self) -> None:
        """Clean up resources."""
        logger.info("Disposing PubSubDataQueueHandler...")
        
        # Cancel all subscriptions
        for symbol_str, future in self.subscriptions.items():
            try:
                future.cancel()
            except Exception:
                pass
        
        self.subscriptions.clear()
        self.running = False
        
        if self.subscriber:
            self.subscriber.close()
        
        logger.info("PubSubDataQueueHandler disposed")
