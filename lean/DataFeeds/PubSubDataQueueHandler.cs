/*
 * LEAN Data Queue Handler for GCP Pub/Sub
 * 
 * This implements IDataQueueHandler to subscribe to Pub/Sub topics
 * and feed real-time data to LEAN's engine.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Google.Cloud.PubSub.V1;
using QuantConnect.Data;
using QuantConnect.Interfaces;
using QuantConnect.Logging;
using Newtonsoft.Json;
using System.IO;

namespace QuantConnect.Lean.Engine.DataFeeds
{
    /// <summary>
    /// Data queue handler for GCP Pub/Sub
    /// </summary>
    public class PubSubDataQueueHandler : IDataQueueHandler
    {
        private string _projectId;
        private SubscriberClient _subscriber;
        private Dictionary<Symbol, SubscriptionDataConfig> _subscriptions;
        private Dictionary<Symbol, Task> _subscriptionTasks;
        private CancellationTokenSource _cancellationTokenSource;
        private readonly object _lock = new object();
        private bool _isConnected;

        /// <summary>
        /// Gets whether the handler is connected
        /// </summary>
        public bool IsConnected => _isConnected;

        /// <summary>
        /// Initializes the data queue handler
        /// </summary>
        public PubSubDataQueueHandler()
        {
            _projectId = Environment.GetEnvironmentVariable("GOOGLE_CLOUD_PROJECT") ?? "";
            _subscriptions = new Dictionary<Symbol, SubscriptionDataConfig>();
            _subscriptionTasks = new Dictionary<Symbol, Task>();
            _cancellationTokenSource = new CancellationTokenSource();
            _isConnected = false;

            if (string.IsNullOrEmpty(_projectId))
            {
                Log.Error("PubSubDataQueueHandler: GOOGLE_CLOUD_PROJECT not set");
                return;
            }

            try
            {
                _subscriber = SubscriberClient.Create();
                _isConnected = true;
                Log.Trace($"PubSubDataQueueHandler: Initialized (project: {_projectId})");
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Failed to initialize: {ex.Message}");
                _isConnected = false;
            }
        }

        /// <summary>
        /// Gets the next batch of data from the queue
        /// </summary>
        public IEnumerable<BaseData> GetNextTicks()
        {
            // This method is called by LEAN to get new data
            // In a real implementation, we'd return data from a queue
            // For now, return empty (data will be fed via Subscribe callback)
            return Enumerable.Empty<BaseData>();
        }

        /// <summary>
        /// Subscribe to the specified data configuration
        /// </summary>
        public void Subscribe(SubscriptionDataConfig dataConfig)
        {
            if (!_isConnected || _subscriber == null)
            {
                Log.Error("PubSubDataQueueHandler: Cannot subscribe - not connected");
                return;
            }

            try
            {
                var symbol = dataConfig.Symbol;
                var topicName = GetTopicName(symbol.Value);
                var subscriptionName = GetSubscriptionName(topicName);

                // Store subscription config
                lock (_lock)
                {
                    _subscriptions[symbol] = dataConfig;
                }

                // Create subscription if it doesn't exist
                var subscriptionPath = SubscriptionName.FromProjectSubscription(_projectId, subscriptionName);
                var topicPath = TopicName.FromProjectTopic(_projectId, topicName);

                try
                {
                    // Try to get existing subscription
                    _subscriber.GetSubscription(subscriptionPath);
                    Log.Trace($"PubSubDataQueueHandler: Using existing subscription: {subscriptionName}");
                }
                catch
                {
                    // Create new subscription
                    try
                    {
                        _subscriber.CreateSubscription(subscriptionPath, topicPath, null, 60);
                        Log.Trace($"PubSubDataQueueHandler: Created subscription: {subscriptionName}");
                    }
                    catch (Exception ex)
                    {
                        Log.Error($"PubSubDataQueueHandler: Failed to create subscription: {ex.Message}");
                        return;
                    }
                }

                // Start async subscription task
                var task = Task.Run(async () => await SubscribeAsync(symbol, subscriptionPath, dataConfig, _cancellationTokenSource.Token));
                lock (_lock)
                {
                    _subscriptionTasks[symbol] = task;
                }

                Log.Trace($"PubSubDataQueueHandler: Subscribed to {topicName} for {symbol}");
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Failed to subscribe to {dataConfig.Symbol}: {ex.Message}");
            }
        }

        /// <summary>
        /// Unsubscribe from the specified data configuration
        /// </summary>
        public void Unsubscribe(SubscriptionDataConfig dataConfig)
        {
            var symbol = dataConfig.Symbol;
            
            lock (_lock)
            {
                if (_subscriptions.ContainsKey(symbol))
                {
                    _subscriptions.Remove(symbol);
                }

                if (_subscriptionTasks.ContainsKey(symbol))
                {
                    _subscriptionTasks.Remove(symbol);
                }
            }

            Log.Trace($"PubSubDataQueueHandler: Unsubscribed from {symbol}");
        }

        /// <summary>
        /// Sets the job for the handler
        /// </summary>
        public void SetJob(LiveNodePacket job)
        {
            Log.Trace($"PubSubDataQueueHandler: Job set: {job?.AlgorithmId ?? "unknown"}");
        }

        /// <summary>
        /// Disposes of the handler
        /// </summary>
        public void Dispose()
        {
            Log.Trace("PubSubDataQueueHandler: Disposing...");
            
            _cancellationTokenSource?.Cancel();
            
            lock (_lock)
            {
                _subscriptions.Clear();
                _subscriptionTasks.Clear();
            }

            _subscriber?.Dispose();
            _isConnected = false;
            
            Log.Trace("PubSubDataQueueHandler: Disposed");
        }

        /// <summary>
        /// Gets the Pub/Sub topic name for a symbol
        /// </summary>
        private string GetTopicName(string symbol)
        {
            // Normalize symbol: BTCUSD -> BTC-USD -> btc-usd
            var normalized = symbol;
            if (!normalized.Contains("-"))
            {
                if (normalized.Length >= 6)
                {
                    var baseCurrency = normalized.Substring(0, 3);
                    var quoteCurrency = normalized.Substring(3);
                    normalized = $"{baseCurrency}-{quoteCurrency}";
                }
            }
            
            return $"vibe-trade-candles-{normalized.ToLower()}-1m";
        }

        /// <summary>
        /// Gets the subscription name for a topic
        /// </summary>
        private string GetSubscriptionName(string topicName)
        {
            // Format: vibe-trade-lean-{symbol}-1m
            var parts = topicName.Split('-');
            if (parts.Length >= 4)
            {
                var symbolPart = string.Join("-", parts.Skip(2).Take(parts.Length - 3));
                return $"vibe-trade-lean-{symbolPart}-1m";
            }
            return $"vibe-trade-lean-{topicName}";
        }

        /// <summary>
        /// Async subscription handler
        /// </summary>
        private async Task SubscribeAsync(Symbol symbol, SubscriptionName subscriptionPath, SubscriptionDataConfig config, CancellationToken cancellationToken)
        {
            try
            {
                await _subscriber.StartAsync((message, cancellationToken) =>
                {
                    try
                    {
                        // Parse message
                        var data = ParseMessage(message.Data.ToByteArray(), symbol, config);
                        if (data != null)
                        {
                            // Feed data to LEAN
                            // Note: In a real implementation, we'd use an event or queue
                            // For now, we'll log that we received data
                            Log.Trace($"PubSubDataQueueHandler: Received data for {symbol}: {data.Time} @ {data.Value}");
                        }

                        // Acknowledge message
                        return Task.FromResult(SubscriberClient.Reply.Ack);
                    }
                    catch (Exception ex)
                    {
                        Log.Error($"PubSubDataQueueHandler: Error processing message: {ex.Message}");
                        return Task.FromResult(SubscriberClient.Reply.Nack);
                    }
                }, cancellationToken);
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Subscription error for {symbol}: {ex.Message}");
            }
        }

        /// <summary>
        /// Parses a Pub/Sub message into LEAN BaseData
        /// </summary>
        private BaseData ParseMessage(byte[] messageData, Symbol symbol, SubscriptionDataConfig config)
        {
            try
            {
                var json = System.Text.Encoding.UTF8.GetString(messageData);
                var data = JsonConvert.DeserializeObject<PubSubMessage>(json);

                if (data == null)
                {
                    return null;
                }

                // Parse timestamp
                var timestamp = DateTime.Parse(data.Timestamp);
                if (data.Timestamp.EndsWith("Z"))
                {
                    timestamp = timestamp.ToUniversalTime();
                }

                // Create TradeBar
                var tradeBar = new TradeBar(
                    timestamp,
                    symbol,
                    data.Open,
                    data.High,
                    data.Low,
                    data.Close,
                    data.Volume
                );

                return tradeBar;
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Failed to parse message: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Pub/Sub message format
        /// </summary>
        private class PubSubMessage
        {
            [JsonProperty("symbol")]
            public string Symbol { get; set; }

            [JsonProperty("timestamp")]
            public string Timestamp { get; set; }

            [JsonProperty("open")]
            public decimal Open { get; set; }

            [JsonProperty("high")]
            public decimal High { get; set; }

            [JsonProperty("low")]
            public decimal Low { get; set; }

            [JsonProperty("close")]
            public decimal Close { get; set; }

            [JsonProperty("volume")]
            public decimal Volume { get; set; }

            [JsonProperty("granularity")]
            public string Granularity { get; set; }
        }
    }
}

