#!/usr/bin/env python3
"""Test the backtest service and measure performance."""

import subprocess
import time

import requests

# Get service URL
SERVICE_URL = "https://vibe-trade-backtest-kff5sbwvca-uc.a.run.app"


def get_auth_token():
    """Get identity token for Cloud Run auth."""
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def create_test_strategy_ir() -> dict:
    """Create a simple EMA crossover strategy IR for testing."""
    return {
        "strategy_id": "test-ema-crossover",
        "strategy_name": "Test: EMA Crossover",
        "symbol": "BTC-USD",
        "resolution": "Hour",
        "indicators": [
            {"id": "ema_fast", "type": "EMA", "period": 9},
            {"id": "ema_slow", "type": "EMA", "period": 21},
        ],
        "state": [],
        "entry": {
            "id": "entry_ema_cross",
            "condition": {
                "type": "compare",
                "left": {"type": "indicator", "indicator_id": "ema_fast"},
                "op": ">",
                "right": {"type": "indicator", "indicator_id": "ema_slow"},
            },
            "action": {"type": "set_holdings", "allocation": 0.95},
            "on_fill": [],
        },
        "exits": [
            {
                "id": "exit_ema_cross",
                "priority": 1,
                "condition": {
                    "type": "compare",
                    "left": {"type": "indicator", "indicator_id": "ema_fast"},
                    "op": "<",
                    "right": {"type": "indicator", "indicator_id": "ema_slow"},
                },
                "action": {"type": "liquidate"},
            }
        ],
        "gates": [],
        "on_bar": [],
        "on_bar_invested": [],
    }


def main():
    print("=" * 60)
    print("🧪 Testing Backtest Service")
    print("=" * 60)
    print()

    # Get auth token
    print("🔑 Getting auth token...")
    token = get_auth_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Health check
    print("❤️  Health check...")
    health_start = time.time()
    resp = requests.get(f"{SERVICE_URL}/health", headers=headers)
    health_time = time.time() - health_start
    print(f"   Status: {resp.status_code} ({health_time:.2f}s)")

    # Run backtest
    print()
    print("🚀 Running backtest...")

    request_body = {
        "strategy_id": "test-ema-crossover",  # Links all backtests for this strategy
        "strategy_ir": create_test_strategy_ir(),
        "symbol": "BTC-USD",
        "start_date": "20260101",
        "end_date": "20260108",
        "initial_cash": 100000.0,
    }

    start_time = time.time()
    resp = requests.post(
        f"{SERVICE_URL}/backtest",
        headers=headers,
        json=request_body,
        timeout=600,
    )
    total_time = time.time() - start_time

    print()
    print("=" * 60)
    print("📊 RESULTS")
    print("=" * 60)
    print()

    if resp.status_code == 200:
        result = resp.json()
        print(f"✅ Status: {result['status']}")
        print(f"🆔 Backtest ID: {result['backtest_id']}")
        print(f"📋 Strategy ID: {result['strategy_id']}")
        print(f"🕐 Created: {result['created_at']}")
        print(f"⏱️  Duration: {result.get('duration_seconds', 0):.1f}s")
        print(f"📁 Results: {result.get('results_path')}")

        if result.get("summary"):
            summary = result["summary"]
            print()
            print("📈 Summary:")
            print(f"   Final equity: ${summary['final_equity']:,.2f}")
            print(f"   Total return: {summary['total_return_pct']:.2f}%")
            print(f"   Max drawdown: {summary['max_drawdown_pct']:.2f}%")
            print(f"   Total trades: {summary['total_trades']}")
            print(f"   Win rate: {summary['win_rate']:.1f}%")
            print(f"   Profit factor: {summary['profit_factor']:.2f}")
    else:
        print(f"❌ Error: {resp.status_code}")
        print(resp.text)

    print()
    print("=" * 60)
    print(f"Total request time: {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
