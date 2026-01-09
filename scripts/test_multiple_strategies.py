#!/usr/bin/env python3
"""Test multiple strategies through the backtest service."""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SERVICE_URL = "https://vibe-trade-backtest-kff5sbwvca-uc.a.run.app"


def get_auth_token():
    """Get identity token for Cloud Run auth."""
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

def ema_crossover_strategy() -> dict:
    """EMA Crossover - classic trend following."""
    return {
        "strategy_id": "test-ema-crossover",
        "strategy_name": "EMA 9/21 Crossover",
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


def rsi_mean_reversion_strategy() -> dict:
    """RSI Mean Reversion - buy oversold, sell overbought (relaxed thresholds)."""
    return {
        "strategy_id": "test-rsi-mean-reversion",
        "strategy_name": "RSI Mean Reversion",
        "symbol": "BTC-USD",
        "resolution": "Hour",
        "indicators": [
            {"id": "rsi", "type": "RSI", "period": 14},
        ],
        "state": [],
        "entry": {
            "id": "entry_rsi_oversold",
            "condition": {
                "type": "compare",
                "left": {"type": "indicator", "indicator_id": "rsi"},
                "op": "<",
                "right": {"type": "literal", "value": 40},  # More sensitive
            },
            "action": {"type": "set_holdings", "allocation": 0.95},
            "on_fill": [],
        },
        "exits": [
            {
                "id": "exit_rsi_overbought",
                "priority": 1,
                "condition": {
                    "type": "compare",
                    "left": {"type": "indicator", "indicator_id": "rsi"},
                    "op": ">",
                    "right": {"type": "literal", "value": 60},  # More sensitive
                },
                "action": {"type": "liquidate"},
            }
        ],
        "gates": [],
        "on_bar": [],
        "on_bar_invested": [],
    }


def sma_crossover_strategy() -> dict:
    """SMA Crossover - slower trend following."""
    return {
        "strategy_id": "test-sma-crossover",
        "strategy_name": "SMA 20/50 Crossover",
        "symbol": "BTC-USD",
        "resolution": "Hour",
        "indicators": [
            {"id": "sma_fast", "type": "SMA", "period": 20},
            {"id": "sma_slow", "type": "SMA", "period": 50},
        ],
        "state": [],
        "entry": {
            "id": "entry_sma_cross",
            "condition": {
                "type": "compare",
                "left": {"type": "indicator", "indicator_id": "sma_fast"},
                "op": ">",
                "right": {"type": "indicator", "indicator_id": "sma_slow"},
            },
            "action": {"type": "set_holdings", "allocation": 0.95},
            "on_fill": [],
        },
        "exits": [
            {
                "id": "exit_sma_cross",
                "priority": 1,
                "condition": {
                    "type": "compare",
                    "left": {"type": "indicator", "indicator_id": "sma_fast"},
                    "op": "<",
                    "right": {"type": "indicator", "indicator_id": "sma_slow"},
                },
                "action": {"type": "liquidate"},
            }
        ],
        "gates": [],
        "on_bar": [],
        "on_bar_invested": [],
    }


def macd_strategy() -> dict:
    """MACD Signal Crossover."""
    return {
        "strategy_id": "test-macd-crossover",
        "strategy_name": "MACD Crossover",
        "symbol": "BTC-USD",
        "resolution": "Hour",
        "indicators": [
            {"id": "macd", "type": "MACD", "fast_period": 12, "slow_period": 26, "signal_period": 9},
        ],
        "state": [],
        "entry": {
            "id": "entry_macd_cross",
            "condition": {
                "type": "compare",
                "left": {"type": "indicator", "indicator_id": "macd", "field": "macd"},
                "op": ">",
                "right": {"type": "indicator", "indicator_id": "macd", "field": "signal"},
            },
            "action": {"type": "set_holdings", "allocation": 0.95},
            "on_fill": [],
        },
        "exits": [
            {
                "id": "exit_macd_cross",
                "priority": 1,
                "condition": {
                    "type": "compare",
                    "left": {"type": "indicator", "indicator_id": "macd", "field": "macd"},
                    "op": "<",
                    "right": {"type": "indicator", "indicator_id": "macd", "field": "signal"},
                },
                "action": {"type": "liquidate"},
            }
        ],
        "gates": [],
        "on_bar": [],
        "on_bar_invested": [],
    }


def bollinger_band_strategy() -> dict:
    """Bollinger Band Mean Reversion (tighter bands)."""
    return {
        "strategy_id": "test-bollinger-reversion",
        "strategy_name": "Bollinger Band Reversion",
        "symbol": "BTC-USD",
        "resolution": "Hour",
        "indicators": [
            {"id": "bb", "type": "BB", "period": 20, "std_dev": 1.5},  # Tighter bands
        ],
        "state": [],
        "entry": {
            "id": "entry_bb_lower",
            "condition": {
                "type": "compare",
                "left": {"type": "price", "field": "close"},
                "op": "<",
                "right": {"type": "indicator_band", "indicator_id": "bb", "band": "lower"},
            },
            "action": {"type": "set_holdings", "allocation": 0.95},
            "on_fill": [],
        },
        "exits": [
            {
                "id": "exit_bb_middle",
                "priority": 1,
                "condition": {
                    "type": "compare",
                    "left": {"type": "price", "field": "close"},
                    "op": ">",
                    "right": {"type": "indicator_band", "indicator_id": "bb", "band": "middle"},
                },
                "action": {"type": "liquidate"},
            }
        ],
        "gates": [],
        "on_bar": [],
        "on_bar_invested": [],
    }


def triple_ema_strategy() -> dict:
    """Triple EMA - enter on alignment, exit on misalignment."""
    return {
        "strategy_id": "test-triple-ema",
        "strategy_name": "Triple EMA Alignment",
        "symbol": "BTC-USD",
        "resolution": "Hour",
        "indicators": [
            {"id": "ema_fast", "type": "EMA", "period": 8},
            {"id": "ema_mid", "type": "EMA", "period": 21},
            {"id": "ema_slow", "type": "EMA", "period": 55},
        ],
        "state": [],
        "entry": {
            "id": "entry_triple_ema",
            "condition": {
                "type": "and",
                "conditions": [
                    {
                        "type": "compare",
                        "left": {"type": "indicator", "indicator_id": "ema_fast"},
                        "op": ">",
                        "right": {"type": "indicator", "indicator_id": "ema_mid"},
                    },
                    {
                        "type": "compare",
                        "left": {"type": "indicator", "indicator_id": "ema_mid"},
                        "op": ">",
                        "right": {"type": "indicator", "indicator_id": "ema_slow"},
                    },
                ],
            },
            "action": {"type": "set_holdings", "allocation": 0.95},
            "on_fill": [],
        },
        "exits": [
            {
                "id": "exit_triple_ema",
                "priority": 1,
                "condition": {
                    "type": "compare",
                    "left": {"type": "indicator", "indicator_id": "ema_fast"},
                    "op": "<",
                    "right": {"type": "indicator", "indicator_id": "ema_mid"},
                },
                "action": {"type": "liquidate"},
            }
        ],
        "gates": [],
        "on_bar": [],
        "on_bar_invested": [],
    }


# =============================================================================
# TEST RUNNER
# =============================================================================

STRATEGIES = [
    ("EMA Crossover", ema_crossover_strategy),
    ("RSI Mean Reversion", rsi_mean_reversion_strategy),
    ("SMA Crossover", sma_crossover_strategy),
    ("MACD Crossover", macd_strategy),
    ("Bollinger Band", bollinger_band_strategy),
    ("Triple EMA", triple_ema_strategy),
]


def run_backtest(name: str, strategy_fn, token: str, start_date: str, end_date: str) -> dict:
    """Run a single backtest and return results."""
    strategy_ir = strategy_fn()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    request_body = {
        "strategy_id": strategy_ir["strategy_id"],
        "strategy_ir": strategy_ir,
        "symbol": "BTC-USD",
        "start_date": start_date,
        "end_date": end_date,
        "initial_cash": 100000.0,
    }

    start_time = time.time()
    try:
        resp = requests.post(
            f"{SERVICE_URL}/backtest",
            headers=headers,
            json=request_body,
            timeout=600,
        )
        duration = time.time() - start_time

        if resp.status_code == 200:
            result = resp.json()
            return {
                "name": name,
                "status": "success",
                "backtest_id": result["backtest_id"],
                "duration": duration,
                "summary": result.get("summary"),
                "error": None,
            }
        else:
            return {
                "name": name,
                "status": "error",
                "backtest_id": None,
                "duration": duration,
                "summary": None,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
    except Exception as e:
        return {
            "name": name,
            "status": "error",
            "backtest_id": None,
            "duration": time.time() - start_time,
            "summary": None,
            "error": str(e),
        }


def main():
    print("=" * 70)
    print("  MULTI-STRATEGY BACKTEST TEST")
    print("=" * 70)
    print()

    # Test parameters - use available data range (data starts 2026-01-02)
    start_date = "20260102"
    end_date = "20260109"

    print(f"Date range: {start_date} - {end_date}")
    print(f"Strategies: {len(STRATEGIES)}")
    print()

    # Get auth token
    print("Getting auth token...")
    token = get_auth_token()

    # Run backtests in parallel
    print("Running backtests in parallel...")
    print()

    results = []
    total_start = time.time()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_backtest, name, fn, token, start_date, end_date): name
            for name, fn in STRATEGIES
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # Print progress
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"{status_icon} {result['name']}: {result['duration']:.1f}s")

    total_time = time.time() - total_start

    # Print summary table
    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Strategy':<25} {'Return':<12} {'MaxDD':<10} {'Trades':<8} {'Win%':<8} {'PF':<8}")
    print("-" * 70)

    for result in sorted(results, key=lambda x: x["name"]):
        name = result["name"]
        if result["status"] == "success" and result["summary"]:
            s = result["summary"]
            print(
                f"{name:<25} "
                f"{s['total_return_pct']:>+10.2f}% "
                f"{s['max_drawdown_pct']:>8.2f}% "
                f"{s['total_trades']:>6} "
                f"{s['win_rate']:>6.1f}% "
                f"{s['profit_factor']:>6.2f}"
            )
        else:
            error_msg = result.get("error") or "Unknown error"
            print(f"{name:<25} {'ERROR':<12} {error_msg[:40]}")

    # Stats
    print()
    print("-" * 70)
    successful = [r for r in results if r["status"] == "success"]
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Total time: {total_time:.1f}s (parallel)")
    print(f"Avg per backtest: {sum(r['duration'] for r in results) / len(results):.1f}s")

    # Best performers
    if successful:
        print()
        print("Top performers by return:")
        by_return = sorted(
            [r for r in successful if r["summary"]],
            key=lambda x: x["summary"]["total_return_pct"],
            reverse=True,
        )
        for i, r in enumerate(by_return[:3], 1):
            print(f"  {i}. {r['name']}: {r['summary']['total_return_pct']:+.2f}%")


if __name__ == "__main__":
    main()
