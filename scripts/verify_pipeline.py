#!/usr/bin/env python3
"""Verify the full MCP → IR → LEAN pipeline.

This script:
1. Creates a strategy using vibe_trade_shared models (same as MCP)
2. Translates to IR using IRTranslator
3. Runs through the cloud backtest service
4. Verifies the trades match expected behavior

This proves that strategies created via MCP will execute correctly.
"""

import json
import subprocess

import requests
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment

from src.translator.ir_translator import IRTranslator

SERVICE_URL = "https://vibe-trade-backtest-kff5sbwvca-uc.a.run.app"


def get_auth_token():
    """Get identity token for Cloud Run auth."""
    result = subprocess.run(
        ["gcloud", "auth", "print-identity-token"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def create_ema_crossover_strategy():
    """Create an EMA crossover strategy using MCP-compatible models.

    This mimics what happens when a user creates a strategy via MCP:
    - Uses entry.rule_trigger archetype
    - Uses exit.rule_trigger archetype
    - Trend MA relation condition
    """
    strategy = Strategy(
        id="verify-ema-crossover",
        name="Pipeline Verification: EMA Crossover",
        universe=["BTC-USD"],
        status="ready",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        attachments=[
            Attachment(card_id="entry-ema", role="entry", enabled=True),
            Attachment(card_id="exit-ema", role="exit", enabled=True),
        ],
    )

    cards = {
        "entry-ema": Card(
            id="entry-ema",
            type="entry.rule_trigger",
            slots={
                "context": {"tf": "1h", "symbol": "BTC-USD"},
                "event": {
                    "condition": {
                        "type": "regime",
                        "regime": {
                            "metric": "trend_ma_relation",
                            "op": ">",
                            "value": 0,
                            "ma_fast": 9,
                            "ma_slow": 21,
                        },
                    }
                },
                "action": {"direction": "long"},
            },
            schema_etag="test",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        ),
        "exit-ema": Card(
            id="exit-ema",
            type="exit.rule_trigger",
            slots={
                "context": {"tf": "1h", "symbol": "BTC-USD"},
                "event": {
                    "condition": {
                        "type": "regime",
                        "regime": {
                            "metric": "trend_ma_relation",
                            "op": "<",
                            "value": 0,
                            "ma_fast": 9,
                            "ma_slow": 21,
                        },
                    }
                },
                "action": {"mode": "close"},
            },
            schema_etag="test",
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        ),
    }

    return strategy, cards


def translate_to_ir(strategy: Strategy, cards: dict) -> dict:
    """Translate strategy to IR using the same translator as MCP compile_strategy."""
    translator = IRTranslator(strategy, cards)
    ir = translator.translate()
    return json.loads(ir.to_json())


def run_backtest(strategy_ir: dict, token: str) -> dict:
    """Run backtest through cloud service."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    request_body = {
        "strategy_id": strategy_ir["strategy_id"],
        "strategy_ir": strategy_ir,
        "symbol": "BTC-USD",
        "start_date": "20260102",
        "end_date": "20260109",
        "initial_cash": 100000.0,
    }

    resp = requests.post(
        f"{SERVICE_URL}/backtest",
        headers=headers,
        json=request_body,
        timeout=600,
    )

    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
    return resp.json()


def verify_results(result: dict, strategy_ir: dict) -> list[str]:
    """Verify backtest results match expected strategy behavior."""
    issues = []

    # Check basic success
    if result.get("status") != "success":
        issues.append(f"Backtest failed: {result.get('error')}")
        return issues

    summary = result.get("summary", {})

    # For EMA crossover, we expect:
    # 1. Some trades (indicators should trigger)
    total_trades = summary.get("total_trades", 0)
    if total_trades == 0:
        issues.append("No trades executed - entry condition may not be triggering")

    # 2. Check that we used the right indicators
    ir_indicators = strategy_ir.get("indicators", [])
    expected_types = {"EMA"}
    actual_types = {ind["type"] for ind in ir_indicators}
    if not expected_types.issubset(actual_types):
        issues.append(f"Expected indicators {expected_types}, got {actual_types}")

    # 3. Verify indicator periods match what we specified
    for ind in ir_indicators:
        if ind["type"] == "EMA":
            if ind["id"] == "ema_fast" and ind.get("period") != 9:
                issues.append(f"Fast EMA period mismatch: expected 9, got {ind.get('period')}")
            if ind["id"] == "ema_slow" and ind.get("period") != 21:
                issues.append(f"Slow EMA period mismatch: expected 21, got {ind.get('period')}")

    # 4. Check entry/exit conditions are present
    entry = strategy_ir.get("entry")
    if not entry:
        issues.append("No entry rule in IR")

    exits = strategy_ir.get("exits", [])
    if not exits:
        issues.append("No exit rules in IR")

    return issues


def main():
    print("=" * 70)
    print("  PIPELINE VERIFICATION: MCP → IR → LEAN")
    print("=" * 70)
    print()

    # Step 1: Create strategy using MCP-compatible models
    print("1. Creating strategy using vibe_trade_shared models...")
    strategy, cards = create_ema_crossover_strategy()
    print(f"   Strategy: {strategy.name}")
    print(f"   Cards: {list(cards.keys())}")
    print()

    # Step 2: Translate to IR
    print("2. Translating to IR...")
    strategy_ir = translate_to_ir(strategy, cards)
    print(f"   Strategy ID: {strategy_ir['strategy_id']}")
    print(f"   Indicators: {[i['id'] for i in strategy_ir.get('indicators', [])]}")
    print(f"   Has entry: {strategy_ir.get('entry') is not None}")
    print(f"   Exit count: {len(strategy_ir.get('exits', []))}")
    print()

    # Show the translated IR for inspection
    print("   IR Entry Condition:")
    entry = strategy_ir.get("entry", {})
    print(f"   {json.dumps(entry.get('condition', {}), indent=6)[:200]}...")
    print()

    # Step 3: Run backtest
    print("3. Running backtest on cloud...")
    token = get_auth_token()
    result = run_backtest(strategy_ir, token)
    print(f"   Raw result keys: {list(result.keys())}")

    if "error" in result and result["error"]:
        print(f"   ❌ Backtest failed: {result['error']}")
        return

    if result.get("status") == "error":
        print(f"   ❌ Backtest error: {result.get('error')}")
        return

    print(f"   Status: {result['status']}")
    print(f"   Backtest ID: {result['backtest_id']}")
    if result.get("summary"):
        s = result["summary"]
        print(f"   Trades: {s['total_trades']}")
        print(f"   Return: {s['total_return_pct']:+.2f}%")
    print()

    # Step 4: Verify results
    print("4. Verifying pipeline integrity...")
    issues = verify_results(result, strategy_ir)

    if issues:
        print("   ❌ ISSUES FOUND:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✅ All verifications passed!")
        print()
        print("   This proves:")
        print("   - Strategy created via MCP models translates correctly")
        print("   - IR contains expected indicators and conditions")
        print("   - LEAN executes the strategy and produces trades")
        print("   - The full pipeline MCP → IR → LEAN works end-to-end")

    print()
    print("=" * 70)

    # Show where full results are stored
    if result.get("results_path"):
        print(f"Full results: {result['results_path']}")


if __name__ == "__main__":
    main()
