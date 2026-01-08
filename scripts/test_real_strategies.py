#!/usr/bin/env python3
"""Test translation of real strategies from Firestore."""

import os
import sys

# Set up for production Firestore
os.environ["GOOGLE_CLOUD_PROJECT"] = "vibe-trade-475704"
os.environ["FIRESTORE_DATABASE"] = "strategy"
# Remove emulator host if set
os.environ.pop("FIRESTORE_EMULATOR_HOST", None)

from vibe_trade_shared.db import CardRepository, FirestoreClient, StrategyRepository
from vibe_trade_shared.models import Card, Strategy

from src.translator import IRTranslator

# Initialize Firestore client
PROJECT = "vibe-trade-475704"
DATABASE = "strategy"
client = FirestoreClient.get_client(PROJECT, DATABASE)


def fetch_strategies() -> list[Strategy]:
    """Fetch all strategies from Firestore."""
    repo = StrategyRepository(client)
    strategies = repo.get_all()
    return strategies


def fetch_cards_for_strategy(strategy: Strategy) -> dict[str, Card]:
    """Fetch all cards attached to a strategy."""
    repo = CardRepository(client)
    cards = {}
    for attachment in strategy.attachments:
        card = repo.get_by_id(attachment.card_id)
        if card:
            cards[card.id] = card
    return cards


def test_translation(strategy: Strategy, cards: dict[str, Card]) -> tuple[bool, str, list[str]]:
    """
    Test if a strategy translates successfully.

    Returns: (success, ir_json or error, warnings)
    """
    try:
        translator = IRTranslator(strategy, cards)
        result = translator.translate()
        ir_json = result.ir.to_json(indent=2)
        return True, ir_json, result.warnings
    except Exception as e:
        return False, str(e), []


def main():
    print("=" * 60)
    print("Testing Real Strategy Translation")
    print("=" * 60)
    print()

    # Fetch all strategies
    print("Fetching strategies from Firestore...")
    strategies = fetch_strategies()
    print(f"Found {len(strategies)} strategies")
    print()

    results = []

    for strategy in strategies:
        print("-" * 60)
        print(f"Strategy: {strategy.name}")
        print(f"  ID: {strategy.id}")
        print(f"  Universe: {strategy.universe}")
        print(f"  Status: {strategy.status}")
        print(f"  Attachments: {len(strategy.attachments)}")

        # Fetch cards
        cards = fetch_cards_for_strategy(strategy)
        print(f"  Cards fetched: {len(cards)}")

        for card_id, card in cards.items():
            print(f"    - {card.type} ({card_id})")

        # Test translation
        success, output, warnings = test_translation(strategy, cards)

        if success:
            print("  ✅ Translation successful")
            if warnings:
                print(f"  ⚠️  Warnings: {len(warnings)}")
                for w in warnings:
                    print(f"      - {w}")
            # Show a snippet of the IR
            lines = output.split("\n")
            print("  IR Preview (first 20 lines):")
            for line in lines[:20]:
                print(f"    {line}")
            if len(lines) > 20:
                print(f"    ... ({len(lines) - 20} more lines)")
        else:
            print(f"  ❌ Translation failed: {output}")

        results.append(
            {
                "name": strategy.name,
                "id": strategy.id,
                "success": success,
                "warnings": warnings,
                "error": None if success else output,
            }
        )
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    no_warnings = sum(1 for r in results if r["success"] and not r["warnings"])
    with_warnings = successful - no_warnings

    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"  - Fully translated (no warnings): {no_warnings}")
    print(f"  - Partial (unsupported archetypes): {with_warnings}")
    print(f"Failed: {failed}")

    # Count unsupported archetypes
    archetype_counts = {}
    for r in results:
        for w in r["warnings"]:
            if "Unsupported" in w:
                # Extract archetype from warning like "Unsupported entry archetype: entry.squeeze_expansion"
                parts = w.split(": ")
                if len(parts) == 2:
                    archetype = parts[1]
                    archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

    if archetype_counts:
        print("\nUnsupported archetypes (by frequency):")
        for arch, count in sorted(archetype_counts.items(), key=lambda x: -x[1]):
            print(f"  {count:3d}x {arch}")

    if no_warnings > 0:
        print("\n✅ Fully translated strategies:")
        for r in results:
            if r["success"] and not r["warnings"]:
                print(f"  - {r['name']} ({r['id']})")

    if failed > 0:
        print("\n❌ Failed strategies:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['name']}: {r['error']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
