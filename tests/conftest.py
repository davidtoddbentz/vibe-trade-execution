"""Shared test fixtures and helpers."""

from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment


def make_strategy(
    cards_dict: dict[str, dict],
    name: str = "Test Strategy",
    symbol: str = "BTC-USD",
) -> tuple[Strategy, dict[str, Card]]:
    """Create a strategy with cards from a simplified dict specification.

    Args:
        cards_dict: Dict mapping card_id to card spec with keys:
            - type: Card type (e.g., "entry.rule_trigger")
            - slots: Card slots dict
            - name: Optional card name (defaults to card_id)
        name: Strategy name (defaults to "Test Strategy")
        symbol: Trading symbol (defaults to "BTC-USD")

    Returns:
        Tuple of (Strategy, dict of Cards)

    Example:
        strategy, cards = make_strategy({
            "entry_1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {...}},
                    "action": {"direction": "long", "position_policy": {"mode": "single"}}
                }
            }
        })
    """
    attachments = []
    cards = {}

    for card_id, card_spec in cards_dict.items():
        role = card_spec["type"].split(".")[0]  # entry, exit, gate, overlay
        attachments.append(Attachment(card_id=card_id, role=role, enabled=True, overrides={}))
        cards[card_id] = Card(
            id=card_id,
            type=card_spec["type"],
            name=card_spec.get("name", card_id),
            schema_etag="test",
            slots=card_spec["slots"],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

    # Generate strategy ID from name
    strategy_id = f"test-{name.lower().replace(' ', '-')}" if name != "Test Strategy" else "test-strategy"

    strategy = Strategy(
        id=strategy_id,
        name=name,
        universe=[symbol],
        attachments=attachments,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )

    return strategy, cards
