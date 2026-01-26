"""Shared test fixtures and helpers."""

from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment


def calculate_warmup_bars(indicators: list[dict]) -> int:
    """Calculate the warmup period required for a set of indicators.

    Based on QuantConnect LEAN indicator WarmUpPeriod values:
    - SMA, EMA, ATR, BollingerBands, KeltnerChannels, DonchianChannels: WarmUpPeriod = period
    - RSI, ROC: WarmUpPeriod = period + 1
    - ADX: WarmUpPeriod = period * 2

    The warmup_bars is the 0-indexed data bar when indicators first become ready,
    which equals WarmUpPeriod - 1. In StrategyRuntime.py, bar_count only starts
    incrementing after _indicators_ready() returns True, so:
        trading_bar = data_bar - warmup_bars

    Args:
        indicators: List of indicator dicts with 'type' and 'period' keys

    Returns:
        Number of warmup bars (0-indexed first ready bar)
    """
    if not indicators:
        return 0

    max_warmup = 0
    for ind in indicators:
        ind_type = ind.get("type", "").upper()
        period = ind.get("period", 0)

        # Map indicator type to WarmUpPeriod formula
        if ind_type in ("RSI", "ROC"):
            # WarmUpPeriod = period + 1
            warmup_period = period + 1
        elif ind_type == "ADX":
            # WarmUpPeriod = period * 2
            warmup_period = period * 2
        else:
            # SMA, EMA, ATR, BB, KC, DC, etc: WarmUpPeriod = period
            warmup_period = period

        # First ready bar (0-indexed) = WarmUpPeriod - 1
        first_ready_bar = warmup_period - 1

        if first_ready_bar > max_warmup:
            max_warmup = first_ready_bar

    return max_warmup


def calculate_trading_bar(data_bar: int, indicators: list[dict]) -> int:
    """Convert a data bar index to a trading bar index.

    In StrategyRuntime.py, bar_count (trading bar) only starts incrementing
    after _indicators_ready() returns True. This function converts from
    the raw data bar index to the trading bar index.

    Args:
        data_bar: 0-indexed data bar
        indicators: List of indicator dicts with 'type' and 'period' keys

    Returns:
        Trading bar index (bar_count in StrategyRuntime.py)
    """
    warmup = calculate_warmup_bars(indicators)
    return data_bar - warmup


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
