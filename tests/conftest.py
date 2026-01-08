"""Shared test fixtures and mock classes.

This module centralizes all mock classes and helper functions used across tests.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment

from src.translator.evaluator import EvalContext
from src.translator.ir import PriceField


# =============================================================================
# Mock Classes for Evaluator Tests
# =============================================================================


class MockIndicatorCurrent:
    """Mock for indicator.Current property."""

    def __init__(self, value: float):
        self.Value = value


class MockIndicator:
    """Mock for a simple indicator (EMA, SMA, ROC, etc.)."""

    def __init__(self, value: float, is_ready: bool = True):
        self.Current = MockIndicatorCurrent(value)
        self.IsReady = is_ready


class MockBandIndicator:
    """Mock for band indicators (BB, KC, Donchian)."""

    def __init__(self, upper: float, middle: float, lower: float, is_ready: bool = True):
        self.UpperBand = MockIndicator(upper)
        self.MiddleBand = MockIndicator(middle)
        self.LowerBand = MockIndicator(lower)
        self.IsReady = is_ready


class MockPriceBar:
    """Mock for LEAN price bar."""

    def __init__(
        self,
        open_: float = 100.0,
        high: float = 105.0,
        low: float = 95.0,
        close: float = 102.0,
        volume: float = 1000.0,
    ):
        self.Open = open_
        self.High = high
        self.Low = low
        self.Close = close
        self.Volume = volume


class MockAlgorithm:
    """Mock for LEAN algorithm."""

    def __init__(self):
        self.holdings_calls: list[tuple[Any, float]] = []
        self.liquidate_calls: list[Any] = []
        self.market_order_calls: list[tuple[Any, float]] = []

    def SetHoldings(self, symbol: Any, allocation: float) -> None:
        self.holdings_calls.append((symbol, allocation))

    def Liquidate(self, symbol: Any) -> None:
        self.liquidate_calls.append(symbol)

    def MarketOrder(self, symbol: Any, quantity: float) -> None:
        self.market_order_calls.append((symbol, quantity))


# =============================================================================
# Dataclass-based Mocks for Simulation Tests
# =============================================================================


@dataclass
class MockBar:
    """Mock price bar for simulation scenarios."""

    open: float
    high: float
    low: float
    close: float
    volume: float = 1000.0


@dataclass
class Signal:
    """Expected signal at a specific bar index."""

    bar_index: int
    signal_type: str  # "entry" or "exit"
    reason: str = ""


@dataclass
class BarData:
    """Bar data with indicator values for a single simulation bar."""

    bar: MockBar
    indicators: dict[str, Any]  # indicator_id -> value or MockIndicator
    hour: int = 12  # Default midday
    minute: int = 0
    day_of_week: int = 2  # Default Wednesday


@dataclass
class SimulationScenario:
    """A complete simulation test scenario."""

    name: str
    description: str
    bars: list[BarData]
    expected_signals: list[Signal]
    initial_state: dict[str, Any] = field(default_factory=dict)


class SimulationPriceBar:
    """Adapter for MockBar to match PriceBarProtocol."""

    def __init__(self, bar: MockBar):
        self._bar = bar

    @property
    def Open(self) -> float:
        return self._bar.open

    @property
    def High(self) -> float:
        return self._bar.high

    @property
    def Low(self) -> float:
        return self._bar.low

    @property
    def Close(self) -> float:
        return self._bar.close

    @property
    def Volume(self) -> float:
        return self._bar.volume


# =============================================================================
# Strategy/Card Creation Helpers
# =============================================================================


def make_strategy(cards_dict: dict[str, dict]) -> tuple[Strategy, dict[str, Card]]:
    """Create a strategy with cards from a simplified dict specification.

    Args:
        cards_dict: Dict mapping card_id to card spec with keys:
            - type: Card type (e.g., "entry.rule_trigger")
            - slots: Card slots dict
            - name: Optional card name (defaults to card_id)

    Returns:
        Tuple of (Strategy, dict of Cards)

    Example:
        strategy, cards = make_strategy({
            "entry_1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {...}},
                    "action": {"direction": "long"}
                }
            }
        })
    """
    attachments = []
    cards = {}

    for card_id, card_spec in cards_dict.items():
        role = card_spec["type"].split(".")[0]  # entry, exit, gate, overlay
        attachments.append(
            Attachment(card_id=card_id, role=role, enabled=True, overrides={})
        )
        cards[card_id] = Card(
            id=card_id,
            type=card_spec["type"],
            name=card_spec.get("name", card_id),
            schema_etag="test",
            slots=card_spec["slots"],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

    strategy = Strategy(
        id="test-strategy",
        name="Test Strategy",
        universe=["BTC-USD"],
        attachments=attachments,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )

    return strategy, cards


def create_entry_strategy(
    condition_type: str,
    condition_spec: dict[str, Any],
    symbol: str = "BTC-USD",
    timeframe: str = "1h",
    direction: str = "long",
) -> tuple[Strategy, dict[str, Card]]:
    """Create a simple entry-only strategy for testing conditions.

    Args:
        condition_type: The condition type (e.g., "regime", "band_event")
        condition_spec: The condition specification dict
        symbol: Trading symbol
        timeframe: Timeframe string
        direction: Trade direction ("long" or "short")

    Returns:
        Tuple of (Strategy, dict of Cards)
    """
    return make_strategy({
        "entry_1": {
            "type": "entry.rule_trigger",
            "slots": {
                "context": {"tf": timeframe, "symbol": symbol},
                "event": {
                    "condition": {
                        "type": condition_type,
                        condition_type: condition_spec,
                    }
                },
                "action": {"direction": direction},
            },
        }
    })


# =============================================================================
# Common Fixtures
# =============================================================================


@pytest.fixture
def simple_eval_context() -> EvalContext:
    """Create a simple evaluation context for unit tests."""
    return EvalContext(
        indicators={
            "ema_10": MockIndicator(105.0),
            "ema_20": MockIndicator(100.0),
            "roc_5": MockIndicator(0.05),
        },
        state={"bars_since_entry": 0},
        price_bar=MockPriceBar(100.0, 105.0, 95.0, 102.0),
    )


@pytest.fixture
def band_eval_context() -> EvalContext:
    """Create an evaluation context with band indicators."""
    return EvalContext(
        indicators={
            "bb_20": MockBandIndicator(upper=110.0, middle=100.0, lower=90.0),
            "ema_fast": MockIndicator(105.0),
            "ema_slow": MockIndicator(100.0),
        },
        state={},
        price_bar=MockPriceBar(100.0, 105.0, 95.0, 102.0),
    )


@pytest.fixture
def ema_crossover_strategy() -> tuple[Strategy, dict[str, Card]]:
    """Create the standard EMA Crossover Long strategy for testing."""
    return make_strategy({
        "entry_1": {
            "type": "entry.rule_trigger",
            "slots": {
                "context": {"tf": "1h", "symbol": "BTC-USD"},
                "event": {
                    "condition": {
                        "type": "regime",
                        "regime": {
                            "metric": "trend_ma_relation",
                            "op": ">",
                            "value": 0,
                            "ma_fast": 20,
                            "ma_slow": 50,
                        },
                    }
                },
                "action": {"direction": "long"},
            },
        },
        "exit_1": {
            "type": "exit.rule_trigger",
            "slots": {
                "context": {"tf": "1h", "symbol": "BTC-USD"},
                "event": {
                    "condition": {
                        "type": "regime",
                        "regime": {
                            "metric": "trend_ma_relation",
                            "op": "<",
                            "value": 0,
                            "ma_fast": 20,
                            "ma_slow": 50,
                        },
                    }
                },
                "action": {"mode": "close"},
            },
        },
    })
