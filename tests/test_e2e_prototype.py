"""E2E test prototype: Typed Pydantic archetype models → BacktestService → LEAN → verified results.

Tests the full execution path without MCP or Firestore:
  1. Build Strategy + Cards using actual archetype Pydantic models
  2. Seed MockDataService with test OHLCV bars (same path as production)
  3. BacktestService internally runs IRTranslator → LEAN container
  4. Assert on trade details, summary metrics, and equity curve

Requirements:
  - LEAN container running: cd vibe-trade-lean && make run-api
  - No Firestore or MCP needed
"""

from datetime import date, datetime, timezone

import httpx
import pytest
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.archetypes.entry.rule_trigger import (
    EntryRuleTrigger,
    EventSlot,
)
from vibe_trade_shared.models.archetypes.exit.rule_trigger import (
    ExitEventSlot,
    ExitRuleTrigger,
)
from vibe_trade_shared.models.archetypes.primitives import (
    CompareSpec,
    ConditionSpec,
    ContextSpec,
    EntryActionSpec,
    ExitActionSpec,
    SignalRef,
)
from vibe_trade_shared.models.data import OHLCVBar
from vibe_trade_shared.models.ir import PositionPolicy
from vibe_trade_shared.models.strategy import Attachment

from src.models.lean_backtest import BacktestConfig
from src.service.backtest_service import BacktestService
from src.service.data_service import MockDataService

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------


def _lean_url() -> str | None:
    """Return the LEAN backtest URL if available, else None."""
    for port in [8083, 8081]:
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            if r.status_code == 200:
                return f"http://localhost:{port}/backtest"
        except (httpx.ConnectError, httpx.TimeoutException):
            continue
    return None


requires_lean = pytest.mark.skipif(
    _lean_url() is None,
    reason="LEAN not running. Start with: cd vibe-trade-lean && make run-api",
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_bars(
    prices: list[float],
    start: datetime | None = None,
    interval_ms: int = 3_600_000,
) -> list[OHLCVBar]:
    """Build hourly OHLCV bars from close prices.

    Each bar: open=close, high=close+1, low=close-1, volume=1000.
    """
    if start is None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    base_ms = int(start.timestamp() * 1000)
    return [
        OHLCVBar(t=base_ms + i * interval_ms, o=p, h=p + 1, l=p - 1, c=p, v=1000.0)
        for i, p in enumerate(prices)
    ]


def card_from_archetype(card_id: str, archetype: EntryRuleTrigger | ExitRuleTrigger) -> Card:
    """Build a Card from a typed archetype instance.

    Dumps the archetype's fields into Card.slots, preserving full type safety
    up to the Card boundary.
    """
    return Card(
        id=card_id,
        type=archetype.TYPE_ID,
        name=card_id,
        schema_etag="test",
        slots=archetype.model_dump(exclude_none=True),
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )


def make_strategy(
    cards: list[Card],
    name: str = "Test Strategy",
    symbol: str = "BTC-USD",
) -> tuple[Strategy, dict[str, Card]]:
    """Build Strategy from a list of Cards."""
    attachments = [
        Attachment(
            card_id=card.id,
            role=card.type.split(".")[0],
            enabled=True,
            overrides={},
        )
        for card in cards
    ]
    strategy = Strategy(
        id=f"test-{name.lower().replace(' ', '-')}",
        name=name,
        universe=[symbol],
        attachments=attachments,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )
    cards_dict = {card.id: card for card in cards}
    return strategy, cards_dict


# ---------------------------------------------------------------------------
# Reusable condition builders
# ---------------------------------------------------------------------------


def price_above(threshold: float) -> ConditionSpec:
    """Close price > threshold."""
    return ConditionSpec(
        type="compare",
        compare=CompareSpec(
            lhs=SignalRef(type="price", field="close"),
            op=">",
            rhs=threshold,
        ),
    )


def price_below(threshold: float) -> ConditionSpec:
    """Close price < threshold."""
    return ConditionSpec(
        type="compare",
        compare=CompareSpec(
            lhs=SignalRef(type="price", field="close"),
            op="<",
            rhs=threshold,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_lean
class TestPriceThresholdEntryExit:
    """Simplest possible E2E: enter when price > 105, exit when price < 95.

    Data shape (100 bars, hourly):
        Bars 0-59:   price = 100 (flat, below entry threshold)
        Bars 60-79:  price = 110 (above 105 → triggers entry)
        Bars 80-99:  price = 90  (below 95 → triggers exit)

    Expected: 1 trade, long, entry ~110, exit ~90, losing trade.
    """

    def test_single_entry_exit_cycle(self):
        # -- Data: seed into mock data service --
        bars = make_bars(
            [100.0] * 60    # flat, no signal
            + [110.0] * 20  # above 105 → entry
            + [90.0] * 20   # below 95 → exit
        )
        data_service = MockDataService()
        data_service.seed("BTC-USD", "1h", bars)

        service = BacktestService(
            data_service=data_service,
            backtest_url=_lean_url(),
        )

        # -- Strategy: actual typed archetype models --
        entry = EntryRuleTrigger(
            context=ContextSpec(symbol="BTC-USD", tf="1h"),
            event=EventSlot(condition=price_above(105.0)),
            action=EntryActionSpec(
                direction="long",
                position_policy=PositionPolicy(mode="single"),
            ),
        )
        exit_ = ExitRuleTrigger(
            context=ContextSpec(symbol="BTC-USD"),
            event=ExitEventSlot(condition=price_below(95.0)),
            action=ExitActionSpec(mode="close"),
        )

        entry_card = card_from_archetype("entry_1", entry)
        exit_card = card_from_archetype("exit_1", exit_)
        strategy, cards = make_strategy([entry_card, exit_card])

        # -- Execute: same call as production --
        result = service.run_backtest(
            strategy=strategy,
            cards=cards,
            config=BacktestConfig(
                start_date=date(2024, 1, 1),
                end_date=date(2024, 1, 5),
                symbol="BTC-USD",
                resolution="1h",
                initial_cash=100_000.0,
                fee_pct=0.1,
                slippage_pct=0.05,
            ),
        )

        # -- Assertions --

        # 1. Backtest succeeded
        assert result.status == "success", f"Backtest failed: {result.error}"
        assert result.response is not None, "No LEAN response"
        resp = result.response

        # 2. Summary
        assert resp.summary is not None, "No summary in response"
        summary = resp.summary
        assert summary.total_trades == 1
        assert summary.winning_trades == 0
        assert summary.losing_trades == 1
        assert summary.total_pnl < 0, "Should lose money (bought 110, sold 90)"
        assert summary.total_pnl_pct < 0

        # 3. Trade details
        assert len(resp.trades) == 1
        trade = resp.trades[0]
        assert trade.direction == "long"
        assert trade.entry_price == pytest.approx(110.0, abs=1.0)
        assert trade.exit_price == pytest.approx(90.0, abs=1.0)
        assert trade.exit_reason is not None
        assert trade.pnl is not None
        assert trade.pnl < 0
        assert trade.quantity > 0
        assert trade.entry_time is not None
        assert trade.exit_time is not None
        assert trade.exit_time > trade.entry_time

        # 4. Equity curve (direct assertions, no hasattr guards)
        assert resp.equity_curve is not None
        assert len(resp.equity_curve) > 0
        first_point = resp.equity_curve[0]
        last_point = resp.equity_curve[-1]
        assert first_point.equity == pytest.approx(100_000.0, rel=0.01)
        assert last_point.equity < 100_000.0, "Should lose money"
        assert last_point.equity > 70_000.0, "Shouldn't lose more than 30%"
        assert last_point.equity < 95_000.0, "Should lose meaningful amount"
        assert last_point.equity > 0, "Not bankrupt"
