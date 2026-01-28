"""Tests for the new IRTranslator implementation."""

from datetime import datetime, timezone

import pytest
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment

from src.translator.ir import Resolution
from src.translator.ir_translator import TranslationError
from src.translator.translator import IRTranslator

NOW = datetime.now(timezone.utc).isoformat()


class TestSimpleEntry:
    """Tests for basic entry translation."""

    def test_translates_simple_entry(self):
        """Translates a simple entry.rule_trigger strategy."""
        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True, overrides={})
            ],
        )
        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                slots={
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
                        },
                    },
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        assert ir.strategy_id == "test"
        assert ir.symbol == "BTC-USD"
        assert ir.resolution == Resolution.HOUR
        assert ir.entry is not None
        # Entry should have on_fill ops for entry_price capture
        assert len(ir.entry.on_fill) > 0

        # Should have EMA indicators from regime condition
        indicator_ids = [ind.id for ind in ir.indicators]
        assert "ema_20" in indicator_ids
        assert "ema_50" in indicator_ids

        # Should have state vars for entry tracking
        state_ids = [s.id for s in ir.state]
        assert "entry_price" in state_ids
        assert "bars_since_entry" in state_ids


class TestExitTranslation:
    """Tests for exit rule translation."""

    def test_translates_entry_with_exit(self):
        """Translates strategy with both entry and trailing_stop exit."""
        strategy = Strategy(
            id="test",
            name="Entry + Exit Strategy",
            universe=["ETH-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                Attachment(card_id="entry1", role="entry", enabled=True, overrides={}),
                Attachment(card_id="exit1", role="exit", enabled=True, overrides={}),
            ],
        )
        cards = {
            "entry1": Card(
                id="entry1",
                type="entry.rule_trigger",
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                slots={
                    "context": {"tf": "4h", "symbol": "ETH-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": ">",
                                "rhs": {"type": "constant", "value": 100},
                            },
                        },
                    },
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
                },
            ),
            "exit1": Card(
                id="exit1",
                type="exit.trailing_stop",
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                slots={
                    "context": {"tf": "4h", "symbol": "ETH-USD"},
                    "event": {
                        "trail_band": {"band": "keltner", "length": 20, "mult": 2.0},
                    },
                    "action": {"mode": "close"},
                },
            ),
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        assert ir.strategy_id == "test"
        assert ir.symbol == "ETH-USD"
        assert ir.resolution == Resolution.HOUR  # 4h maps to HOUR
        assert ir.entry is not None
        assert len(ir.exits) == 1

        # Trailing stop should register highest_since_entry state
        state_ids = [s.id for s in ir.state]
        assert "highest_since_entry" in state_ids

        # Should have on_bar_invested ops from trailing stop
        assert len(ir.on_bar_invested) > 0


class TestDisabledAttachments:
    """Tests for attachment enabled/disabled behavior."""

    def test_disabled_attachments_skipped(self):
        """Disabled attachments are not translated."""
        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                Attachment(card_id="entry1", role="entry", enabled=True, overrides={}),
                Attachment(card_id="entry2", role="entry", enabled=False, overrides={}),
            ],
        )
        cards = {
            "entry1": Card(
                id="entry1",
                type="entry.rule_trigger",
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": ">",
                                "rhs": {"type": "constant", "value": 100},
                            },
                        },
                    },
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
                },
            ),
            # This card won't be used but must be valid
            "entry2": Card(
                id="entry2",
                type="entry.rule_trigger",
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "indicator", "indicator": "rsi", "period": 14},
                                "op": "<",
                                "rhs": {"type": "constant", "value": 30},
                            },
                        },
                    },
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
                },
            ),
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        # Only one entry should be translated (the enabled one)
        assert ir.entry is not None
        # RSI indicator from disabled entry should NOT be present
        indicator_types = [ind.type for ind in ir.indicators]
        assert "RSI" not in indicator_types


class TestErrorHandling:
    """Tests for error cases."""

    def test_missing_card_raises(self):
        """Missing card raises TranslationError."""
        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                Attachment(card_id="nonexistent", role="entry", enabled=True, overrides={})
            ],
        )
        cards = {}  # Empty - no cards

        translator = IRTranslator(strategy, cards)
        with pytest.raises(TranslationError, match="Card not found: nonexistent"):
            translator.translate()


class TestGateTranslation:
    """Tests for gate rule translation."""

    def test_translates_regime_gate(self):
        """Translates a regime gate that filters entries."""
        strategy = Strategy(
            id="test",
            name="Gated Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                Attachment(card_id="gate1", role="gate", enabled=True, overrides={}),
                Attachment(card_id="entry1", role="entry", enabled=True, overrides={}),
            ],
        )
        cards = {
            "gate1": Card(
                id="gate1",
                type="gate.regime",
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                slots={
                    "context": {"tf": "1d", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "metric": "trend_adx",
                            "op": ">",
                            "value": 25,
                            "lookback_bars": 14,
                        },
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]},
                },
            ),
            "entry1": Card(
                id="entry1",
                type="entry.rule_trigger",
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                slots={
                    "context": {"tf": "1d", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": ">",
                                "rhs": {"type": "constant", "value": 50000},
                            },
                        },
                    },
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
                },
            ),
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        assert ir.strategy_id == "test"
        assert len(ir.gates) == 1
        assert ir.gates[0].mode == "allow"
        assert ir.gates[0].target_roles == ["entry"]
        assert ir.entry is not None

        # ADX indicator should be registered
        indicator_ids = [ind.id for ind in ir.indicators]
        assert "adx_14" in indicator_ids
