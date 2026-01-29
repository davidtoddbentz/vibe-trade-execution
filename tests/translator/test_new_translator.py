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


class TestEntryFieldRejection:
    """Tests for rejection of unsupported entry action fields."""

    def _make_entry_strategy(self, action_overrides: dict):
        """Helper to create a minimal entry strategy with custom action fields."""
        action = {"direction": "long", "position_policy": {"mode": "single"}}
        action.update(action_overrides)
        return Strategy(
            id="test",
            name="Test",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True, overrides={})
            ],
        ), {
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
                    "action": action,
                },
            )
        }

    def test_non_market_execution_raises(self):
        """Non-market order types raise until implemented."""
        strategy, cards = self._make_entry_strategy({
            "execution": {"order_type": "limit", "limit_price": 100.0}
        })
        translator = IRTranslator(strategy, cards)
        with pytest.raises(TranslationError, match="not yet supported"):
            translator.translate()

    def test_market_execution_allowed(self):
        """Market execution should not raise."""
        strategy, cards = self._make_entry_strategy({
            "execution": {"order_type": "market"}
        })
        translator = IRTranslator(strategy, cards)
        ir = translator.translate()
        assert ir is not None

    def test_close_confirm_accepted(self):
        """close_confirm is a no-op (engine evaluates on bar close by default)."""
        strategy, cards = self._make_entry_strategy({
            "confirm": "close_confirm"
        })
        translator = IRTranslator(strategy, cards)
        ir = translator.translate()
        assert ir.entry is not None

    def test_unsupported_confirm_raises(self):
        """Unknown confirm modes raise TranslationError (via Pydantic validation)."""
        strategy, cards = self._make_entry_strategy({
            "confirm": "immediate"
        })
        translator = IRTranslator(strategy, cards)
        with pytest.raises(TranslationError, match="Invalid slots"):
            translator.translate()

    def test_cooldown_bars_maps_to_min_bars_between(self):
        """cooldown_bars maps to position_policy.min_bars_between."""
        strategy, cards = self._make_entry_strategy({
            "cooldown_bars": 5
        })
        translator = IRTranslator(strategy, cards)
        ir = translator.translate()
        assert ir.entry.action.position_policy.min_bars_between == 5

    def test_cooldown_bars_does_not_override_explicit(self):
        """cooldown_bars does not override explicit min_bars_between."""
        strategy, cards = self._make_entry_strategy({
            "cooldown_bars": 5,
            "position_policy": {"mode": "accumulate", "min_bars_between": 10},
        })
        translator = IRTranslator(strategy, cards)
        ir = translator.translate()
        assert ir.entry.action.position_policy.min_bars_between == 10

    def test_max_entries_per_day_maps_to_policy(self):
        """max_entries_per_day maps to position_policy.max_entries_per_day."""
        strategy, cards = self._make_entry_strategy({
            "max_entries_per_day": 3
        })
        translator = IRTranslator(strategy, cards)
        ir = translator.translate()
        assert ir.entry.action.position_policy.max_entries_per_day == 3

    def test_sizing_min_usd_passes_through(self):
        """min_usd passes through to SetHoldingsAction."""
        strategy, cards = self._make_entry_strategy({
            "sizing": {"type": "pct_equity", "pct": 50, "min_usd": 100.0}
        })
        translator = IRTranslator(strategy, cards)
        ir = translator.translate()
        assert ir.entry.action.min_usd == 100.0

    def test_sizing_max_usd_passes_through(self):
        """max_usd passes through to SetHoldingsAction."""
        strategy, cards = self._make_entry_strategy({
            "sizing": {"type": "pct_equity", "pct": 50, "max_usd": 50000.0}
        })
        translator = IRTranslator(strategy, cards)
        ir = translator.translate()
        assert ir.entry.action.max_usd == 50000.0


class TestExitFieldRejection:
    """Tests for rejection of unsupported exit action fields."""

    def _make_exit_strategy(self, exit_action: dict):
        """Helper to create a strategy with entry + exit where exit action has custom fields."""
        return Strategy(
            id="test",
            name="Test",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True, overrides={}),
                Attachment(card_id="exit", role="exit", enabled=True, overrides={}),
            ],
        ), {
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
            ),
            "exit": Card(
                id="exit",
                type="exit.trailing_stop",
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"trail_band": {"band": "keltner", "length": 20, "mult": 2.0}},
                    "action": exit_action,
                },
            ),
        }

    def test_exit_close_confirm_accepted(self):
        """Exit close_confirm is a no-op (engine evaluates on bar close by default)."""
        strategy, cards = self._make_exit_strategy({"mode": "close", "confirm": "close_confirm"})
        translator = IRTranslator(strategy, cards)
        ir = translator.translate()
        assert len(ir.exits) >= 1

    def test_exit_unsupported_confirm_raises(self):
        """Exit with unknown confirm mode raises TranslationError (via Pydantic validation)."""
        strategy, cards = self._make_exit_strategy({"mode": "close", "confirm": "immediate"})
        translator = IRTranslator(strategy, cards)
        with pytest.raises(TranslationError, match="Invalid slots"):
            translator.translate()


class TestGateFieldRejection:
    """Tests for rejection of unsupported gate action fields."""

    def _make_gate_strategy(self, gate_action: dict):
        """Helper to create a strategy with gate + entry where gate action has custom fields."""
        return Strategy(
            id="test",
            name="Test",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                Attachment(card_id="gate", role="gate", enabled=True, overrides={}),
                Attachment(card_id="entry", role="entry", enabled=True, overrides={}),
            ],
        ), {
            "gate": Card(
                id="gate",
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
                    "action": gate_action,
                },
            ),
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                slots={
                    "context": {"tf": "1d", "symbol": "BTC-USD"},
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
            ),
        }

    def test_gate_target_tags_raises(self):
        """Gate target_tags raises until implemented."""
        strategy, cards = self._make_gate_strategy({
            "mode": "allow", "target_roles": ["entry"], "target_tags": ["trend"]
        })
        translator = IRTranslator(strategy, cards)
        with pytest.raises(TranslationError, match="target_tags"):
            translator.translate()

    def test_gate_target_ids_raises(self):
        """Gate target_ids raises until implemented."""
        strategy, cards = self._make_gate_strategy({
            "mode": "allow", "target_roles": ["entry"], "target_ids": ["entry_1"]
        })
        translator = IRTranslator(strategy, cards)
        with pytest.raises(TranslationError, match="target_ids"):
            translator.translate()


class TestOverlayFieldRejection:
    """Tests for rejection of unsupported overlay action fields."""

    def _make_overlay_strategy(self, overlay_action: dict):
        """Helper to create a strategy with entry + overlay where overlay has custom fields."""
        return Strategy(
            id="test",
            name="Test",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True, overrides={}),
                Attachment(card_id="overlay", role="overlay", enabled=True, overrides={}),
            ],
        ), {
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
            ),
            "overlay": Card(
                id="overlay",
                type="overlay.regime_scaler",
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "regime": {
                            "type": "regime",
                            "regime": {
                                "metric": "vol_atr_pct",
                            "op": ">",
                            "value": 2.0,
                                "lookback_bars": 14,
                            },
                        },
                    },
                    "action": overlay_action,
                },
            ),
        }

    def test_overlay_scale_risk_frac_raises(self):
        """scale_risk_frac raises until implemented."""
        strategy, cards = self._make_overlay_strategy({
            "scale_size_frac": 1.0, "scale_risk_frac": 0.5, "target_roles": ["entry"]
        })
        translator = IRTranslator(strategy, cards)
        with pytest.raises(TranslationError, match="scale_risk_frac"):
            translator.translate()

    def test_overlay_target_tags_raises(self):
        """Overlay target_tags raises until implemented."""
        strategy, cards = self._make_overlay_strategy({
            "scale_size_frac": 0.5, "target_tags": ["trend"], "target_roles": ["entry"]
        })
        translator = IRTranslator(strategy, cards)
        with pytest.raises(TranslationError, match="target_tags"):
            translator.translate()

    def test_overlay_target_ids_raises(self):
        """Overlay target_ids raises until implemented."""
        strategy, cards = self._make_overlay_strategy({
            "scale_size_frac": 0.5, "target_ids": ["entry_1"], "target_roles": ["entry"]
        })
        translator = IRTranslator(strategy, cards)
        with pytest.raises(TranslationError, match="target_ids"):
            translator.translate()
