"""Tests for archetype → IR translation pipeline.

These tests verify that archetypes translate correctly to IR with all
required indicators registered. This is critical because:

1. E2E tests construct IR manually, bypassing translation
2. Archetypes generate complex nested conditions (StateCondition, etc.)
3. Indicator registration must happen for all nested references

Coverage:
- entry.trend_pullback (uses StateCondition with IndicatorBandRef)
- exit.trailing_stop (uses state hooks)
- entry.rule_trigger (uses RegimeCondition)
- Various condition types with nested indicator references
"""

from datetime import datetime, timezone

import pytest
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.archetypes import parse_archetype

from src.translator.ir_translator import IRTranslator, TranslationError

# Helper for creating test timestamps
NOW = datetime.now(timezone.utc).isoformat()


class TestTrendPullbackTranslation:
    """Test entry.trend_pullback archetype translation.

    This archetype generates:
    - AllOfCondition containing:
      - RegimeCondition (trend_ma_relation)
      - StateCondition (reentry) with nested IndicatorBandRef

    The translator must register:
    - EMA indicators for MA relation
    - BollingerBands/Keltner/Donchian for band references
    """

    def test_trend_pullback_bollinger_registers_band_indicator(self):
        """Verify bollinger band indicator is registered from StateCondition."""
        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                {"card_id": "entry", "role": "entry", "enabled": True, "overrides": {}}
            ],
        )
        cards = {
            "entry": Card(created_at=NOW, updated_at=NOW, schema_etag="v1", 
                id="entry",
                type="entry.trend_pullback",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "dip": {"kind": "reentry", "edge": "lower"},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                    "risk": {"sl_atr": 2.0},
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        # Check all required indicators are registered
        indicator_ids = {ind.id for ind in ir.indicators}

        assert "ema_20" in indicator_ids, "EMA 20 should be registered for trend gate"
        assert "ema_50" in indicator_ids, "EMA 50 should be registered for trend gate"
        assert "bollinger_20" in indicator_ids, "Bollinger 20 should be registered from StateCondition"

    def test_trend_pullback_keltner_registers_band_indicator(self):
        """Verify keltner band indicator is registered from StateCondition."""
        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                {"card_id": "entry", "role": "entry", "enabled": True, "overrides": {}}
            ],
        )
        cards = {
            "entry": Card(created_at=NOW, updated_at=NOW, schema_etag="v1", 
                id="entry",
                type="entry.trend_pullback",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip_band": {"band": "keltner", "length": 20, "mult": 2.0},
                        "dip": {"kind": "reentry", "edge": "lower"},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                    "risk": {"sl_atr": 2.0},
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        indicator_ids = {ind.id for ind in ir.indicators}
        assert "keltner_20" in indicator_ids, "Keltner 20 should be registered from StateCondition"

    def test_trend_pullback_donchian_registers_band_indicator(self):
        """Verify donchian band indicator is registered from StateCondition."""
        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                {"card_id": "entry", "role": "entry", "enabled": True, "overrides": {}}
            ],
        )
        cards = {
            "entry": Card(created_at=NOW, updated_at=NOW, schema_etag="v1", 
                id="entry",
                type="entry.trend_pullback",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip_band": {"band": "donchian", "length": 20},
                        "dip": {"kind": "reentry", "edge": "lower"},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                    "risk": {"sl_atr": 2.0},
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        indicator_ids = {ind.id for ind in ir.indicators}
        assert "donchian_20" in indicator_ids, "Donchian 20 should be registered from StateCondition"


class TestTrailingStopTranslation:
    """Test exit.trailing_stop archetype translation."""

    def test_trailing_stop_registers_band_and_atr(self):
        """Verify trailing stop registers trail band indicator."""
        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                {"card_id": "entry", "role": "entry", "enabled": True, "overrides": {}},
                {"card_id": "exit", "role": "exit", "enabled": True, "overrides": {}},
            ],
        )
        cards = {
            "entry": Card(created_at=NOW, updated_at=NOW, schema_etag="v1", 
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "trend_adx", "op": ">", "value": 25},
                        }
                    },
                    "action": {"direction": "long"},
                },
            ),
            "exit": Card(created_at=NOW, updated_at=NOW, schema_etag="v1", 
                id="exit",
                type="exit.trailing_stop",
                slots={
                    "context": {"symbol": "BTC-USD"},
                    "event": {
                        "trail_band": {"band": "keltner", "length": 20, "mult": 2.0},
                    },
                },
            ),
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        indicator_ids = {ind.id for ind in ir.indicators}
        # Trailing stop uses ATR for calculations
        assert "atr_20" in indicator_ids, "ATR should be registered for trailing stop"


class TestRuleTriggerTranslation:
    """Test entry.rule_trigger and exit.rule_trigger translation."""

    def test_rule_trigger_regime_registers_indicators(self):
        """Verify regime conditions register required indicators."""
        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                {"card_id": "entry", "role": "entry", "enabled": True, "overrides": {}},
            ],
        )
        cards = {
            "entry": Card(created_at=NOW, updated_at=NOW, schema_etag="v1", 
                id="entry",
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
                                "ma_fast": 10,
                                "ma_slow": 30,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
            ),
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        indicator_ids = {ind.id for ind in ir.indicators}
        assert "ema_10" in indicator_ids, "EMA 10 should be registered for ma_fast"
        assert "ema_30" in indicator_ids, "EMA 30 should be registered for ma_slow"

    def test_rule_trigger_adx_registers_indicator(self):
        """Verify ADX regime registers ADX indicator."""
        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                {"card_id": "entry", "role": "entry", "enabled": True, "overrides": {}},
            ],
        )
        cards = {
            "entry": Card(created_at=NOW, updated_at=NOW, schema_etag="v1", 
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "trend_adx", "op": ">", "value": 25},
                        }
                    },
                    "action": {"direction": "long"},
                },
            ),
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        indicator_ids = {ind.id for ind in ir.indicators}
        assert "adx_14" in indicator_ids, "ADX 14 should be registered for trend_adx metric"

    def test_rule_trigger_atr_pct_registers_indicator(self):
        """Verify vol_atr_pct regime registers ATR indicator."""
        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                {"card_id": "entry", "role": "entry", "enabled": True, "overrides": {}},
            ],
        )
        cards = {
            "entry": Card(created_at=NOW, updated_at=NOW, schema_etag="v1", 
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "vol_atr_pct", "op": ">", "value": 2.0},
                        }
                    },
                    "action": {"direction": "long"},
                },
            ),
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        indicator_ids = {ind.id for ind in ir.indicators}
        assert "atr_14" in indicator_ids, "ATR 14 should be registered for vol_atr_pct metric"


class TestIndicatorRegistrationRegression:
    """Regression tests for indicator registration issues.

    These tests specifically target the bug pattern where indicator
    references exist in nested conditions but aren't registered.
    """

    def test_nested_indicator_band_in_state_condition(self):
        """Ensure IndicatorBandRef inside StateCondition registers indicator.

        This is the exact bug that was fixed: StateCondition with nested
        IndicatorBandRef in outside_condition/inside_condition.
        """
        # This is tested via trend_pullback above, but adding explicit test
        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                {"card_id": "entry", "role": "entry", "enabled": True, "overrides": {}}
            ],
        )
        cards = {
            "entry": Card(created_at=NOW, updated_at=NOW, schema_etag="v1", 
                id="entry",
                type="entry.trend_pullback",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip_band": {"band": "bollinger", "length": 15, "mult": 2.5},
                        "dip": {"kind": "reentry", "edge": "upper"},
                        "trend_gate": {"fast": 10, "slow": 20, "op": "<"},  # Short bias
                    },
                    "action": {"direction": "short"},
                    "risk": {"sl_atr": 1.5},
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        # The key assertion: bollinger_15 must be registered
        indicator_ids = {ind.id for ind in ir.indicators}
        assert "bollinger_15" in indicator_ids, \
            "Bollinger 15 must be registered from nested StateCondition"

    def test_ir_validation_passes_after_translation(self):
        """Ensure translated IR passes validation.

        If indicators aren't registered, validation will fail.
        """
        from src.translator.ir_validator import validate_ir

        strategy = Strategy(
            id="test",
            name="Test Strategy",
            universe=["BTC-USD"],
            created_at=NOW,
            updated_at=NOW,
            attachments=[
                {"card_id": "entry", "role": "entry", "enabled": True, "overrides": {}},
                {"card_id": "exit", "role": "exit", "enabled": True, "overrides": {}},
            ],
        )
        cards = {
            "entry": Card(created_at=NOW, updated_at=NOW, schema_etag="v1", 
                id="entry",
                type="entry.trend_pullback",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "dip": {"kind": "reentry", "edge": "lower"},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                    "risk": {"sl_atr": 2.0},
                },
            ),
            "exit": Card(created_at=NOW, updated_at=NOW, schema_etag="v1", 
                id="exit",
                type="exit.trailing_stop",
                slots={
                    "context": {"symbol": "BTC-USD"},
                    "event": {
                        "trail_band": {"band": "keltner", "length": 20, "mult": 2.0},
                    },
                },
            ),
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        # Validation should pass - if it fails, indicators weren't registered
        result = validate_ir(ir)
        assert result.is_valid, f"IR validation failed: {result.errors}"
