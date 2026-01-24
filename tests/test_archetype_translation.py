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
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
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
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
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
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
                    "risk": {"sl_atr": 2.0},
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        indicator_ids = {ind.id for ind in ir.indicators}
        assert "donchian_20" in indicator_ids, "Donchian 20 should be registered from StateCondition"

    def test_trend_pullback_reentry_declares_state_variable(self):
        """Verify StateCondition declares its required state variable in IR.

        This is a regression test for a critical bug: StateCondition (used for
        reentry patterns) requires a boolean state variable to track whether
        the previous bar was outside/inside a band. Without this declaration,
        the LEAN runtime can't track state and reentry signals fail silently.

        The bug caused 0 trades in production because:
        1. StateCondition generated correct IR structure
        2. But state_var wasn't declared in ir.state
        3. LEAN runtime couldn't track state, so condition never triggered
        """
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
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
                    "risk": {"sl_atr": 2.0},
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        # Verify state variable is declared for StateCondition
        state_var_ids = {s.id for s in ir.state}
        expected_state_var = "outside_bollinger_20_lower"

        assert expected_state_var in state_var_ids, (
            f"StateCondition requires state variable '{expected_state_var}' to be declared in ir.state. "
            f"Found: {state_var_ids}. "
            "Without this, LEAN can't track outside/inside state and reentry signals fail."
        )

        # Verify the state variable is boolean type
        state_var = next(s for s in ir.state if s.id == expected_state_var)
        assert state_var.var_type == "bool", (
            f"State variable '{expected_state_var}' should be bool type, got {state_var.var_type}"
        )


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
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
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
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
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
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
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
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
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
                    "action": {"direction": "short", "position_policy": {"mode": "single"}},
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
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
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


class TestCompareConditionTranslation:
    """Test compare condition translation through archetypes."""

    def test_compare_condition_translates_correctly(self):
        """Verify compare condition translates to IR CompareCondition."""
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
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": ">",
                                "rhs": 100.0,
                            },
                        }
                    },
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        # Entry condition should be CompareCondition
        assert ir.entry is not None
        assert ir.entry.condition.type == "compare"
        assert ir.entry.condition.left.type == "price"
        assert ir.entry.condition.right.value == 100.0

    def test_compare_with_allof_translates_correctly(self):
        """Verify compare inside allOf translates correctly."""
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
                            "type": "allOf",
                            "allOf": [
                                {
                                    "type": "compare",
                                    "compare": {
                                        "lhs": {"type": "price", "field": "close"},
                                        "op": ">",
                                        "rhs": 100.0,
                                    },
                                },
                                {
                                    "type": "compare",
                                    "compare": {
                                        "lhs": {"type": "price", "field": "close"},
                                        "op": "<",
                                        "rhs": 200.0,
                                    },
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        # Entry condition should be AllOfCondition with two CompareConditions
        assert ir.entry is not None
        assert ir.entry.condition.type == "allOf"
        assert len(ir.entry.condition.conditions) == 2
        assert ir.entry.condition.conditions[0].type == "compare"
        assert ir.entry.condition.conditions[1].type == "compare"


class TestSizingSpecTranslation:
    """Test that SizingSpec in EntryActionSpec is honored by translator.

    SizingSpec allows users to specify position sizing:
    - pct_equity: Allocate X% of portfolio equity
    - fixed_usd: Allocate fixed USD amount
    - fixed_units: Allocate fixed number of units
    - pct_position: Size relative to current position

    The translator was hardcoding 0.95 allocation regardless of sizing spec.
    """

    def test_pct_equity_sizing_translates_to_correct_allocation(self):
        """Verify pct_equity sizing produces correct allocation.

        sizing: {type: "pct_equity", pct: 5} should produce allocation=0.05
        NOT the hardcoded 0.95.
        """
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
            "entry": Card(
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": ">",
                                "rhs": 100.0,
                            },
                        }
                    },
                    "action": {
                        "direction": "long",
                        "sizing": {"type": "pct_equity", "pct": 5},
                        "position_policy": {"mode": "single"},
                    },
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        # Entry action should have allocation=0.05 (5% of equity)
        assert ir.entry is not None
        assert ir.entry.action.allocation == 0.05, (
            f"Expected allocation=0.05 for 5% pct_equity, got {ir.entry.action.allocation}"
        )

    def test_pct_equity_sizing_short_direction(self):
        """Verify pct_equity sizing with short direction produces negative allocation.

        sizing: {type: "pct_equity", pct: 10} with direction="short"
        should produce allocation=-0.10
        """
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
            "entry": Card(
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": "<",
                                "rhs": 100.0,
                            },
                        }
                    },
                    "action": {
                        "direction": "short",
                        "sizing": {"type": "pct_equity", "pct": 10},
                        "position_policy": {"mode": "single"},
                    },
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        # Short with 10% sizing should produce -0.10
        assert ir.entry is not None
        assert ir.entry.action.allocation == -0.10, (
            f"Expected allocation=-0.10 for short 10% pct_equity, got {ir.entry.action.allocation}"
        )

    def test_default_allocation_without_sizing(self):
        """Verify default allocation when no sizing spec provided.

        When sizing is not specified, should use 95% allocation (current default).
        """
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
            "entry": Card(
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": ">",
                                "rhs": 100.0,
                            },
                        }
                    },
                    "action": {"direction": "long", "position_policy": {"mode": "single"}},  # No sizing
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        # Should maintain backwards compatibility with 95% default
        assert ir.entry is not None
        assert ir.entry.action.allocation == 0.95, (
            f"Expected default allocation=0.95, got {ir.entry.action.allocation}"
        )


class TestFixedSizingModes:
    """Test fixed_usd and fixed_units sizing modes.

    These modes pass through to the SetHoldingsAction with sizing_mode set
    and the appropriate value field populated. The runtime converts these
    to actual allocation at execution time using current prices.
    """

    def test_fixed_usd_sizing_produces_correct_action(self):
        """Verify fixed_usd sizing produces SetHoldingsAction with fixed_usd."""
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
            "entry": Card(
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": ">",
                                "rhs": 100.0,
                            },
                        }
                    },
                    "action": {
                        "direction": "long",
                        "sizing": {"type": "fixed_usd", "usd": 5000},
                        "position_policy": {"mode": "single"},
                    },
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        assert ir.entry is not None
        action = ir.entry.action
        assert action.sizing_mode == "fixed_usd", (
            f"Expected sizing_mode='fixed_usd', got '{action.sizing_mode}'"
        )
        assert action.fixed_usd == 5000.0, (
            f"Expected fixed_usd=5000.0, got {action.fixed_usd}"
        )

    def test_fixed_usd_short_produces_negative_value(self):
        """Verify fixed_usd with short direction produces negative fixed_usd."""
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
            "entry": Card(
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": "<",
                                "rhs": 100.0,
                            },
                        }
                    },
                    "action": {
                        "direction": "short",
                        "sizing": {"type": "fixed_usd", "usd": 2500},
                        "position_policy": {"mode": "single"},
                    },
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        assert ir.entry is not None
        action = ir.entry.action
        assert action.sizing_mode == "fixed_usd"
        assert action.fixed_usd == -2500.0, (
            f"Expected fixed_usd=-2500.0 for short, got {action.fixed_usd}"
        )

    def test_fixed_units_sizing_produces_correct_action(self):
        """Verify fixed_units sizing produces SetHoldingsAction with fixed_units."""
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
            "entry": Card(
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": ">",
                                "rhs": 100.0,
                            },
                        }
                    },
                    "action": {
                        "direction": "long",
                        "sizing": {"type": "fixed_units", "units": 0.5},
                        "position_policy": {"mode": "single"},
                    },
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        assert ir.entry is not None
        action = ir.entry.action
        assert action.sizing_mode == "fixed_units", (
            f"Expected sizing_mode='fixed_units', got '{action.sizing_mode}'"
        )
        assert action.fixed_units == 0.5, (
            f"Expected fixed_units=0.5, got {action.fixed_units}"
        )

    def test_fixed_units_short_produces_negative_value(self):
        """Verify fixed_units with short direction produces negative fixed_units."""
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
            "entry": Card(
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": "<",
                                "rhs": 100.0,
                            },
                        }
                    },
                    "action": {
                        "direction": "short",
                        "sizing": {"type": "fixed_units", "units": 1.0},
                        "position_policy": {"mode": "single"},
                    },
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        assert ir.entry is not None
        action = ir.entry.action
        assert action.sizing_mode == "fixed_units"
        assert action.fixed_units == -1.0, (
            f"Expected fixed_units=-1.0 for short, got {action.fixed_units}"
        )

    def test_pct_equity_explicitly_sets_sizing_mode(self):
        """Verify pct_equity sizing explicitly sets sizing_mode field."""
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
            "entry": Card(
                created_at=NOW,
                updated_at=NOW,
                schema_etag="v1",
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "compare",
                            "compare": {
                                "lhs": {"type": "price", "field": "close"},
                                "op": ">",
                                "rhs": 100.0,
                            },
                        }
                    },
                    "action": {
                        "direction": "long",
                        "sizing": {"type": "pct_equity", "pct": 25},
                        "position_policy": {"mode": "single"},
                    },
                },
            )
        }

        translator = IRTranslator(strategy, cards)
        ir = translator.translate()

        assert ir.entry is not None
        action = ir.entry.action
        assert action.sizing_mode == "pct_equity", (
            f"Expected sizing_mode='pct_equity', got '{action.sizing_mode}'"
        )
        assert action.allocation == 0.25
