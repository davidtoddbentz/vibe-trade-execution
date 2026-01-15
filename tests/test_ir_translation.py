"""Tests for the IR translator.

This module consolidates all IR translation tests including:
- Basic translator functionality (empty, disabled, missing)
- Entry archetypes (rule_trigger, trend_pullback)
- Exit archetypes (rule_trigger, band_exit)
- Band event conditions (touch, cross_in, reentry, distance)
- Sequence conditions (multi-step patterns)
- New metrics (session_phase, volume_spike, price_level, etc.)

NOTE: We test only the IR path which produces structured data.
The legacy LEAN code generator (string-to-code) is deprecated.
"""

import pytest
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment

from src.translator.ir import (
    EMA,
    AllOfCondition,
    BandField,
    BollingerBands,
    CompareCondition,
    CompareOp,
    ExpressionValue,
    IndicatorBandValue,
    IndicatorValue,
    LiquidateAction,
    LiteralValue,
    PriceValue,
    RateOfChange,
    Resolution,
    SetHoldingsAction,
    SetStateFromConditionOp,
    StateValue,
    StrategyIR,
)
from src.translator.ir_translator import IRTranslator, TranslationError

# =============================================================================
# Basic Translator Tests
# =============================================================================


class TestIRTranslatorBasic:
    """Basic tests for IR translator."""

    def test_empty_strategy(self):
        """Test translating a strategy with no attachments."""
        strategy = Strategy(
            id="test-001",
            name="Empty Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[],
        )

        translator = IRTranslator(strategy, {})
        result = translator.translate()

        assert result.strategy_id == "test-001"
        assert result.strategy_name == "Empty Test"
        assert result.symbol == "BTC-USD"
        assert result.entry is None
        assert len(result.exits) == 0

    def test_disabled_attachment_ignored(self):
        """Test that disabled attachments are not translated."""
        strategy = Strategy(
            id="test-002",
            name="Disabled Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="card-001", role="entry", enabled=False),
            ],
        )

        cards = {
            "card-001": Card(
                id="card-001",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0},
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        assert result.entry is None

    def test_missing_card_warning(self):
        """Test that missing cards produce warnings."""
        strategy = Strategy(
            id="test-003",
            name="Missing Card Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="nonexistent", role="entry", enabled=True),
            ],
        )

        translator = IRTranslator(strategy, {})
        with pytest.raises(TranslationError, match="not found"):
            translator.translate()


# =============================================================================
# IR Condition Tests
# =============================================================================


class TestIRConditions:
    """Tests for IR condition generation."""

    def test_compare_condition_structure(self):
        """Test that compare conditions have correct structure."""
        condition = CompareCondition(
            left=IndicatorValue(indicator_id="ema_fast"),
            op=CompareOp.GT,
            right=IndicatorValue(indicator_id="ema_slow"),
        )

        assert condition.type == "compare"
        assert condition.left.indicator_id == "ema_fast"
        assert condition.op == CompareOp.GT
        assert condition.right.indicator_id == "ema_slow"

    def test_literal_value(self):
        """Test literal values in conditions."""
        condition = CompareCondition(
            left=IndicatorValue(indicator_id="roc"),
            op=CompareOp.LT,
            right=LiteralValue(value=-2.0),
        )

        assert condition.right.value == -2.0


# =============================================================================
# Entry Rule Trigger Tests
# =============================================================================


class TestIRTranslatorEntryRuleTrigger:
    """Tests for entry.rule_trigger archetype."""

    def test_ret_pct_entry(self):
        """Test translating ret_pct condition."""
        strategy = Strategy(
            id="test-ret",
            name="Ret Pct Entry",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1d", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "op": "<=",
                                "value": 1.0,
                                "lookback_bars": 1,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have ROC indicator
        assert any(isinstance(ind, RateOfChange) for ind in result.indicators)

        # Should have entry
        assert result.entry is not None
        assert isinstance(result.entry.action, SetHoldingsAction)
        assert result.entry.action.allocation == 0.95

        # Entry condition should be CompareCondition
        assert isinstance(result.entry.condition, CompareCondition)
        assert result.entry.condition.op == CompareOp.LTE

    def test_trend_ma_relation_entry(self):
        """Test translating trend_ma_relation condition."""
        strategy = Strategy(
            id="test-ma",
            name="MA Crossover",
            universe=["ETH-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "ETH-USD"},
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
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have two EMA indicators
        emas = [ind for ind in result.indicators if isinstance(ind, EMA)]
        assert len(emas) == 2

        # Check periods
        periods = {ema.period for ema in emas}
        assert periods == {20, 50}

        # Should have entry with CompareCondition
        assert result.entry is not None
        assert isinstance(result.entry.condition, CompareCondition)
        assert result.entry.condition.op == CompareOp.GT

    def test_short_direction(self):
        """Test short direction produces negative allocation."""
        strategy = Strategy(
            id="test-short",
            name="Short Entry",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "ret_pct", "op": ">=", "value": 5.0},
                        }
                    },
                    "action": {"direction": "short"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        assert result.entry is not None
        assert isinstance(result.entry.action, SetHoldingsAction)
        assert result.entry.action.allocation == -0.95

    def test_composite_allof_condition(self):
        """Test translating composite allOf conditions."""
        strategy = Strategy(
            id="test-composite",
            name="Composite Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "composite",
                            "composite": {
                                "op": "allOf",
                                "conditions": [
                                    {
                                        "type": "regime",
                                        "regime": {"metric": "ret_pct", "op": "<", "value": -2.0},
                                    },
                                    {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "trend_ma_relation",
                                            "ma_fast": 10,
                                            "ma_slow": 30,
                                            "op": ">",
                                            "value": 0,
                                        },
                                    },
                                ],
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        assert result.entry is not None
        # Should have indicators for both conditions
        indicator_ids = [ind.id for ind in result.indicators]
        assert "roc" in indicator_ids  # For ret_pct
        assert "ema_10" in indicator_ids  # For trend_ma_relation (ma_fast=10)
        assert "ema_30" in indicator_ids  # For trend_ma_relation (ma_slow=30)


# =============================================================================
# Trend Pullback Tests
# =============================================================================


class TestIRTranslatorTrendPullback:
    """Tests for entry.trend_pullback archetype."""

    def test_trend_pullback_long(self):
        """Test trend pullback long entry."""
        strategy = Strategy(
            id="test-pullback",
            name="Pullback Entry",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.trend_pullback",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have EMA and BB indicators
        assert any(isinstance(ind, EMA) for ind in result.indicators)
        assert any(isinstance(ind, BollingerBands) for ind in result.indicators)

        # Should have AllOfCondition (trend AND dip)
        assert result.entry is not None
        assert isinstance(result.entry.condition, AllOfCondition)
        assert len(result.entry.condition.conditions) == 2


# =============================================================================
# Exit Archetype Tests
# =============================================================================


class TestIRTranslatorExits:
    """Tests for exit archetypes."""

    def test_exit_rule_trigger(self):
        """Test exit.rule_trigger archetype."""
        strategy = Strategy(
            id="test-exit",
            name="Exit Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
                Attachment(card_id="exit_1", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
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
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
            "exit_1": Card(
                id="exit_1",
                type="exit.rule_trigger",
                slots={
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
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have one exit
        assert len(result.exits) == 1
        assert isinstance(result.exits[0].action, LiquidateAction)
        assert result.exits[0].condition.op == CompareOp.LT

    def test_exit_band(self):
        """Test exit.band_exit archetype."""
        strategy = Strategy(
            id="test-band-exit",
            name="Band Exit Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="exit_1", role="exit", enabled=True),
            ],
        )

        cards = {
            "exit_1": Card(
                id="exit_1",
                type="exit.band_exit",
                slots={
                    "event": {
                        "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "exit_trigger": {"edge": "upper"},
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have BB indicator
        assert any(isinstance(ind, BollingerBands) for ind in result.indicators)

        # Should have exit with upper band comparison
        assert len(result.exits) == 1
        exit_cond = result.exits[0].condition
        assert isinstance(exit_cond, CompareCondition)
        assert exit_cond.op == CompareOp.GTE
        assert isinstance(exit_cond.right, IndicatorBandValue)
        assert exit_cond.right.band == BandField.UPPER


# =============================================================================
# Gate Archetype Tests
# =============================================================================


class TestIRTranslatorGates:
    """Tests for gate archetypes."""

    def test_gate_translation(self):
        """Test translating gate cards."""
        strategy = Strategy(
            id="test-gate",
            name="Gated Strategy",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="gate_1", role="gate", enabled=True),
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "gate_1": Card(
                id="gate_1",
                type="gate.regime",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "metric": "trend_adx",
                            "op": ">",
                            "value": 25,
                            "lookback_bars": 14,
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "ret_pct", "op": "<", "value": -2.0},
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have gates
        assert len(result.gates) >= 1


# =============================================================================
# EMA Crossover Complete Tests
# =============================================================================


class TestIRTranslatorEMACrossover:
    """End-to-end test for EMA Crossover strategy."""

    def test_ema_crossover_long_complete(self):
        """Test complete EMA Crossover Long strategy translation."""
        strategy = Strategy(
            id="strat-ema-crossover",
            name="EMA Crossover Long",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
                Attachment(card_id="exit_1", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
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
                                "ma_fast": 20,
                                "ma_slow": 50,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
            "exit_1": Card(
                id="exit_1",
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
                                "ma_fast": 20,
                                "ma_slow": 50,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Verify metadata
        assert result.strategy_id == "strat-ema-crossover"
        assert result.strategy_name == "EMA Crossover Long"
        assert result.symbol == "BTC-USD"
        assert result.resolution == Resolution.HOUR

        # Verify indicators - should be deduplicated
        emas = [ind for ind in result.indicators if isinstance(ind, EMA)]
        assert len(emas) == 2
        assert {ema.id for ema in emas} == {"ema_20", "ema_50"}
        assert {ema.period for ema in emas} == {20, 50}

        # Verify entry
        assert result.entry is not None
        entry_cond = result.entry.condition
        assert isinstance(entry_cond, CompareCondition)
        assert isinstance(entry_cond.left, IndicatorValue)
        assert entry_cond.left.indicator_id == "ema_20"
        assert entry_cond.op == CompareOp.GT
        assert isinstance(entry_cond.right, IndicatorValue)
        assert entry_cond.right.indicator_id == "ema_50"

        # Verify entry action
        assert isinstance(result.entry.action, SetHoldingsAction)
        assert result.entry.action.allocation == 0.95

        # Verify exit
        assert len(result.exits) == 1
        exit_cond = result.exits[0].condition
        assert isinstance(exit_cond, CompareCondition)
        assert exit_cond.op == CompareOp.LT
        assert isinstance(result.exits[0].action, LiquidateAction)

        # Verify on_fill hooks
        assert len(result.entry.on_fill) == 2  # entry_price and bars_since_entry

        # Verify state vars
        state_ids = {sv.id for sv in result.state}
        assert "entry_price" in state_ids
        assert "bars_since_entry" in state_ids

    def test_ir_serialization_roundtrip(self):
        """Test that IR can be serialized and deserialized correctly."""
        strategy = Strategy(
            id="test-serial",
            name="Serialization Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
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
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Serialize to JSON
        json_str = result.to_json()

        # Deserialize back
        restored = StrategyIR.from_json(json_str)

        # Verify roundtrip
        assert restored.strategy_id == result.strategy_id
        assert restored.strategy_name == result.strategy_name
        assert len(restored.indicators) == len(result.indicators)
        assert restored.entry is not None
        assert restored.entry.condition.op == result.entry.condition.op


# =============================================================================
# Band Event Tests
# =============================================================================


class TestIRTranslatorBandEvent:
    """Tests for band_event condition translation."""

    def test_band_event_touch_lower(self):
        """Test band_event touch lower band condition."""
        strategy = Strategy(
            id="test-band-touch",
            name="Band Touch Entry",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                                "kind": "edge_event",
                                "event": "touch",
                                "edge": "lower",
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have no warnings
        # Translation succeeded without exception

        # Should have BB indicator
        indicator_types = {ind.type for ind in result.indicators}
        assert "BB" in indicator_types

        # Entry condition should be price <= lower_band
        assert result.entry is not None
        cond = result.entry.condition
        assert isinstance(cond, CompareCondition)
        assert cond.op == CompareOp.LTE
        assert isinstance(cond.left, PriceValue)
        assert isinstance(cond.right, IndicatorBandValue)
        assert cond.right.band == BandField.LOWER

    def test_band_event_touch_upper(self):
        """Test band_event touch upper band condition."""
        strategy = Strategy(
            id="test-band-upper",
            name="Band Touch Upper",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "keltner", "length": 20, "mult": 2.0},
                                "kind": "edge_event",
                                "event": "touch",
                                "edge": "upper",
                            },
                        }
                    },
                    "action": {"direction": "short"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Translation succeeded without exception

        # Should have KC indicator
        indicator_types = {ind.type for ind in result.indicators}
        assert "KC" in indicator_types

        # Condition should be price >= upper_band
        cond = result.entry.condition
        assert isinstance(cond, CompareCondition)
        assert cond.op == CompareOp.GTE
        assert cond.right.band == BandField.UPPER

    def test_band_event_distance_zscore(self):
        """Test band_event distance (z-score) condition."""
        strategy = Strategy(
            id="test-band-zscore",
            name="Z-Score Entry",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                                "kind": "distance",
                                "op": "<",
                                "value": -2.0,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Translation succeeded without exception

        # Entry condition should be a comparison
        cond = result.entry.condition
        assert isinstance(cond, CompareCondition)
        assert cond.op == CompareOp.LT

        # Left side should be an expression (z-score calculation)
        assert isinstance(cond.left, ExpressionValue)
        assert cond.left.op == "/"

        # Right side should be literal -2.0
        assert isinstance(cond.right, LiteralValue)
        assert cond.right.value == -2.0

    def test_band_event_cross_in_creates_state(self):
        """Test band_event cross_in creates state tracking."""
        strategy = Strategy(
            id="test-band-cross",
            name="Cross Entry",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                                "kind": "edge_event",
                                "event": "cross_in",
                                "edge": "lower",
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Translation succeeded without exception

        # Should have state variables for tracking
        state_ids = {s.id for s in result.state}
        assert any("was_" in sid for sid in state_ids), f"Expected state var, got: {state_ids}"

        # Should have on_bar hooks for state tracking
        assert len(result.on_bar) > 0, "Expected on_bar hooks for cross detection"

        # on_bar should have SetStateFromConditionOp
        op = result.on_bar[0]
        assert isinstance(op, SetStateFromConditionOp)

        # Entry condition should be AllOf (state AND price condition)
        cond = result.entry.condition
        assert isinstance(cond, AllOfCondition)
        assert len(cond.conditions) == 2

    def test_band_event_reentry_creates_state(self):
        """Test band_event reentry creates state tracking."""
        strategy = Strategy(
            id="test-band-reentry",
            name="Reentry Entry",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                                "kind": "reentry",
                                "edge": "lower",
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Translation succeeded without exception

        # Should have state variables for tracking
        state_ids = {s.id for s in result.state}
        assert any("was_outside" in sid for sid in state_ids)

        # Should have on_bar hooks
        assert len(result.on_bar) > 0

        # Entry condition should be AllOf (was_outside AND crosses_middle)
        cond = result.entry.condition
        assert isinstance(cond, AllOfCondition)
        # One condition checks state, other checks price vs middle band
        has_state_check = any(
            isinstance(c, CompareCondition) and isinstance(c.left, StateValue)
            for c in cond.conditions
        )
        has_price_check = any(
            isinstance(c, CompareCondition)
            and isinstance(c.left, PriceValue)
            and isinstance(c.right, IndicatorBandValue)
            and c.right.band == BandField.MIDDLE
            for c in cond.conditions
        )
        assert has_state_check, "Expected state check in condition"
        assert has_price_check, "Expected price vs middle band check"

    def test_band_event_donchian_channel(self):
        """Test band_event with Donchian channel."""
        strategy = Strategy(
            id="test-donchian",
            name="Donchian Entry",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "donchian", "length": 20},
                                "kind": "edge_event",
                                "event": "touch",
                                "edge": "upper",
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Translation succeeded without exception

        # Should have DC indicator
        indicator_types = {ind.type for ind in result.indicators}
        assert "DC" in indicator_types

    def test_band_event_combined_with_regime(self):
        """Test band_event combined with regime condition via allOf."""
        strategy = Strategy(
            id="test-combined",
            name="Combined Entry",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "allOf",
                            "allOf": [
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "trend_ma_relation",
                                        "op": ">",
                                        "value": 0,
                                        "ma_fast": 20,
                                        "ma_slow": 50,
                                    },
                                },
                                {
                                    "type": "band_event",
                                    "band_event": {
                                        "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                                        "kind": "edge_event",
                                        "event": "touch",
                                        "edge": "lower",
                                    },
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Translation succeeded without exception

        # Should have EMA and BB indicators
        indicator_types = {ind.type for ind in result.indicators}
        assert "EMA" in indicator_types
        assert "BB" in indicator_types

        # Entry should be AllOf with two conditions
        cond = result.entry.condition
        assert isinstance(cond, AllOfCondition)
        assert len(cond.conditions) == 2


# =============================================================================
# Sequence Tests
# =============================================================================


class TestIRTranslatorSequence:
    """Tests for sequence ConditionSpec translation."""

    def test_sequence_two_steps_basic(self):
        """Test basic 2-step sequence without timeout."""
        strategy = Strategy(
            id="test-sequence",
            name="Sequence Entry",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "sequence",
                            "sequence": [
                                # Step 0: Price dips (oversold)
                                {
                                    "cond": {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "ret_pct",
                                            "op": "<",
                                            "value": -3.0,
                                            "lookback_bars": 5,
                                        },
                                    }
                                },
                                # Step 1: Price bounces back
                                {
                                    "cond": {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "ret_pct",
                                            "op": ">",
                                            "value": 1.0,
                                            "lookback_bars": 5,
                                        },
                                    }
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Translation succeeded without exception

        # Should have state variables for sequence tracking
        state_ids = {sv.id for sv in result.state}
        assert "seq_0_step_0_trigger" in state_ids
        assert "seq_0_step_0_done" in state_ids
        assert "seq_0_bars_since_0" in state_ids

        # Should have on_bar hooks for state tracking
        assert len(result.on_bar) >= 3  # trigger, done, bars_since

        # Entry condition should be AllOf
        cond = result.entry.condition
        assert isinstance(cond, AllOfCondition)

    def test_sequence_with_timeout(self):
        """Test sequence with within_bars timeout."""
        strategy = Strategy(
            id="test-sequence-timeout",
            name="Sequence with Timeout",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "sequence",
                            "sequence": [
                                # Step 0: Touch lower band
                                {
                                    "cond": {
                                        "type": "band_event",
                                        "band_event": {
                                            "band": {
                                                "band": "bollinger",
                                                "length": 20,
                                                "mult": 2.0,
                                            },
                                            "kind": "edge_event",
                                            "event": "touch",
                                            "edge": "lower",
                                        },
                                    }
                                },
                                # Step 1: Bounce back within 5 bars
                                {
                                    "cond": {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "ret_pct",
                                            "op": ">",
                                            "value": 2.0,
                                            "lookback_bars": 3,
                                        },
                                    },
                                    "within_bars": 5,
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Translation succeeded without exception

        # Entry condition should include timeout check
        # The condition checks: step_0_done AND bars_since_0 <= 5 AND step_1_condition
        cond = result.entry.condition
        assert isinstance(cond, AllOfCondition)
        # Should have at least 3 conditions: step_0_done, timeout check, step_1 condition
        assert len(cond.conditions) >= 3

    def test_sequence_three_steps(self):
        """Test 3-step sequence."""
        strategy = Strategy(
            id="test-sequence-3",
            name="Three Step Sequence",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "sequence",
                            "sequence": [
                                # Step 0: Uptrend
                                {
                                    "cond": {
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
                                # Step 1: Dip
                                {
                                    "cond": {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "ret_pct",
                                            "op": "<",
                                            "value": -2.0,
                                            "lookback_bars": 3,
                                        },
                                    },
                                    "within_bars": 10,
                                },
                                # Step 2: Recovery
                                {
                                    "cond": {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "ret_pct",
                                            "op": ">",
                                            "value": 1.0,
                                            "lookback_bars": 3,
                                        },
                                    },
                                    "within_bars": 5,
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Translation succeeded without exception

        # Should have state for steps 0 and 1 (step 2 is the final condition)
        state_ids = {sv.id for sv in result.state}
        assert "seq_0_step_0_done" in state_ids
        assert "seq_0_step_1_done" in state_ids

        # Entry condition should check both previous steps done
        cond = result.entry.condition
        assert isinstance(cond, AllOfCondition)

    def test_sequence_single_step_warning(self):
        """Test that single-step sequence produces warning."""
        strategy = Strategy(
            id="test-sequence-invalid",
            name="Invalid Sequence",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "sequence",
                            "sequence": [
                                # Only one step - invalid
                                {
                                    "cond": {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "ret_pct",
                                            "op": "<",
                                            "value": -3.0,
                                            "lookback_bars": 5,
                                        },
                                    }
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        # Single-step sequence should raise an error
        with pytest.raises(TranslationError, match="at least 2"):
            translator.translate()


# =============================================================================
# New Metrics Tests
# =============================================================================


class TestNewMetrics:
    """Tests for newly implemented metrics."""

    def test_session_phase_metric(self):
        """Test session_phase metric translation."""
        strategy = Strategy(
            id="test-session",
            name="Session Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "session_phase",
                                "session": "us",
                                "op": "==",
                                "value": "open",
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should produce an allOf condition with time checks
        assert result.entry is not None
        assert result.entry.condition.type == "allOf"

    def test_volume_spike_metric(self):
        """Test volume_spike metric translation."""
        strategy = Strategy(
            id="test-vol-spike",
            name="Volume Spike Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "volume_spike",
                                "volume_threshold_pctile": 80,
                                "lookback_bars": 20,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have vol_sma indicator
        indicator_types = [ind.type for ind in result.indicators]
        assert "VOL_SMA" in indicator_types

        # Entry condition should be a compare
        assert result.entry is not None
        assert result.entry.condition.type == "compare"

    def test_volume_dip_metric(self):
        """Test volume_dip metric translation."""
        strategy = Strategy(
            id="test-vol-dip",
            name="Volume Dip Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "volume_dip",
                                "volume_threshold_pctile": 20,
                                "lookback_bars": 20,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have vol_sma indicator
        indicator_types = [ind.type for ind in result.indicators]
        assert "VOL_SMA" in indicator_types

        # Entry condition should compare volume < threshold
        assert result.entry is not None
        assert result.entry.condition.type == "compare"
        assert result.entry.condition.op == CompareOp.LT

    def test_price_level_touch_fixed(self):
        """Test price_level_touch metric with fixed price."""
        strategy = Strategy(
            id="test-level-touch",
            name="Price Level Touch Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "price_level_touch",
                                "level_price": 50000,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should produce allOf with high >= level >= low
        assert result.entry is not None
        assert result.entry.condition.type == "allOf"
        assert len(result.entry.condition.conditions) == 2

    def test_price_level_cross_up(self):
        """Test price_level_cross metric with up direction."""
        strategy = Strategy(
            id="test-level-cross",
            name="Price Level Cross Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "price_level_cross",
                                "level_price": 50000,
                                "direction": "up",
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should produce compare condition
        assert result.entry is not None
        assert result.entry.condition.type == "compare"
        assert result.entry.condition.op == CompareOp.GT

    def test_price_level_cross_dynamic(self):
        """Test price_level_cross with dynamic level reference."""
        strategy = Strategy(
            id="test-level-dynamic",
            name="Dynamic Level Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "price_level_cross",
                                "level_reference": "previous_high",
                                "direction": "up",
                                "lookback_bars": 20,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have RollingMinMax indicator
        indicator_types = [ind.type for ind in result.indicators]
        assert "RMM" in indicator_types

        # Should produce compare condition
        assert result.entry is not None
        assert result.entry.condition.type == "compare"

    def test_liquidity_sweep_metric(self):
        """Test liquidity_sweep metric translation."""
        strategy = Strategy(
            id="test-sweep",
            name="Liquidity Sweep Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "liquidity_sweep",
                                "level_reference": "previous_low",
                                "reclaim_within_bars": 3,
                                "lookback_bars": 20,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have RollingMinMax indicator for level tracking
        indicator_types = [ind.type for ind in result.indicators]
        assert "RMM" in indicator_types

        # Should produce regime condition for runtime evaluation
        assert result.entry is not None
        assert result.entry.condition.type == "regime"

    def test_flag_pattern_metric(self):
        """Test flag_pattern metric translation."""
        strategy = Strategy(
            id="test-flag",
            name="Flag Pattern Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "flag_pattern",
                                "flag_momentum_bars": 5,
                                "flag_consolidation_bars": 10,
                                "flag_breakout_direction": "same",
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have pattern indicators (ROC, ATR, MAX, MIN)
        indicator_types = [ind.type for ind in result.indicators]
        assert "ROC" in indicator_types
        assert "ATR" in indicator_types
        assert "MAX" in indicator_types
        assert "MIN" in indicator_types

    def test_vol_regime_quiet(self):
        """Test vol_regime metric with 'quiet' value."""
        strategy = Strategy(
            id="test-vol-regime",
            name="Vol Regime Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "vol_regime",
                                "op": "==",
                                "value": "quiet",
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have BB indicator
        indicator_types = [ind.type for ind in result.indicators]
        assert "BB" in indicator_types

        # Should produce compare condition with LT
        assert result.entry is not None
        assert result.entry.condition.type == "compare"
        assert result.entry.condition.op == CompareOp.LT

    def test_dist_from_vwap_pct(self):
        """Test dist_from_vwap_pct metric."""
        strategy = Strategy(
            id="test-vwap",
            name="VWAP Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "dist_from_vwap_pct",
                                "op": "<",
                                "value": -2.0,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have VWAP indicator
        indicator_types = [ind.type for ind in result.indicators]
        assert "VWAP" in indicator_types

        # Entry condition should be a compare with expression
        assert result.entry is not None
        assert result.entry.condition.type == "compare"

    def test_risk_event_prob_warning(self):
        """Test that risk_event_prob produces appropriate warning."""
        strategy = Strategy(
            id="test-risk",
            name="Risk Event Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry_1", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry_1": Card(
                id="entry_1",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "risk_event_prob",
                                "op": "<",
                                "value": 0.5,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        # risk_event_prob requires external data and should raise an error
        with pytest.raises(TranslationError, match="risk_event_prob|calendar"):
            translator.translate()
