"""Tests for the IR translator.

NOTE: We test only the IR path which produces structured data.
The legacy LEAN code generator (string-to-code) is deprecated.
"""

import json

import pytest

from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment

from src.translator.ir_translator import IRTranslator
from src.translator.ir import (
    CompareCondition,
    CompareOp,
    IndicatorValue,
    LiteralValue,
)


class TestIRTranslator:
    """Tests for the IR translator."""

    def test_translate_simple_entry(self):
        """Test translating a simple entry card to IR."""
        strategy = Strategy(
            id="strat-001",
            name="Sunday Dip Buyer",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(
                    card_id="card-001",
                    role="entry",
                    enabled=True,
                )
            ],
        )

        cards = {
            "card-001": Card(
                id="card-001",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1d", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "tf": "1d",
                                "op": "<=",
                                "value": -1.0,
                                "lookback_bars": 1,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="W/\"v1.entry.rule_trigger\"",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        assert result.ir.strategy_id == "strat-001"
        assert result.ir.entry is not None
        assert len(result.ir.indicators) > 0  # Should have ROC indicator

    def test_translate_ema_crossover(self):
        """Test translating EMA crossover strategy to IR."""
        strategy = Strategy(
            id="strat-ema",
            name="EMA Crossover",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry-card", role="entry", enabled=True),
                Attachment(card_id="exit-card", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry-card": Card(
                id="entry-card",
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
                created_at="",
                updated_at="",
            ),
            "exit-card": Card(
                id="exit-card",
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
                created_at="",
                updated_at="",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # No warnings for supported archetypes
        assert len(result.warnings) == 0, f"Unexpected warnings: {result.warnings}"

        # Check entry
        assert result.ir.entry is not None
        assert result.ir.entry.condition is not None

        # Check exit
        assert len(result.ir.exits) == 1

        # Check indicators - should have ema_fast and ema_slow
        indicator_ids = [ind.id for ind in result.ir.indicators]
        assert "ema_fast" in indicator_ids
        assert "ema_slow" in indicator_ids

        # Verify EMA periods
        ema_fast = next(i for i in result.ir.indicators if i.id == "ema_fast")
        ema_slow = next(i for i in result.ir.indicators if i.id == "ema_slow")
        assert ema_fast.period == 20
        assert ema_slow.period == 50

    def test_ir_serialization_roundtrip(self):
        """Test that IR can be serialized to JSON and back."""
        strategy = Strategy(
            id="test-roundtrip",
            name="Roundtrip Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "op": "<",
                                "value": -2.0,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Serialize to JSON
        ir_json = result.ir.to_json()
        assert isinstance(ir_json, str)
        assert len(ir_json) > 0

        # Parse JSON back
        parsed = json.loads(ir_json)
        assert parsed["strategy_id"] == "test-roundtrip"
        assert parsed["symbol"] == "BTC-USD"
        assert "entry" in parsed
        assert "indicators" in parsed

    def test_disabled_attachment_ignored(self):
        """Test that disabled attachments are not translated."""
        strategy = Strategy(
            id="test-disabled",
            name="Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=False),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<", "value": -1}}},
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Entry should be None because attachment is disabled
        assert result.ir.entry is None

    def test_missing_card_warning(self):
        """Test that missing cards produce warnings."""
        strategy = Strategy(
            id="test-missing",
            name="Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="nonexistent", role="entry", enabled=True),
            ],
        )

        translator = IRTranslator(strategy, {})
        result = translator.translate()

        assert any("not found" in w.lower() for w in result.warnings)

    def test_composite_allof_condition(self):
        """Test translating composite allOf conditions."""
        strategy = Strategy(
            id="test-composite",
            name="Composite Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                                        "regime": {"metric": "ret_pct", "op": "<", "value": -2.0}
                                    },
                                    {
                                        "type": "regime",
                                        "regime": {"metric": "trend_ma_relation", "ma_fast": 10, "ma_slow": 30, "op": ">", "value": 0}
                                    }
                                ]
                            }
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        assert result.ir.entry is not None
        # Should have indicators for both conditions
        indicator_ids = [ind.id for ind in result.ir.indicators]
        assert "roc" in indicator_ids  # For ret_pct
        assert "ema_fast" in indicator_ids  # For trend_ma_relation
        assert "ema_slow" in indicator_ids

    def test_gate_translation(self):
        """Test translating gate cards."""
        strategy = Strategy(
            id="test-gate",
            name="Gated Strategy",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="gate", role="gate", enabled=True),
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "gate": Card(
                id="gate",
                type="gate.regime",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "trend_adx", "op": ">", "value": 25}
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"symbol": "BTC-USD", "tf": "1h"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "ret_pct", "op": "<", "value": -2.0}
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have gates
        assert len(result.ir.gates) >= 1 or any("gate" in w.lower() for w in result.warnings)


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


class TestNewMetrics:
    """Tests for newly implemented metrics."""

    def test_session_phase_metric(self):
        """Test session_phase metric translation."""
        strategy = Strategy(
            id="test-session",
            name="Session Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should produce an allOf condition with time checks
        assert result.ir.entry is not None
        assert result.ir.entry.condition.type == "allOf"
        # No errors should occur
        assert len([w for w in result.warnings if "error" in w.lower()]) == 0

    def test_volume_spike_metric(self):
        """Test volume_spike metric translation."""
        strategy = Strategy(
            id="test-vol-spike",
            name="Volume Spike Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have vol_sma indicator
        indicator_types = [ind.type for ind in result.ir.indicators]
        assert "VOL_SMA" in indicator_types

        # Entry condition should be a compare
        assert result.ir.entry is not None
        assert result.ir.entry.condition.type == "compare"

    def test_volume_dip_metric(self):
        """Test volume_dip metric translation."""
        strategy = Strategy(
            id="test-vol-dip",
            name="Volume Dip Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have vol_sma indicator
        indicator_types = [ind.type for ind in result.ir.indicators]
        assert "VOL_SMA" in indicator_types

        # Entry condition should compare volume < threshold
        assert result.ir.entry is not None
        assert result.ir.entry.condition.type == "compare"
        assert result.ir.entry.condition.op == CompareOp.LT

    def test_price_level_touch_fixed(self):
        """Test price_level_touch metric with fixed price."""
        strategy = Strategy(
            id="test-level-touch",
            name="Price Level Touch Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should produce allOf with high >= level >= low
        assert result.ir.entry is not None
        assert result.ir.entry.condition.type == "allOf"
        assert len(result.ir.entry.condition.conditions) == 2

    def test_price_level_cross_up(self):
        """Test price_level_cross metric with up direction."""
        strategy = Strategy(
            id="test-level-cross",
            name="Price Level Cross Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should produce compare condition
        assert result.ir.entry is not None
        assert result.ir.entry.condition.type == "compare"
        assert result.ir.entry.condition.op == CompareOp.GT

    def test_price_level_cross_dynamic(self):
        """Test price_level_cross with dynamic level reference."""
        strategy = Strategy(
            id="test-level-dynamic",
            name="Dynamic Level Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have RollingMinMax indicator
        indicator_types = [ind.type for ind in result.ir.indicators]
        assert "RMM" in indicator_types

        # Should produce compare condition
        assert result.ir.entry is not None
        assert result.ir.entry.condition.type == "compare"

    def test_liquidity_sweep_metric(self):
        """Test liquidity_sweep metric translation."""
        strategy = Strategy(
            id="test-sweep",
            name="Liquidity Sweep Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have RollingMinMax indicator for level tracking
        indicator_types = [ind.type for ind in result.ir.indicators]
        assert "RMM" in indicator_types

        # Should produce regime condition for runtime evaluation
        assert result.ir.entry is not None
        assert result.ir.entry.condition.type == "regime"

    def test_flag_pattern_metric(self):
        """Test flag_pattern metric translation."""
        strategy = Strategy(
            id="test-flag",
            name="Flag Pattern Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have pattern indicators (ROC, ATR, MAX, MIN)
        indicator_types = [ind.type for ind in result.ir.indicators]
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
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have BB indicator
        indicator_types = [ind.type for ind in result.ir.indicators]
        assert "BB" in indicator_types

        # Should produce compare condition with LT
        assert result.ir.entry is not None
        assert result.ir.entry.condition.type == "compare"
        assert result.ir.entry.condition.op == CompareOp.LT

    def test_dist_from_vwap_pct(self):
        """Test dist_from_vwap_pct metric."""
        strategy = Strategy(
            id="test-vwap",
            name="VWAP Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have VWAP indicator
        indicator_types = [ind.type for ind in result.ir.indicators]
        assert "VWAP" in indicator_types

        # Entry condition should be a compare with expression
        assert result.ir.entry is not None
        assert result.ir.entry.condition.type == "compare"

    def test_risk_event_prob_warning(self):
        """Test that risk_event_prob produces appropriate warning."""
        strategy = Strategy(
            id="test-risk",
            name="Risk Event Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="",
            updated_at="",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
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
                created_at="",
                updated_at="",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should produce warning about external data
        assert any("calendar" in w.lower() or "risk_event_prob" in w.lower() for w in result.warnings)
