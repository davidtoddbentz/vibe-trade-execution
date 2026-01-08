"""End-to-end tests for the IR-based translation pipeline.

These tests verify the complete flow from Strategy/Cards to IR JSON
that can be consumed by the LEAN StrategyRuntime algorithm.
"""

import json
import pytest

from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment

from src.translator import IRTranslator, StrategyIR, ConditionEvaluator, ValueResolver, EvalContext


class TestEndToEndEmaCrossover:
    """End-to-end tests for EMA Crossover strategy."""

    @pytest.fixture
    def ema_crossover_strategy(self) -> tuple[Strategy, dict[str, Card]]:
        """Create the EMA Crossover Long strategy."""
        strategy = Strategy(
            id="strat-ema-crossover",
            name="EMA Crossover Long",
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
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
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
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }
        return strategy, cards

    def test_full_pipeline_produces_valid_ir(self, ema_crossover_strategy):
        """Test that the full pipeline produces valid IR."""
        strategy, cards = ema_crossover_strategy

        # Step 1: Translate to IR
        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Verify no warnings
        assert len(result.warnings) == 0, f"Unexpected warnings: {result.warnings}"

        # Step 2: Serialize to JSON
        ir_json = result.ir.to_json()
        assert isinstance(ir_json, str)
        assert len(ir_json) > 100  # Should have meaningful content

        # Step 3: Parse JSON to verify structure
        ir_dict = json.loads(ir_json)

        assert ir_dict["strategy_id"] == "strat-ema-crossover"
        assert ir_dict["strategy_name"] == "EMA Crossover Long"
        assert ir_dict["symbol"] == "BTC-USD"
        assert ir_dict["resolution"] == "Hour"

        # Verify indicators
        assert len(ir_dict["indicators"]) == 2
        indicator_ids = {ind["id"] for ind in ir_dict["indicators"]}
        assert indicator_ids == {"ema_fast", "ema_slow"}

        # Verify entry
        assert ir_dict["entry"] is not None
        assert ir_dict["entry"]["action"]["type"] == "set_holdings"
        assert ir_dict["entry"]["action"]["allocation"] == 0.95

        # Verify exit
        assert len(ir_dict["exits"]) == 1
        assert ir_dict["exits"][0]["action"]["type"] == "liquidate"

    def test_ir_can_be_deserialized(self, ema_crossover_strategy):
        """Test that serialized IR can be deserialized back."""
        strategy, cards = ema_crossover_strategy

        # Translate and serialize
        translator = IRTranslator(strategy, cards)
        result = translator.translate()
        ir_json = result.ir.to_json()

        # Deserialize
        restored_ir = StrategyIR.from_json(ir_json)

        # Verify roundtrip
        assert restored_ir.strategy_id == result.ir.strategy_id
        assert restored_ir.strategy_name == result.ir.strategy_name
        assert len(restored_ir.indicators) == len(result.ir.indicators)
        assert restored_ir.entry.action.allocation == result.ir.entry.action.allocation

    def test_evaluator_works_with_translated_conditions(self, ema_crossover_strategy):
        """Test that translated conditions can be evaluated."""
        strategy, cards = ema_crossover_strategy

        # Translate
        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Create mock context
        class MockIndicatorCurrent:
            def __init__(self, value):
                self.Value = value

        class MockIndicator:
            def __init__(self, value):
                self.Current = MockIndicatorCurrent(value)
                self.IsReady = True

        class MockPriceBar:
            def __init__(self):
                self.Open = 100.0
                self.High = 105.0
                self.Low = 98.0
                self.Close = 103.0

        # Scenario 1: EMA fast > EMA slow (should trigger entry)
        ctx_bullish = EvalContext(
            indicators={
                "ema_fast": MockIndicator(105.0),
                "ema_slow": MockIndicator(100.0),
            },
            state={"entry_price": None, "bars_since_entry": 0},
            price_bar=MockPriceBar(),
        )

        evaluator = ConditionEvaluator()
        entry_result = evaluator.evaluate(result.ir.entry.condition, ctx_bullish)
        assert entry_result is True, "Entry should trigger when EMA fast > EMA slow"

        # Scenario 2: EMA fast < EMA slow (should trigger exit)
        ctx_bearish = EvalContext(
            indicators={
                "ema_fast": MockIndicator(95.0),
                "ema_slow": MockIndicator(100.0),
            },
            state={"entry_price": 100.0, "bars_since_entry": 5},
            price_bar=MockPriceBar(),
        )

        exit_result = evaluator.evaluate(result.ir.exits[0].condition, ctx_bearish)
        assert exit_result is True, "Exit should trigger when EMA fast < EMA slow"

    def test_ir_json_compatible_with_lean_runtime(self, ema_crossover_strategy):
        """Test that IR JSON has correct structure for LEAN runtime."""
        strategy, cards = ema_crossover_strategy

        translator = IRTranslator(strategy, cards)
        result = translator.translate()
        ir_dict = json.loads(result.ir.to_json())

        # Verify all required fields for LEAN runtime
        required_fields = [
            "strategy_id",
            "strategy_name",
            "symbol",
            "resolution",
            "indicators",
            "state",
            "gates",
            "entry",
            "exits",
        ]
        for field in required_fields:
            assert field in ir_dict, f"Missing required field: {field}"

        # Verify indicator structure
        for ind in ir_dict["indicators"]:
            assert "type" in ind
            assert "id" in ind
            assert "period" in ind

        # Verify condition structure
        entry_cond = ir_dict["entry"]["condition"]
        assert "type" in entry_cond
        assert entry_cond["type"] == "compare"
        assert "left" in entry_cond
        assert "op" in entry_cond
        assert "right" in entry_cond

        # Verify value references
        assert entry_cond["left"]["type"] == "indicator"
        assert entry_cond["left"]["indicator_id"] == "ema_fast"
        assert entry_cond["right"]["type"] == "indicator"
        assert entry_cond["right"]["indicator_id"] == "ema_slow"


class TestEndToEndComplexStrategy:
    """End-to-end tests for more complex strategies."""

    def test_trend_pullback_with_bollinger(self):
        """Test trend pullback strategy with Bollinger bands."""
        strategy = Strategy(
            id="strat-pullback",
            name="Trend Pullback",
            universe=["ETH-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.trend_pullback",
                slots={
                    "context": {"tf": "1h", "symbol": "ETH-USD"},
                    "event": {
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
            "exit": Card(
                id="exit",
                type="exit.band_exit",
                slots={
                    "event": {
                        "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "exit_trigger": {"edge": "upper"},
                    },
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have EMA and BB indicators
        indicator_types = {ind.type for ind in result.ir.indicators}
        assert "EMA" in indicator_types
        assert "BB" in indicator_types

        # Entry should be AllOf (trend AND dip)
        assert result.ir.entry.condition.type == "allOf"
        assert len(result.ir.entry.condition.conditions) == 2

        # Serialize and verify
        ir_json = result.ir.to_json()
        ir_dict = json.loads(ir_json)

        assert ir_dict["entry"]["condition"]["type"] == "allOf"

    def test_breakout_with_trailing_stop(self):
        """Test breakout entry with trailing stop exit."""
        strategy = Strategy(
            id="strat-breakout",
            name="Breakout Trend",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.breakout_trendfollow",
                slots={
                    "event": {"breakout": {"lookback_bars": 50}},
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
            "exit": Card(
                id="exit",
                type="exit.trailing_stop",
                slots={
                    "event": {
                        "trail_band": {"band": "keltner", "length": 20, "mult": 2.0},
                    },
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have MAX, MIN, and ATR indicators
        indicator_types = {ind.type for ind in result.ir.indicators}
        assert "MAX" in indicator_types
        assert "MIN" in indicator_types
        assert "ATR" in indicator_types

        # Entry should compare price to highest
        assert result.ir.entry.condition.type == "compare"

        # Verify JSON
        ir_json = result.ir.to_json()
        assert "highest" in ir_json
        assert "atr" in ir_json


class TestIRJsonFormat:
    """Tests for IR JSON format details."""

    def test_json_uses_snake_case(self):
        """Verify JSON uses snake_case field names."""
        strategy = Strategy(
            id="test",
            name="Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "trend_ma_relation", "op": ">", "value": 0, "ma_fast": 20, "ma_slow": 50},
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
        ir_json = result.ir.to_json()

        # All these should be snake_case in JSON
        assert "strategy_id" in ir_json
        assert "strategy_name" in ir_json
        assert "indicator_id" in ir_json

        # These should NOT be camelCase
        assert "strategyId" not in ir_json
        assert "indicatorId" not in ir_json

    def test_json_compact_mode(self):
        """Test compact JSON (no indentation)."""
        strategy = Strategy(
            id="test",
            name="Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[],
        )

        translator = IRTranslator(strategy, {})
        result = translator.translate()

        # Default is indented
        ir_json_pretty = result.ir.to_json(indent=2)
        ir_json_compact = result.ir.to_json(indent=None)

        # Compact should be shorter (no whitespace)
        assert len(ir_json_compact) < len(ir_json_pretty)
        assert "\n" in ir_json_pretty
        # Compact JSON may still have newlines depending on Pydantic version
