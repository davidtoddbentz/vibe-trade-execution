"""Tests for event-based archetypes.

These tests verify:
1. entry.event_followthrough translates to IR correctly
2. gate.event_risk_window translates to IR correctly
3. Event conditions evaluate correctly with mock calendar data
"""

import pytest

from tests.conftest import MockPriceBar, make_strategy
from src.translator.evaluator import (
    ConditionEvaluator,
    EvalContext,
    EventCalendarEntry,
)
from src.translator.ir import IREventWindowCondition
from src.translator.ir_translator import IRTranslator
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment


# =============================================================================
# IREventWindowCondition Tests
# =============================================================================


class TestIREventWindowCondition:
    """Tests for IREventWindowCondition evaluation."""

    @pytest.fixture
    def evaluator(self):
        return ConditionEvaluator()

    @pytest.fixture
    def mock_calendar(self):
        """Event calendar with an earnings event at bar 10."""
        return [
            EventCalendarEntry(event_type="earnings", bar_index=10, symbol="AAPL"),
            EventCalendarEntry(event_type="fomc", bar_index=20, symbol=None),
        ]

    def test_within_post_window_after_event(self, evaluator, mock_calendar):
        """Condition triggers when within post-event window."""
        condition = IREventWindowCondition(
            event_types=["earnings"],
            pre_window_bars=0,
            post_window_bars=8,
            mode="within",
        )

        # Bar 12 is 2 bars after event at bar 10 (within 8-bar window)
        ctx = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            current_bar_index=12,
            event_calendar=mock_calendar,
        )

        result = evaluator.evaluate(condition, ctx)
        assert result is True

    def test_outside_post_window_after_event(self, evaluator, mock_calendar):
        """Condition doesn't trigger when outside post-event window."""
        condition = IREventWindowCondition(
            event_types=["earnings"],
            pre_window_bars=0,
            post_window_bars=8,
            mode="within",
        )

        # Bar 25 is 15 bars after event at bar 10 (outside 8-bar window)
        ctx = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            current_bar_index=25,
            event_calendar=mock_calendar,
        )

        result = evaluator.evaluate(condition, ctx)
        assert result is False

    def test_within_pre_window_before_event(self, evaluator, mock_calendar):
        """Condition triggers when within pre-event window."""
        condition = IREventWindowCondition(
            event_types=["earnings"],
            pre_window_bars=4,
            post_window_bars=0,
            mode="within",
        )

        # Bar 8 is 2 bars before event at bar 10 (within 4-bar pre-window)
        ctx = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            current_bar_index=8,
            event_calendar=mock_calendar,
        )

        result = evaluator.evaluate(condition, ctx)
        assert result is True

    def test_outside_mode_returns_inverse(self, evaluator, mock_calendar):
        """Mode='outside' returns True when NOT in window."""
        condition = IREventWindowCondition(
            event_types=["earnings"],
            pre_window_bars=4,
            post_window_bars=8,
            mode="outside",
        )

        # Bar 25 is far from event - should be True for "outside" mode
        ctx = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            current_bar_index=25,
            event_calendar=mock_calendar,
        )

        result = evaluator.evaluate(condition, ctx)
        assert result is True

    def test_no_calendar_returns_false(self, evaluator):
        """Returns False when no event calendar is provided."""
        condition = IREventWindowCondition(
            event_types=["earnings"],
            pre_window_bars=0,
            post_window_bars=8,
            mode="within",
        )

        ctx = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            current_bar_index=12,
            event_calendar=None,  # No calendar
        )

        result = evaluator.evaluate(condition, ctx)
        assert result is False

    def test_event_type_filter(self, evaluator, mock_calendar):
        """Only matches specified event types."""
        condition = IREventWindowCondition(
            event_types=["dividend"],  # Not in calendar
            pre_window_bars=0,
            post_window_bars=100,
            mode="within",
        )

        ctx = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            current_bar_index=10,  # Right at earnings event
            event_calendar=mock_calendar,
        )

        result = evaluator.evaluate(condition, ctx)
        assert result is False  # No dividend events


# =============================================================================
# Entry Event Followthrough Tests
# =============================================================================


class TestEntryEventFollowthrough:
    """Tests for entry.event_followthrough archetype."""

    def test_translates_to_event_window_condition(self):
        """event_followthrough produces IREventWindowCondition."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.event_followthrough",
                "slots": {
                    "context": {"symbol": "AAPL"},
                    "event": {
                        "event": {
                            "event_kind": "earnings",
                            "entry_window_bars": 24,
                        }
                    },
                    "action": {"direction": "long"},
                },
            }
        })

        ir = IRTranslator(strategy, cards).translate()

        assert ir.entry is not None
        # The condition should be an event window condition
        assert ir.entry.condition.type == "event_window"
        assert ir.entry.condition.event_types == ["earnings"]
        assert ir.entry.condition.post_window_bars == 24
        assert ir.entry.condition.pre_window_bars == 0
        assert ir.entry.condition.mode == "within"

    def test_evaluates_with_mock_calendar(self):
        """Full pipeline: translate and evaluate with mock calendar."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.event_followthrough",
                "slots": {
                    "context": {"symbol": "AAPL"},
                    "event": {
                        "event": {
                            "event_kind": "earnings",
                            "entry_window_bars": 8,
                        }
                    },
                    "action": {"direction": "long"},
                },
            }
        })

        ir = IRTranslator(strategy, cards).translate()
        evaluator = ConditionEvaluator()

        # Event at bar 10, we're at bar 15 (within 8-bar post window)
        ctx_in_window = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            current_bar_index=15,
            event_calendar=[
                EventCalendarEntry(event_type="earnings", bar_index=10),
            ],
        )
        assert evaluator.evaluate(ir.entry.condition, ctx_in_window) is True

        # Event at bar 10, we're at bar 25 (outside 8-bar post window)
        ctx_outside_window = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            current_bar_index=25,
            event_calendar=[
                EventCalendarEntry(event_type="earnings", bar_index=10),
            ],
        )
        assert evaluator.evaluate(ir.entry.condition, ctx_outside_window) is False


# =============================================================================
# Gate Event Risk Window Tests
# =============================================================================


class TestGateEventRiskWindow:
    """Tests for gate.event_risk_window archetype."""

    def test_block_mode_translates_to_outside(self):
        """Block mode produces event window with mode='outside'."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "ret_pct", "op": ">", "value": 0, "lookback_bars": 1},
                        }
                    },
                    "action": {"direction": "long"},
                },
            },
            "gate1": {
                "type": "gate.event_risk_window",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "catalyst": {
                            "event_kind": "macro",
                            "entry_window_bars": 24,
                        },
                        "pre_event_bars": 12,
                        "post_event_bars": 24,
                    },
                    "action": {
                        "mode": "block",
                        "target_roles": ["entry"],
                    },
                },
            },
        })

        ir = IRTranslator(strategy, cards).translate()

        assert len(ir.gates) == 1
        gate_condition = ir.gates[0].condition
        assert gate_condition.type == "event_window"
        assert gate_condition.event_types == ["macro"]
        assert gate_condition.pre_window_bars == 12
        assert gate_condition.post_window_bars == 24
        # Block mode = allow when OUTSIDE window
        assert gate_condition.mode == "outside"

    def test_allow_mode_translates_to_within(self):
        """Allow mode produces event window with mode='within'."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "ret_pct", "op": ">", "value": 0, "lookback_bars": 1},
                        }
                    },
                    "action": {"direction": "long"},
                },
            },
            "gate1": {
                "type": "gate.event_risk_window",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "catalyst": {
                            "event_kind": "earnings",
                            "entry_window_bars": 8,
                        },
                        "pre_event_bars": 4,
                        "post_event_bars": 8,
                    },
                    "action": {
                        "mode": "allow",
                        "target_roles": ["entry"],
                    },
                },
            },
        })

        ir = IRTranslator(strategy, cards).translate()

        gate_condition = ir.gates[0].condition
        # Allow mode = allow when INSIDE window
        assert gate_condition.mode == "within"

    def test_gate_blocks_entry_around_event(self):
        """Gate correctly blocks entry around events."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "ret_pct", "op": ">", "value": 0, "lookback_bars": 1},
                        }
                    },
                    "action": {"direction": "long"},
                },
            },
            "gate1": {
                "type": "gate.event_risk_window",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "catalyst": {
                            "event_kind": "macro",
                            "entry_window_bars": 8,
                        },
                        "pre_event_bars": 4,
                        "post_event_bars": 8,
                    },
                    "action": {
                        "mode": "block",
                        "target_roles": ["entry"],
                    },
                },
            },
        })

        ir = IRTranslator(strategy, cards).translate()
        evaluator = ConditionEvaluator()

        # Macro event (e.g., FOMC) at bar 20
        calendar = [EventCalendarEntry(event_type="macro", bar_index=20)]

        # Bar 10: far from event, gate should ALLOW (True for outside mode)
        ctx_far = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            current_bar_index=10,
            event_calendar=calendar,
        )
        assert evaluator.evaluate(ir.gates[0].condition, ctx_far) is True

        # Bar 18: 2 bars before event, within pre-window, gate should BLOCK (False)
        ctx_near = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            current_bar_index=18,
            event_calendar=calendar,
        )
        assert evaluator.evaluate(ir.gates[0].condition, ctx_near) is False

        # Bar 25: 5 bars after event, within post-window, gate should BLOCK (False)
        ctx_after = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            current_bar_index=25,
            event_calendar=calendar,
        )
        assert evaluator.evaluate(ir.gates[0].condition, ctx_after) is False
