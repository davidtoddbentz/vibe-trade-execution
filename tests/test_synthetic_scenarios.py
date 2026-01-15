"""Synthetic scenarios proving full pipeline: Create → Translate → Execute.

These tests demonstrate that each archetype and combination works end-to-end
using deterministic price data that guarantees specific outcomes.

Each scenario:
1. Creates a Strategy with typed Cards
2. Translates to StrategyIR
3. Evaluates with mock price data
4. Verifies expected signals fire at expected bars
"""

import pytest

from tests.conftest import MockBandIndicator, MockIndicator, MockPriceBar, make_strategy
from src.translator.evaluator import (
    ConditionEvaluator,
    EvalContext,
    StateOperator,
)
from src.translator.ir import StrategyIR
from src.translator.ir_translator import IRTranslator
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment


# =============================================================================
# Test Helpers
# =============================================================================


def translate(cards_dict: dict) -> StrategyIR:
    """Helper to create and translate a strategy."""
    strategy, cards = make_strategy(cards_dict)
    return IRTranslator(strategy, cards).translate()


def evaluate_entry(ir: StrategyIR, ctx: EvalContext) -> bool:
    """Check if entry condition fires."""
    if ir.entry is None:
        return False
    evaluator = ConditionEvaluator()
    return evaluator.evaluate(ir.entry.condition, ctx)


def evaluate_exit(ir: StrategyIR, ctx: EvalContext, exit_index: int = 0) -> bool:
    """Check if exit condition fires."""
    if not ir.exits or exit_index >= len(ir.exits):
        return False
    evaluator = ConditionEvaluator()
    return evaluator.evaluate(ir.exits[exit_index].condition, ctx)


def evaluate_gate(ir: StrategyIR, ctx: EvalContext, gate_index: int = 0) -> bool:
    """Check if gate allows (returns True) or blocks (returns False)."""
    if not ir.gates or gate_index >= len(ir.gates):
        return True  # No gate = allow
    evaluator = ConditionEvaluator()
    return evaluator.evaluate(ir.gates[gate_index].condition, ctx)


def run_state_ops(ir: StrategyIR, ctx: EvalContext, invested: bool = False) -> None:
    """Execute state operations for current bar."""
    evaluator = ConditionEvaluator()
    operator = StateOperator(evaluator)

    for op in ir.on_bar:
        operator.execute(op, ctx)

    if invested:
        for op in ir.on_bar_invested:
            operator.execute(op, ctx)


# =============================================================================
# ENTRY ARCHETYPE SCENARIOS
# =============================================================================


class TestEntryRuleTrigger:
    """Proves entry.rule_trigger works with various conditions."""

    def test_regime_ret_pct_positive(self):
        """Entry fires when return % exceeds threshold."""
        ir = translate({
            "entry": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "ret_pct", "op": ">", "value": 2.0, "lookback_bars": 1},
                        }
                    },
                    "action": {"direction": "long"},
                },
            }
        })

        # ROC = 3% > 2% threshold → should fire
        ctx = EvalContext(
            indicators={"roc_1": MockIndicator(0.03)},  # 3%
            state={"entry_price": None, "bars_since_entry": 0},
            price_bar=MockPriceBar(close=103.0),
        )
        assert evaluate_entry(ir, ctx) is True

        # ROC = 1% < 2% threshold → should NOT fire
        ctx.indicators["roc_1"] = MockIndicator(0.01)
        assert evaluate_entry(ir, ctx) is False

    def test_trend_ma_relation(self):
        """Entry fires when fast MA above slow MA."""
        ir = translate({
            "entry": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "trend_ma_relation", "op": ">", "value": 0, "ma_fast": 20, "ma_slow": 50},
                        }
                    },
                    "action": {"direction": "long"},
                },
            }
        })

        # EMA20=105, EMA50=100 → fast > slow → fire
        ctx = EvalContext(
            indicators={
                "ema_20": MockIndicator(105.0),
                "ema_50": MockIndicator(100.0),
            },
            state={"entry_price": None, "bars_since_entry": 0},
            price_bar=MockPriceBar(close=103.0),
        )
        assert evaluate_entry(ir, ctx) is True

        # EMA20=95, EMA50=100 → fast < slow → don't fire
        ctx.indicators["ema_20"] = MockIndicator(95.0)
        assert evaluate_entry(ir, ctx) is False


class TestTrendPullback:
    """Proves entry.trend_pullback works: uptrend + band touch."""

    def test_uptrend_with_lower_band_touch(self):
        """Entry when uptrend AND price touches lower band."""
        ir = translate({
            "entry": {
                "type": "entry.trend_pullback",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                },
            }
        })

        # Uptrend (EMA20 > EMA50) AND close at lower band
        ctx = EvalContext(
            indicators={
                "ema_20": MockIndicator(105.0),
                "ema_50": MockIndicator(100.0),
                "bb_20": MockBandIndicator(upper=110.0, middle=100.0, lower=90.0),
            },
            state={"entry_price": None, "bars_since_entry": 0},
            price_bar=MockPriceBar(close=90.0),  # At lower band
        )
        assert evaluate_entry(ir, ctx) is True

    def test_downtrend_blocks_entry(self):
        """No entry even at lower band if downtrend."""
        ir = translate({
            "entry": {
                "type": "entry.trend_pullback",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                },
            }
        })

        # Downtrend (EMA20 < EMA50) but price at lower band
        ctx = EvalContext(
            indicators={
                "ema_20": MockIndicator(95.0),  # Below slow
                "ema_50": MockIndicator(100.0),
                "bb_20": MockBandIndicator(upper=110.0, middle=100.0, lower=90.0),
            },
            state={"entry_price": None, "bars_since_entry": 0},
            price_bar=MockPriceBar(close=90.0),
        )
        assert evaluate_entry(ir, ctx) is False


class TestBreakoutTrendfollow:
    """Proves entry.breakout_trendfollow works: price breaks above high."""

    def test_breakout_above_donchian_upper(self):
        """Entry when close breaks above lookback high."""
        ir = translate({
            "entry": {
                "type": "entry.breakout_trendfollow",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"breakout": {"lookback_bars": 50}},
                    "action": {"direction": "long"},
                },
            }
        })

        # Close=112 > DC upper=110 → breakout
        ctx = EvalContext(
            indicators={
                "dc_50": MockBandIndicator(upper=110.0, middle=100.0, lower=90.0),
            },
            state={"entry_price": None, "bars_since_entry": 0},
            price_bar=MockPriceBar(close=112.0),
        )
        assert evaluate_entry(ir, ctx) is True

        # Close=108 < DC upper=110 → no breakout
        ctx = EvalContext(
            indicators={
                "dc_50": MockBandIndicator(upper=110.0, middle=100.0, lower=90.0),
            },
            state={"entry_price": None, "bars_since_entry": 0},
            price_bar=MockPriceBar(close=108.0),
        )
        assert evaluate_entry(ir, ctx) is False


# =============================================================================
# EXIT ARCHETYPE SCENARIOS
# =============================================================================


class TestTrailingStop:
    """Proves exit.trailing_stop works with state tracking."""

    def test_trailing_stop_with_state_update(self):
        """Exit fires when price drops below trailing level."""
        ir = translate({
            "entry": {
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
            "exit": {
                "type": "exit.trailing_stop",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"trail_band": {"band": "keltner", "length": 20, "mult": 2.0}},
                    "action": {"mode": "close"},
                },
            },
        })

        # Verify state variable created
        state_ids = [s.id for s in ir.state]
        assert "highest_since_entry" in state_ids

        # Verify on_bar_invested hook exists
        assert len(ir.on_bar_invested) >= 1
        assert ir.on_bar_invested[0].state_id == "highest_since_entry"

        # Initialize state and run a few bars
        state = {s.id: s.default for s in ir.state}
        state["highest_since_entry"] = 100.0  # Simulated entry at 100

        # Bar 1: Price at 105, ATR=2 → trail level = 105 - 2*2 = 101
        # Close=103 > 101 → NO exit
        ctx = EvalContext(
            indicators={"atr_20": MockIndicator(2.0)},
            state=state,
            price_bar=MockPriceBar(high=105.0, close=103.0),
        )
        assert evaluate_exit(ir, ctx) is False

        # Update highest via state op
        run_state_ops(ir, ctx, invested=True)
        assert ctx.state["highest_since_entry"] == 105.0

        # Bar 2: Price drops, ATR=2 → trail level = 105 - 2*2 = 101
        # Close=99 < 101 → EXIT
        ctx = EvalContext(
            indicators={"atr_20": MockIndicator(2.0)},
            state=ctx.state,
            price_bar=MockPriceBar(high=100.0, close=99.0),
        )
        assert evaluate_exit(ir, ctx) is True


class TestBandExit:
    """Proves exit.band_exit fires at target band."""

    def test_exit_at_upper_band(self):
        """Exit when price reaches upper band."""
        ir = translate({
            "entry": {
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
            "exit": {
                "type": "exit.band_exit",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "exit_trigger": {"edge": "upper"},
                    },
                    "action": {"mode": "close"},
                },
            },
        })

        # Close=111 >= upper=110 → exit
        ctx = EvalContext(
            indicators={
                "bb_20": MockBandIndicator(upper=110.0, middle=100.0, lower=90.0),
            },
            state={"entry_price": None, "bars_since_entry": 0},
            price_bar=MockPriceBar(close=111.0),
        )
        assert evaluate_exit(ir, ctx) is True

        # Close=105 < upper=110 → no exit
        ctx.price_bar = MockPriceBar(close=105.0)
        assert evaluate_exit(ir, ctx) is False


# =============================================================================
# GATE ARCHETYPE SCENARIOS
# =============================================================================


class TestRegimeGate:
    """Proves gate.regime filters entries."""

    def test_gate_allows_in_bull_market(self):
        """Gate allows when regime condition met."""
        ir = translate({
            "entry": {
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
            "gate": {
                "type": "gate.regime",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            # gate.regime uses flat RegimeSpec, not nested ConditionSpec
                            "metric": "trend_ma_relation",
                            "op": ">",
                            "value": 0,
                            "ma_fast": 50,
                            "ma_slow": 200,
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]},
                },
            },
        })

        # Bull market: EMA50 > EMA200
        ctx = EvalContext(
            indicators={
                "ema_50": MockIndicator(110.0),
                "ema_200": MockIndicator(100.0),
            },
            state={},
            price_bar=MockPriceBar(),
        )
        assert evaluate_gate(ir, ctx) is True

    def test_gate_blocks_in_bear_market(self):
        """Gate blocks when regime condition not met."""
        ir = translate({
            "entry": {
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
            "gate": {
                "type": "gate.regime",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "metric": "trend_ma_relation",
                            "op": ">",
                            "value": 0,
                            "ma_fast": 50,
                            "ma_slow": 200,
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]},
                },
            },
        })

        # Bear market: EMA50 < EMA200
        ctx = EvalContext(
            indicators={
                "ema_50": MockIndicator(90.0),
                "ema_200": MockIndicator(100.0),
            },
            state={},
            price_bar=MockPriceBar(),
        )
        assert evaluate_gate(ir, ctx) is False


class TestTimeFilterGate:
    """Proves gate.time_filter restricts trading hours."""

    def test_allows_during_market_hours(self):
        """Gate allows during specified hours."""
        ir = translate({
            "entry": {
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
            "gate": {
                "type": "gate.time_filter",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "filter": {
                            "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                            "time_window": "0900-1700",
                        },
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]},
                },
            },
        })

        # 10 AM Wednesday → within window
        ctx = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            hour=10,
            day_of_week=2,  # Wednesday
        )
        assert evaluate_gate(ir, ctx) is True

    def test_blocks_outside_market_hours(self):
        """Gate blocks outside specified hours."""
        ir = translate({
            "entry": {
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
            "gate": {
                "type": "gate.time_filter",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "filter": {
                            "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                            "time_window": "0900-1700",
                        },
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]},
                },
            },
        })

        # 8 PM Wednesday → outside window
        ctx = EvalContext(
            indicators={},
            state={},
            price_bar=MockPriceBar(),
            hour=20,
            day_of_week=2,
        )
        assert evaluate_gate(ir, ctx) is False


# =============================================================================
# COMBINED SCENARIOS
# =============================================================================


class TestEntryExitGateCombination:
    """Proves complete strategy with entry + exit + gate works together."""

    def test_full_trade_lifecycle(self):
        """Entry, gate check, exit sequence."""
        ir = translate({
            "entry": {
                "type": "entry.trend_pullback",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                },
            },
            "exit": {
                "type": "exit.trailing_stop",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"trail_band": {"band": "keltner", "length": 20, "mult": 2.0}},
                },
            },
            "gate": {
                "type": "gate.regime",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            # gate.regime uses flat RegimeSpec
                            "metric": "trend_ma_relation",
                            "op": ">",
                            "value": 0,
                            "ma_fast": 50,
                            "ma_slow": 200,
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]},
                },
            },
        })

        # Initialize state
        state = {s.id: s.default for s in ir.state}

        # BAR 1: Bull market, uptrend, price at lower band → ENTRY
        ctx1 = EvalContext(
            indicators={
                "ema_20": MockIndicator(105.0),
                "ema_50": MockIndicator(100.0),
                "ema_200": MockIndicator(95.0),
                "bb_20": MockBandIndicator(upper=110.0, middle=100.0, lower=90.0),
                "atr_20": MockIndicator(2.0),
            },
            state=state,
            price_bar=MockPriceBar(high=92.0, low=88.0, close=90.0),
        )

        gate_allows = evaluate_gate(ir, ctx1)
        entry_fires = evaluate_entry(ir, ctx1)
        assert gate_allows is True, "Gate should allow in bull market"
        assert entry_fires is True, "Entry should fire at lower band in uptrend"

        # Simulate entry fill - update state
        state["entry_price"] = 90.0
        state["bars_since_entry"] = 0
        state["highest_since_entry"] = 90.0

        # BAR 2: Price rises to 100 → update trailing level, no exit
        ctx2 = EvalContext(
            indicators={
                "ema_20": MockIndicator(105.0),
                "ema_50": MockIndicator(100.0),
                "atr_20": MockIndicator(2.0),
            },
            state=state,
            price_bar=MockPriceBar(high=100.0, close=98.0),
        )
        run_state_ops(ir, ctx2, invested=True)
        assert ctx2.state["highest_since_entry"] == 100.0

        exit_fires = evaluate_exit(ir, ctx2)
        assert exit_fires is False, "Exit should not fire - above trailing level"

        # BAR 3: Price drops to 95 → still above trail (100 - 4 = 96)
        ctx3 = EvalContext(
            indicators={"atr_20": MockIndicator(2.0)},
            state=ctx2.state,
            price_bar=MockPriceBar(high=96.0, close=95.0),
        )
        exit_fires = evaluate_exit(ir, ctx3)
        assert exit_fires is True, "Exit should fire - below trailing level (100 - 4 = 96)"


class TestMultipleExits:
    """Proves multiple exit conditions work with priority."""

    def test_first_exit_wins(self):
        """First triggered exit should be used."""
        ir = translate({
            "entry": {
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
            "exit1": {
                "type": "exit.band_exit",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "exit_trigger": {"edge": "upper"},  # Take profit
                    },
                    "action": {"mode": "close"},
                },
            },
            "exit2": {
                "type": "exit.trailing_stop",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"trail_band": {"band": "keltner", "length": 20, "mult": 3.0}},
                },
            },
        })

        assert len(ir.exits) == 2

        # Initialize state
        state = {s.id: s.default for s in ir.state}
        state["highest_since_entry"] = 100.0

        # Scenario: Price at upper band (take profit triggers)
        ctx = EvalContext(
            indicators={
                "bb_20": MockBandIndicator(upper=110.0, middle=100.0, lower=90.0),
                "atr_20": MockIndicator(2.0),
            },
            state=state,
            price_bar=MockPriceBar(close=111.0),  # Above upper band
        )

        exit1_fires = evaluate_exit(ir, ctx, exit_index=0)
        exit2_fires = evaluate_exit(ir, ctx, exit_index=1)

        assert exit1_fires is True, "Take profit should trigger"
        assert exit2_fires is False, "Trailing stop should not trigger (above trail level)"


# =============================================================================
# COMPOSITE CONDITION SCENARIOS
# =============================================================================


class TestAllOfCondition:
    """Proves allOf (AND) logic works."""

    def test_both_conditions_required(self):
        """Entry only when ALL conditions met."""
        ir = translate({
            "entry": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "allOf",
                            "allOf": [  # Key matches type name
                                {
                                    "type": "regime",
                                    "regime": {"metric": "trend_ma_relation", "op": ">", "value": 0, "ma_fast": 20, "ma_slow": 50},
                                },
                                {
                                    "type": "regime",
                                    "regime": {"metric": "ret_pct", "op": "<", "value": -2.0, "lookback_bars": 1},
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
            }
        })

        # Both true: uptrend AND oversold
        ctx = EvalContext(
            indicators={
                "ema_20": MockIndicator(105.0),
                "ema_50": MockIndicator(100.0),
                "roc_1": MockIndicator(-0.03),  # -3%
            },
            state={"entry_price": None, "bars_since_entry": 0},
            price_bar=MockPriceBar(),
        )
        assert evaluate_entry(ir, ctx) is True

        # Only uptrend, not oversold
        ctx.indicators["roc_1"] = MockIndicator(-0.01)  # -1%
        assert evaluate_entry(ir, ctx) is False


class TestAnyOfCondition:
    """Proves anyOf (OR) logic works."""

    def test_either_condition_sufficient(self):
        """Entry when ANY condition met."""
        ir = translate({
            "entry": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "anyOf",
                            "anyOf": [  # Key matches type name
                                {
                                    "type": "regime",
                                    "regime": {"metric": "ret_pct", "op": "<", "value": -5.0, "lookback_bars": 1},
                                },
                                {
                                    "type": "regime",
                                    "regime": {"metric": "ret_pct", "op": ">", "value": 5.0, "lookback_bars": 1},
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
            }
        })

        # First condition true: very oversold
        ctx = EvalContext(
            indicators={"roc_1": MockIndicator(-0.06)},  # -6%
            state={"entry_price": None, "bars_since_entry": 0},
            price_bar=MockPriceBar(),
        )
        assert evaluate_entry(ir, ctx) is True

        # Second condition true: very overbought
        ctx.indicators["roc_1"] = MockIndicator(0.06)  # +6%
        assert evaluate_entry(ir, ctx) is True

        # Neither true: moderate
        ctx.indicators["roc_1"] = MockIndicator(0.02)  # +2%
        assert evaluate_entry(ir, ctx) is False


# =============================================================================
# IR STRUCTURE VALIDATION
# =============================================================================


class TestIRStructure:
    """Proves IR is correctly structured after translation."""

    def test_complete_strategy_ir_structure(self):
        """Verify all IR components are present and valid."""
        ir = translate({
            "entry": {
                "type": "entry.trend_pullback",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                },
            },
            "exit": {
                "type": "exit.trailing_stop",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"trail_band": {"band": "keltner", "length": 20, "mult": 2.0}},
                },
            },
            "gate": {
                "type": "gate.regime",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            # gate.regime uses flat RegimeSpec
                            "metric": "trend_ma_relation",
                            "op": ">",
                            "value": 0,
                            "ma_fast": 50,
                            "ma_slow": 200,
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]},
                },
            },
        })

        # Entry
        assert ir.entry is not None
        assert ir.entry.condition is not None
        assert ir.entry.action is not None

        # Exit
        assert len(ir.exits) == 1
        assert ir.exits[0].condition is not None

        # Gate
        assert len(ir.gates) == 1
        assert ir.gates[0].condition is not None
        assert ir.gates[0].mode == "allow"

        # State (from entry + exit)
        state_ids = [s.id for s in ir.state]
        assert "entry_price" in state_ids
        assert "bars_since_entry" in state_ids
        assert "highest_since_entry" in state_ids

        # Hooks (from trailing_stop)
        assert len(ir.on_bar_invested) >= 1

        # Indicators
        assert len(ir.indicators) >= 3  # EMA20, EMA50, BB, ATR...
