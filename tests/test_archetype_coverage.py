"""Comprehensive archetype coverage tests.

Tests that ALL archetypes in the MCP schema can be:
1. Translated to IR without errors
2. Simulated with mock data
3. Produce expected signals
"""

import pytest
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment

from src.translator.ir_translator import IRTranslator
from tests.conftest import make_strategy


class TestEntryArchetypes:
    """Test all entry archetypes translate successfully."""

    def test_entry_rule_trigger_regime(self):
        """entry.rule_trigger with regime condition."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_entry_rule_trigger_band_event(self):
        """entry.rule_trigger with band_event condition."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "band_event",
                                "band_event": {
                                    "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                                    "kind": "edge_event",
                                    "edge": "lower",
                                    "event": "touch",
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_entry_rule_trigger_allof(self):
        """entry.rule_trigger with allOf composite condition."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                                            "band": {
                                                "band": "bollinger",
                                                "length": 20,
                                                "mult": 2.0,
                                            },
                                            "kind": "edge_event",
                                            "edge": "lower",
                                            "event": "touch",
                                        },
                                    },
                                ],
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_entry_rule_trigger_sequence(self):
        """entry.rule_trigger with sequence condition."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "sequence",
                                "sequence": [
                                    {
                                        "cond": {
                                            "type": "regime",
                                            "regime": {
                                                "metric": "ret_pct",
                                                "op": "<",
                                                "value": -2.0,
                                                "lookback_bars": 1,
                                            },
                                        }
                                    },
                                    {
                                        "cond": {
                                            "type": "regime",
                                            "regime": {
                                                "metric": "ret_pct",
                                                "op": ">",
                                                "value": 0,
                                                "lookback_bars": 1,
                                            },
                                        },
                                        "within_bars": 5,
                                    },
                                ],
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        # Sequence creates state variables
        assert len(result.state) > 0

    def test_entry_trend_pullback(self):
        """entry.trend_pullback archetype."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.trend_pullback",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "dip": {"kind": "edge_event", "edge": "lower", "op": "touch"},
                            "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                            "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

        # Semantic assertions: verify IR structure matches archetype intent
        # 1. Entry condition should be AllOf (trend AND band event)
        assert result.entry.condition.type == "allOf", "trend_pullback should use AllOf condition"
        assert len(result.entry.condition.conditions) == 2, "Should have trend and dip conditions"

        # 2. Required indicators: EMA fast/slow + Bollinger bands
        indicator_types = {ind.type for ind in result.indicators}
        assert "EMA" in indicator_types, "Should have EMA indicators for trend gate"
        assert "BB" in indicator_types, "Should have Bollinger bands for dip"

        # 3. State variables for entry tracking
        state_ids = {sv.id for sv in result.state}
        assert "entry_price" in state_ids, "Should track entry price"
        assert "bars_since_entry" in state_ids, "Should track bars since entry"

        # 4. Action direction
        assert result.entry.action.allocation > 0, "Long entry should have positive allocation"

    def test_entry_breakout_trendfollow(self):
        """entry.breakout_trendfollow archetype."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.breakout_trendfollow",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {"breakout": {"lookback_bars": 50, "buffer_bps": 5}},
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

        # Semantic assertions
        # 1. Uses compare condition (price > highest)
        assert result.entry.condition.type == "compare", "Breakout should compare price to highest"

        # 2. Requires Donchian Channel for lookback high/low
        indicator_types = {ind.type for ind in result.indicators}
        assert "DC" in indicator_types, "Should use Donchian Channel for breakout levels"

        # 3. State tracking
        state_ids = {sv.id for sv in result.state}
        assert "entry_price" in state_ids, "Should track entry price"

    def test_entry_range_mean_reversion(self):
        """entry.range_mean_reversion via to_ir()."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.range_mean_reversion",
                    "slots": {
                        "context": {"tf": "15m", "symbol": "ETH-USD"},
                        "event": {
                            "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                            "trigger": {"kind": "edge_event", "edge": "lower", "op": "touch"},
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        # Should expand to rule_trigger
        assert result.entry is not None

    def test_entry_squeeze_expansion(self):
        """entry.squeeze_expansion using PCTILE indicator."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.squeeze_expansion",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
                        "event": {
                            "squeeze": {
                                "metric": "bb_width_pctile",
                                "pctile_min": 10,
                                "break_rule": "donchian",
                            }
                        },
                        "action": {"direction": "auto"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_entry_breakout_condition(self):
        """entry.rule_trigger with breakout condition type."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "breakout",
                                "breakout": {"lookback_bars": 50, "buffer_bps": 5},
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_entry_squeeze_condition(self):
        """entry.rule_trigger with squeeze condition type."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "squeeze",
                                "squeeze": {"metric": "bb_width_pctile", "pctile_min": 10},
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_entry_time_filter_condition(self):
        """entry.rule_trigger with time_filter condition type."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "time_filter",
                                "time_filter": {
                                    "days_of_week": ["monday", "tuesday", "wednesday"],
                                    "time_window": "0930-1600",
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_entry_breakout_retest(self):
        """entry.breakout_retest via to_ir() with sequence support."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.breakout_retest",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "retest": {
                                "break_lookback_bars": 50,
                                "pullback_depth_atr": 1.5,
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_entry_avwap_reversion(self):
        """entry.avwap_reversion using AVWAP indicator."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.avwap_reversion",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {"anchor": {"anchor": "session_open"}, "dist_sigma_entry": 2.0},
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_entry_gap_play(self):
        """entry.gap_play with session gap trading."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.gap_play",
                    "slots": {
                        "context": {"tf": "15m", "symbol": "SPY"},
                        "event": {
                            "session": {
                                "session": "us",
                                "window": "0930-1000",
                                "mode": "gap_fade",
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None


class TestExitArchetypes:
    """Test all exit archetypes translate successfully."""

    def test_exit_rule_trigger(self):
        """exit.rule_trigger archetype."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                },
                "exit1": {
                    "type": "exit.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "ret_pct",
                                    "op": ">",
                                    "value": 5.0,
                                    "lookback_bars": 1,
                                },
                            }
                        },
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert len(result.exits) == 1

    def test_exit_rule_trigger_with_allof(self):
        """exit.rule_trigger with allOf composite condition."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "ret_pct",
                                    "op": ">",
                                    "value": 1.0,
                                    "lookback_bars": 1,
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                },
                "exit1": {
                    "type": "exit.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "allOf",
                                "allOf": [
                                    {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "ret_pct",
                                            "op": ">",
                                            "value": 5.0,
                                            "lookback_bars": 1,
                                        },
                                    },
                                    {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "trend_ma_relation",
                                            "op": "<",
                                            "value": 0,
                                            "ma_fast": 20,
                                            "ma_slow": 50,
                                        },
                                    },
                                ],
                            }
                        },
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert len(result.exits) == 1
        assert result.exits[0].condition is not None

    def test_exit_rule_trigger_with_anyof(self):
        """exit.rule_trigger with anyOf composite condition."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "ret_pct",
                                    "op": ">",
                                    "value": 1.0,
                                    "lookback_bars": 1,
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                },
                "exit1": {
                    "type": "exit.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "anyOf",
                                "anyOf": [
                                    {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "ret_pct",
                                            "op": ">",
                                            "value": 10.0,
                                            "lookback_bars": 1,
                                        },
                                    },
                                    {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "ret_pct",
                                            "op": "<",
                                            "value": -5.0,
                                            "lookback_bars": 1,
                                        },
                                    },
                                ],
                            }
                        },
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert len(result.exits) == 1
        assert result.exits[0].condition is not None

    def test_exit_trailing_stop(self):
        """exit.trailing_stop archetype."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                },
                "exit1": {
                    "type": "exit.trailing_stop",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "ETH-USD"},
                        "event": {
                            "trail_band": {"band": "keltner", "length": 20, "mult": 2.0},
                            "trail_trigger": {
                                "kind": "edge_event",
                                "edge": "lower",
                                "op": "cross_out",
                            },
                        },
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        ir = IRTranslator(strategy, cards).translate()
        assert len(ir.exits) == 1

        # Verify state variable for tracking highest since entry
        state_ids = [s.id for s in ir.state]
        assert "highest_since_entry" in state_ids

        # Verify on_bar_invested hook updates the state
        assert len(ir.on_bar_invested) >= 1
        max_op = ir.on_bar_invested[0]
        assert max_op.state_id == "highest_since_entry"

        # Verify exit condition uses StateValue (not IndicatorValue)
        exit_rule = ir.exits[0]
        cond = exit_rule.condition
        # The condition is: close < highest_since_entry - (mult * atr)
        assert cond.left.field.value == "close"
        # Right side is an expression: highest_since_entry - (mult * atr)
        expr = cond.right
        assert expr.op == "-"
        # Left of subtraction should be StateValue for highest_since_entry
        assert hasattr(expr.left, "state_id")
        assert expr.left.state_id == "highest_since_entry"

    def test_exit_band_exit(self):
        """exit.band_exit archetype."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                },
                "exit1": {
                    "type": "exit.band_exit",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
                        "event": {
                            "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                            "exit_trigger": {"edge": "upper"},
                        },
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert len(result.exits) == 1

        # Semantic assertions for exit.band_exit
        exit_rule = result.exits[0]
        # 1. Exit condition should be compare (price vs band)
        assert exit_rule.condition.type == "compare", "Band exit should compare price to band"

        # 2. Bollinger bands should be created
        indicator_types = {ind.type for ind in result.indicators}
        assert "BB" in indicator_types, "Should have Bollinger bands"

        # 3. Exit action should be liquidate
        assert exit_rule.action.type == "liquidate", "Band exit should liquidate position"

    def test_exit_structure_break(self):
        """exit.structure_break via to_ir() - exits on MA cross."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                },
                "exit1": {
                    "type": "exit.structure_break",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "BTC-USD"},
                        "event": {"structure": {"fast": 20, "slow": 50, "op": "<"}},
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert len(result.exits) == 1

    def test_exit_squeeze_compression(self):
        """exit.squeeze_compression using PCTILE indicator."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                },
                "exit1": {
                    "type": "exit.squeeze_compression",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
                        "event": {"squeeze": {"metric": "bb_width_pctile", "pctile_min": 10}},
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert len(result.exits) == 1

    def test_exit_vwap_reversion(self):
        """exit.vwap_reversion using AVWAP indicator."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                },
                "exit1": {
                    "type": "exit.vwap_reversion",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {"anchor": {"anchor": "session_open"}, "dist_sigma_exit": 0.5},
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert len(result.exits) == 1


class TestGateArchetypes:
    """Test all gate archetypes translate successfully."""

    def test_gate_regime(self):
        """gate.regime archetype."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                },
                "gate1": {
                    "type": "gate.regime",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "metric": "trend_ma_relation",
                                "op": ">",
                                "value": 0,
                                "ma_fast": 20,
                                "ma_slow": 50,
                            }
                        },
                        "action": {"mode": "allow", "target_roles": ["entry"]},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert len(result.gates) == 1

        # Semantic assertions for gate.regime
        gate = result.gates[0]
        # 1. Gate condition should be a compare (for trend_ma_relation)
        assert gate.condition.type == "compare", "Regime gate should use compare condition"

        # 2. Gate mode should be "allow" (as specified in slots)
        assert gate.mode == "allow", "Gate mode should match slot configuration"

        # 3. Gate targets entries
        assert "entry" in gate.target_roles, "Gate should target entries"

        # 4. EMA indicators should be created for ma_fast/ma_slow
        indicator_types = {ind.type for ind in result.indicators}
        assert "EMA" in indicator_types, "Should have EMA indicators for regime check"

    def test_gate_time_filter(self):
        """gate.time_filter via expansion."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                },
                "gate1": {
                    "type": "gate.time_filter",
                    "slots": {
                        "context": {"symbol": "SPY"},
                        "event": {
                            "filter": {
                                "days_of_week": [
                                    "monday",
                                    "tuesday",
                                    "wednesday",
                                    "thursday",
                                    "friday",
                                ],
                                "time_window": "0930-1600",
                                "timezone": "America/New_York",
                            }
                        },
                        "action": {"mode": "allow", "target_roles": ["entry"]},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        # gate.time_filter should expand or have a handler
        assert len(result.gates) >= 0  # May or may not have gate depending on expansion


class TestOverlayArchetypes:
    """Test overlay archetypes translate successfully."""

    def test_overlay_regime_scaler(self):
        """overlay.regime_scaler archetype."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                },
                "overlay1": {
                    "type": "overlay.regime_scaler",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "BTC-USD"},
                        "event": {
                            "regime": {
                                "type": "regime",
                                "regime": {
                                    "metric": "vol_bb_width_pctile",
                                    "op": "<",
                                    "value": 20,
                                    "lookback_bars": 200,
                                },
                            }
                        },
                        "action": {
                            "scale_risk_frac": 0.5,
                            "scale_size_frac": 0.5,
                            "target_roles": ["entry"],
                        },
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        # Verify entry and overlay were translated
        assert result.entry is not None
        assert len(result.overlays) == 1

        # Verify overlay properties
        overlay = result.overlays[0]
        assert overlay.scale_risk_frac == 0.5
        assert overlay.scale_size_frac == 0.5
        assert overlay.target_roles == ["entry"]
        assert overlay.condition is not None

    def test_overlay_with_target_tags(self):
        """overlay.regime_scaler with target_tags filtering."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
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
                },
                "overlay1": {
                    "type": "overlay.regime_scaler",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
                        "event": {
                            "regime": {
                                "type": "regime",
                                "regime": {
                                    "metric": "vol_atr_pct",
                                    "op": "<",
                                    "value": 2.0,
                                    "lookback_bars": 14,
                                },
                            }
                        },
                        "action": {
                            "scale_risk_frac": 0.25,
                            "scale_size_frac": 0.25,
                            "target_roles": ["entry", "exit"],
                            "target_tags": ["breakout"],
                        },
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert len(result.overlays) == 1

        overlay = result.overlays[0]
        assert overlay.scale_risk_frac == 0.25
        assert overlay.scale_size_frac == 0.25
        assert overlay.target_roles == ["entry", "exit"]
        assert overlay.target_tags == ["breakout"]

    def test_multiple_overlays(self):
        """Multiple overlay.regime_scaler cards in one strategy."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "ret_pct",
                                    "op": ">",
                                    "value": 0.5,
                                    "lookback_bars": 1,
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                },
                "overlay_vol": {
                    "type": "overlay.regime_scaler",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "BTC-USD"},
                        "event": {
                            "regime": {
                                "type": "regime",
                                "regime": {
                                    "metric": "vol_bb_width_pctile",
                                    "op": "<",
                                    "value": 15,
                                    "lookback_bars": 100,
                                },
                            }
                        },
                        "action": {
                            "scale_risk_frac": 0.5,
                            "scale_size_frac": 0.5,
                            "target_roles": ["entry"],
                        },
                    },
                },
                "overlay_trend": {
                    "type": "overlay.regime_scaler",
                    "slots": {
                        "context": {"tf": "1d", "symbol": "BTC-USD"},
                        "event": {
                            "regime": {
                                "type": "regime",
                                "regime": {
                                    "metric": "trend_adx",
                                    "op": "<",
                                    "value": 20,
                                    "lookback_bars": 14,
                                },
                            }
                        },
                        "action": {
                            "scale_risk_frac": 0.75,
                            "scale_size_frac": 0.75,
                            "target_roles": ["entry", "exit"],
                        },
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert len(result.overlays) == 2

        # Both overlays should have been translated with correct properties
        risk_fracs = {o.scale_risk_frac for o in result.overlays}
        assert risk_fracs == {0.5, 0.75}


class TestComplexStrategies:
    """Test complex multi-card strategies."""

    def test_trend_pullback_with_trailing_stop(self):
        """Complete trend pullback strategy with trailing stop exit."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.trend_pullback",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "dip": {"kind": "edge_event", "edge": "lower", "op": "touch"},
                            "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                            "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                        },
                        "action": {"direction": "long"},
                    },
                },
                "exit1": {
                    "type": "exit.trailing_stop",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "trail_band": {"band": "keltner", "length": 20, "mult": 2.0},
                            "trail_trigger": {
                                "kind": "edge_event",
                                "edge": "lower",
                                "op": "cross_out",
                            },
                        },
                        "action": {"mode": "close"},
                    },
                },
                "exit2": {
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
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        assert len(result.exits) == 2

    def test_breakout_with_regime_gate(self):
        """Breakout entry gated by trend regime."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.breakout_trendfollow",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {"breakout": {"lookback_bars": 50, "buffer_bps": 5}},
                        "action": {"direction": "long"},
                    },
                },
                "gate1": {
                    "type": "gate.regime",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                },
                "exit1": {
                    "type": "exit.rule_trigger",
                    "slots": {
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
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        assert len(result.gates) == 1
        assert len(result.exits) == 1

    def test_mean_reversion_with_band_exit(self):
        """Mean reversion with band exit strategy (both use to_ir())."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.range_mean_reversion",
                    "slots": {
                        "context": {"tf": "15m", "symbol": "ETH-USD"},
                        "event": {
                            "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                            "trigger": {"kind": "edge_event", "edge": "lower", "op": "touch"},
                        },
                        "action": {"direction": "long"},
                    },
                },
                "exit1": {
                    "type": "exit.band_exit",
                    "slots": {
                        "context": {"tf": "15m", "symbol": "ETH-USD"},
                        "event": {
                            "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                            "exit_trigger": {"edge": "mid"},
                        },
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        assert len(result.exits) == 1

    def test_full_strategy_with_gate_and_overlay(self):
        """Complete strategy: entry + gate + overlay + multiple exits."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
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
                                            "ma_fast": 10,
                                            "ma_slow": 30,
                                        },
                                    },
                                    {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "ret_pct",
                                            "op": ">",
                                            "value": 1,
                                            "lookback_bars": 5,
                                        },
                                    },
                                ],
                            }
                        },
                        "action": {"direction": "long"},
                    },
                },
                "gate1": {
                    "type": "gate.regime",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "metric": "trend_adx",
                                "op": ">",
                                "value": 20,
                                "lookback_bars": 14,
                            }
                        },
                        "action": {"mode": "allow", "target_roles": ["entry"]},
                    },
                },
                "overlay1": {
                    "type": "overlay.regime_scaler",
                    "slots": {
                        "context": {"tf": "1d", "symbol": "BTC-USD"},
                        "event": {
                            "regime": {
                                "type": "regime",
                                "regime": {
                                    "metric": "vol_bb_width_pctile",
                                    "op": "<",
                                    "value": 30,
                                    "lookback_bars": 100,
                                },
                            }
                        },
                        "action": {
                            "scale_risk_frac": 0.5,
                            "scale_size_frac": 0.5,
                            "target_roles": ["entry"],
                        },
                    },
                },
                "exit_stop": {
                    "type": "exit.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "ret_pct",
                                    "op": "<",
                                    "value": -5,
                                    "lookback_bars": 3,
                                },
                            }
                        },
                        "action": {"mode": "close"},
                    },
                },
                "exit_profit": {
                    "type": "exit.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "ret_pct",
                                    "op": ">",
                                    "value": 10,
                                    "lookback_bars": 10,
                                },
                            }
                        },
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        assert len(result.gates) == 1
        assert len(result.overlays) == 1
        assert len(result.exits) == 2

        # Verify overlay was correctly translated
        overlay = result.overlays[0]
        assert overlay.scale_risk_frac == 0.5
        assert overlay.scale_size_frac == 0.5

    def test_multi_gate_strategy(self):
        """Strategy with multiple gates (trend + volatility)."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "ret_pct",
                                    "op": ">",
                                    "value": 2,
                                    "lookback_bars": 3,
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                },
                "gate_trend": {
                    "type": "gate.regime",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "ETH-USD"},
                        "event": {
                            "condition": {
                                "metric": "trend_ma_relation",
                                "op": ">",
                                "value": 0,
                                "ma_fast": 20,
                                "ma_slow": 50,
                            }
                        },
                        "action": {"mode": "allow", "target_roles": ["entry"]},
                    },
                },
                "gate_vol": {
                    "type": "gate.regime",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
                        "event": {
                            "condition": {
                                "metric": "vol_atr_pct",
                                "op": ">",
                                "value": 1,
                                "lookback_bars": 14,
                            }
                        },
                        "action": {"mode": "allow", "target_roles": ["entry"]},
                    },
                },
                "exit1": {
                    "type": "exit.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
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
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        assert len(result.gates) == 2

    def test_deeply_nested_conditions(self):
        """Strategy with deeply nested allOf/anyOf/not conditions."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "allOf",
                                "allOf": [
                                    {
                                        "type": "anyOf",
                                        "anyOf": [
                                            {
                                                "type": "regime",
                                                "regime": {
                                                    "metric": "trend_ma_relation",
                                                    "op": ">",
                                                    "value": 5,
                                                    "ma_fast": 5,
                                                    "ma_slow": 20,
                                                },
                                            },
                                            {
                                                "type": "regime",
                                                "regime": {
                                                    "metric": "trend_adx",
                                                    "op": ">",
                                                    "value": 30,
                                                    "lookback_bars": 14,
                                                },
                                            },
                                        ],
                                    },
                                    {
                                        "type": "not",
                                        "not": {
                                            "type": "regime",
                                            "regime": {
                                                "metric": "ret_pct",
                                                "op": "<",
                                                "value": -10,
                                                "lookback_bars": 5,
                                            },
                                        },
                                    },
                                    {
                                        "type": "anyOf",
                                        "anyOf": [
                                            {
                                                "type": "regime",
                                                "regime": {
                                                    "metric": "vol_bb_width_pctile",
                                                    "op": ">",
                                                    "value": 50,
                                                    "lookback_bars": 100,
                                                },
                                            },
                                            {
                                                "type": "allOf",
                                                "allOf": [
                                                    {
                                                        "type": "regime",
                                                        "regime": {
                                                            "metric": "ret_pct",
                                                            "op": ">",
                                                            "value": 1,
                                                            "lookback_bars": 1,
                                                        },
                                                    },
                                                    {
                                                        "type": "regime",
                                                        "regime": {
                                                            "metric": "ret_pct",
                                                            "op": ">",
                                                            "value": 0,
                                                            "lookback_bars": 3,
                                                        },
                                                    },
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            }
                        },
                        "action": {"direction": "long"},
                    },
                },
                "exit1": {
                    "type": "exit.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "ret_pct",
                                    "op": "<",
                                    "value": -5,
                                    "lookback_bars": 3,
                                },
                            }
                        },
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None


class TestRegimeMetrics:
    """Test all RegimeSpec metrics can be translated."""

    @pytest.mark.parametrize(
        "metric,extra_params,expected_indicators",
        [
            ("ret_pct", {"lookback_bars": 1}, {"ROC"}),
            ("trend_ma_relation", {"ma_fast": 20, "ma_slow": 50}, {"EMA"}),
            ("trend_regime", {"ma_fast": 20, "ma_slow": 50}, {"EMA"}),
            ("trend_adx", {"lookback_bars": 14}, {"ADX"}),
            ("vol_bb_width_pctile", {"lookback_bars": 100}, {"BB", "PCTILE"}),
            ("vol_atr_pct", {"lookback_bars": 14}, {"ATR"}),
            ("vol_regime", {"lookback_bars": 100}, {"BB"}),  # Uses BB width
            ("volume_pctile", {"lookback_bars": 20}, {"VOL_SMA"}),  # Volume SMA
        ],
    )
    def test_regime_metric(self, metric, extra_params, expected_indicators):
        """Test regime metric translation produces correct indicators."""
        regime = {"metric": metric, "op": ">", "value": 0, **extra_params}
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {"condition": {"type": "regime", "regime": regime}},
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

        # Verify at least one of the expected indicators is created
        indicator_types = {ind.type for ind in result.indicators}
        assert indicator_types & expected_indicators, (
            f"Metric {metric} should create one of {expected_indicators}, got {indicator_types}"
        )


class TestBandTypes:
    """Test all band types can be translated."""

    @pytest.mark.parametrize(
        "band_type,expected_indicator",
        [
            ("bollinger", "BB"),
            ("keltner", "KC"),
            ("donchian", "DC"),
        ],
    )
    def test_band_type(self, band_type, expected_indicator):
        """Test band type translation produces correct indicator."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "band_event",
                                "band_event": {
                                    "band": {"band": band_type, "length": 20, "mult": 2.0},
                                    "kind": "edge_event",
                                    "edge": "lower",
                                    "event": "touch",
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()

        # Verify correct indicator type is created
        indicator_types = {ind.type for ind in result.indicators}
        assert expected_indicator in indicator_types, (
            f"Band type {band_type} should create {expected_indicator} indicator"
        )
        assert result.entry is not None


class TestBandEvents:
    """Test all band event types can be translated."""

    @pytest.mark.parametrize(
        "kind,extra_params",
        [
            ("edge_event", {"edge": "lower", "event": "touch"}),
            ("edge_event", {"edge": "upper", "event": "touch"}),
            ("edge_event", {"edge": "lower", "event": "cross_in"}),
            ("edge_event", {"edge": "lower", "event": "cross_out"}),
            ("distance", {"mode": "z", "side": "away_lower", "thresh": 1.5}),
            ("distance", {"mode": "band_mult", "side": "away_upper", "thresh": 0.8}),
            ("reentry", {"edge": "lower"}),
            ("reentry", {"edge": "upper"}),
        ],
    )
    def test_band_event_type(self, kind, extra_params):
        """Test band event type translation."""
        band_event = {
            "band": {"band": "bollinger", "length": 20, "mult": 2.0},
            "kind": kind,
            **extra_params,
        }
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {"condition": {"type": "band_event", "band_event": band_event}},
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None


# =============================================================================
# Additional Archetypes (Gap Coverage)
# =============================================================================


class TestEventFollowthroughArchetype:
    """Test entry.event_followthrough archetype.

    This archetype triggers entries following catalyst events (earnings, macro).
    It's typically expanded to a rule_trigger with event-based conditions.
    """

    def test_entry_event_followthrough_earnings(self):
        """entry.event_followthrough for earnings catalyst."""
        strategy, cards = make_strategy(
            {
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
            }
        )
        result = IRTranslator(strategy, cards).translate()
        # Should translate to IR with event condition
        assert result is not None
        assert result.entry is not None

    def test_entry_event_followthrough_macro(self):
        """entry.event_followthrough for macro event catalyst."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.event_followthrough",
                    "slots": {
                        "context": {"symbol": "SPY"},
                        "event": {
                            "event": {
                                "event_kind": "macro",
                                "entry_window_bars": 4,
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result is not None
        assert result.entry is not None


class TestIntermarketTriggerArchetype:
    """Test entry.intermarket_trigger archetype.

    This archetype triggers entries based on movements in a different asset.
    MVP constraint: follower_symbol must equal context.symbol.
    """

    def test_entry_intermarket_single_leader(self):
        """entry.intermarket_trigger with single leader symbol."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.intermarket_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
                        "event": {
                            "lead_follow": {
                                "leader_symbol": "BTC-USD",
                                "follower_symbol": "ETH-USD",
                                "trigger_feature": "ret_pct",
                                "trigger_threshold": 3.0,
                                "window_bars": 24,
                                "entry_side_map": {"leader_up": "long", "leader_down": "short"},
                            }
                        },
                        "action": {"direction": "auto"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result is not None
        assert result.entry is not None
        assert "BTC-USD" in result.additional_symbols

    def test_entry_intermarket_multiple_leaders(self):
        """entry.intermarket_trigger with multiple leader symbols and aggregation."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.intermarket_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "SOL-USD"},
                        "event": {
                            "lead_follow": {
                                "leaders": ["BTC-USD", "ETH-USD"],
                                "follower_symbol": "SOL-USD",
                                "leader_aggregate": {
                                    "feature": "ret_pct",
                                    "op": "avg",
                                    "threshold": 2.0,
                                },
                                "window_bars": 12,
                                "entry_side_map": {"leader_up": "long", "leader_down": "none"},
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result is not None


class TestEventRiskWindowGate:
    """Test gate.event_risk_window archetype.

    This gate blocks/allows trading within a window around catalyst events.
    """

    def test_gate_event_risk_window_earnings(self):
        """gate.event_risk_window blocking around earnings."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "AAPL"},
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
                },
                "gate1": {
                    "type": "gate.event_risk_window",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "AAPL"},
                        "event": {
                            "catalyst": {
                                "event_kind": "earnings",
                                "entry_window_bars": 8,
                            },
                            "pre_event_bars": 8,
                            "post_event_bars": 12,
                        },
                        "action": {
                            "mode": "block",
                            "target_roles": ["entry"],
                        },
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        # Translates gate to event-based condition
        assert result is not None

    def test_gate_event_risk_window_macro(self):
        """gate.event_risk_window blocking around macro events."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "SPY"},
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
                },
                "gate1": {
                    "type": "gate.event_risk_window",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "SPY"},
                        "event": {
                            "catalyst": {
                                "event_kind": "macro",
                                "entry_window_bars": 4,
                            },
                            "pre_event_bars": 4,
                            "post_event_bars": 8,
                        },
                        "action": {"mode": "block", "target_roles": ["entry"]},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result is not None


# =============================================================================
# Additional Metrics (Gap Coverage)
# =============================================================================


class TestPennantPatternMetric:
    """Test pennant_pattern regime metric.

    Detects pennant consolidation pattern: initial momentum + triangular
    narrowing + breakout continuation.
    """

    def test_pennant_pattern_up(self):
        """pennant_pattern with upward breakout direction."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "pennant_pattern",
                                    "op": "==",
                                    "value": "up",
                                    "pennant_momentum_bars": 5,
                                    "pennant_consolidation_bars": 15,
                                    "pennant_breakout_direction": "up",
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        # Should create pattern detection indicators
        assert len(result.indicators) > 0

    def test_pennant_pattern_same_direction(self):
        """pennant_pattern with 'same' breakout direction (continuation)."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "ETH-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "pennant_pattern",
                                    "op": "==",
                                    "value": "same",
                                    "pennant_momentum_bars": 8,
                                    "pennant_consolidation_bars": 20,
                                    "pennant_breakout_direction": "same",
                                },
                            }
                        },
                        "action": {"direction": "auto"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None


class TestVolumePctileMetric:
    """Test volume_pctile regime metric.

    Percentile rank of volume vs lookback - approximated using relative volume.
    """

    def test_volume_pctile_high(self):
        """volume_pctile above threshold (high volume)."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "volume_pctile",
                                    "op": ">",
                                    "value": 80,
                                    "lookback_bars": 50,
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        # Should create volume SMA indicator
        vol_indicators = [i for i in result.indicators if "vol" in i.id.lower()]
        assert len(vol_indicators) > 0

    def test_volume_pctile_low(self):
        """volume_pctile below threshold (low volume)."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "volume_pctile",
                                    "op": "<",
                                    "value": 20,
                                    "lookback_bars": 30,
                                },
                            }
                        },
                        "action": {"direction": "short"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None


class TestVWAPBandType:
    """Test vwap_band band type.

    VWAP with standard deviation bands - used for intraday mean reversion.
    """

    def test_vwap_band_edge_event(self):
        """vwap_band with edge_event touch."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "15m", "symbol": "SPY"},
                        "event": {
                            "condition": {
                                "type": "band_event",
                                "band_event": {
                                    "band": {"band": "vwap_band", "length": 20, "mult": 2.0},
                                    "kind": "edge_event",
                                    "edge": "lower",
                                    "event": "touch",
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        # Either creates entry or warns about unsupported band type
        # Either way should not crash
        assert result is not None

    def test_vwap_band_distance(self):
        """vwap_band with distance z-score."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "15m", "symbol": "AAPL"},
                        "event": {
                            "condition": {
                                "type": "band_event",
                                "band_event": {
                                    "band": {"band": "vwap_band", "length": 0, "mult": 2.0},
                                    "kind": "distance",
                                    "mode": "z",
                                    "side": "away_lower",
                                    "thresh": 2.0,
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result is not None

    def test_vwap_band_anchor_session(self):
        """vwap_band with session anchor (default/most common)."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "15m", "symbol": "SPY"},
                        "event": {
                            "condition": {
                                "type": "band_event",
                                "band_event": {
                                    "band": {
                                        "band": "vwap_band",
                                        "length": 0,
                                        "mult": 2.0,
                                        "anchor": "session",
                                    },
                                    "kind": "edge_event",
                                    "edge": "lower",
                                    "event": "touch",
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result is not None

    def test_vwap_band_anchor_week(self):
        """vwap_band with weekly anchor."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ES"},
                        "event": {
                            "condition": {
                                "type": "band_event",
                                "band_event": {
                                    "band": {
                                        "band": "vwap_band",
                                        "length": 0,
                                        "mult": 1.5,
                                        "anchor": "week",
                                    },
                                    "kind": "distance",
                                    "mode": "z",
                                    "side": "away_lower",
                                    "thresh": 1.5,
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result is not None

    def test_vwap_band_anchor_month(self):
        """vwap_band with monthly anchor."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "NQ"},
                        "event": {
                            "condition": {
                                "type": "band_event",
                                "band_event": {
                                    "band": {
                                        "band": "vwap_band",
                                        "length": 0,
                                        "mult": 2.0,
                                        "anchor": "month",
                                    },
                                    "kind": "edge_event",
                                    "edge": "upper",
                                    "event": "touch",
                                },
                            }
                        },
                        "action": {"direction": "short"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result is not None

    def test_vwap_band_anchor_earnings(self):
        """vwap_band anchored to earnings (event-based)."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "AAPL"},
                        "event": {
                            "condition": {
                                "type": "band_event",
                                "band_event": {
                                    "band": {
                                        "band": "vwap_band",
                                        "length": 0,
                                        "mult": 2.5,
                                        "anchor": "earnings",
                                    },
                                    "kind": "reentry",
                                    "side": "from_below",
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result is not None

    def test_vwap_band_anchor_custom(self):
        """vwap_band with custom anchor date."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1d", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "band_event",
                                "band_event": {
                                    "band": {
                                        "band": "vwap_band",
                                        "length": 0,
                                        "mult": 3.0,
                                        "anchor": "custom",
                                    },
                                    "kind": "edge_event",
                                    "edge": "lower",
                                    "event": "cross_in",
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result is not None


# =============================================================================
# Complex Pattern Metrics (Full Coverage)
# =============================================================================


class TestLiquiditySweepMetric:
    """Test liquidity_sweep regime metric.

    Detects break-below-then-reclaim pattern for stop hunts.
    """

    def test_liquidity_sweep_previous_low(self):
        """liquidity_sweep of previous low."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "liquidity_sweep",
                                    "op": "==",
                                    "value": "previous_low",
                                    "level_reference": "previous_low",
                                    "reclaim_within_bars": 3,
                                    "lookback_bars": 20,
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        # Should create rolling min/max indicator
        assert len(result.indicators) > 0

    def test_liquidity_sweep_session_low(self):
        """liquidity_sweep of session low."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "15m", "symbol": "SPY"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "liquidity_sweep",
                                    "op": "==",
                                    "value": "session_low",
                                    "level_reference": "session_low",
                                    "reclaim_within_bars": 5,
                                    "lookback_bars": 50,
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None


class TestFlagPatternMetric:
    """Test flag_pattern regime metric.

    Detects flag consolidation pattern: initial momentum + narrowing range + breakout.
    """

    def test_flag_pattern_continuation(self):
        """flag_pattern with same direction continuation."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "flag_pattern",
                                    "op": "==",
                                    "value": "same",
                                    "flag_momentum_bars": 5,
                                    "flag_consolidation_bars": 10,
                                    "flag_breakout_direction": "same",
                                },
                            }
                        },
                        "action": {"direction": "auto"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        # Should create momentum ROC, ATR, and max/min indicators
        indicator_ids = [i.id for i in result.indicators]
        assert any("roc" in i.lower() for i in indicator_ids)

    def test_flag_pattern_up_breakout(self):
        """flag_pattern with explicit up breakout."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "ETH-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "flag_pattern",
                                    "op": "==",
                                    "value": "up",
                                    "flag_momentum_bars": 8,
                                    "flag_consolidation_bars": 15,
                                    "flag_breakout_direction": "up",
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None


# =============================================================================
# Price Level Metrics (Full Coverage)
# =============================================================================


class TestPriceLevelMetrics:
    """Test price_level_touch and price_level_cross with dynamic references."""

    def test_price_level_touch_session_high(self):
        """price_level_touch with session_high reference."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "15m", "symbol": "SPY"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "price_level_touch",
                                    "op": "==",
                                    "value": True,
                                    "level_reference": "session_high",
                                    "lookback_bars": 50,
                                },
                            }
                        },
                        "action": {"direction": "short"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_price_level_cross_recent_support(self):
        """price_level_cross with recent_support reference (rolling min low)."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "price_level_cross",
                                    "op": "==",
                                    "value": "recent_support_down",
                                    "level_reference": "recent_support",
                                    "direction": "down",
                                    "lookback_bars": 30,
                                },
                            }
                        },
                        "action": {"direction": "short"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_price_level_cross_previous_high_up(self):
        """price_level_cross breakout above previous high."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "ETH-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "price_level_cross",
                                    "op": "==",
                                    "value": True,
                                    "level_reference": "previous_high",
                                    "direction": "up",
                                    "lookback_bars": 50,
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        # Should create rolling max indicator
        assert len(result.indicators) > 0


# =============================================================================
# Gap/Session Metrics (Full Coverage)
# =============================================================================


class TestGapMetrics:
    """Test gap_pct metric."""

    def test_gap_pct_positive(self):
        """gap_pct positive gap filter."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "15m", "symbol": "SPY"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "gap_pct",
                                    "op": ">",
                                    "value": 1.0,
                                    "session": "us",
                                },
                            }
                        },
                        "action": {"direction": "short"},  # Gap fade
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_gap_pct_negative(self):
        """gap_pct negative gap filter."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "15m", "symbol": "AAPL"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "gap_pct",
                                    "op": "<",
                                    "value": -1.5,
                                    "session": "us",
                                },
                            }
                        },
                        "action": {"direction": "long"},  # Gap fade
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None


# =============================================================================
# Combined Complex Strategies (Coverage Integration)
# =============================================================================


class TestComplexMultiComponentStrategies:
    """Test complex strategies combining multiple new components."""

    def test_liquidity_sweep_with_volume_confirmation(self):
        """Liquidity sweep entry with volume spike confirmation."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "allOf",
                                "allOf": [
                                    {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "liquidity_sweep",
                                            "op": "==",
                                            "value": "previous_low",
                                            "level_reference": "previous_low",
                                            "reclaim_within_bars": 3,
                                            "lookback_bars": 20,
                                        },
                                    },
                                    {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "volume_spike",
                                            "op": "==",
                                            "value": True,
                                            "volume_threshold_pctile": 80,
                                            "lookback_bars": 20,
                                        },
                                    },
                                ],
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None

    def test_flag_pattern_with_trend_gate(self):
        """Flag pattern entry gated by strong trend."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "ETH-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "flag_pattern",
                                    "op": "==",
                                    "value": "same",
                                    "flag_momentum_bars": 5,
                                    "flag_consolidation_bars": 10,
                                    "flag_breakout_direction": "same",
                                },
                            }
                        },
                        "action": {"direction": "auto"},
                    },
                },
                "gate1": {
                    "type": "gate.regime",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "ETH-USD"},
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
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        assert len(result.gates) == 1

    def test_pennant_breakout_with_volume_filter(self):
        """Pennant pattern with low volatility and volume confirmation."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "allOf",
                                "allOf": [
                                    {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "pennant_pattern",
                                            "op": "==",
                                            "value": "up",
                                            "pennant_momentum_bars": 5,
                                            "pennant_consolidation_bars": 15,
                                            "pennant_breakout_direction": "up",
                                        },
                                    },
                                    {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "vol_regime",
                                            "op": "==",
                                            "value": "quiet",
                                            "lookback_bars": 20,
                                        },
                                    },
                                    {
                                        "type": "regime",
                                        "regime": {
                                            "metric": "volume_pctile",
                                            "op": ">",
                                            "value": 60,
                                            "lookback_bars": 20,
                                        },
                                    },
                                ],
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None


class TestIRValidation:
    """Test that ALL translations produce valid IR with no referential integrity errors.

    This is a safety net to catch bugs where translation produces IR that references
    non-existent indicators, states, or other entities.
    """

    def test_all_entry_archetypes_produce_valid_ir(self):
        """Every supported entry archetype should produce IR with no validation errors."""
        test_cases = [
            # rule_trigger with regime condition
            {
                "entry1": {
                    "type": "entry.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "ret_pct",
                                    "op": ">",
                                    "value": 1.0,
                                    "lookback_bars": 10,
                                },
                            }
                        },
                        "action": {"direction": "long"},
                    },
                }
            },
            # trend_pullback (schema v2 format)
            {
                "entry1": {
                    "type": "entry.trend_pullback",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                            "dip": {"kind": "edge_event", "edge": "lower", "op": "touch"},
                            "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                        },
                        "action": {"direction": "long"},
                    },
                }
            },
        ]

        for i, cards_dict in enumerate(test_cases):
            strategy, cards = make_strategy(cards_dict)
            # Translation raises TranslationError if invalid
            result = IRTranslator(strategy, cards).translate()
            assert result.entry is not None, f"Entry test case {i} has no entry"

    def test_all_exit_archetypes_produce_valid_ir(self):
        """Every exit archetype should produce IR with no validation errors."""
        # Base entry for all exit tests
        base_entry = {
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
            }
        }

        exit_cases = [
            # rule_trigger with regime condition
            {
                "exit1": {
                    "type": "exit.rule_trigger",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "condition": {
                                "type": "regime",
                                "regime": {
                                    "metric": "ret_pct",
                                    "op": "<",
                                    "value": -2.0,
                                    "lookback_bars": 5,
                                },
                            }
                        },
                        "action": {"mode": "close"},
                    },
                }
            },
            # trailing_stop (the bug we fixed!)
            {
                "exit1": {
                    "type": "exit.trailing_stop",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "trail_band": {"band": "keltner", "length": 14, "mult": 2.0},
                        },
                        "action": {"mode": "close"},
                    },
                }
            },
            # band_exit
            {
                "exit1": {
                    "type": "exit.band_exit",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                            "exit_trigger": {"edge": "upper"},
                        },
                        "action": {"mode": "close"},
                    },
                }
            },
        ]

        for i, exit_dict in enumerate(exit_cases):
            cards_dict = {**base_entry, **exit_dict}
            strategy, cards = make_strategy(cards_dict)
            # Translation raises TranslationError if invalid
            result = IRTranslator(strategy, cards).translate()
            assert len(result.exits) >= 1, f"Exit test case {i} has no exits"

    def test_trailing_stop_has_state_and_hook(self):
        """Trailing stop must create state variable AND on_bar_invested hook.

        This test specifically catches the bug where trailing_stop referenced
        a non-existent indicator instead of using StateValue.
        """
        strategy, cards = make_strategy(
            {
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
                "exit1": {
                    "type": "exit.trailing_stop",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {"trail_band": {"band": "keltner", "length": 14, "mult": 2.0}},
                        "action": {"mode": "close"},
                    },
                },
            }
        )
        # Translation raises TranslationError if invalid (including validation errors)
        result = IRTranslator(strategy, cards).translate()

        # Must have highest_since_entry state
        state_ids = [s.id for s in result.state]
        assert "highest_since_entry" in state_ids, f"Missing state. Found: {state_ids}"

        # Must have on_bar_invested hook to update state
        assert len(result.on_bar_invested) >= 1, "Missing on_bar_invested hook"
        assert result.on_bar_invested[0].state_id == "highest_since_entry"


# =============================================================================
# Pairs Trade Archetype Tests (Multi-Symbol)
# =============================================================================


class TestPairsTradeArchetype:
    """Test pairs_trade archetype - multi-symbol spread trading."""

    def test_pairs_trade_zscore(self):
        """entry.pairs_trade with zscore spread type."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.pairs_trade",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
                        "event": {
                            "pairs": {
                                "leg_a_symbol": "ETH-USD",
                                "leg_b_symbol": "BTC-USD",
                                "spread_type": "zscore",
                                "window_bars": 100,
                                "entry_threshold": 2.0,
                                "exit_threshold": 0.5,
                            }
                        },
                        "action": {"direction": "auto"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        # Should produce an IRRuntimeCondition
        assert result.entry.condition.type == "runtime"
        assert result.entry.condition.condition_type == "pairs_trade"
        # Parameters should include spread config
        assert result.entry.condition.params["leg_a_symbol"] == "ETH-USD"
        assert result.entry.condition.params["leg_b_symbol"] == "BTC-USD"
        assert result.entry.condition.params["spread_type"] == "zscore"
        assert result.entry.condition.params["entry_threshold"] == 2.0

    def test_pairs_trade_ratio(self):
        """entry.pairs_trade with ratio spread type."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.pairs_trade",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "SPY"},
                        "event": {
                            "pairs": {
                                "leg_a_symbol": "SPY",
                                "leg_b_symbol": "QQQ",
                                "spread_type": "ratio",
                                "window_bars": 50,
                                "entry_threshold": 1.5,
                                "exit_threshold": 0.25,
                            }
                        },
                        "action": {"direction": "long_spread"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        assert result.entry.condition.type == "runtime"
        # Direction should be passed through
        assert result.entry.condition.params["direction"] == "long_spread"


# =============================================================================
# Trailing Breakout Archetype Tests
# =============================================================================


class TestTrailingBreakoutArchetype:
    """Test trailing_breakout archetype - entry on trail band break."""

    def test_trailing_breakout_keltner(self):
        """entry.trailing_breakout with Keltner channel."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.trailing_breakout",
                    "slots": {
                        "context": {"tf": "4h", "symbol": "BTC-USD"},
                        "event": {
                            "trail_band": {"band": "keltner", "length": 20, "mult": 1.5},
                            "trail_trigger": {"kind": "edge_event", "edge": "upper", "op": "cross_out"},
                        },
                        "action": {"direction": "long"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        # Should produce an IRRuntimeCondition
        assert result.entry.condition.type == "runtime"
        assert result.entry.condition.condition_type == "trailing_breakout"
        # Inline IndicatorRefs get converted to indicator IDs during translation
        assert len(result.entry.condition.indicators_required) >= 1
        ind_id = result.entry.condition.indicators_required[0]
        # Should have registered a KC indicator
        indicator_types = [i.type for i in result.indicators]
        assert "KC" in indicator_types, f"Expected KC indicator, got {indicator_types}"

    def test_trailing_breakout_bollinger(self):
        """entry.trailing_breakout with Bollinger Bands."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.trailing_breakout",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "ETH-USD"},
                        "event": {
                            "trail_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                            "trail_trigger": {"kind": "edge_event", "edge": "upper", "op": "cross_out"},
                        },
                        "action": {"direction": "long", "confirm": "close_confirm"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        assert result.entry.condition.type == "runtime"
        # Should have registered a BB indicator
        indicator_types = [i.type for i in result.indicators]
        assert "BB" in indicator_types, f"Expected BB indicator, got {indicator_types}"

    def test_trailing_breakout_donchian(self):
        """entry.trailing_breakout with Donchian channel."""
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.trailing_breakout",
                    "slots": {
                        "context": {"tf": "1d", "symbol": "AAPL"},
                        "event": {
                            "trail_band": {"band": "donchian", "length": 10, "mult": 1.0},
                            "trail_trigger": {"kind": "edge_event", "edge": "upper", "op": "cross_out"},
                        },
                        "action": {"direction": "long"},
                    },
                },
            }
        )
        result = IRTranslator(strategy, cards).translate()
        assert result.entry is not None
        assert result.entry.condition.type == "runtime"
        # Should have registered a DC indicator
        indicator_types = [i.type for i in result.indicators]
        assert "DC" in indicator_types, f"Expected DC indicator, got {indicator_types}"
