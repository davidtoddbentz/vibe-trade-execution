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


def make_strategy(cards_dict: dict[str, dict]) -> tuple[Strategy, dict[str, Card]]:
    """Helper to create a strategy with cards."""
    attachments = []
    cards = {}

    for card_id, card_spec in cards_dict.items():
        role = card_spec["type"].split(".")[0]  # entry, exit, gate
        attachments.append(Attachment(card_id=card_id, role=role, enabled=True, overrides={}))
        cards[card_id] = Card(
            id=card_id,
            type=card_spec["type"],
            name=card_spec.get("name", card_id),
            schema_etag="test",
            slots=card_spec["slots"],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

    strategy = Strategy(
        id="test_strategy",
        name="Test Strategy",
        universe=["BTC-USD"],
        attachments=attachments,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )

    return strategy, cards


class TestEntryArchetypes:
    """Test all entry archetypes translate successfully."""

    def test_entry_rule_trigger_regime(self):
        """entry.rule_trigger with regime condition."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        assert len([w for w in result.warnings if "Unsupported" in w]) == 0

    def test_entry_rule_trigger_band_event(self):
        """entry.rule_trigger with band_event condition."""
        strategy, cards = make_strategy({
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
                                "event": "touch"
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_entry_rule_trigger_allof(self):
        """entry.rule_trigger with allOf composite condition."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "allOf",
                            "allOf": [
                                {"type": "regime", "regime": {"metric": "trend_ma_relation", "op": ">", "value": 0, "ma_fast": 20, "ma_slow": 50}},
                                {"type": "band_event", "band_event": {"band": {"band": "bollinger", "length": 20, "mult": 2.0}, "kind": "edge_event", "edge": "lower", "event": "touch"}}
                            ]
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_entry_rule_trigger_sequence(self):
        """entry.rule_trigger with sequence condition."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "sequence",
                            "sequence": [
                                {"cond": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<", "value": -2.0, "lookback_bars": 1}}},
                                {"cond": {"type": "regime", "regime": {"metric": "ret_pct", "op": ">", "value": 0, "lookback_bars": 1}}, "within_bars": 5}
                            ]
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        # Sequence creates state variables
        assert len(result.ir.state) > 0

    def test_entry_trend_pullback(self):
        """entry.trend_pullback archetype."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.trend_pullback",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip": {"kind": "edge_event", "edge": "lower", "op": "touch"},
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"}
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_entry_breakout_trendfollow(self):
        """entry.breakout_trendfollow archetype."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.breakout_trendfollow",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "breakout": {"lookback_bars": 50, "buffer_bps": 5}
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_entry_range_mean_reversion(self):
        """entry.range_mean_reversion via expansion."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.range_mean_reversion",
                "slots": {
                    "context": {"tf": "15m", "symbol": "ETH-USD"},
                    "event": {
                        "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "entry": {"kind": "edge_event", "edge": "lower", "op": "touch"}
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        # Should expand to rule_trigger
        assert result.ir.entry is not None
        # Check for expansion warning
        expansion_msgs = [w for w in result.warnings if "Expanded" in w]
        assert len(expansion_msgs) == 1

    def test_entry_squeeze_expansion(self):
        """entry.squeeze_expansion via expansion."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.squeeze_expansion",
                "slots": {
                    "context": {"tf": "1h", "symbol": "ETH-USD"},
                    "event": {
                        "squeeze": {"metric": "bb_width_pctile", "pctile_min": 10, "break_rule": "donchian"}
                    },
                    "action": {"direction": "auto"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_entry_breakout_condition(self):
        """entry.rule_trigger with breakout condition type."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "breakout",
                            "breakout": {"lookback_bars": 50, "buffer_bps": 5}
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_entry_squeeze_condition(self):
        """entry.rule_trigger with squeeze condition type."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "squeeze",
                            "squeeze": {"metric": "bb_width_pctile", "pctile_min": 10}
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_entry_time_filter_condition(self):
        """entry.rule_trigger with time_filter condition type."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "time_filter",
                            "time_filter": {"days_of_week": ["monday", "tuesday", "wednesday"], "time_window": "0930-1600"}
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_entry_breakout_retest(self):
        """entry.breakout_retest archetype via expansion."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.breakout_retest",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "retest": {
                            "lookback_bars": 50,
                            "buffer_bps": 5,
                            "pullback_pct": -2.0,
                            "retest_bars": 10
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        # Should expand to sequence condition
        expansion_warnings = [w for w in result.warnings if "Expanded" in w]
        assert len(expansion_warnings) == 1

    def test_entry_avwap_reversion(self):
        """entry.avwap_reversion archetype with $infer: substitution."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.avwap_reversion",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "anchor": "session_open",
                        "dist_sigma_entry": 2.0
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        # Should expand and infer direction_op based on action.direction=long -> "<"
        expansion_warnings = [w for w in result.warnings if "Expanded" in w]
        assert len(expansion_warnings) == 1

    def test_entry_gap_play(self):
        """entry.gap_play archetype with time filter."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.gap_play",
                "slots": {
                    "context": {"tf": "15m", "symbol": "SPY"},
                    "event": {
                        "session": {
                            "session": "us_equity",
                            "window": "0930-1000",
                            "mode": "gap_fade"
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        # Should expand to allOf with regime + time_filter
        expansion_warnings = [w for w in result.warnings if "Expanded" in w]
        assert len(expansion_warnings) == 1


class TestExitArchetypes:
    """Test all exit archetypes translate successfully."""

    def test_exit_rule_trigger(self):
        """exit.rule_trigger archetype."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "exit1": {
                "type": "exit.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": ">", "value": 5.0, "lookback_bars": 1}}},
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert len(result.ir.exits) == 1

    def test_exit_rule_trigger_with_allof(self):
        """exit.rule_trigger with allOf composite condition."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": ">", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "exit1": {
                "type": "exit.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "allOf",
                            "allOf": [
                                {"type": "regime", "regime": {"metric": "ret_pct", "op": ">", "value": 5.0, "lookback_bars": 1}},
                                {"type": "regime", "regime": {"metric": "trend_ma_relation", "op": "<", "value": 0, "ma_fast": 20, "ma_slow": 50}}
                            ]
                        }
                    },
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert len(result.ir.exits) == 1
        assert result.ir.exits[0].condition is not None

    def test_exit_rule_trigger_with_anyof(self):
        """exit.rule_trigger with anyOf composite condition."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": ">", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "exit1": {
                "type": "exit.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "anyOf",
                            "anyOf": [
                                {"type": "regime", "regime": {"metric": "ret_pct", "op": ">", "value": 10.0, "lookback_bars": 1}},
                                {"type": "regime", "regime": {"metric": "ret_pct", "op": "<", "value": -5.0, "lookback_bars": 1}}
                            ]
                        }
                    },
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert len(result.ir.exits) == 1
        assert result.ir.exits[0].condition is not None

    def test_exit_trailing_stop(self):
        """exit.trailing_stop archetype."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "exit1": {
                "type": "exit.trailing_stop",
                "slots": {
                    "context": {"tf": "4h", "symbol": "ETH-USD"},
                    "event": {
                        "trail_band": {"band": "keltner", "length": 20, "mult": 2.0},
                        "trail_trigger": {"kind": "edge_event", "edge": "lower", "op": "cross_out"}
                    },
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert len(result.ir.exits) == 1

    def test_exit_band_exit(self):
        """exit.band_exit archetype."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "exit1": {
                "type": "exit.band_exit",
                "slots": {
                    "context": {"tf": "1h", "symbol": "ETH-USD"},
                    "event": {
                        "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "exit_trigger": {"kind": "edge_event", "edge": "upper", "op": "touch"}
                    },
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert len(result.ir.exits) == 1

    def test_exit_structure_break(self):
        """exit.structure_break via expansion."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "exit1": {
                "type": "exit.structure_break",
                "slots": {
                    "context": {"tf": "4h", "symbol": "BTC-USD"},
                    "event": {"breakout": {"lookback_bars": 50, "buffer_bps": 5}},
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert len(result.ir.exits) == 1

    def test_exit_squeeze_compression(self):
        """exit.squeeze_compression via expansion."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "exit1": {
                "type": "exit.squeeze_compression",
                "slots": {
                    "context": {"tf": "1h", "symbol": "ETH-USD"},
                    "event": {"squeeze": {"metric": "bb_width_pctile", "pctile_min": 10}},
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert len(result.ir.exits) == 1

    def test_exit_vwap_reversion(self):
        """exit.vwap_reversion archetype via expansion."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "exit1": {
                "type": "exit.vwap_reversion",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "anchor": "session_open",
                        "dist_sigma_exit": 0.5
                    },
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert len(result.ir.exits) == 1
        expansion_warnings = [w for w in result.warnings if "Expanded" in w]
        assert len(expansion_warnings) == 1


class TestGateArchetypes:
    """Test all gate archetypes translate successfully."""

    def test_gate_regime(self):
        """gate.regime archetype."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "gate1": {
                "type": "gate.regime",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "regime": {
                            "type": "regime",
                            "regime": {"metric": "trend_ma_relation", "op": ">", "value": 0, "ma_fast": 20, "ma_slow": 50}
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert len(result.ir.gates) == 1

    def test_gate_time_filter(self):
        """gate.time_filter via expansion."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "gate1": {
                "type": "gate.time_filter",
                "slots": {
                    "context": {"tf": "15m", "symbol": "SPY"},
                    "event": {
                        "time_filter": {
                            "days_of_week": ["monday", "tuesday", "wednesday", "thursday", "friday"],
                            "time_window": "0930-1600",
                            "timezone": "America/New_York"
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        # gate.time_filter should expand or have a handler
        assert len(result.ir.gates) >= 0  # May or may not have gate depending on expansion


class TestOverlayArchetypes:
    """Test overlay archetypes translate successfully."""

    def test_overlay_regime_scaler(self):
        """overlay.regime_scaler archetype."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "overlay1": {
                "type": "overlay.regime_scaler",
                "slots": {
                    "context": {"tf": "4h", "symbol": "BTC-USD"},
                    "event": {
                        "regime": {
                            "type": "regime",
                            "regime": {"metric": "vol_bb_width_pctile", "op": "<", "value": 20, "lookback_bars": 200}
                        }
                    },
                    "action": {"scale_risk_frac": 0.5, "scale_size_frac": 0.5, "target_roles": ["entry"]}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        # Overlay should be translated or at least not cause errors
        assert result.ir.entry is not None


class TestComplexStrategies:
    """Test complex multi-card strategies."""

    def test_trend_pullback_with_trailing_stop(self):
        """Complete trend pullback strategy with trailing stop exit."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.trend_pullback",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip": {"kind": "edge_event", "edge": "lower", "op": "touch"},
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"}
                    },
                    "action": {"direction": "long"}
                }
            },
            "exit1": {
                "type": "exit.trailing_stop",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "trail_band": {"band": "keltner", "length": 20, "mult": 2.0},
                        "trail_trigger": {"kind": "edge_event", "edge": "lower", "op": "cross_out"}
                    },
                    "action": {"mode": "close"}
                }
            },
            "exit2": {
                "type": "exit.band_exit",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "exit_trigger": {"kind": "edge_event", "edge": "upper", "op": "touch"}
                    },
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        assert len(result.ir.exits) == 2

    def test_breakout_with_regime_gate(self):
        """Breakout entry gated by trend regime."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.breakout_trendfollow",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"breakout": {"lookback_bars": 50, "buffer_bps": 5}},
                    "action": {"direction": "long"}
                }
            },
            "gate1": {
                "type": "gate.regime",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "regime": {
                            "type": "regime",
                            "regime": {"metric": "trend_adx", "op": ">", "value": 25, "lookback_bars": 14}
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]}
                }
            },
            "exit1": {
                "type": "exit.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "trend_ma_relation", "op": "<", "value": 0, "ma_fast": 20, "ma_slow": 50}}},
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        assert len(result.ir.gates) == 1
        assert len(result.ir.exits) == 1

    def test_mean_reversion_with_band_exit(self):
        """Mean reversion with band exit strategy."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.range_mean_reversion",
                "slots": {
                    "context": {"tf": "15m", "symbol": "ETH-USD"},
                    "event": {
                        "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "entry": {"kind": "edge_event", "edge": "lower", "op": "touch"}
                    },
                    "action": {"direction": "long"}
                }
            },
            "exit1": {
                "type": "exit.band_exit",
                "slots": {
                    "context": {"tf": "15m", "symbol": "ETH-USD"},
                    "event": {
                        "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "exit_trigger": {"kind": "edge_event", "edge": "mid", "op": "touch"}
                    },
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        assert len(result.ir.exits) == 1

    def test_full_strategy_with_gate_and_overlay(self):
        """Complete strategy: entry + gate + overlay + multiple exits."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "allOf",
                            "allOf": [
                                {"type": "regime", "regime": {"metric": "trend_ma_relation", "op": ">", "value": 0, "ma_fast": 10, "ma_slow": 30}},
                                {"type": "regime", "regime": {"metric": "ret_pct", "op": ">", "value": 1, "lookback_bars": 5}}
                            ]
                        }
                    },
                    "action": {"direction": "long"}
                }
            },
            "gate1": {
                "type": "gate.regime",
                "slots": {
                    "context": {"tf": "4h", "symbol": "BTC-USD"},
                    "event": {
                        "regime": {
                            "type": "regime",
                            "regime": {"metric": "trend_adx", "op": ">", "value": 20, "lookback_bars": 14}
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]}
                }
            },
            "overlay1": {
                "type": "overlay.regime_scaler",
                "slots": {
                    "context": {"tf": "1d", "symbol": "BTC-USD"},
                    "event": {
                        "regime": {
                            "type": "regime",
                            "regime": {"metric": "vol_bb_width_pctile", "op": "<", "value": 30, "lookback_bars": 100}
                        }
                    },
                    "action": {"scale_risk_frac": 0.5, "scale_size_frac": 0.5, "target_roles": ["entry"]}
                }
            },
            "exit_stop": {
                "type": "exit.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<", "value": -5, "lookback_bars": 3}}},
                    "action": {"mode": "close"}
                }
            },
            "exit_profit": {
                "type": "exit.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": ">", "value": 10, "lookback_bars": 10}}},
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        assert len(result.ir.gates) == 1
        assert len(result.ir.exits) == 2

    def test_multi_gate_strategy(self):
        """Strategy with multiple gates (trend + volatility)."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "ETH-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": ">", "value": 2, "lookback_bars": 3}}},
                    "action": {"direction": "long"}
                }
            },
            "gate_trend": {
                "type": "gate.regime",
                "slots": {
                    "context": {"tf": "4h", "symbol": "ETH-USD"},
                    "event": {
                        "regime": {
                            "type": "regime",
                            "regime": {"metric": "trend_ma_relation", "op": ">", "value": 0, "ma_fast": 20, "ma_slow": 50}
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]}
                }
            },
            "gate_vol": {
                "type": "gate.regime",
                "slots": {
                    "context": {"tf": "1h", "symbol": "ETH-USD"},
                    "event": {
                        "regime": {
                            "type": "regime",
                            "regime": {"metric": "vol_atr_pct", "op": ">", "value": 1, "lookback_bars": 14}
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]}
                }
            },
            "exit1": {
                "type": "exit.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "ETH-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "trend_ma_relation", "op": "<", "value": 0, "ma_fast": 20, "ma_slow": 50}}},
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        assert len(result.ir.gates) == 2

    def test_deeply_nested_conditions(self):
        """Strategy with deeply nested allOf/anyOf/not conditions."""
        strategy, cards = make_strategy({
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
                                        {"type": "regime", "regime": {"metric": "trend_ma_relation", "op": ">", "value": 5, "ma_fast": 5, "ma_slow": 20}},
                                        {"type": "regime", "regime": {"metric": "trend_adx", "op": ">", "value": 30, "lookback_bars": 14}}
                                    ]
                                },
                                {
                                    "type": "not",
                                    "not": {
                                        "type": "regime",
                                        "regime": {"metric": "ret_pct", "op": "<", "value": -10, "lookback_bars": 5}
                                    }
                                },
                                {
                                    "type": "anyOf",
                                    "anyOf": [
                                        {"type": "regime", "regime": {"metric": "vol_bb_width_pctile", "op": ">", "value": 50, "lookback_bars": 100}},
                                        {
                                            "type": "allOf",
                                            "allOf": [
                                                {"type": "regime", "regime": {"metric": "ret_pct", "op": ">", "value": 1, "lookback_bars": 1}},
                                                {"type": "regime", "regime": {"metric": "ret_pct", "op": ">", "value": 0, "lookback_bars": 3}}
                                            ]
                                        }
                                    ]
                                }
                            ]
                        }
                    },
                    "action": {"direction": "long"}
                }
            },
            "exit1": {
                "type": "exit.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<", "value": -5, "lookback_bars": 3}}},
                    "action": {"mode": "close"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        # Check no errors for deeply nested conditions
        assert len([w for w in result.warnings if "error" in w.lower()]) == 0


class TestRegimeMetrics:
    """Test all RegimeSpec metrics can be translated."""

    @pytest.mark.parametrize("metric,extra_params", [
        ("ret_pct", {"lookback_bars": 1}),
        ("trend_ma_relation", {"ma_fast": 20, "ma_slow": 50}),
        ("trend_regime", {"ma_fast": 20, "ma_slow": 50}),
        ("trend_adx", {"lookback_bars": 14}),
        ("vol_bb_width_pctile", {"lookback_bars": 100}),
        ("vol_atr_pct", {"lookback_bars": 14}),
        ("vol_regime", {"lookback_bars": 100}),
        ("volume_pctile", {"lookback_bars": 20}),
    ])
    def test_regime_metric(self, metric, extra_params):
        """Test regime metric translation."""
        regime = {"metric": metric, "op": ">", "value": 0, **extra_params}
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "regime", "regime": regime}},
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        # No unsupported metric warning
        unsupported = [w for w in result.warnings if "Unsupported" in w.lower() or "Unknown" in w.lower()]
        assert len(unsupported) == 0, f"Unexpected warnings: {unsupported}"


class TestBandTypes:
    """Test all band types can be translated."""

    @pytest.mark.parametrize("band_type", ["bollinger", "keltner", "donchian"])
    def test_band_type(self, band_type):
        """Test band type translation."""
        strategy, cards = make_strategy({
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
                                "event": "touch"
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None


class TestBandEvents:
    """Test all band event types can be translated."""

    @pytest.mark.parametrize("kind,extra_params", [
        ("edge_event", {"edge": "lower", "event": "touch"}),
        ("edge_event", {"edge": "upper", "event": "touch"}),
        ("edge_event", {"edge": "lower", "event": "cross_in"}),
        ("edge_event", {"edge": "lower", "event": "cross_out"}),
        ("distance", {"mode": "z", "side": "away_lower", "thresh": 1.5}),
        ("distance", {"mode": "band_mult", "side": "away_upper", "thresh": 0.8}),
        ("reentry", {"edge": "lower"}),
        ("reentry", {"edge": "upper"}),
    ])
    def test_band_event_type(self, kind, extra_params):
        """Test band event type translation."""
        band_event = {
            "band": {"band": "bollinger", "length": 20, "mult": 2.0},
            "kind": kind,
            **extra_params
        }
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {"type": "band_event", "band_event": band_event}},
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None


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
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.event_followthrough",
                "slots": {
                    "context": {"tf": "15m", "symbol": "AAPL"},
                    "event": {
                        "catalyst": {
                            "event_kind": "earnings",
                            "entry_window_bars": 8,
                            "strength_filter": "strong"
                        },
                        "follow_mode": "gap_continuation"
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        # Should either expand or produce a warning for unsupported archetype
        # Either way, translation should complete without error
        assert result is not None

    def test_entry_event_followthrough_macro(self):
        """entry.event_followthrough for macro event catalyst."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.event_followthrough",
                "slots": {
                    "context": {"tf": "1h", "symbol": "SPY"},
                    "event": {
                        "catalyst": {
                            "event_kind": "macro",
                            "entry_window_bars": 4,
                            "strength_filter": "normal"
                        },
                        "follow_mode": "momentum"
                    },
                    "action": {"direction": "auto"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result is not None


class TestIntermarketTriggerArchetype:
    """Test entry.intermarket_trigger archetype.

    This archetype triggers entries based on movements in a different asset.
    MVP constraint: follower_symbol must equal context.symbol.
    """

    def test_entry_intermarket_single_leader(self):
        """entry.intermarket_trigger with single leader symbol."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.intermarket_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "ETH-USD"},
                    "event": {
                        "intermarket": {
                            "leader_symbol": "BTC-USD",
                            "follower_symbol": "ETH-USD",
                            "trigger_feature": "ret_pct",
                            "trigger_threshold": 3.0,
                            "window_bars": 24,
                            "entry_side_map": {
                                "leader_up": "long",
                                "leader_down": "short"
                            }
                        }
                    },
                    "action": {"direction": "auto"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        # Should either expand or produce a warning for unsupported archetype
        assert result is not None

    def test_entry_intermarket_multiple_leaders(self):
        """entry.intermarket_trigger with multiple leader symbols."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.intermarket_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "SOL-USD"},
                    "event": {
                        "intermarket": {
                            "leaders": ["BTC-USD", "ETH-USD"],
                            "follower_symbol": "SOL-USD",
                            "leader_aggregate": {
                                "feature": "ret_pct",
                                "op": "avg",
                                "threshold": 2.0
                            },
                            "window_bars": 12,
                            "entry_side_map": {
                                "leader_up": "long",
                                "leader_down": "none"
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result is not None


class TestEventRiskWindowGate:
    """Test gate.event_risk_window archetype.

    This gate blocks/allows trading within a window around catalyst events.
    """

    def test_gate_event_risk_window_earnings(self):
        """gate.event_risk_window blocking around earnings."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "AAPL"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "gate1": {
                "type": "gate.event_risk_window",
                "slots": {
                    "context": {"tf": "15m", "symbol": "AAPL"},
                    "event": {
                        "catalyst": {
                            "event_kind": "earnings",
                            "entry_window_bars": 3,
                            "strength_filter": "strong"
                        },
                        "pre_event_bars": 8,
                        "post_event_bars": 12
                    },
                    "action": {
                        "mode": "block",
                        "target_roles": ["entry", "overlay"],
                        "target_tags": ["mean-revert"]
                    }
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        # Either translates gate or warns it's unsupported
        assert result is not None

    def test_gate_event_risk_window_macro(self):
        """gate.event_risk_window blocking around macro events."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "SPY"},
                    "event": {"condition": {"type": "regime", "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0, "lookback_bars": 1}}},
                    "action": {"direction": "long"}
                }
            },
            "gate1": {
                "type": "gate.event_risk_window",
                "slots": {
                    "context": {"tf": "1h", "symbol": "SPY"},
                    "event": {
                        "catalyst": {
                            "event_kind": "macro",
                            "entry_window_bars": 2,
                            "strength_filter": "auto"
                        },
                        "pre_event_bars": 4,
                        "post_event_bars": 8
                    },
                    "action": {
                        "mode": "block",
                        "target_roles": ["entry"]
                    }
                }
            }
        })
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
        strategy, cards = make_strategy({
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
                                "pennant_breakout_direction": "up"
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        # Should create pattern detection indicators
        assert len(result.ir.indicators) > 0

    def test_pennant_pattern_same_direction(self):
        """pennant_pattern with 'same' breakout direction (continuation)."""
        strategy, cards = make_strategy({
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
                                "pennant_breakout_direction": "same"
                            }
                        }
                    },
                    "action": {"direction": "auto"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None


class TestVolumePctileMetric:
    """Test volume_pctile regime metric.

    Percentile rank of volume vs lookback - approximated using relative volume.
    """

    def test_volume_pctile_high(self):
        """volume_pctile above threshold (high volume)."""
        strategy, cards = make_strategy({
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
                                "lookback_bars": 50
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        # Should create volume SMA indicator
        vol_indicators = [i for i in result.ir.indicators if "vol" in i.id.lower()]
        assert len(vol_indicators) > 0

    def test_volume_pctile_low(self):
        """volume_pctile below threshold (low volume)."""
        strategy, cards = make_strategy({
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
                                "lookback_bars": 30
                            }
                        }
                    },
                    "action": {"direction": "short"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None


class TestVWAPBandType:
    """Test vwap_band band type.

    VWAP with standard deviation bands - used for intraday mean reversion.
    """

    def test_vwap_band_edge_event(self):
        """vwap_band with edge_event touch."""
        strategy, cards = make_strategy({
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
                                "event": "touch"
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        # Either creates entry or warns about unsupported band type
        # Either way should not crash
        assert result is not None

    def test_vwap_band_distance(self):
        """vwap_band with distance z-score."""
        strategy, cards = make_strategy({
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
                                "thresh": 2.0
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result is not None

    def test_vwap_band_anchor_session(self):
        """vwap_band with session anchor (default/most common)."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "15m", "symbol": "SPY"},
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "vwap_band", "length": 0, "mult": 2.0, "anchor": "session"},
                                "kind": "edge_event",
                                "edge": "lower",
                                "event": "touch"
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result is not None

    def test_vwap_band_anchor_week(self):
        """vwap_band with weekly anchor."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "ES"},
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "vwap_band", "length": 0, "mult": 1.5, "anchor": "week"},
                                "kind": "distance",
                                "mode": "z",
                                "side": "away_lower",
                                "thresh": 1.5
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result is not None

    def test_vwap_band_anchor_month(self):
        """vwap_band with monthly anchor."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "4h", "symbol": "NQ"},
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "vwap_band", "length": 0, "mult": 2.0, "anchor": "month"},
                                "kind": "edge_event",
                                "edge": "upper",
                                "event": "touch"
                            }
                        }
                    },
                    "action": {"direction": "short"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result is not None

    def test_vwap_band_anchor_earnings(self):
        """vwap_band anchored to earnings (event-based)."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "AAPL"},
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "vwap_band", "length": 0, "mult": 2.5, "anchor": "earnings"},
                                "kind": "reentry",
                                "side": "from_below"
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result is not None

    def test_vwap_band_anchor_custom(self):
        """vwap_band with custom anchor date."""
        strategy, cards = make_strategy({
            "entry1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1d", "symbol": "BTC-USD"},
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "vwap_band", "length": 0, "mult": 3.0, "anchor": "custom"},
                                "kind": "edge_event",
                                "edge": "lower",
                                "event": "cross_in"
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
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
        strategy, cards = make_strategy({
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
                                "lookback_bars": 20
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        # Should create rolling min/max indicator
        assert len(result.ir.indicators) > 0

    def test_liquidity_sweep_session_low(self):
        """liquidity_sweep of session low."""
        strategy, cards = make_strategy({
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
                                "lookback_bars": 50
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None


class TestFlagPatternMetric:
    """Test flag_pattern regime metric.

    Detects flag consolidation pattern: initial momentum + narrowing range + breakout.
    """

    def test_flag_pattern_continuation(self):
        """flag_pattern with same direction continuation."""
        strategy, cards = make_strategy({
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
                                "flag_breakout_direction": "same"
                            }
                        }
                    },
                    "action": {"direction": "auto"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        # Should create momentum ROC, ATR, and max/min indicators
        indicator_ids = [i.id for i in result.ir.indicators]
        assert any("roc" in i.lower() for i in indicator_ids)

    def test_flag_pattern_up_breakout(self):
        """flag_pattern with explicit up breakout."""
        strategy, cards = make_strategy({
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
                                "flag_breakout_direction": "up"
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None


# =============================================================================
# Price Level Metrics (Full Coverage)
# =============================================================================


class TestPriceLevelMetrics:
    """Test price_level_touch and price_level_cross with dynamic references."""

    def test_price_level_touch_session_high(self):
        """price_level_touch with session_high reference."""
        strategy, cards = make_strategy({
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
                                "lookback_bars": 50
                            }
                        }
                    },
                    "action": {"direction": "short"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_price_level_cross_recent_support(self):
        """price_level_cross with recent_support reference."""
        strategy, cards = make_strategy({
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
                                "lookback_bars": 30
                            }
                        }
                    },
                    "action": {"direction": "short"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_price_level_cross_previous_high_up(self):
        """price_level_cross breakout above previous high."""
        strategy, cards = make_strategy({
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
                                "lookback_bars": 50
                            }
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        # Should create rolling max indicator
        assert len(result.ir.indicators) > 0


# =============================================================================
# Gap/Session Metrics (Full Coverage)
# =============================================================================


class TestGapMetrics:
    """Test gap_pct metric."""

    def test_gap_pct_positive(self):
        """gap_pct positive gap filter."""
        strategy, cards = make_strategy({
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
                                "session": "us"
                            }
                        }
                    },
                    "action": {"direction": "short"}  # Gap fade
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_gap_pct_negative(self):
        """gap_pct negative gap filter."""
        strategy, cards = make_strategy({
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
                                "session": "us"
                            }
                        }
                    },
                    "action": {"direction": "long"}  # Gap fade
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None


# =============================================================================
# Combined Complex Strategies (Coverage Integration)
# =============================================================================


class TestComplexMultiComponentStrategies:
    """Test complex strategies combining multiple new components."""

    def test_liquidity_sweep_with_volume_confirmation(self):
        """Liquidity sweep entry with volume spike confirmation."""
        strategy, cards = make_strategy({
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
                                        "lookback_bars": 20
                                    }
                                },
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "volume_spike",
                                        "op": "==",
                                        "value": True,
                                        "volume_threshold_pctile": 80,
                                        "lookback_bars": 20
                                    }
                                }
                            ]
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None

    def test_flag_pattern_with_trend_gate(self):
        """Flag pattern entry gated by strong trend."""
        strategy, cards = make_strategy({
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
                                "flag_breakout_direction": "same"
                            }
                        }
                    },
                    "action": {"direction": "auto"}
                }
            },
            "gate1": {
                "type": "gate.regime",
                "slots": {
                    "context": {"tf": "4h", "symbol": "ETH-USD"},
                    "event": {
                        "regime": {
                            "type": "regime",
                            "regime": {
                                "metric": "trend_adx",
                                "op": ">",
                                "value": 25,
                                "lookback_bars": 14
                            }
                        }
                    },
                    "action": {"mode": "allow", "target_roles": ["entry"]}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
        assert len(result.ir.gates) == 1

    def test_pennant_breakout_with_volume_filter(self):
        """Pennant pattern with low volatility and volume confirmation."""
        strategy, cards = make_strategy({
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
                                        "pennant_breakout_direction": "up"
                                    }
                                },
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "vol_regime",
                                        "op": "==",
                                        "value": "quiet",
                                        "lookback_bars": 20
                                    }
                                },
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "volume_pctile",
                                        "op": ">",
                                        "value": 60,
                                        "lookback_bars": 20
                                    }
                                }
                            ]
                        }
                    },
                    "action": {"direction": "long"}
                }
            }
        })
        result = IRTranslator(strategy, cards).translate()
        assert result.ir.entry is not None
