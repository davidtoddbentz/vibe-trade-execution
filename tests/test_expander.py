"""Tests for archetype expansion system."""

from src.translator.archetype_expander import ArchetypeExpander, expand_card


class TestArchetypeExpander:
    """Tests for ArchetypeExpander."""

    def setup_method(self):
        """Set up test fixtures."""
        self.expander = ArchetypeExpander()

    def test_primitive_not_expanded(self):
        """Primitive archetypes should not be expanded."""
        card = {
            "type_id": "entry.rule_trigger",
            "slots": {
                "context": {"tf": "1h", "symbol": "BTC-USD"},
                "event": {
                    "condition": {
                        "type": "regime",
                        "regime": {"metric": "ret_pct", "op": "<=", "value": 1.0},
                    }
                },
                "action": {"direction": "long"},
            },
        }

        expanded, provenance = self.expander.expand(card)

        # Should return original card unchanged
        assert expanded["type_id"] == "entry.rule_trigger"
        assert provenance is None

    def test_is_primitive_for_rule_trigger(self):
        """entry.rule_trigger should be recognized as primitive."""
        assert self.expander.is_primitive("entry.rule_trigger")
        assert self.expander.is_primitive("exit.rule_trigger")
        assert self.expander.is_primitive("gate.regime")

    def test_is_not_primitive_for_specialized(self):
        """Specialized archetypes should not be primitive."""
        assert not self.expander.is_primitive("entry.trend_pullback")
        assert not self.expander.is_primitive("entry.range_mean_reversion")
        assert not self.expander.is_primitive("exit.band_exit")

    def test_get_primitive_type(self):
        """get_primitive_type should return correct expansion target."""
        assert self.expander.get_primitive_type("entry.rule_trigger") == "entry.rule_trigger"
        assert self.expander.get_primitive_type("entry.trend_pullback") == "entry.rule_trigger"
        assert self.expander.get_primitive_type("exit.band_exit") == "exit.rule_trigger"

    def test_expand_trend_pullback(self):
        """entry.trend_pullback should expand to entry.rule_trigger."""
        card = {
            "type_id": "entry.trend_pullback",
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

        expanded, provenance = self.expander.expand(card)

        # Should expand to rule_trigger
        assert expanded["type_id"] == "entry.rule_trigger"
        assert provenance is not None
        assert provenance["source_archetype"] == "entry.trend_pullback"

        # Event should have a condition
        event = expanded["slots"]["event"]
        assert "condition" in event
        condition = event["condition"]

        # Should be an allOf with trend + band_event
        assert condition["type"] == "allOf"
        assert len(condition["allOf"]) == 2

    def test_expand_range_mean_reversion(self):
        """entry.range_mean_reversion should expand to entry.rule_trigger with band_event."""
        card = {
            "type_id": "entry.range_mean_reversion",
            "slots": {
                "context": {"tf": "15m", "symbol": "ETH-USD"},
                "event": {
                    "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                    "entry": {"kind": "distance", "mode": "z", "side": "away_upper", "thresh": 1.8},
                },
                "action": {"direction": "short"},
            },
        }

        expanded, provenance = self.expander.expand(card)

        # Should expand to rule_trigger
        assert expanded["type_id"] == "entry.rule_trigger"

        # Event should have a band_event condition
        event = expanded["slots"]["event"]
        condition = event["condition"]
        assert condition["type"] == "band_event"
        assert condition["band_event"]["band"]["band"] == "bollinger"
        assert condition["band_event"]["kind"] == "distance"

    def test_expand_breakout_trendfollow(self):
        """entry.breakout_trendfollow should expand to entry.rule_trigger with breakout."""
        card = {
            "type_id": "entry.breakout_trendfollow",
            "slots": {
                "context": {"tf": "1h", "symbol": "BTC-USD"},
                "event": {"breakout": {"lookback_bars": 50, "buffer_bps": 5}},
                "action": {"direction": "long"},
            },
        }

        expanded, provenance = self.expander.expand(card)

        # Should expand to rule_trigger
        assert expanded["type_id"] == "entry.rule_trigger"

        # Event should have a breakout condition
        event = expanded["slots"]["event"]
        condition = event["condition"]
        assert condition["type"] == "breakout"
        assert condition["breakout"]["lookback_bars"] == 50

    def test_expand_exit_band_exit(self):
        """exit.band_exit should expand to exit.rule_trigger."""
        card = {
            "type_id": "exit.band_exit",
            "slots": {
                "context": {"tf": "1h", "symbol": "ETH-USD"},
                "event": {
                    "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                    "exit_trigger": {"kind": "edge_event", "edge": "upper", "op": "touch"},
                },
                "action": {"mode": "close"},
            },
        }

        expanded, provenance = self.expander.expand(card)

        # Should expand to exit.rule_trigger
        assert expanded["type_id"] == "exit.rule_trigger"
        assert provenance["source_archetype"] == "exit.band_exit"

        # Event should have a band_event condition
        event = expanded["slots"]["event"]
        condition = event["condition"]
        assert condition["type"] == "band_event"

    def test_expand_squeeze_expansion(self):
        """entry.squeeze_expansion should expand to entry.rule_trigger."""
        card = {
            "type_id": "entry.squeeze_expansion",
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

        expanded, provenance = self.expander.expand(card)

        # Should expand to rule_trigger
        assert expanded["type_id"] == "entry.rule_trigger"

        # Event should have a squeeze condition
        event = expanded["slots"]["event"]
        condition = event["condition"]
        assert condition["type"] == "squeeze"

    def test_expand_preserves_context_and_action(self):
        """Expansion should preserve context, action, and risk slots."""
        card = {
            "type_id": "entry.range_mean_reversion",
            "slots": {
                "context": {"tf": "15m", "symbol": "ETH-USD"},
                "event": {
                    "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                    "entry": {"kind": "distance", "mode": "z", "side": "away_upper", "thresh": 1.8},
                },
                "action": {"direction": "short", "confirm": "close_confirm"},
                "risk": {"sl_atr": 2.0, "tp_pct": 3.0},
            },
        }

        expanded, _ = self.expander.expand(card)

        # Context should be preserved
        assert expanded["slots"]["context"]["tf"] == "15m"
        assert expanded["slots"]["context"]["symbol"] == "ETH-USD"

        # Action should be preserved
        assert expanded["slots"]["action"]["direction"] == "short"
        assert expanded["slots"]["action"]["confirm"] == "close_confirm"

        # Risk should be preserved
        assert expanded["slots"]["risk"]["sl_atr"] == 2.0
        assert expanded["slots"]["risk"]["tp_pct"] == 3.0

    def test_expand_unknown_archetype_is_primitive(self):
        """Unknown archetypes should be treated as primitives (no expansion)."""
        card = {
            "type_id": "entry.unknown_archetype",
            "slots": {"event": {"something": "here"}},
        }

        expanded, provenance = self.expander.expand(card)

        # Should return unchanged
        assert expanded["type_id"] == "entry.unknown_archetype"
        assert provenance is None


class TestExpandCardConvenience:
    """Tests for expand_card convenience function."""

    def test_expand_card_function(self):
        """expand_card should work as a convenience function."""
        card = {
            "type_id": "entry.rule_trigger",
            "slots": {"event": {"condition": {"type": "regime"}}},
        }

        expanded, provenance = expand_card(card)

        assert expanded["type_id"] == "entry.rule_trigger"
        assert provenance is None


class TestArchetypeExpansionIntegration:
    """Integration tests for expansion with translator."""

    def test_expand_then_translate_range_mean_reversion(self):
        """Expanded range_mean_reversion should translate successfully."""
        from vibe_trade_shared.models import Card, Strategy
        from vibe_trade_shared.models.strategy import Attachment

        from src.translator.ir_translator import IRTranslator

        # Create strategy with range_mean_reversion card
        strategy = Strategy(
            id="test_strategy",
            name="Test Range Mean Reversion",
            universe=["ETH-USD"],
            attachments=[Attachment(card_id="card1", role="entry", enabled=True, overrides={})],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

        cards = {
            "card1": Card(
                id="card1",
                type="entry.range_mean_reversion",
                name="Range MR Entry",
                schema_etag="test",
                slots={
                    "context": {"tf": "15m", "symbol": "ETH-USD"},
                    "event": {
                        "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "entry": {"kind": "edge_event", "edge": "lower", "op": "touch"},
                    },
                    "action": {"direction": "long"},
                },
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            )
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        # Should have an entry rule
        assert result.ir.entry is not None
        # Should have expansion warning (logging the expansion)
        expansion_warnings = [w for w in result.warnings if "Expanded" in w]
        assert len(expansion_warnings) == 1
        assert "entry.range_mean_reversion" in expansion_warnings[0]
