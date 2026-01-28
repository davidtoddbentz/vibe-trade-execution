"""Tests for ActionBuilder."""

import pytest

from src.translator.builders import ActionBuilder
from src.translator.ir_translator import TranslationError


class TestActionBuilder:
    """Tests for ActionBuilder.build_holdings_action."""

    def test_default_allocation(self):
        """No sizing defaults to 95% allocation."""
        action = {}
        result = ActionBuilder.build_holdings_action(action, direction="long")

        assert result.sizing_mode == "pct_equity"
        assert result.allocation == 0.95

    def test_pct_equity_sizing(self):
        """pct_equity sizing converts percentage correctly."""
        action = {}
        sizing = {"type": "pct_equity", "pct": 50}

        result = ActionBuilder.build_holdings_action(action, direction="long", sizing=sizing)

        assert result.sizing_mode == "pct_equity"
        assert result.allocation == 0.5

    def test_pct_equity_short_negative(self):
        """Short direction produces negative allocation."""
        action = {}
        sizing = {"type": "pct_equity", "pct": 80}

        result = ActionBuilder.build_holdings_action(action, direction="short", sizing=sizing)

        assert result.sizing_mode == "pct_equity"
        assert result.allocation == -0.8

    def test_fixed_usd_sizing(self):
        """fixed_usd sizing sets fixed_usd field."""
        action = {}
        sizing = {"type": "fixed_usd", "usd": 5000.0}

        result = ActionBuilder.build_holdings_action(action, direction="long", sizing=sizing)

        assert result.sizing_mode == "fixed_usd"
        assert result.fixed_usd == 5000.0
        assert result.allocation == 0.0

    def test_fixed_units_sizing(self):
        """fixed_units sizing sets fixed_units field."""
        action = {}
        sizing = {"type": "fixed_units", "units": 10.0}

        result = ActionBuilder.build_holdings_action(action, direction="long", sizing=sizing)

        assert result.sizing_mode == "fixed_units"
        assert result.fixed_units == 10.0
        assert result.allocation == 0.0

    def test_position_policy_passed_through(self):
        """Position policy from action dict is preserved."""
        action = {
            "position_policy": {
                "mode": "accumulate",
                "max_positions": 3,
            }
        }

        result = ActionBuilder.build_holdings_action(action, direction="long")

        assert result.position_policy is not None
        assert result.position_policy.mode == "accumulate"
        assert result.position_policy.max_positions == 3

    def test_unknown_sizing_raises(self):
        """Unknown sizing type raises TranslationError."""
        action = {}
        sizing = {"type": "unknown_type"}

        with pytest.raises(TranslationError, match="Unknown sizing type: unknown_type"):
            ActionBuilder.build_holdings_action(action, direction="long", sizing=sizing)
