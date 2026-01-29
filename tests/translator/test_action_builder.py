"""Tests for ActionBuilder."""

import pytest

from src.translator.builders import ActionBuilder
from src.translator.ir import LiquidateAction, ReducePositionAction
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


class TestBuildExitAction:
    """Tests for ActionBuilder.build_exit_action."""

    def test_close_full_returns_liquidate(self):
        """mode=close with size_frac=1.0 returns LiquidateAction."""
        result = ActionBuilder.build_exit_action({"mode": "close"})
        assert isinstance(result, LiquidateAction)

    def test_close_default_returns_liquidate(self):
        """Empty spec defaults to full close."""
        result = ActionBuilder.build_exit_action({})
        assert isinstance(result, LiquidateAction)

    def test_close_partial_returns_reduce(self):
        """mode=close with size_frac<1 returns ReducePositionAction."""
        result = ActionBuilder.build_exit_action({"mode": "close", "size_frac": 0.5})
        assert isinstance(result, ReducePositionAction)
        assert result.size_frac == 0.5

    def test_reduce_mode_returns_reduce(self):
        """mode=reduce returns ReducePositionAction."""
        result = ActionBuilder.build_exit_action({"mode": "reduce", "size_frac": 0.25})
        assert isinstance(result, ReducePositionAction)
        assert result.size_frac == 0.25

    def test_reduce_default_frac(self):
        """mode=reduce with no size_frac defaults to 1.0."""
        result = ActionBuilder.build_exit_action({"mode": "reduce"})
        assert isinstance(result, ReducePositionAction)
        assert result.size_frac == 1.0

    def test_unknown_mode_raises(self):
        """Unknown exit mode raises TranslationError."""
        with pytest.raises(TranslationError, match="Unknown exit mode"):
            ActionBuilder.build_exit_action({"mode": "reverse"})

    def test_invalid_mode_raises(self):
        """Completely invalid mode raises TranslationError."""
        with pytest.raises(TranslationError, match="Unknown exit mode"):
            ActionBuilder.build_exit_action({"mode": "explode"})
