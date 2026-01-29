"""Action builder utilities."""

from __future__ import annotations

from typing import Any

from src.translator.errors import TranslationError
from src.translator.ir import LiquidateAction, PositionPolicy, ReducePositionAction, SetHoldingsAction


class ActionBuilder:
    """Builds action objects from slot configurations."""

    DEFAULT_ALLOCATION = 0.95  # 95%

    @staticmethod
    def build_holdings_action(
        action: dict[str, Any],
        direction: str,
        sizing: dict[str, Any] | None = None,
    ) -> SetHoldingsAction:
        """Build SetHoldingsAction from action spec.

        Args:
            action: Action dict from archetype slots
            direction: "long" or "short"
            sizing: Optional sizing spec

        Returns:
            SetHoldingsAction configured for the sizing mode

        Raises:
            TranslationError: If sizing type is unknown
        """
        sign = 1.0 if direction == "long" else -1.0

        # Extract position policy if present
        position_policy = None
        if policy_dict := action.get("position_policy"):
            position_policy = PositionPolicy(**policy_dict)

        # Handle no sizing - default to percentage
        if sizing is None:
            return SetHoldingsAction(
                sizing_mode="pct_equity",
                allocation=sign * ActionBuilder.DEFAULT_ALLOCATION,
                position_policy=position_policy,
            )

        sizing_type = sizing.get("type")

        if sizing_type == "pct_equity":
            pct = sizing.get("pct", ActionBuilder.DEFAULT_ALLOCATION * 100)
            return SetHoldingsAction(
                sizing_mode="pct_equity",
                allocation=sign * (pct / 100.0),
                position_policy=position_policy,
            )

        if sizing_type == "fixed_usd":
            usd = sizing.get("usd", 1000.0)
            return SetHoldingsAction(
                sizing_mode="fixed_usd",
                allocation=0.0,  # Not used
                fixed_usd=sign * usd,
                position_policy=position_policy,
            )

        if sizing_type == "fixed_units":
            units = sizing.get("units", 1.0)
            return SetHoldingsAction(
                sizing_mode="fixed_units",
                allocation=0.0,  # Not used
                fixed_units=sign * units,
                position_policy=position_policy,
            )

        raise TranslationError(f"Unknown sizing type: {sizing_type}")

    @staticmethod
    def build_exit_action(action_spec: dict[str, Any]) -> LiquidateAction | ReducePositionAction:
        """Build exit action from ExitActionSpec slots.

        Args:
            action_spec: Dict with 'mode' and optional 'size_frac'

        Returns:
            LiquidateAction for full close, ReducePositionAction for partial
        """
        mode = action_spec.get("mode", "close")
        size_frac = action_spec.get("size_frac", 1.0)

        # Full close: use LiquidateAction (backward compatible)
        if mode == "close" and size_frac == 1.0:
            return LiquidateAction()

        # Partial close or reduce mode: use ReducePositionAction
        if mode in ("close", "reduce"):
            return ReducePositionAction(size_frac=size_frac)

        # Reverse mode: for now, treat as full liquidate (future: ReversePositionAction)
        if mode == "reverse":
            return LiquidateAction()

        return LiquidateAction()
