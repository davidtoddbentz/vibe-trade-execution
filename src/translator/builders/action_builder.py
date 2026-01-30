"""Action builder utilities."""

from __future__ import annotations

from typing import Any

from src.translator.errors import TranslationError
from src.translator.ir import (
    IRExpression,
    LiquidateAction,
    LiteralRef,
    PositionPolicy,
    PriceField,
    PriceRef,
    ReducePositionAction,
    SetHoldingsAction,
)


class ActionBuilder:
    """Builds action objects from slot configurations."""

    DEFAULT_ALLOCATION = 0.95  # 95%

    @staticmethod
    def build_execution_params(
        execution: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build execution parameters from ExecutionSpec slots.

        Converts archetype-level ExecutionSpec into IR-level fields:
        - order_type: passed through
        - limit_price -> LiteralRef
        - limit_offset_pct -> IRExpression(close * (1 + offset/100))
        - stop_price -> LiteralRef
        - time_in_force: passed through

        Returns:
            Dict with order_type, limit_price_ref, stop_price_ref, time_in_force
            to be merged into SetHoldingsAction kwargs.
        """
        if not execution:
            return {}

        order_type = execution.get("order_type", "market")
        if order_type == "market":
            return {}

        result: dict[str, Any] = {"order_type": order_type}

        # Resolve limit price
        limit_price = execution.get("limit_price")
        limit_offset_pct = execution.get("limit_offset_pct")

        if limit_price is not None:
            result["limit_price_ref"] = LiteralRef(value=limit_price)
        elif limit_offset_pct is not None:
            # close * (1 + offset/100)
            # Buy limit below market: offset is negative (e.g., -2.0 => close * 0.98)
            # Sell limit above market: offset is positive (e.g., 2.0 => close * 1.02)
            multiplier = 1.0 + (limit_offset_pct / 100.0)
            result["limit_price_ref"] = IRExpression(
                op="*",
                left=PriceRef(field=PriceField.CLOSE),
                right=LiteralRef(value=multiplier),
            )

        # Resolve stop price
        stop_price = execution.get("stop_price")
        if stop_price is not None:
            result["stop_price_ref"] = LiteralRef(value=stop_price)

        # Time in force
        tif = execution.get("time_in_force", "gtc")
        if tif in ("gtc", "day"):
            result["time_in_force"] = tif

        return result

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

        # Extract sizing constraints
        min_usd = sizing.get("min_usd") if sizing else None
        max_usd = sizing.get("max_usd") if sizing else None

        # Build execution params from ExecutionSpec
        execution = action.get("execution")
        exec_params = ActionBuilder.build_execution_params(execution)

        common = dict(
            position_policy=position_policy,
            min_usd=min_usd,
            max_usd=max_usd,
            **exec_params,
        )

        # Handle no sizing - default to percentage
        if sizing is None:
            return SetHoldingsAction(
                sizing_mode="pct_equity",
                allocation=sign * ActionBuilder.DEFAULT_ALLOCATION,
                **common,
            )

        sizing_type = sizing.get("type")

        if sizing_type == "pct_equity":
            pct = sizing.get("pct", ActionBuilder.DEFAULT_ALLOCATION * 100)
            return SetHoldingsAction(
                sizing_mode="pct_equity",
                allocation=sign * (pct / 100.0),
                **common,
            )

        if sizing_type == "fixed_usd":
            usd = sizing.get("usd", 1000.0)
            return SetHoldingsAction(
                sizing_mode="fixed_usd",
                allocation=0.0,  # Not used
                fixed_usd=sign * usd,
                **common,
            )

        if sizing_type == "fixed_units":
            units = sizing.get("units", 1.0)
            return SetHoldingsAction(
                sizing_mode="fixed_units",
                allocation=0.0,  # Not used
                fixed_units=sign * units,
                **common,
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

        raise TranslationError(f"Unknown exit mode: {mode}")
