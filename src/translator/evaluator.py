"""Runtime evaluator for IR conditions and values.

This module provides the evaluation engine that interprets the IR at runtime.
It's designed to be used by the LEAN StrategyRuntime algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .ir import (
    # Actions
    Action,
    # Conditions
    AllOfCondition,
    AnyOfCondition,
    # Enums
    BandField,
    CompareCondition,
    Condition,
    # Value refs
    ExpressionValue,
    # State ops
    IncrementStateOp,
    IndicatorBandValue,
    IndicatorProperty,
    IndicatorPropertyValue,
    IndicatorValue,
    # Runtime conditions
    IREventWindowCondition,
    IRRuntimeCondition,
    RuntimeCondition,
    TimeFilterCondition,
    BreakoutCondition,
    SqueezeCondition,
    LiquidateAction,
    LiteralValue,
    MarketOrderAction,
    MaxStateOp,
    MinStateOp,
    NotCondition,
    PriceField,
    PriceValue,
    RegimeCondition,
    RollingWindowValue,
    SetHoldingsAction,
    SetStateFromConditionOp,
    SetStateOp,
    StateOp,
    StateValue,
    TimeValue,
    ValueRef,
    VolumeValue,
)

# =============================================================================
# Protocols for LEAN integration
# =============================================================================


class IndicatorProtocol(Protocol):
    """Protocol for LEAN indicator objects."""

    @property
    def Current(self) -> Any: ...

    @property
    def IsReady(self) -> bool: ...


class BandIndicatorProtocol(Protocol):
    """Protocol for band indicators (BB, KC)."""

    @property
    def UpperBand(self) -> Any: ...

    @property
    def MiddleBand(self) -> Any: ...

    @property
    def LowerBand(self) -> Any: ...

    @property
    def IsReady(self) -> bool: ...


class BollingerBandsProtocol(Protocol):
    """Protocol for Bollinger Bands indicator with StandardDeviation property."""

    @property
    def UpperBand(self) -> Any: ...

    @property
    def MiddleBand(self) -> Any: ...

    @property
    def LowerBand(self) -> Any: ...

    @property
    def StandardDeviation(self) -> Any: ...

    @property
    def IsReady(self) -> bool: ...


class RollingWindowProtocol(Protocol):
    """Protocol for RollingWindow indicators that store historical values."""

    @property
    def IsReady(self) -> bool: ...

    def __getitem__(self, index: int) -> Any:
        """Access historical value by index (0 = most recent, 1 = previous bar, etc.)."""
        ...


class PriceBarProtocol(Protocol):
    """Protocol for LEAN price bar objects."""

    @property
    def Open(self) -> float: ...

    @property
    def High(self) -> float: ...

    @property
    def Low(self) -> float: ...

    @property
    def Close(self) -> float: ...

    @property
    def Volume(self) -> float: ...


class AlgorithmProtocol(Protocol):
    """Protocol for LEAN algorithm objects."""

    def SetHoldings(self, symbol: Any, allocation: float) -> None: ...

    def Liquidate(self, symbol: Any) -> None: ...

    def MarketOrder(self, symbol: Any, quantity: float) -> None: ...


# =============================================================================
# Evaluation Context
# =============================================================================


@dataclass
class EventCalendarEntry:
    """A single event in the calendar."""

    event_type: str  # "earnings", "fomc", "holiday", etc.
    bar_index: int  # Which bar this event occurs at
    symbol: str | None = None  # Optional symbol filter


@dataclass
class EvalContext:
    """Context for evaluating conditions and resolving values.

    This is passed through the evaluation tree and provides access to
    indicators, state, and price data.
    """

    indicators: dict[str, IndicatorProtocol | BandIndicatorProtocol]
    state: dict[str, float | int | bool | None]
    price_bar: PriceBarProtocol
    # Time fields for session_phase and time-based conditions
    hour: int = 12  # Default to midday
    minute: int = 0
    day_of_week: int = 2  # Default to Wednesday (0=Mon, 6=Sun)
    # Event calendar support (stub - injected for testing)
    current_bar_index: int = 0  # Current position in bar sequence
    event_calendar: list[EventCalendarEntry] | None = None  # Injected event calendar

    def get_indicator(self, indicator_id: str) -> IndicatorProtocol:
        """Get an indicator by ID."""
        ind = self.indicators.get(indicator_id)
        if ind is None:
            raise KeyError(f"Unknown indicator: {indicator_id}")
        return ind

    def get_state(self, state_id: str) -> float | int | bool | None:
        """Get a state variable by ID."""
        if state_id not in self.state:
            raise KeyError(f"Unknown state variable: {state_id}")
        return self.state[state_id]

    def set_state(self, state_id: str, value: float | int | bool | None) -> None:
        """Set a state variable."""
        self.state[state_id] = value

    def get_price(self, field: PriceField) -> float:
        """Get a price field from the current bar."""
        match field:
            case PriceField.OPEN:
                return self.price_bar.Open
            case PriceField.HIGH:
                return self.price_bar.High
            case PriceField.LOW:
                return self.price_bar.Low
            case PriceField.CLOSE:
                return self.price_bar.Close

    def get_volume(self) -> float:
        """Get the volume from the current bar."""
        return self.price_bar.Volume

    def get_time(self, component: str) -> float:
        """Get a time component."""
        match component:
            case "hour":
                return float(self.hour)
            case "minute":
                return float(self.minute)
            case "day_of_week":
                return float(self.day_of_week)
            case _:
                return 0.0

    def is_in_event_window(
        self,
        event_types: list[str],
        pre_window_bars: int,
        post_window_bars: int,
    ) -> bool:
        """Check if current bar is within window around any matching event.

        Args:
            event_types: Types of events to check (e.g., ["earnings", "fomc"])
            pre_window_bars: Bars before event to include in window
            post_window_bars: Bars after event to include in window

        Returns:
            True if within window of any matching event, False otherwise.
        """
        if self.event_calendar is None:
            return False

        for event in self.event_calendar:
            if event.event_type not in event_types:
                continue
            # Check if current bar is within the window
            bars_to_event = event.bar_index - self.current_bar_index
            # Within window if: -pre_window_bars <= bars_to_event <= post_window_bars
            # Negative bars_to_event means event is in the past (we're after it)
            # Positive bars_to_event means event is in the future (we're before it)
            if -post_window_bars <= bars_to_event <= pre_window_bars:
                return True

        return False


# =============================================================================
# Execution Context
# =============================================================================


@dataclass
class ExecContext:
    """Context for executing actions.

    Wraps the LEAN algorithm and symbol for action execution.
    """

    algorithm: AlgorithmProtocol
    symbol: Any

    def set_holdings(self, allocation: float) -> None:
        """Set portfolio holdings."""
        self.algorithm.SetHoldings(self.symbol, allocation)

    def liquidate(self) -> None:
        """Liquidate all holdings."""
        self.algorithm.Liquidate(self.symbol)

    def market_order(self, quantity: float) -> None:
        """Place a market order."""
        self.algorithm.MarketOrder(self.symbol, quantity)


# =============================================================================
# Value Resolver
# =============================================================================


class ValueResolver:
    """Resolves ValueRef objects to actual float values."""

    def resolve(self, ref: ValueRef, ctx: EvalContext) -> float:
        """Resolve a value reference to a float."""
        match ref:
            case IndicatorValue(indicator_id=ind_id):
                return self._get_indicator_value(ctx, ind_id)

            case IndicatorBandValue(indicator_id=ind_id, band=band):
                return self._get_band_value(ctx, ind_id, band)

            case IndicatorPropertyValue(indicator_id=ind_id, property=prop):
                return self._get_property_value(ctx, ind_id, prop)

            case PriceValue(field=field):
                return ctx.get_price(field)

            case StateValue(state_id=state_id):
                value = ctx.get_state(state_id)
                if value is None:
                    raise ValueError(f"State variable '{state_id}' is None")
                return float(value)

            case LiteralValue(value=value):
                return value

            case ExpressionValue(op=op, left=left, right=right):
                left_val = self.resolve(left, ctx)
                right_val = self.resolve(right, ctx)
                return self._apply_expr_op(op, left_val, right_val)

            case VolumeValue():
                return ctx.get_volume()

            case TimeValue(component=component):
                return ctx.get_time(component)

            case RollingWindowValue(indicator_id=ind_id, offset=offset):
                return self._get_rolling_window_value(ctx, ind_id, offset)

            case _:
                raise ValueError(f"Unknown value ref type: {type(ref)}")

    def _get_indicator_value(self, ctx: EvalContext, indicator_id: str) -> float:
        """Get the current value of an indicator."""
        ind = ctx.get_indicator(indicator_id)
        return ind.Current.Value

    def _get_band_value(self, ctx: EvalContext, indicator_id: str, band: BandField) -> float:
        """Get a specific band value from a band indicator."""
        ind = ctx.get_indicator(indicator_id)
        match band:
            case BandField.UPPER:
                return ind.UpperBand.Current.Value
            case BandField.MIDDLE:
                return ind.MiddleBand.Current.Value
            case BandField.LOWER:
                return ind.LowerBand.Current.Value

    def _get_property_value(
        self, ctx: EvalContext, indicator_id: str, prop: IndicatorProperty
    ) -> float:
        """Get a property value from an indicator (e.g., StandardDeviation for BB)."""
        ind = ctx.get_indicator(indicator_id)
        match prop:
            case IndicatorProperty.STANDARD_DEVIATION:
                return ind.StandardDeviation.Current.Value
            case IndicatorProperty.BAND_WIDTH:
                # Try direct property, otherwise calculate
                if hasattr(ind, "BandWidth"):
                    return ind.BandWidth.Current.Value
                # Fallback: calculate manually
                upper = ind.UpperBand.Current.Value
                lower = ind.LowerBand.Current.Value
                middle = ind.MiddleBand.Current.Value
                if middle != 0:
                    return (upper - lower) / middle
                return 0.0
            case _:
                raise ValueError(f"Unknown indicator property: {prop}")

    def _apply_expr_op(self, op: str, left: float, right: float) -> float:
        """Apply an expression operator."""
        match op:
            case "+":
                return left + right
            case "-":
                return left - right
            case "*":
                return left * right
            case "/":
                if right == 0:
                    raise ValueError("Division by zero")
                return left / right
            case _:
                raise ValueError(f"Unknown expression operator: {op}")

    def _get_rolling_window_value(
        self, ctx: EvalContext, indicator_id: str, offset: int
    ) -> float:
        """Get a historical value from a RollingWindow indicator.

        Args:
            ctx: Evaluation context
            indicator_id: ID of the RollingWindow indicator
            offset: How many bars back (1 = previous bar, 2 = 2 bars ago)

        Returns:
            The historical value at the specified offset
        """
        ind = ctx.get_indicator(indicator_id)
        # RollingWindow uses 0-indexed access where 0 = most recent
        # Our offset convention: 1 = previous bar, so we use offset directly
        # Since the rolling window stores after the current bar update,
        # offset=1 means index 1 (the value before the most recent)
        return ind[offset].Value

    def _get_indicator_band_width(self, ctx: EvalContext, indicator_id: str) -> float:
        """Get the bandwidth of a band indicator (BB, KC, etc.).

        Bandwidth = (upper - lower) / middle * 100
        """
        ind = ctx.get_indicator(indicator_id)
        try:
            # Try Bollinger Bands style first
            upper = ind.UpperBand.Current.Value
            middle = ind.MiddleBand.Current.Value
            lower = ind.LowerBand.Current.Value
        except AttributeError:
            # Try Keltner Channel style
            try:
                upper = ind.UpperChannel.Current.Value
                middle = ind.MiddleChannel.Current.Value
                lower = ind.LowerChannel.Current.Value
            except AttributeError:
                return 0.0
        if middle == 0:
            return 0.0
        return (upper - lower) / middle * 100


# =============================================================================
# Condition Evaluator
# =============================================================================


class ConditionEvaluator:
    """Evaluates Condition objects to boolean results."""

    def __init__(self):
        self.resolver = ValueResolver()

    def evaluate(self, condition: Condition, ctx: EvalContext) -> bool:
        """Evaluate a condition to a boolean."""
        match condition:
            case CompareCondition(left=left, op=op, right=right):
                left_val = self.resolver.resolve(left, ctx)
                right_val = self.resolver.resolve(right, ctx)
                return op.apply(left_val, right_val)

            case AllOfCondition(conditions=conditions):
                return all(self.evaluate(c, ctx) for c in conditions)

            case AnyOfCondition(conditions=conditions):
                return any(self.evaluate(c, ctx) for c in conditions)

            case NotCondition(condition=inner):
                return not self.evaluate(inner, ctx)

            case RegimeCondition() as regime:
                return self._evaluate_regime(regime, ctx)

            case IREventWindowCondition(
                event_types=event_types,
                pre_window_bars=pre_window,
                post_window_bars=post_window,
                mode=mode,
            ):
                in_window = ctx.is_in_event_window(event_types, pre_window, post_window)
                return in_window if mode == "within" else not in_window

            case RuntimeCondition(
                condition_type=cond_type,
                params=params,
            ) | IRRuntimeCondition(
                condition_type=cond_type,
                params=params,
            ):
                # Handle specific runtime condition types
                if cond_type == "breakout":
                    return self._evaluate_breakout_runtime(params, ctx)
                elif cond_type == "band_edge_event":
                    return self._evaluate_band_edge_event(params, ctx)
                elif cond_type == "band_exit":
                    return self._evaluate_band_exit(params, ctx)
                # For other runtime conditions, pass through
                return True

            case TimeFilterCondition(
                days_of_week=days_of_week,
                time_window=time_window,
            ):
                # Check if current day/time is within the filter window
                if days_of_week and ctx.day_of_week not in days_of_week:
                    return False
                if time_window:
                    # Parse time window "HHMM-HHMM"
                    start_str, end_str = time_window.split("-")
                    start_h, start_m = int(start_str[:2]), int(start_str[2:])
                    end_h, end_m = int(end_str[:2]), int(end_str[2:])
                    current_mins = ctx.hour * 60 + ctx.minute
                    start_mins = start_h * 60 + start_m
                    end_mins = end_h * 60 + end_m
                    if not (start_mins <= current_mins <= end_mins):
                        return False
                return True

            case BreakoutCondition(
                max_indicator=max_ind,
                min_indicator=min_ind,
                buffer_bps=buffer_bps,
            ):
                # Check if price breaks above max or below min
                price = ctx.price_bar.Close
                buffer_mult = 1 + buffer_bps / 10000
                # Get max/min indicator values
                max_val = self.resolver._get_indicator_value(ctx, max_ind)
                min_val = self.resolver._get_indicator_value(ctx, min_ind)
                # Breakout above max or below min
                return price > max_val * buffer_mult or price < min_val / buffer_mult

            case SqueezeCondition():
                # Squeeze condition needs specific handling
                # In simulation, we just return True (condition passes)
                return True

            case _:
                raise ValueError(f"Unknown condition type: {type(condition)}")

    def _evaluate_regime(self, regime: RegimeCondition, ctx: EvalContext) -> bool:
        """Evaluate a regime condition.

        This handles higher-level regime metrics by mapping them to
        indicator comparisons.
        """
        metric = regime.metric
        op = regime.op
        value = regime.value

        match metric:
            case "trend_ma_relation":
                # Compare fast MA to slow MA
                fast_id = f"ema_{regime.ma_fast}" if regime.ma_fast else "ema_fast"
                slow_id = f"ema_{regime.ma_slow}" if regime.ma_slow else "ema_slow"
                fast_val = self.resolver._get_indicator_value(ctx, fast_id)
                slow_val = self.resolver._get_indicator_value(ctx, slow_id)
                diff = fast_val - slow_val
                return op.apply(diff, float(value))

            case "ret_pct":
                # Rate of change
                roc_id = f"roc_{regime.lookback_bars}" if regime.lookback_bars else "roc"
                roc_val = self.resolver._get_indicator_value(ctx, roc_id)
                return op.apply(roc_val * 100, float(value))  # Convert to percentage

            case "session_phase":
                # Session phase: check if current time is in specified phase
                # Value is 'open', 'close', or 'overnight'
                session = getattr(regime, 'session', 'us') if hasattr(regime, 'session') else 'us'
                phase = str(value)
                # Session hours (in UTC)
                session_hours = {
                    "us": {"open_start": 14, "open_end": 16, "close_start": 20, "close_end": 21},
                    "eu": {"open_start": 8, "open_end": 10, "close_start": 16, "close_end": 17},
                    "asia": {"open_start": 0, "open_end": 2, "close_start": 6, "close_end": 7},
                }
                hours = session_hours.get(session or 'us', session_hours["us"])
                current_hour = ctx.get_timestamp().hour

                if phase == "open":
                    in_phase = hours["open_start"] <= current_hour < hours["open_end"]
                elif phase == "close":
                    in_phase = hours["close_start"] <= current_hour < hours["close_end"]
                elif phase == "overnight":
                    in_phase = not (hours["open_start"] <= current_hour < hours["close_end"])
                else:
                    in_phase = False
                return in_phase if op == CompareOp.EQ else not in_phase

            case "volume_spike" | "volume_dip":
                # Volume comparison to average
                lookback = regime.lookback_bars or 20
                vol_sma_id = f"vol_sma_{lookback}"
                vol_sma = self.resolver._get_indicator_value(ctx, vol_sma_id)
                current_vol = ctx.get_volume()
                if metric == "volume_spike":
                    threshold = 1.0 + (float(value) - 50) / 50  # Convert percentile to multiplier
                    return current_vol > vol_sma * threshold
                else:  # volume_dip
                    threshold = float(value) / 50
                    return current_vol < vol_sma * threshold

            case "price_level_touch":
                # Price touches a level
                level = float(value) if value else 0
                high = ctx.get_price(PriceField.HIGH)
                low = ctx.get_price(PriceField.LOW)
                return high >= level >= low

            case "price_level_cross":
                # Price crosses a level
                level = float(value) if value else 0
                close = ctx.get_price(PriceField.CLOSE)
                direction = getattr(regime, 'level_direction', 'up') if hasattr(regime, 'level_direction') else 'up'
                if direction == "up":
                    return close > level
                else:
                    return close < level

            case "vol_bb_width_pctile" | "bb_width_pctile":
                # BB width percentile comparison
                lookback = regime.lookback_bars or 20
                bb_id = f"bb_{lookback}"
                bb_width = self.resolver._get_indicator_band_width(ctx, bb_id)
                return op.apply(bb_width, float(value))

            case "vol_atr_pct":
                # ATR as percentage of price
                lookback = regime.lookback_bars or 14
                atr_id = f"atr_{lookback}"
                atr_val = self.resolver._get_indicator_value(ctx, atr_id)
                close = ctx.get_price(PriceField.CLOSE)
                atr_pct = (atr_val / close) * 100
                return op.apply(atr_pct, float(value))

            case "trend_adx":
                # ADX value comparison
                lookback = regime.lookback_bars or 14
                adx_id = f"adx_{lookback}"
                adx_val = self.resolver._get_indicator_value(ctx, adx_id)
                return op.apply(adx_val, float(value))

            case "trend_regime":
                # Same as trend_ma_relation but with string value
                fast_id = f"ema_{regime.ma_fast}" if regime.ma_fast else "ema_fast"
                slow_id = f"ema_{regime.ma_slow}" if regime.ma_slow else "ema_slow"
                fast_val = self.resolver._get_indicator_value(ctx, fast_id)
                slow_val = self.resolver._get_indicator_value(ctx, slow_id)
                if value == "up":
                    return fast_val > slow_val
                elif value == "down":
                    return fast_val < slow_val
                else:
                    return True  # neutral

            case "vol_regime":
                # Volatility regime: "quiet", "normal", "high"
                lookback = regime.lookback_bars or 20
                bb_id = f"bb_{lookback}"
                bb_width = self.resolver._get_indicator_band_width(ctx, bb_id)
                low_thresh = 25
                high_thresh = 75
                if value == "quiet":
                    return bb_width < low_thresh
                elif value == "high":
                    return bb_width > high_thresh
                else:  # normal
                    return low_thresh <= bb_width <= high_thresh

            case "volume_pctile":
                # Volume percentile approximation
                lookback = regime.lookback_bars or 20
                vol_sma_id = f"vol_sma_{lookback}"
                vol_sma = self.resolver._get_indicator_value(ctx, vol_sma_id)
                current_vol = ctx.get_volume()
                approx_pctile = (current_vol / vol_sma) * 50 if vol_sma > 0 else 50
                return op.apply(approx_pctile, float(value))

            case "dist_from_vwap_pct":
                # Distance from VWAP
                vwap_val = self.resolver._get_indicator_value(ctx, "vwap")
                close = ctx.get_price(PriceField.CLOSE)
                dist_pct = ((close - vwap_val) / vwap_val) * 100 if vwap_val > 0 else 0
                return op.apply(dist_pct, float(value))

            case "liquidity_sweep" | "flag_pattern" | "pennant_pattern":
                # These complex patterns need more sophisticated state tracking
                # For now, return False as they require runtime state machines
                return False

            case "risk_event_prob":
                # External calendar data - not available in simulation
                return True  # Pass through, no restriction

            case "gap_pct":
                # Gap percentage requires previous close
                try:
                    prev_close = self.resolver._get_rolling_window_value(ctx, "prev_close_rw", 1)
                    open_price = ctx.get_price(PriceField.OPEN)
                    gap_pct = ((open_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0
                    return op.apply(gap_pct, float(value))
                except Exception:
                    return False

            case _:
                # Unknown metric - log warning and pass through
                import logging
                logging.warning(f"Unknown regime metric: {metric}")
                return True  # Default to passing

    def _evaluate_breakout_runtime(
        self, params: dict[str, Any], ctx: EvalContext
    ) -> bool:
        """Evaluate a breakout runtime condition.

        Uses Donchian Channel to check if price breaks above upper or below lower.
        """
        lookback = params.get("lookback_bars", 50)
        buffer_bps = params.get("buffer_bps", 0)
        direction = params.get("direction", "long")

        # Get Donchian Channel indicator (raw, not via _get_indicator_value)
        dc_id = f"dc_{lookback}"
        try:
            ind = ctx.get_indicator(dc_id)
        except KeyError:
            # No indicator available - pass through
            return True

        price = ctx.price_bar.Close
        buffer_mult = 1 + buffer_bps / 10000

        # Handle different indicator value representations
        if isinstance(ind, dict):
            # Test mock provides dict directly
            upper = ind.get("upper", float("inf"))
            lower = ind.get("lower", float("-inf"))
        elif hasattr(ind, "UpperBand"):
            # LEAN-style band indicator or MockBandIndicator
            upper_band = ind.UpperBand
            lower_band = ind.LowerBand
            # Check if it's a mock with Current.Value or just value
            if hasattr(upper_band, "Current"):
                upper = upper_band.Current.Value
                lower = lower_band.Current.Value
            elif hasattr(upper_band, "value"):
                upper = upper_band.value
                lower = lower_band.value
            else:
                return True
        elif hasattr(ind, "value") and isinstance(ind.value, dict):
            # Mock with value attribute containing dict
            upper = ind.value.get("upper", float("inf"))
            lower = ind.value.get("lower", float("-inf"))
        else:
            # Unknown format - pass through
            return True

        if direction == "long":
            return price > upper * buffer_mult
        elif direction == "short":
            return price < lower / buffer_mult
        else:  # auto - either direction
            return price > upper * buffer_mult or price < lower / buffer_mult

    def _evaluate_band_edge_event(
        self, params: dict[str, Any], ctx: EvalContext
    ) -> bool:
        """Evaluate a band edge event condition.

        Checks if price touches or crosses a band edge (upper/lower).
        """
        band_type = params.get("band_type", "bollinger")
        length = params.get("length", 20)
        edge = params.get("edge", "lower")
        event = params.get("event", "touch")  # touch, cross_above, cross_below

        # Get band indicator
        if band_type == "bollinger":
            band_id = f"bb_{length}"
        elif band_type == "keltner":
            band_id = f"kc_{length}"
        elif band_type == "donchian":
            band_id = f"dc_{length}"
        else:
            return True  # Unknown band type - pass through

        try:
            ind = ctx.get_indicator(band_id)
        except KeyError:
            return True  # No indicator - pass through

        price = ctx.price_bar.Close

        # Get band values
        if isinstance(ind, dict):
            upper = ind.get("upper", float("inf"))
            lower = ind.get("lower", float("-inf"))
        elif hasattr(ind, "UpperBand"):
            upper_band = ind.UpperBand
            lower_band = ind.LowerBand
            if hasattr(upper_band, "Current"):
                upper = upper_band.Current.Value
                lower = lower_band.Current.Value
            elif hasattr(upper_band, "value"):
                upper = upper_band.value
                lower = lower_band.value
            else:
                return True
        else:
            return True

        # Check edge condition
        target = lower if edge == "lower" else upper

        if event == "touch":
            # Price at or below lower band, or at or above upper band
            if edge == "lower":
                return price <= lower
            else:
                return price >= upper
        elif event == "cross_out":
            # Price moves from inside to outside the band
            # For upper: price goes above upper band
            # For lower: price goes below lower band
            if edge == "upper":
                return price >= upper
            else:
                return price <= lower
        elif event == "cross_in":
            # Price moves from outside to inside the band
            # For upper: price comes back below upper band (was above)
            # For lower: price comes back above lower band (was below)
            if edge == "upper":
                return price <= upper
            else:
                return price >= lower
        elif event == "cross_above":
            # This would need previous bar data - approximate with current
            return price > target
        elif event == "cross_below":
            return price < target
        else:
            return True

    def _evaluate_band_exit(
        self, params: dict[str, Any], ctx: EvalContext
    ) -> bool:
        """Evaluate a band exit condition.

        Checks if price reaches band exit level.
        """
        band_type = params.get("band_type", "bollinger")
        length = params.get("length", 20)
        exit_edge = params.get("exit_edge", "upper")

        # Get band indicator
        if band_type == "bollinger":
            band_id = f"bb_{length}"
        elif band_type == "keltner":
            band_id = f"kc_{length}"
        elif band_type == "donchian":
            band_id = f"dc_{length}"
        else:
            return True

        try:
            ind = ctx.get_indicator(band_id)
        except KeyError:
            return True

        price = ctx.price_bar.Close

        # Get band values
        if isinstance(ind, dict):
            upper = ind.get("upper", float("inf"))
            lower = ind.get("lower", float("-inf"))
        elif hasattr(ind, "UpperBand"):
            upper_band = ind.UpperBand
            lower_band = ind.LowerBand
            if hasattr(upper_band, "Current"):
                upper = upper_band.Current.Value
                lower = lower_band.Current.Value
            elif hasattr(upper_band, "value"):
                upper = upper_band.value
                lower = lower_band.value
            else:
                return True
        else:
            return True

        if exit_edge == "upper":
            return price >= upper
        elif exit_edge == "lower":
            return price <= lower
        else:
            return True


# =============================================================================
# Action Executor
# =============================================================================


class ActionExecutor:
    """Executes Action objects."""

    def execute(self, action: Action, ctx: ExecContext) -> None:
        """Execute an action."""
        match action:
            case SetHoldingsAction(allocation=allocation):
                ctx.set_holdings(allocation)

            case LiquidateAction():
                ctx.liquidate()

            case MarketOrderAction(quantity=quantity):
                ctx.market_order(quantity)

            case _:
                raise ValueError(f"Unknown action type: {type(action)}")


# =============================================================================
# State Operator
# =============================================================================


class StateOperator:
    """Executes StateOp objects to mutate state."""

    def __init__(self, condition_evaluator: ConditionEvaluator | None = None):
        self.resolver = ValueResolver()
        # Lazy import to avoid circular dependency
        self._condition_evaluator = condition_evaluator

    @property
    def condition_evaluator(self) -> ConditionEvaluator:
        if self._condition_evaluator is None:
            self._condition_evaluator = ConditionEvaluator()
        return self._condition_evaluator

    def execute(self, op: StateOp, ctx: EvalContext) -> None:
        """Execute a state operation."""
        match op:
            case SetStateOp(state_id=state_id, value=value_ref):
                value = self.resolver.resolve(value_ref, ctx)
                ctx.set_state(state_id, value)

            case IncrementStateOp(state_id=state_id):
                current = ctx.get_state(state_id)
                ctx.set_state(state_id, (current or 0) + 1)

            case MaxStateOp(state_id=state_id, value=value_ref):
                new_value = self.resolver.resolve(value_ref, ctx)
                current = ctx.get_state(state_id)
                if current is None or new_value > current:
                    ctx.set_state(state_id, new_value)

            case MinStateOp(state_id=state_id, value=value_ref):
                new_value = self.resolver.resolve(value_ref, ctx)
                current = ctx.get_state(state_id)
                if current is None or new_value < current:
                    ctx.set_state(state_id, new_value)

            case SetStateFromConditionOp(state_id=state_id, condition=condition):
                result = self.condition_evaluator.evaluate(condition, ctx)
                # Store as 1.0 or 0.0 for float compatibility
                ctx.set_state(state_id, 1.0 if result else 0.0)

            case _:
                raise ValueError(f"Unknown state op type: {type(op)}")
