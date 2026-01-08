"""
Generic Strategy Runtime for LEAN.

This algorithm interprets StrategyIR JSON at runtime, allowing a single algorithm
to execute any strategy defined in the IR format.

The IR is passed via:
1. A JSON file in the algorithm's data directory
2. Or as a parameter when launching the algorithm
"""

from AlgorithmImports import *
import json
from enum import Enum


# =============================================================================
# IR Types (mirrors src/translator/ir.py for LEAN environment)
# =============================================================================


class CompareOp(str, Enum):
    """Comparison operators."""
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    EQ = "=="
    NEQ = "!="

    def apply(self, left: float, right: float) -> bool:
        """Apply the comparison operator."""
        if self == CompareOp.LT:
            return left < right
        elif self == CompareOp.LTE:
            return left <= right
        elif self == CompareOp.GT:
            return left > right
        elif self == CompareOp.GTE:
            return left >= right
        elif self == CompareOp.EQ:
            return left == right
        elif self == CompareOp.NEQ:
            return left != right
        return False


# =============================================================================
# Runtime Algorithm
# =============================================================================


class StrategyRuntime(QCAlgorithm):
    """Generic strategy runtime that interprets IR JSON."""

    def Initialize(self):
        """Initialize the algorithm."""
        # Default backtest period
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)

        # Load strategy IR
        ir_json = self.GetParameter("strategy_ir")
        if ir_json:
            self.ir = json.loads(ir_json)
        else:
            # Try loading from file
            ir_path = self.GetParameter("strategy_ir_path")
            if ir_path:
                self.ir = self._load_ir_from_file(ir_path)
            else:
                raise ValueError("No strategy_ir or strategy_ir_path parameter provided")

        # Set up symbol
        symbol_str = self.ir.get("symbol", "BTC-USD")
        self.symbol = self._add_symbol(symbol_str)

        # Set resolution
        resolution_str = self.ir.get("resolution", "Hour")
        self.resolution = getattr(Resolution, resolution_str.capitalize(), Resolution.Hour)

        # Initialize indicators
        self.indicators = {}
        self.rolling_windows = {}  # For RollingWindow indicators
        self.vol_sma_indicators = {}  # For volume SMA indicators
        self.rolling_minmax = {}  # For rolling min/max trackers
        self._create_indicators()

        # Initialize state
        self.state = {}
        self._initialize_state()

        # Parse entry and exit rules
        self.entry_rule = self.ir.get("entry")
        self.exit_rules = self.ir.get("exits", [])
        self.gates = self.ir.get("gates", [])
        self.on_bar_invested_ops = self.ir.get("on_bar_invested", [])
        self.on_bar_ops = self.ir.get("on_bar", [])

        self.Log(f"✅ StrategyRuntime initialized")
        self.Log(f"   Strategy: {self.ir.get('strategy_name', 'Unknown')}")
        self.Log(f"   Symbol: {self.symbol}")
        self.Log(f"   Indicators: {len(self.indicators)}")

    def _add_symbol(self, symbol_str: str) -> Symbol:
        """Add symbol based on string (e.g., BTC-USD)."""
        # Convert vibe-trade symbol format to LEAN format
        if "-USD" in symbol_str:
            base = symbol_str.replace("-USD", "")
            lean_symbol = f"{base}USD"
            return self.AddCrypto(lean_symbol, self.resolution, Market.GDAX).Symbol
        else:
            return self.AddCrypto(symbol_str, self.resolution, Market.GDAX).Symbol

    def _load_ir_from_file(self, path: str) -> dict:
        """Load IR JSON from a file path."""
        # In LEAN, you'd typically use ObjectStore or include in algorithm data
        raise NotImplementedError("File loading not yet implemented")

    def _create_indicators(self):
        """Create all indicators defined in the IR."""
        for ind_def in self.ir.get("indicators", []):
            ind_type = ind_def.get("type")
            ind_id = ind_def.get("id")

            if ind_type == "EMA":
                period = ind_def.get("period", 20)
                self.indicators[ind_id] = self.EMA(self.symbol, period, self.resolution)
            elif ind_type == "SMA":
                period = ind_def.get("period", 20)
                self.indicators[ind_id] = self.SMA(self.symbol, period, self.resolution)
            elif ind_type == "BB":
                period = ind_def.get("period", 20)
                mult = ind_def.get("multiplier", 2.0)
                self.indicators[ind_id] = self.BB(self.symbol, period, mult, self.resolution)
            elif ind_type == "KC":
                period = ind_def.get("period", 20)
                mult = ind_def.get("multiplier", 2.0)
                self.indicators[ind_id] = self.KeltnerChannels(self.symbol, period, mult, self.resolution)
            elif ind_type == "ATR":
                period = ind_def.get("period", 14)
                self.indicators[ind_id] = self.ATR(self.symbol, period, self.resolution)
            elif ind_type == "MAX":
                period = ind_def.get("period", 50)
                self.indicators[ind_id] = self.MAX(self.symbol, period, self.resolution)
            elif ind_type == "MIN":
                period = ind_def.get("period", 50)
                self.indicators[ind_id] = self.MIN(self.symbol, period, self.resolution)
            elif ind_type == "ROC":
                period = ind_def.get("period", 1)
                self.indicators[ind_id] = self.ROC(self.symbol, period, self.resolution)
            elif ind_type == "ADX":
                period = ind_def.get("period", 14)
                self.indicators[ind_id] = self.ADX(self.symbol, period, self.resolution)
            elif ind_type == "DC":
                period = ind_def.get("period", 20)
                self.indicators[ind_id] = self.DCH(self.symbol, period, self.resolution)
            elif ind_type == "VWAP":
                period = ind_def.get("period", 0)
                if period == 0:
                    # Intraday VWAP (resets daily)
                    self.indicators[ind_id] = self.VWAP(self.symbol)
                else:
                    # Rolling VWAP with period
                    self.indicators[ind_id] = self.VWAP(self.symbol, period)
            elif ind_type == "RW":
                # Rolling window for historical values (e.g., previous close)
                period = ind_def.get("period", 2)
                field = ind_def.get("field", "close")
                # Store as a RollingWindow - handled specially in OnData
                self.rolling_windows[ind_id] = {
                    "window": RollingWindow[float](period),
                    "field": field,
                }
            elif ind_type == "VOL_SMA":
                # Simple Moving Average of volume
                period = ind_def.get("period", 20)
                # Use SMA indicator on volume data
                self.vol_sma_indicators[ind_id] = {
                    "sma": SimpleMovingAverage(period),
                    "period": period,
                }
            elif ind_type == "RMM":
                # Rolling Min/Max tracker
                period = ind_def.get("period", 20)
                mode = ind_def.get("mode", "min")
                field = ind_def.get("field", "close")
                self.rolling_minmax[ind_id] = {
                    "window": RollingWindow[float](period),
                    "mode": mode,
                    "field": field,
                }
            else:
                self.Log(f"⚠️ Unknown indicator type: {ind_type}")

    def _initialize_state(self):
        """Initialize state variables from IR."""
        for state_var in self.ir.get("state", []):
            state_id = state_var.get("id")
            default = state_var.get("default")
            self.state[state_id] = default

    def OnData(self, data: Slice):
        """Called when new market data arrives."""
        # Skip if no data for our symbol
        if self.symbol not in data:
            return

        bar = data[self.symbol]

        # Update rolling windows before checking indicators
        for rw_id, rw_data in self.rolling_windows.items():
            field = rw_data["field"]
            if field == "close":
                rw_data["window"].Add(bar.Close)
            elif field == "open":
                rw_data["window"].Add(bar.Open)
            elif field == "high":
                rw_data["window"].Add(bar.High)
            elif field == "low":
                rw_data["window"].Add(bar.Low)

        # Update volume SMA indicators
        for vol_id, vol_data in self.vol_sma_indicators.items():
            vol_data["sma"].Update(self.Time, float(bar.Volume))

        # Update rolling min/max trackers
        for rmm_id, rmm_data in self.rolling_minmax.items():
            field = rmm_data["field"]
            if field == "close":
                rmm_data["window"].Add(bar.Close)
            elif field == "open":
                rmm_data["window"].Add(bar.Open)
            elif field == "high":
                rmm_data["window"].Add(bar.High)
            elif field == "low":
                rmm_data["window"].Add(bar.Low)

        # Wait for all indicators to be ready
        if not self._indicators_ready():
            return

        # Run on_bar hooks every bar (for state tracking like cross detection)
        self._run_on_bar(bar)

        # Evaluate gates first
        if not self._evaluate_gates(bar):
            return

        # Check position state
        is_invested = self.Portfolio[self.symbol].Invested

        # If invested, run on_bar_invested hooks and check exits
        if is_invested:
            self._run_on_bar_invested(bar)
            self._evaluate_exits(bar)
        else:
            # Check entry
            self._evaluate_entry(bar)

    def _indicators_ready(self) -> bool:
        """Check if all indicators are ready."""
        for ind in self.indicators.values():
            if not ind.IsReady:
                return False
        # Also check rolling windows
        for rw_data in self.rolling_windows.values():
            if not rw_data["window"].IsReady:
                return False
        # Check volume SMA indicators
        for vol_data in self.vol_sma_indicators.values():
            if not vol_data["sma"].IsReady:
                return False
        # Check rolling min/max
        for rmm_data in self.rolling_minmax.values():
            if not rmm_data["window"].IsReady:
                return False
        return True

    def _evaluate_gates(self, bar) -> bool:
        """Evaluate gate conditions. Returns True if all gates pass."""
        for gate in self.gates:
            condition = gate.get("condition")
            mode = gate.get("mode", "allow")

            result = self._evaluate_condition(condition, bar)

            if mode == "allow" and not result:
                return False
            elif mode == "block" and result:
                return False

        return True

    def _evaluate_entry(self, bar):
        """Evaluate entry rule and execute if conditions met."""
        if not self.entry_rule:
            return

        condition = self.entry_rule.get("condition")
        if self._evaluate_condition(condition, bar):
            action = self.entry_rule.get("action", {})
            self._execute_action(action)

            # Run on_fill hooks
            for op in self.entry_rule.get("on_fill", []):
                self._execute_state_op(op, bar)

            self.Log(f"🟢 ENTRY @ ${bar.Close:.2f}")

    def _evaluate_exits(self, bar):
        """Evaluate exit rules in priority order."""
        # Sort by priority (lower priority number = higher priority)
        sorted_exits = sorted(self.exit_rules, key=lambda x: x.get("priority", 0))

        for exit_rule in sorted_exits:
            condition = exit_rule.get("condition")
            if self._evaluate_condition(condition, bar):
                action = exit_rule.get("action", {})
                self._execute_action(action)
                self.Log(f"🔴 EXIT ({exit_rule.get('id', 'unknown')}) @ ${bar.Close:.2f}")
                break  # Only execute first matching exit

    def _run_on_bar_invested(self, bar):
        """Run on_bar_invested state operations."""
        for op in self.on_bar_invested_ops:
            self._execute_state_op(op, bar)

    def _run_on_bar(self, bar):
        """Run on_bar state operations (every bar, for state tracking)."""
        for op in self.on_bar_ops:
            self._execute_state_op(op, bar)

    def _evaluate_condition(self, condition: dict, bar) -> bool:
        """Evaluate a condition from IR."""
        if not condition:
            return True

        cond_type = condition.get("type")

        if cond_type == "compare":
            left_val = self._resolve_value(condition.get("left"), bar)
            right_val = self._resolve_value(condition.get("right"), bar)
            op_str = condition.get("op")
            op = CompareOp(op_str)
            return op.apply(left_val, right_val)

        elif cond_type == "allOf":
            for sub in condition.get("conditions", []):
                if not self._evaluate_condition(sub, bar):
                    return False
            return True

        elif cond_type == "anyOf":
            for sub in condition.get("conditions", []):
                if self._evaluate_condition(sub, bar):
                    return True
            return False

        elif cond_type == "not":
            inner = condition.get("condition")
            return not self._evaluate_condition(inner, bar)

        elif cond_type == "regime":
            # Handle regime conditions by mapping to indicator comparisons
            return self._evaluate_regime(condition, bar)

        else:
            self.Log(f"⚠️ Unknown condition type: {cond_type}")
            return True

    def _evaluate_regime(self, regime: dict, bar) -> bool:
        """Evaluate a regime condition."""
        metric = regime.get("metric")
        op_str = regime.get("op", "==")
        value = regime.get("value")
        op = CompareOp(op_str)

        if metric == "trend_ma_relation":
            fast_id = f"ema_{regime.get('ma_fast', 20)}"
            slow_id = f"ema_{regime.get('ma_slow', 50)}"

            # Fall back to named indicators
            fast_ind = self.indicators.get(fast_id) or self.indicators.get("ema_fast")
            slow_ind = self.indicators.get(slow_id) or self.indicators.get("ema_slow")

            if fast_ind and slow_ind:
                diff = fast_ind.Current.Value - slow_ind.Current.Value
                return op.apply(diff, float(value))

        elif metric == "ret_pct":
            roc_ind = self.indicators.get("roc")
            if roc_ind:
                return op.apply(roc_ind.Current.Value * 100, float(value))

        elif metric == "gap_pct":
            # Gap percentage: (Open - PrevClose) / PrevClose * 100
            prev_close_rw = self.rolling_windows.get("prev_close")
            if prev_close_rw and prev_close_rw["window"].IsReady:
                # Rolling window: index 0 = most recent (current bar's close after update)
                # index 1 = previous bar's close
                prev_close = prev_close_rw["window"][1]
                if prev_close != 0:
                    gap = (bar.Open - prev_close) / prev_close * 100
                    return op.apply(gap, float(value))

        elif metric == "liquidity_sweep":
            # Liquidity sweep: break below level then reclaim
            # Requires state tracking (implemented via state vars in IR)
            return self._evaluate_liquidity_sweep(regime, bar)

        elif metric == "flag_pattern":
            # Flag pattern: momentum + consolidation + breakout
            return self._evaluate_flag_pattern(regime, bar)

        elif metric == "pennant_pattern":
            # Pennant pattern: similar to flag with triangular consolidation
            return self._evaluate_pennant_pattern(regime, bar)

        elif metric in ("price_level_touch", "price_level_cross"):
            # These should be lowered by translator, but handle fallback
            # For dynamic levels that need runtime resolution
            return self._evaluate_price_level(metric, regime, bar)

        # Unknown metric - return True to not block
        self.Log(f"⚠️ Unknown regime metric: {metric}")
        return True

    def _resolve_value(self, value_ref: dict, bar) -> float:
        """Resolve a value reference to a float."""
        if not value_ref:
            return 0.0

        val_type = value_ref.get("type")

        if val_type == "literal":
            return float(value_ref.get("value", 0))

        elif val_type == "indicator":
            ind_id = value_ref.get("indicator_id")
            # First check regular indicators
            ind = self.indicators.get(ind_id)
            if ind:
                return ind.Current.Value
            # Check volume SMA indicators
            vol_data = self.vol_sma_indicators.get(ind_id)
            if vol_data:
                return vol_data["sma"].Current.Value
            # Check rolling min/max indicators
            rmm_data = self.rolling_minmax.get(ind_id)
            if rmm_data:
                window = rmm_data["window"]
                if window.IsReady:
                    if rmm_data["mode"] == "min":
                        return min(list(window))
                    else:  # max
                        return max(list(window))
                return 0.0
            self.Log(f"⚠️ Unknown indicator: {ind_id}")
            return 0.0

        elif val_type == "indicator_band":
            ind_id = value_ref.get("indicator_id")
            band = value_ref.get("band")
            ind = self.indicators.get(ind_id)
            if ind:
                if band == "upper":
                    return ind.UpperBand.Current.Value
                elif band == "middle":
                    return ind.MiddleBand.Current.Value
                elif band == "lower":
                    return ind.LowerBand.Current.Value
            return 0.0

        elif val_type == "indicator_property":
            ind_id = value_ref.get("indicator_id")
            prop = value_ref.get("property")
            ind = self.indicators.get(ind_id)
            if ind:
                if prop == "StandardDeviation":
                    # Bollinger Bands have StandardDeviation property
                    return ind.StandardDeviation.Current.Value
                elif prop == "BandWidth":
                    # Band width calculation
                    if hasattr(ind, 'BandWidth'):
                        return ind.BandWidth.Current.Value
                    # Fallback: calculate manually
                    upper = ind.UpperBand.Current.Value
                    lower = ind.LowerBand.Current.Value
                    middle = ind.MiddleBand.Current.Value
                    if middle != 0:
                        return (upper - lower) / middle
            return 0.0

        elif val_type == "price":
            field = value_ref.get("field", "close")
            if field == "open":
                return bar.Open
            elif field == "high":
                return bar.High
            elif field == "low":
                return bar.Low
            else:  # close
                return bar.Close

        elif val_type == "volume":
            # Current bar's volume
            return float(bar.Volume)

        elif val_type == "time":
            # Time component from current bar
            component = value_ref.get("component", "hour")
            if component == "hour":
                return float(self.Time.hour)
            elif component == "minute":
                return float(self.Time.minute)
            elif component == "day_of_week":
                return float(self.Time.weekday())  # 0=Monday, 6=Sunday
            return 0.0

        elif val_type == "state":
            state_id = value_ref.get("state_id")
            val = self.state.get(state_id)
            if val is None:
                return 0.0
            return float(val)

        elif val_type == "expr":
            op = value_ref.get("op")
            left = self._resolve_value(value_ref.get("left"), bar)
            right = self._resolve_value(value_ref.get("right"), bar)

            if op == "+":
                return left + right
            elif op == "-":
                return left - right
            elif op == "*":
                return left * right
            elif op == "/":
                if right == 0:
                    return 0.0
                return left / right

        self.Log(f"⚠️ Unknown value type: {val_type}")
        return 0.0

    def _evaluate_liquidity_sweep(self, regime: dict, bar) -> bool:
        """Evaluate liquidity sweep pattern.

        Liquidity sweep: price breaks below a level (taking out stops),
        then reclaims above it within N bars.

        State tracking: Uses state vars 'sweep_triggered' and 'sweep_bar_count'
        """
        lookback_bars = regime.get("lookback_bars", 3)

        # Get the level indicator (set up by translator)
        level_min = self.rolling_minmax.get("level_min")
        level_max = self.rolling_minmax.get("level_max")

        level_value = None
        if level_min and level_min["window"].IsReady:
            level_value = min(list(level_min["window"]))
        elif level_max and level_max["window"].IsReady:
            level_value = max(list(level_max["window"]))

        if level_value is None:
            return False

        # Check state
        sweep_triggered = self.state.get("sweep_triggered", False)
        sweep_bar_count = self.state.get("sweep_bar_count", 0)

        if not sweep_triggered:
            # Check if price broke below level (sweep)
            if bar.Low < level_value:
                self.state["sweep_triggered"] = True
                self.state["sweep_bar_count"] = 0
                self.state["sweep_level"] = level_value
        else:
            # Increment bar count
            sweep_bar_count += 1
            self.state["sweep_bar_count"] = sweep_bar_count

            if sweep_bar_count > lookback_bars:
                # Timeout - reset
                self.state["sweep_triggered"] = False
                return False

            # Check if price reclaimed above level
            sweep_level = self.state.get("sweep_level", level_value)
            if bar.Close > sweep_level:
                # Sweep complete - reset and signal
                self.state["sweep_triggered"] = False
                return True

        return False

    def _evaluate_flag_pattern(self, regime: dict, bar) -> bool:
        """Evaluate flag pattern.

        Flag pattern: Initial strong momentum move + consolidation with
        narrowing range + breakout in direction of initial move.
        """
        breakout_dir = regime.get("value", "same")  # "same" or "opposite"

        # Get indicators (set up by translator)
        momentum_roc = self.indicators.get("momentum_roc")
        pattern_atr = self.indicators.get("pattern_atr")
        pattern_max = self.indicators.get("pattern_max")
        pattern_min = self.indicators.get("pattern_min")

        if not all([momentum_roc, pattern_atr, pattern_max, pattern_min]):
            return False

        roc_value = momentum_roc.Current.Value
        atr_value = pattern_atr.Current.Value
        range_high = pattern_max.Current.Value
        range_low = pattern_min.Current.Value

        # Check for consolidation (narrowing range relative to ATR)
        current_range = range_high - range_low
        if atr_value == 0 or current_range / atr_value > 2.0:
            # Not consolidating enough
            return False

        # Check for momentum direction
        initial_momentum_up = roc_value > 0.02  # 2% momentum
        initial_momentum_down = roc_value < -0.02

        if not (initial_momentum_up or initial_momentum_down):
            # No clear momentum
            return False

        # Check breakout
        if breakout_dir == "same":
            if initial_momentum_up and bar.Close > range_high:
                return True
            if initial_momentum_down and bar.Close < range_low:
                return True
        else:  # opposite
            if initial_momentum_up and bar.Close < range_low:
                return True
            if initial_momentum_down and bar.Close > range_high:
                return True

        return False

    def _evaluate_pennant_pattern(self, regime: dict, bar) -> bool:
        """Evaluate pennant pattern.

        Similar to flag but with triangular (converging) consolidation.
        """
        # For now, use same logic as flag - true triangular detection
        # would require tracking converging highs/lows over time
        return self._evaluate_flag_pattern(regime, bar)

    def _evaluate_price_level(self, metric: str, regime: dict, bar) -> bool:
        """Evaluate price level touch/cross for dynamic levels.

        This handles cases where the translator couldn't fully lower the condition.
        """
        level_ref = regime.get("value", "")
        direction = "up" if "_up" in level_ref else "down"

        # Get dynamic level from indicators
        level_min = self.rolling_minmax.get("level_min")
        level_max = self.rolling_minmax.get("level_max")

        level_value = None
        if level_min and level_min["window"].IsReady:
            level_value = min(list(level_min["window"]))
        elif level_max and level_max["window"].IsReady:
            level_value = max(list(level_max["window"]))

        if level_value is None:
            return False

        if metric == "price_level_touch":
            # Check if bar touches the level
            return bar.Low <= level_value <= bar.High
        else:  # price_level_cross
            if direction == "up":
                return bar.Close > level_value
            else:
                return bar.Close < level_value

    def _execute_action(self, action: dict):
        """Execute an action from IR."""
        if not action:
            return

        action_type = action.get("type")

        if action_type == "set_holdings":
            allocation = action.get("allocation", 0.95)
            self.SetHoldings(self.symbol, allocation)

        elif action_type == "liquidate":
            self.Liquidate(self.symbol)

        elif action_type == "market_order":
            quantity = action.get("quantity", 0)
            self.MarketOrder(self.symbol, quantity)

        else:
            self.Log(f"⚠️ Unknown action type: {action_type}")

    def _execute_state_op(self, op: dict, bar):
        """Execute a state operation."""
        if not op:
            return

        op_type = op.get("type")
        state_id = op.get("state_id")

        if op_type == "set_state":
            value_ref = op.get("value")
            value = self._resolve_value(value_ref, bar)
            self.state[state_id] = value

        elif op_type == "increment":
            current = self.state.get(state_id, 0) or 0
            self.state[state_id] = current + 1

        elif op_type == "max_state":
            value_ref = op.get("value")
            new_value = self._resolve_value(value_ref, bar)
            current = self.state.get(state_id)
            if current is None or new_value > current:
                self.state[state_id] = new_value

        elif op_type == "set_state_from_condition":
            condition = op.get("condition")
            result = self._evaluate_condition(condition, bar)
            # Store as 1.0 or 0.0 for float compatibility
            self.state[state_id] = 1.0 if result else 0.0

        else:
            self.Log(f"⚠️ Unknown state op type: {op_type}")

    def OnEndOfAlgorithm(self):
        """Called when algorithm ends."""
        portfolio_value = self.Portfolio.TotalPortfolioValue
        self.Log(f"📊 Final Portfolio Value: ${portfolio_value:,.2f}")
        self.Log(f"   Strategy: {self.ir.get('strategy_name', 'Unknown')}")
