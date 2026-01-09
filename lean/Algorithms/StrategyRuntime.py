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
# Custom Data Reader for CSV files
# =============================================================================


class CustomCryptoData(PythonData):
    """Custom data reader for cryptocurrency CSV files."""

    # Class variable to hold the data folder path - set by algorithm
    DataFolder = "/Data"

    # Debug log file path
    DebugLogPath = None

    @staticmethod
    def _log_debug(msg):
        """Write debug message to log file."""
        if CustomCryptoData.DebugLogPath:
            try:
                with open(CustomCryptoData.DebugLogPath, "a") as f:
                    f.write(msg + "\n")
            except:
                pass

    def GetSource(self, config, date, isLiveMode):
        """Return the data source - CSV file in data directory."""
        import os
        # Convert symbol to filename (e.g., BTC-USD -> btc_usd_data.csv)
        # Symbol.ID.Symbol may include data type suffix (e.g., "BTC-USD.CustomCryptoData")
        # We need to strip the suffix to get the base symbol
        symbol = str(config.Symbol.ID.Symbol).lower()
        # Strip the data type suffix if present (e.g., ".customcryptodata")
        if "." in symbol:
            symbol = symbol.split(".")[0]
        # Normalize: replace dashes with underscores
        symbol_normalized = symbol.replace("-", "_")
        filename = f"{symbol_normalized}_data.csv"
        # Use full path - data folder is set by the algorithm
        full_path = f"{CustomCryptoData.DataFolder}/{filename}"
        # Log for debugging
        exists = os.path.exists(full_path)
        CustomCryptoData._log_debug(f"[GetSource] DataFolder: {CustomCryptoData.DataFolder}, Symbol: {symbol}, Date: {date}, File: {full_path}, Exists: {exists}")
        return SubscriptionDataSource(
            full_path,
            SubscriptionTransportMedium.LocalFile,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        """Parse a line from the CSV file."""
        if not line or line.startswith("datetime"):
            return None

        data = CustomCryptoData()
        try:
            parts = line.split(",")
            data.Time = datetime.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            data.Symbol = config.Symbol
            data.Value = float(parts[4])  # close price
            data["Open"] = float(parts[1])
            data["High"] = float(parts[2])
            data["Low"] = float(parts[3])
            data["Close"] = float(parts[4])
            data["Volume"] = float(parts[5])
        except Exception as e:
            CustomCryptoData._log_debug(f"[Reader] Error parsing line: {line[:50]}... Error: {e}")
            return None
        return data


# =============================================================================
# Runtime Algorithm
# =============================================================================


class StrategyRuntime(QCAlgorithm):
    """Generic strategy runtime that interprets IR JSON."""

    def Initialize(self):
        """Initialize the algorithm."""
        # Set data folder FIRST - before any subscriptions
        data_folder = self.GetParameter("data_folder")
        if data_folder:
            CustomCryptoData.DataFolder = data_folder
            self.Log(f"[INIT] Set data folder to: {data_folder}")
        else:
            self.Log(f"[INIT] No data_folder parameter, using default: {CustomCryptoData.DataFolder}")

        # Debug: List files in data folder
        import os
        # Set up debug log in data folder
        CustomCryptoData.DebugLogPath = os.path.join(CustomCryptoData.DataFolder, "debug.log")
        self.Debug(f"[INIT] Debug log path: {CustomCryptoData.DebugLogPath}")
        CustomCryptoData._log_debug(f"[INIT] Debug logging initialized at {CustomCryptoData.DataFolder}")

        try:
            files = os.listdir(CustomCryptoData.DataFolder)
            self.Debug(f"[INIT] Files in data folder: {files}")
            # Check for btc_usd_data.csv specifically
            expected_file = os.path.join(CustomCryptoData.DataFolder, "btc_usd_data.csv")
            if os.path.exists(expected_file):
                self.Debug(f"[INIT] ✅ Found btc_usd_data.csv")
                # Read first few lines to verify format
                with open(expected_file, 'r') as f:
                    lines = f.readlines()[:3]
                    for i, line in enumerate(lines):
                        self.Debug(f"[INIT] Line {i}: {line.strip()}")
            else:
                self.Debug(f"[INIT] ❌ btc_usd_data.csv NOT FOUND at {expected_file}")
        except Exception as e:
            self.Debug(f"[INIT] Error listing data folder: {e}")

        # Get date parameters (format: YYYYMMDD)
        start_date_str = self.GetParameter("start_date")
        end_date_str = self.GetParameter("end_date")
        initial_cash_str = self.GetParameter("initial_cash")

        self.Debug(f"start_date parameter: {start_date_str}")
        self.Debug(f"end_date parameter: {end_date_str}")
        self.Debug(f"initial_cash parameter: {initial_cash_str}")

        # Parse dates or use defaults
        if start_date_str:
            year = int(start_date_str[:4])
            month = int(start_date_str[4:6])
            day = int(start_date_str[6:8])
            self.SetStartDate(year, month, day)
        else:
            self.SetStartDate(2024, 1, 1)

        if end_date_str:
            year = int(end_date_str[:4])
            month = int(end_date_str[4:6])
            day = int(end_date_str[6:8])
            self.SetEndDate(year, month, day)
        else:
            self.SetEndDate(2024, 12, 31)

        if initial_cash_str:
            self.SetCash(float(initial_cash_str))
        else:
            self.SetCash(100000)

        # Load strategy IR
        ir_json = self.GetParameter("strategy_ir")
        self.Debug(f"strategy_ir parameter: {ir_json}")
        if ir_json:
            self.ir = json.loads(ir_json)
        else:
            # Try loading from file (parameter or default path)
            ir_path = self.GetParameter("strategy_ir_path")
            self.Debug(f"strategy_ir_path parameter: {ir_path}")
            if not ir_path:
                # Default path for backtest service
                ir_path = "/Data/strategy_ir.json"
            self.Debug(f"Loading IR from: {ir_path}")
            self.ir = self._load_ir_from_file(ir_path)

        # Set resolution (must be before _add_symbol which uses it)
        resolution_str = self.ir.get("resolution", "Hour")
        self.resolution = getattr(Resolution, resolution_str.capitalize(), Resolution.Hour)

        # Data folder was set at the beginning of Initialize

        # Set up symbol
        symbol_str = self.ir.get("symbol", "BTC-USD")
        self.symbol = self._add_symbol(symbol_str)

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

        # Trade tracking for output
        self.trades = []  # List of completed trades
        self.current_trade = None  # Active trade
        self.equity_curve = []  # Portfolio value over time
        self.peak_equity = float(initial_cash_str) if initial_cash_str else 100000
        self.max_drawdown = 0.0
        self.bar_count = 0  # Count bars for equity sampling

        self.Log(f"✅ StrategyRuntime initialized")
        self.Log(f"   Strategy: {self.ir.get('strategy_name', 'Unknown')}")
        self.Log(f"   Symbol: {self.symbol}")
        self.Log(f"   Indicators: {len(self.indicators)}")

    def _add_symbol(self, symbol_str: str) -> Symbol:
        """Add symbol using custom data reader for CSV files."""
        # Use AddData with CustomCryptoData for CSV files
        return self.AddData(CustomCryptoData, symbol_str, self.resolution).Symbol

    def _load_ir_from_file(self, path: str) -> dict:
        """Load IR JSON from a file path."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Strategy IR file not found: {path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in strategy IR file: {e}")

    def _create_indicators(self):
        """Create all indicators defined in the IR."""
        for ind_def in self.ir.get("indicators", []):
            ind_type = ind_def.get("type")
            ind_id = ind_def.get("id")

            # All indicators use named resolution parameter to avoid signature issues
            if ind_type == "EMA":
                period = ind_def.get("period", 20)
                self.indicators[ind_id] = self.EMA(self.symbol, period, resolution=self.resolution)
            elif ind_type == "SMA":
                period = ind_def.get("period", 20)
                self.indicators[ind_id] = self.SMA(self.symbol, period, resolution=self.resolution)
            elif ind_type == "BB":
                period = ind_def.get("period", 20)
                mult = ind_def.get("multiplier", 2.0)
                self.indicators[ind_id] = self.BB(self.symbol, period, mult, resolution=self.resolution)
            elif ind_type == "KC":
                period = ind_def.get("period", 20)
                mult = ind_def.get("multiplier", 2.0)
                self.indicators[ind_id] = self.KCH(self.symbol, period, mult, resolution=self.resolution)
            elif ind_type == "ATR":
                period = ind_def.get("period", 14)
                self.indicators[ind_id] = self.ATR(self.symbol, period, resolution=self.resolution)
            elif ind_type == "MAX":
                period = ind_def.get("period", 50)
                self.indicators[ind_id] = self.MAX(self.symbol, period, resolution=self.resolution)
            elif ind_type == "MIN":
                period = ind_def.get("period", 50)
                self.indicators[ind_id] = self.MIN(self.symbol, period, resolution=self.resolution)
            elif ind_type == "ROC":
                period = ind_def.get("period", 1)
                self.indicators[ind_id] = self.ROC(self.symbol, period, resolution=self.resolution)
            elif ind_type == "ADX":
                period = ind_def.get("period", 14)
                self.indicators[ind_id] = self.ADX(self.symbol, period, resolution=self.resolution)
            elif ind_type == "RSI":
                period = ind_def.get("period", 14)
                self.indicators[ind_id] = self.RSI(self.symbol, period, resolution=self.resolution)
            elif ind_type == "MACD":
                fast = ind_def.get("fast_period", 12)
                slow = ind_def.get("slow_period", 26)
                signal = ind_def.get("signal_period", 9)
                self.indicators[ind_id] = self.MACD(self.symbol, fast, slow, signal, resolution=self.resolution)
            elif ind_type == "DC":
                period = ind_def.get("period", 20)
                self.indicators[ind_id] = self.DCH(self.symbol, period, resolution=self.resolution)
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

        # Track equity curve (every bar)
        self._track_equity(bar)

        # Check position state
        is_invested = self.Portfolio[self.symbol].Invested

        # If invested, run on_bar_invested hooks and check exits
        if is_invested:
            self._run_on_bar_invested(bar)
            self._evaluate_exits(bar)
        else:
            # Check entry
            self._evaluate_entry(bar)

    def _track_equity(self, bar):
        """Track portfolio equity for equity curve."""
        self.bar_count += 1
        equity = self.Portfolio.TotalPortfolioValue
        cash = self.Portfolio.Cash
        holdings = equity - cash

        # Update peak and drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = (self.peak_equity - equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Sample equity curve (every 60 bars ~= hourly for minute data)
        if self.bar_count == 1 or self.bar_count % 60 == 0:
            self.equity_curve.append({
                "time": str(self.Time),
                "equity": float(equity),
                "cash": float(cash),
                "holdings": float(holdings),
                "drawdown": float(drawdown),
            })

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

            # Track trade
            self.current_trade = {
                "symbol": str(self.symbol),
                "direction": "long",  # TODO: detect short
                "entry_time": str(self.Time),
                "entry_price": float(bar.Close),
                "quantity": float(self.Portfolio[self.symbol].Quantity),
            }

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
                # Complete trade tracking before executing action
                if self.current_trade:
                    entry_price = self.current_trade["entry_price"]
                    exit_price = float(bar.Close)
                    pnl = (exit_price - entry_price) * self.current_trade.get("quantity", 0)
                    pnl_pct = ((exit_price / entry_price) - 1) * 100 if entry_price > 0 else 0

                    self.current_trade["exit_time"] = str(self.Time)
                    self.current_trade["exit_price"] = exit_price
                    self.current_trade["pnl"] = pnl
                    self.current_trade["pnl_percent"] = pnl_pct
                    self.current_trade["exit_reason"] = exit_rule.get("id", "unknown")

                    self.trades.append(self.current_trade)
                    self.current_trade = None

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
            field = value_ref.get("field")  # Optional field for multi-value indicators
            # First check regular indicators
            ind = self.indicators.get(ind_id)
            if ind:
                # Handle MACD fields
                if field == "signal" and hasattr(ind, 'Signal'):
                    return ind.Signal.Current.Value
                elif field == "histogram" and hasattr(ind, 'Histogram'):
                    return ind.Histogram.Current.Value
                elif field == "macd" and hasattr(ind, 'Fast'):
                    # MACD line = Fast - Slow internally, but .Current.Value gives us the MACD line
                    return ind.Current.Value
                # Default: return the main value
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
        initial_cash = float(self.GetParameter("initial_cash") or 100000)

        # Calculate statistics
        total_return = ((portfolio_value / initial_cash) - 1) * 100
        winning_trades = [t for t in self.trades if t.get("pnl", 0) > 0]
        losing_trades = [t for t in self.trades if t.get("pnl", 0) <= 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0

        avg_win = sum(t.get("pnl_percent", 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.get("pnl_percent", 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Log summary
        self.Log(f"")
        self.Log(f"{'='*60}")
        self.Log(f"📊 BACKTEST RESULTS: {self.ir.get('strategy_name', 'Unknown')}")
        self.Log(f"{'='*60}")
        self.Log(f"")
        self.Log(f"PERFORMANCE")
        self.Log(f"  Initial Capital:    ${initial_cash:,.2f}")
        self.Log(f"  Final Equity:       ${portfolio_value:,.2f}")
        self.Log(f"  Total Return:       {total_return:+.2f}%")
        self.Log(f"  Max Drawdown:       {self.max_drawdown:.2f}%")
        self.Log(f"")
        self.Log(f"TRADES")
        self.Log(f"  Total Trades:       {len(self.trades)}")
        self.Log(f"  Winning Trades:     {len(winning_trades)}")
        self.Log(f"  Losing Trades:      {len(losing_trades)}")
        self.Log(f"  Win Rate:           {win_rate:.1f}%")
        self.Log(f"  Avg Win:            {avg_win:+.2f}%")
        self.Log(f"  Avg Loss:           {avg_loss:+.2f}%")
        self.Log(f"  Profit Factor:      {profit_factor:.2f}")
        self.Log(f"")

        # Log individual trades
        if self.trades:
            self.Log(f"TRADE LOG")
            for i, t in enumerate(self.trades):
                self.Log(f"  #{i+1}: {t['direction'].upper()} @ ${t['entry_price']:.2f} -> ${t.get('exit_price', 0):.2f} | P&L: {t.get('pnl_percent', 0):+.2f}% | Exit: {t.get('exit_reason', 'N/A')}")
            self.Log(f"")

        self.Log(f"{'='*60}")

        # Write structured output to JSON file
        import os
        output = {
            "strategy_id": self.ir.get("strategy_id", "unknown"),
            "strategy_name": self.ir.get("strategy_name", "Unknown"),
            "symbol": str(self.symbol),
            "initial_cash": initial_cash,
            "final_equity": float(portfolio_value),
            "total_return_pct": total_return,
            "max_drawdown_pct": self.max_drawdown,
            "statistics": {
                "total_trades": len(self.trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": win_rate,
                "avg_win_pct": avg_win,
                "avg_loss_pct": avg_loss,
                "profit_factor": profit_factor,
            },
            "trades": self.trades,
            "equity_curve": self.equity_curve,
        }

        # Write to data folder (same location as strategy_ir.json and debug.log)
        import os
        output_path = os.path.join(CustomCryptoData.DataFolder, "strategy_output.json")
        try:
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            self.Log(f"📁 Results written to: {output_path}")
        except Exception as e:
            self.Log(f"⚠️ Failed to write output: {e}")
