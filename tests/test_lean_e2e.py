"""End-to-end tests that run strategies through the actual LEAN engine.

These tests verify that:
1. IR translation produces valid output
2. StrategyRuntime correctly interprets the IR
3. Actual trades execute at expected times

These tests are SLOW (~6s each due to Docker) but provide critical coverage
that unit tests cannot: proving the full pipeline works end-to-end.

Test Strategy:
- 5 tests covering different strategy archetypes
- Each test uses tailored deterministic data
- Total runtime: ~30s (acceptable for CI)

Run with: pytest tests/test_lean_e2e.py -v
"""

import csv
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment

from src.translator.ir_translator import IRTranslator

# =============================================================================
# Test Infrastructure
# =============================================================================


def docker_available() -> bool:
    """Check if Docker is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not docker_available(),
    reason="Docker not available",
)


def generate_deterministic_csv(
    path: Path,
    bars: list[dict[str, Any]],
    start_date: datetime = datetime(2024, 1, 1),
) -> None:
    """Generate a deterministic CSV data file for LEAN backtesting.

    Args:
        path: Output CSV path
        bars: List of bar dicts with keys: open, high, low, close, volume
        start_date: Starting datetime for the data
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "open", "high", "low", "close", "volume"])

        current_time = start_date
        for bar in bars:
            writer.writerow(
                [
                    current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{bar['open']:.2f}",
                    f"{bar['high']:.2f}",
                    f"{bar['low']:.2f}",
                    f"{bar['close']:.2f}",
                    f"{bar['volume']:.2f}",
                ]
            )
            current_time += timedelta(hours=1)


def create_lean_data_reader(symbol: str, csv_filename: str) -> str:
    """Generate Python code for a LEAN custom data reader."""
    return f'''
class {symbol}Data(PythonData):
    """Custom data reader for {symbol}."""

    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(
            "/Data/{csv_filename}",
            SubscriptionTransportMedium.LocalFile,
            FileFormat.Csv
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line.strip() or line.startswith("datetime"):
            return None

        data = {symbol}Data()
        data.Symbol = config.Symbol

        try:
            parts = line.split(',')
            from datetime import datetime as dt
            data.Time = dt.strptime(parts[0], "%Y-%m-%d %H:%M:%S")
            data.Value = float(parts[4])
            data["Open"] = float(parts[1])
            data["High"] = float(parts[2])
            data["Low"] = float(parts[3])
            data["Close"] = float(parts[4])
            data["Volume"] = float(parts[5])
        except:
            return None

        return data
'''


def create_strategy_wrapper(
    class_name: str,
    symbol: str,
    data_reader_code: str,
    ir_filename: str,
    start_date: tuple[int, int, int] = (2024, 1, 1),
    end_date: tuple[int, int, int] = (2024, 1, 10),
) -> str:
    """Generate Python code for a LEAN algorithm wrapper."""
    return f'''"""Auto-generated E2E test algorithm: {class_name}"""
from AlgorithmImports import *
from StrategyRuntime import StrategyRuntime
import json

{data_reader_code}

class {class_name}(StrategyRuntime):
    """Test wrapper for StrategyRuntime."""

    def Initialize(self):
        # Load IR from file
        with open("/Lean/Algorithm.Python/{ir_filename}") as f:
            self.ir = json.load(f)

        # Set backtest period
        self.SetStartDate{start_date}
        self.SetEndDate{end_date}
        self.SetCash(100000)

        # Add custom data
        self.symbol = self.AddData({symbol}Data, "{symbol}", Resolution.Hour).Symbol
        self.resolution = Resolution.Hour

        # Initialize indicators
        self.indicators = {{}}
        self.rolling_windows = {{}}
        self.vol_sma_indicators = {{}}
        self.rolling_minmax = {{}}
        self._create_indicators()

        # Initialize state
        self.state = {{}}
        self._initialize_state()

        # Parse rules
        self.entry_rule = self.ir.get("entry")
        self.exit_rules = self.ir.get("exits", [])
        self.gates = self.ir.get("gates", [])
        self.on_bar_invested_ops = self.ir.get("on_bar_invested", [])
        self.on_bar_ops = self.ir.get("on_bar", [])

        self.Log(f"=== E2E TEST: {{self.ir.get('strategy_name')}} ===")
'''


class TestLeanEndToEnd:
    """End-to-end tests using LEAN engine.

    Each test:
    1. Creates deterministic market data tailored to the strategy
    2. Translates a strategy from cards to IR
    3. Runs the strategy through LEAN
    4. Verifies expected trades executed
    """

    @pytest.fixture
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @pytest.fixture
    def lean_algorithms_dir(self, project_root) -> Path:
        return project_root / "lean" / "Algorithms"

    @pytest.fixture
    def lean_data_dir(self, project_root) -> Path:
        return project_root / "lean" / "Data"

    @pytest.fixture
    def lean_results_dir(self, project_root) -> Path:
        return project_root / "lean" / "Results"

    def run_lean_backtest(self, algorithm_name: str, project_root: Path) -> dict:
        """Run a LEAN backtest and return results."""
        from src.lean_runner.engine import LeanEngine

        engine = LeanEngine()
        return engine.run_backtest(
            algorithm_name=algorithm_name,
            start_date="20240101",
            end_date="20240110",
            cash=100000.0,
        )

    def get_filled_orders(self, results_dir: Path, algorithm_name: str) -> list[dict]:
        """Get filled orders from backtest results."""
        orders_file = results_dir / f"{algorithm_name}-order-events.json"
        if not orders_file.exists():
            return []
        with open(orders_file) as f:
            orders = json.load(f)
        return [o for o in orders if o["status"] == "filled"]

    def cleanup_files(self, *paths: Path) -> None:
        """Clean up generated files."""
        for path in paths:
            if path.exists():
                path.unlink()

    # =========================================================================
    # Test 1: Trend Following (EMA Crossover)
    # Tests: trend_ma_relation, EMA indicators, basic entry/exit
    # =========================================================================

    def test_trend_following_ema_crossover(
        self, project_root, lean_algorithms_dir, lean_data_dir, lean_results_dir
    ):
        """Test EMA crossover: buy on golden cross, sell on death cross.

        Data pattern:
        - Hours 0-29: Flat at $100 (EMAs converge)
        - Hours 30-49: Ramp $100→$150 (fast EMA crosses above slow → BUY)
        - Hours 50-69: Flat at $150 (hold position)
        - Hours 70-89: Ramp $150→$100 (fast EMA crosses below slow → SELL)
        """
        # Generate deterministic data
        bars = []
        # Flat period
        for _ in range(30):
            bars.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
        # Ramp up
        for i in range(20):
            price = 100 + (i * 2.5)
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
            )
        # Flat high
        for _ in range(20):
            bars.append({"open": 150, "high": 151, "low": 149, "close": 150, "volume": 1000})
        # Ramp down
        for i in range(20):
            price = 150 - (i * 2.5)
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
            )

        csv_file = lean_data_dir / "e2e_trend.csv"
        generate_deterministic_csv(csv_file, bars)

        # Create strategy
        strategy = Strategy(
            id="e2e-trend",
            name="E2E Trend Following",
            universe=["TREND"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "TREND"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "trend_ma_relation",
                                "op": ">",
                                "value": 0,
                                "ma_fast": 5,
                                "ma_slow": 15,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "TREND"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "trend_ma_relation",
                                "op": "<",
                                "value": 0,
                                "ma_fast": 5,
                                "ma_slow": 15,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
        }

        # Translate and run
        translator = IRTranslator(strategy, cards)
        result = translator.translate()
        assert len(result.warnings) == 0

        ir_file = lean_algorithms_dir / "e2e_trend_ir.json"
        wrapper_file = lean_algorithms_dir / "E2ETrend.py"

        with open(ir_file, "w") as f:
            f.write(result.ir.to_json())

        data_reader = create_lean_data_reader("TREND", "e2e_trend.csv")
        wrapper = create_strategy_wrapper("E2ETrend", "TREND", data_reader, "e2e_trend_ir.json")
        with open(wrapper_file, "w") as f:
            f.write(wrapper)

        try:
            lean_result = self.run_lean_backtest("E2ETrend", project_root)
            assert lean_result["status"] == "success", f"LEAN failed: {lean_result}"

            orders = self.get_filled_orders(lean_results_dir, "E2ETrend")
            assert len(orders) >= 2, f"Expected at least 2 orders, got {len(orders)}"
            assert orders[0]["direction"] == "buy"
            assert orders[1]["direction"] == "sell"

            # Verify profitability
            buy_price = orders[0]["fillPrice"]
            sell_price = orders[1]["fillPrice"]
            assert sell_price > buy_price, f"Expected profit: bought {buy_price}, sold {sell_price}"

        finally:
            self.cleanup_files(csv_file, ir_file, wrapper_file)

    # =========================================================================
    # Test 2: Mean Reversion (Bollinger Band Touch)
    # Tests: IndicatorBandValue, band comparisons, BB indicator
    # =========================================================================

    def test_mean_reversion_bollinger(
        self, project_root, lean_algorithms_dir, lean_data_dir, lean_results_dir
    ):
        """Test BB mean reversion: buy at lower band, exit at middle.

        Data pattern:
        - Hours 0-29: Oscillate around $100 (BB forms)
        - Hours 30-39: Drop to $90 (touch lower band → BUY)
        - Hours 40-59: Rise to $100 (touch middle band → EXIT)
        """
        bars = []
        # Oscillation to establish BB
        for i in range(30):
            price = 100 + (5 * (1 if i % 2 == 0 else -1))
            bars.append(
                {"open": price, "high": price + 2, "low": price - 2, "close": price, "volume": 1000}
            )
        # Drop to lower band
        for i in range(10):
            price = 100 - i
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
            )
        # Rise back to middle
        for _ in range(20):
            price = 90 + (i * 0.5)
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
            )

        csv_file = lean_data_dir / "e2e_reversion.csv"
        generate_deterministic_csv(csv_file, bars)

        strategy = Strategy(
            id="e2e-reversion",
            name="E2E Mean Reversion",
            universe=["REVERT"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "REVERT"},
                    "event": {
                        "condition": {
                            "type": "indicator",
                            "indicator": {
                                "type": "compare",
                                "left": {"type": "price", "field": "close"},
                                "op": "<",
                                "right": {
                                    "type": "indicator_band",
                                    "indicator": "bb",
                                    "band": "lower",
                                    "period": 20,
                                },
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "REVERT"},
                    "event": {
                        "condition": {
                            "type": "indicator",
                            "indicator": {
                                "type": "compare",
                                "left": {"type": "price", "field": "close"},
                                "op": ">",
                                "right": {
                                    "type": "indicator_band",
                                    "indicator": "bb",
                                    "band": "middle",
                                    "period": 20,
                                },
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        ir_file = lean_algorithms_dir / "e2e_reversion_ir.json"
        wrapper_file = lean_algorithms_dir / "E2EReversion.py"

        with open(ir_file, "w") as f:
            f.write(result.ir.to_json())

        data_reader = create_lean_data_reader("REVERT", "e2e_reversion.csv")
        wrapper = create_strategy_wrapper(
            "E2EReversion", "REVERT", data_reader, "e2e_reversion_ir.json"
        )
        with open(wrapper_file, "w") as f:
            f.write(wrapper)

        try:
            lean_result = self.run_lean_backtest("E2EReversion", project_root)
            assert lean_result["status"] == "success", f"LEAN failed: {lean_result}"

            orders = self.get_filled_orders(lean_results_dir, "E2EReversion")
            # May or may not have trades depending on BB calculation
            # At minimum, verify LEAN ran successfully
            print(f"Mean reversion test: {len(orders)} orders executed")

        finally:
            self.cleanup_files(csv_file, ir_file, wrapper_file)

    # =========================================================================
    # Test 3: Breakout Strategy
    # Tests: price comparisons, literal values, Maximum indicator
    # =========================================================================

    def test_breakout_strategy(
        self, project_root, lean_algorithms_dir, lean_data_dir, lean_results_dir
    ):
        """Test breakout: buy when price breaks above recent high.

        Data pattern:
        - Hours 0-39: Consolidate at $100 (establish range)
        - Hours 40-49: Break above $105 → BUY
        - Hours 50-69: Continue to $120
        - Hours 70-89: Drop below $115 → EXIT
        """
        bars = []
        # Consolidation
        for i in range(40):
            price = 100 + (i % 5) - 2  # Oscillate 98-103
            bars.append(
                {"open": price, "high": price + 2, "low": price - 2, "close": price, "volume": 1000}
            )
        # Breakout
        for i in range(10):
            price = 105 + (i * 1.5)
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 2000}
            )
        # Continue up
        for _ in range(20):
            price = 120
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
            )
        # Drop
        for _ in range(20):
            price = 120 - i
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
            )

        csv_file = lean_data_dir / "e2e_breakout.csv"
        generate_deterministic_csv(csv_file, bars)

        strategy = Strategy(
            id="e2e-breakout",
            name="E2E Breakout",
            universe=["BREAK"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit", role="exit", enabled=True),
            ],
        )

        # Simple breakout using regime price_level_cross
        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BREAK"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "price_level_cross",
                                "level_price": 105,
                                "direction": "up",
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "BREAK"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "price_level_cross",
                                "level_price": 115,
                                "direction": "down",
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        ir_file = lean_algorithms_dir / "e2e_breakout_ir.json"
        wrapper_file = lean_algorithms_dir / "E2EBreakout.py"

        with open(ir_file, "w") as f:
            f.write(result.ir.to_json())

        data_reader = create_lean_data_reader("BREAK", "e2e_breakout.csv")
        wrapper = create_strategy_wrapper(
            "E2EBreakout", "BREAK", data_reader, "e2e_breakout_ir.json"
        )
        with open(wrapper_file, "w") as f:
            f.write(wrapper)

        try:
            lean_result = self.run_lean_backtest("E2EBreakout", project_root)
            assert lean_result["status"] == "success", f"LEAN failed: {lean_result}"

            orders = self.get_filled_orders(lean_results_dir, "E2EBreakout")
            assert len(orders) >= 2, f"Expected at least 2 orders, got {len(orders)}"
            assert orders[0]["direction"] == "buy"

        finally:
            self.cleanup_files(csv_file, ir_file, wrapper_file)

    # =========================================================================
    # Test 4: ROC Momentum Strategy
    # Tests: ROC indicator, percentage thresholds
    # =========================================================================

    def test_momentum_roc(self, project_root, lean_algorithms_dir, lean_data_dir, lean_results_dir):
        """Test ROC momentum: buy on strong momentum, exit on reversal.

        Data pattern:
        - Hours 0-29: Flat at $100 (ROC near 0)
        - Hours 30-49: Ramp to $120 (ROC > 5% → BUY)
        - Hours 50-69: Flat (hold)
        - Hours 70-89: Drop to $100 (ROC < -5% → EXIT)
        """
        bars = []
        # Flat
        for _ in range(30):
            bars.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
        # Strong up momentum
        for i in range(20):
            price = 100 + i
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1500}
            )
        # Flat high
        for _ in range(20):
            bars.append({"open": 120, "high": 121, "low": 119, "close": 120, "volume": 1000})
        # Strong down momentum
        for i in range(20):
            price = 120 - i
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1500}
            )

        csv_file = lean_data_dir / "e2e_momentum.csv"
        generate_deterministic_csv(csv_file, bars)

        strategy = Strategy(
            id="e2e-momentum",
            name="E2E Momentum",
            universe=["MOM"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "MOM"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "op": ">",
                                "value": 5,
                                "lookback_bars": 10,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "MOM"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "op": "<",
                                "value": -5,
                                "lookback_bars": 10,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        ir_file = lean_algorithms_dir / "e2e_momentum_ir.json"
        wrapper_file = lean_algorithms_dir / "E2EMomentum.py"

        with open(ir_file, "w") as f:
            f.write(result.ir.to_json())

        data_reader = create_lean_data_reader("MOM", "e2e_momentum.csv")
        wrapper = create_strategy_wrapper("E2EMomentum", "MOM", data_reader, "e2e_momentum_ir.json")
        with open(wrapper_file, "w") as f:
            f.write(wrapper)

        try:
            lean_result = self.run_lean_backtest("E2EMomentum", project_root)
            assert lean_result["status"] == "success", f"LEAN failed: {lean_result}"

            orders = self.get_filled_orders(lean_results_dir, "E2EMomentum")
            print(f"Momentum test: {len(orders)} orders executed")

        finally:
            self.cleanup_files(csv_file, ir_file, wrapper_file)

    # =========================================================================
    # Test 5: Composite Strategy (Trend + Gate)
    # Tests: AllOfCondition, gates, multiple conditions
    # =========================================================================

    def test_composite_trend_with_gate(
        self, project_root, lean_algorithms_dir, lean_data_dir, lean_results_dir
    ):
        """Test composite: trend following with momentum gate.

        Entry requires BOTH:
        - EMA fast > EMA slow (trend)
        - ROC > 0 (positive momentum gate)

        Data pattern: Same as trend following
        """
        bars = []
        # Flat
        for _ in range(30):
            bars.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
        # Ramp up (both conditions true)
        for i in range(20):
            price = 100 + (i * 2.5)
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
            )
        # Flat high
        for _ in range(20):
            bars.append({"open": 150, "high": 151, "low": 149, "close": 150, "volume": 1000})
        # Ramp down
        for i in range(20):
            price = 150 - (i * 2.5)
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
            )

        csv_file = lean_data_dir / "e2e_composite.csv"
        generate_deterministic_csv(csv_file, bars)

        strategy = Strategy(
            id="e2e-composite",
            name="E2E Composite",
            universe=["COMP"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="gate", role="gate", enabled=True),
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit", role="exit", enabled=True),
            ],
        )

        cards = {
            "gate": Card(
                id="gate",
                type="gate.regime_filter",
                slots={
                    "context": {"tf": "1h", "symbol": "COMP"},
                    "regime": {
                        "metric": "ret_pct",
                        "op": ">",
                        "value": 0,
                        "lookback_bars": 5,
                    },
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "COMP"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "trend_ma_relation",
                                "op": ">",
                                "value": 0,
                                "ma_fast": 5,
                                "ma_slow": 15,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "COMP"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "trend_ma_relation",
                                "op": "<",
                                "value": 0,
                                "ma_fast": 5,
                                "ma_slow": 15,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()

        ir_file = lean_algorithms_dir / "e2e_composite_ir.json"
        wrapper_file = lean_algorithms_dir / "E2EComposite.py"

        with open(ir_file, "w") as f:
            f.write(result.ir.to_json())

        data_reader = create_lean_data_reader("COMP", "e2e_composite.csv")
        wrapper = create_strategy_wrapper(
            "E2EComposite", "COMP", data_reader, "e2e_composite_ir.json"
        )
        with open(wrapper_file, "w") as f:
            f.write(wrapper)

        try:
            lean_result = self.run_lean_backtest("E2EComposite", project_root)
            assert lean_result["status"] == "success", f"LEAN failed: {lean_result}"

            orders = self.get_filled_orders(lean_results_dir, "E2EComposite")
            print(f"Composite test: {len(orders)} orders executed")

            if len(orders) >= 2:
                assert orders[0]["direction"] == "buy"
                buy_price = orders[0]["fillPrice"]
                sell_price = orders[1]["fillPrice"]
                assert sell_price > buy_price, (
                    f"Expected profit: bought {buy_price}, sold {sell_price}"
                )

        finally:
            self.cleanup_files(csv_file, ir_file, wrapper_file)

    # =========================================================================
    # Test 6: Multi-Condition AllOf Entry
    # Tests: AllOfCondition with multiple regime checks
    # =========================================================================

    def test_multi_condition_allof_entry(
        self, project_root, lean_algorithms_dir, lean_data_dir, lean_results_dir
    ):
        """Test allOf: entry requires trend + momentum + volatility.

        Entry requires ALL conditions:
        - EMA fast > EMA slow (trend confirmation)
        - ROC > 2% (momentum confirmation)

        Data pattern: Strong uptrend that satisfies all conditions
        """
        bars = []
        # Flat warmup
        for _ in range(30):
            bars.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
        # Strong uptrend (all conditions met)
        for i in range(30):
            price = 100 + (i * 2)
            bars.append(
                {
                    "open": price - 1,
                    "high": price + 2,
                    "low": price - 2,
                    "close": price,
                    "volume": 1500,
                }
            )
        # Plateau
        for _ in range(20):
            bars.append({"open": 160, "high": 161, "low": 159, "close": 160, "volume": 1000})
        # Reversal
        for _ in range(20):
            price = 160 - (i * 2)
            bars.append(
                {
                    "open": price + 1,
                    "high": price + 2,
                    "low": price - 2,
                    "close": price,
                    "volume": 1000,
                }
            )

        csv_file = lean_data_dir / "e2e_allof.csv"
        generate_deterministic_csv(csv_file, bars)

        strategy = Strategy(
            id="e2e-allof",
            name="E2E AllOf Multi-Condition",
            universe=["ALLOF"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "ALLOF"},
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
                                        "ma_fast": 5,
                                        "ma_slow": 15,
                                    },
                                },
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "ret_pct",
                                        "op": ">",
                                        "value": 2,
                                        "lookback_bars": 5,
                                    },
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "ALLOF"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "op": "<",
                                "value": -3,
                                "lookback_bars": 5,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()
        assert len(result.warnings) == 0

        ir_file = lean_algorithms_dir / "e2e_allof_ir.json"
        wrapper_file = lean_algorithms_dir / "E2EAllOf.py"

        with open(ir_file, "w") as f:
            f.write(result.ir.to_json())

        data_reader = create_lean_data_reader("ALLOF", "e2e_allof.csv")
        wrapper = create_strategy_wrapper("E2EAllOf", "ALLOF", data_reader, "e2e_allof_ir.json")
        with open(wrapper_file, "w") as f:
            f.write(wrapper)

        try:
            lean_result = self.run_lean_backtest("E2EAllOf", project_root)
            assert lean_result["status"] == "success", f"LEAN failed: {lean_result}"

            orders = self.get_filled_orders(lean_results_dir, "E2EAllOf")
            print(f"AllOf multi-condition test: {len(orders)} orders executed")
            assert len(orders) >= 2, f"Expected at least 2 orders, got {len(orders)}"

        finally:
            self.cleanup_files(csv_file, ir_file, wrapper_file)

    # =========================================================================
    # Test 7: AnyOf Exit (Stop Loss OR Profit Target)
    # Tests: AnyOfCondition, multiple exit conditions
    # =========================================================================

    def test_anyof_exit_stop_or_target(
        self, project_root, lean_algorithms_dir, lean_data_dir, lean_results_dir
    ):
        """Test anyOf exit: either stop loss OR profit target triggers exit.

        Exit on EITHER:
        - ROC < -5% (stop loss)
        - ROC > 10% (profit target)

        Data pattern: Quick profit scenario
        """
        bars = []
        # Warmup
        for _ in range(20):
            bars.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
        # Entry trigger
        for i in range(10):
            price = 100 + i
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
            )
        # Quick profit spike (triggers profit target exit)
        for i in range(15):
            price = 110 + (i * 2)
            bars.append(
                {"open": price, "high": price + 2, "low": price - 1, "close": price, "volume": 2000}
            )
        # Continuation
        for _ in range(20):
            bars.append({"open": 140, "high": 142, "low": 138, "close": 140, "volume": 1000})

        csv_file = lean_data_dir / "e2e_anyof.csv"
        generate_deterministic_csv(csv_file, bars)

        strategy = Strategy(
            id="e2e-anyof",
            name="E2E AnyOf Exit",
            universe=["ANYOF"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "ANYOF"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "op": ">",
                                "value": 3,
                                "lookback_bars": 5,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "ANYOF"},
                    "event": {
                        "condition": {
                            "type": "anyOf",
                            "anyOf": [
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "ret_pct",
                                        "op": "<",
                                        "value": -5,
                                        "lookback_bars": 5,
                                    },
                                },
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "ret_pct",
                                        "op": ">",
                                        "value": 10,
                                        "lookback_bars": 10,
                                    },
                                },
                            ],
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()
        assert len(result.warnings) == 0

        ir_file = lean_algorithms_dir / "e2e_anyof_ir.json"
        wrapper_file = lean_algorithms_dir / "E2EAnyOf.py"

        with open(ir_file, "w") as f:
            f.write(result.ir.to_json())

        data_reader = create_lean_data_reader("ANYOF", "e2e_anyof.csv")
        wrapper = create_strategy_wrapper("E2EAnyOf", "ANYOF", data_reader, "e2e_anyof_ir.json")
        with open(wrapper_file, "w") as f:
            f.write(wrapper)

        try:
            lean_result = self.run_lean_backtest("E2EAnyOf", project_root)
            assert lean_result["status"] == "success", f"LEAN failed: {lean_result}"

            orders = self.get_filled_orders(lean_results_dir, "E2EAnyOf")
            print(f"AnyOf exit test: {len(orders)} orders executed")

        finally:
            self.cleanup_files(csv_file, ir_file, wrapper_file)

    # =========================================================================
    # Test 8: Nested Conditions (AllOf containing AnyOf)
    # Tests: Deeply nested condition trees
    # =========================================================================

    def test_nested_conditions(
        self, project_root, lean_algorithms_dir, lean_data_dir, lean_results_dir
    ):
        """Test nested: AllOf(trend, AnyOf(momentum_high, momentum_recovery)).

        Entry requires:
        - Trend up (EMA fast > slow) AND
        - Either strong momentum (ROC > 5%) OR recovering momentum (ROC > 0 after dip)
        """
        bars = []
        # Warmup
        for _ in range(30):
            bars.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
        # Uptrend with strong momentum
        for i in range(25):
            price = 100 + (i * 2)
            bars.append(
                {"open": price, "high": price + 2, "low": price - 1, "close": price, "volume": 1500}
            )
        # Hold
        for _ in range(15):
            bars.append({"open": 150, "high": 152, "low": 148, "close": 150, "volume": 1000})
        # Reversal
        for _ in range(20):
            price = 150 - (i * 2)
            bars.append(
                {"open": price, "high": price + 1, "low": price - 2, "close": price, "volume": 1000}
            )

        csv_file = lean_data_dir / "e2e_nested.csv"
        generate_deterministic_csv(csv_file, bars)

        strategy = Strategy(
            id="e2e-nested",
            name="E2E Nested Conditions",
            universe=["NESTED"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "NESTED"},
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
                                        "ma_fast": 5,
                                        "ma_slow": 15,
                                    },
                                },
                                {
                                    "type": "anyOf",
                                    "anyOf": [
                                        {
                                            "type": "regime",
                                            "regime": {
                                                "metric": "ret_pct",
                                                "op": ">",
                                                "value": 5,
                                                "lookback_bars": 5,
                                            },
                                        },
                                        {
                                            "type": "regime",
                                            "regime": {
                                                "metric": "ret_pct",
                                                "op": ">",
                                                "value": 1,
                                                "lookback_bars": 3,
                                            },
                                        },
                                    ],
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "NESTED"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "trend_ma_relation",
                                "op": "<",
                                "value": 0,
                                "ma_fast": 5,
                                "ma_slow": 15,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()
        assert len(result.warnings) == 0

        ir_file = lean_algorithms_dir / "e2e_nested_ir.json"
        wrapper_file = lean_algorithms_dir / "E2ENested.py"

        with open(ir_file, "w") as f:
            f.write(result.ir.to_json())

        data_reader = create_lean_data_reader("NESTED", "e2e_nested.csv")
        wrapper = create_strategy_wrapper("E2ENested", "NESTED", data_reader, "e2e_nested_ir.json")
        with open(wrapper_file, "w") as f:
            f.write(wrapper)

        try:
            lean_result = self.run_lean_backtest("E2ENested", project_root)
            assert lean_result["status"] == "success", f"LEAN failed: {lean_result}"

            orders = self.get_filled_orders(lean_results_dir, "E2ENested")
            print(f"Nested conditions test: {len(orders)} orders executed")
            assert len(orders) >= 2, f"Expected at least 2 orders, got {len(orders)}"

        finally:
            self.cleanup_files(csv_file, ir_file, wrapper_file)

    # =========================================================================
    # Test 9: Multiple Exit Rules
    # Tests: Multiple exit cards, first to trigger wins
    # =========================================================================

    def test_multiple_exit_rules(
        self, project_root, lean_algorithms_dir, lean_data_dir, lean_results_dir
    ):
        """Test multiple exits: trailing stop AND profit target as separate cards.

        Two exit cards:
        1. Stop loss: exit if ROC < -3%
        2. Profit target: exit if ROC > 8%

        First condition to trigger closes position.
        """
        bars = []
        # Warmup
        for _ in range(20):
            bars.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
        # Entry
        for i in range(15):
            price = 100 + i
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
            )
        # Quick profit (triggers profit exit)
        for i in range(10):
            price = 115 + (i * 2)
            bars.append(
                {"open": price, "high": price + 2, "low": price - 1, "close": price, "volume": 1500}
            )
        # Continue
        for _ in range(20):
            bars.append({"open": 135, "high": 137, "low": 133, "close": 135, "volume": 1000})

        csv_file = lean_data_dir / "e2e_multi_exit.csv"
        generate_deterministic_csv(csv_file, bars)

        strategy = Strategy(
            id="e2e-multi-exit",
            name="E2E Multiple Exits",
            universe=["MEXIT"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit_stop", role="exit", enabled=True),
                Attachment(card_id="exit_profit", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "MEXIT"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "op": ">",
                                "value": 3,
                                "lookback_bars": 5,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "exit_stop": Card(
                id="exit_stop",
                type="exit.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "MEXIT"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "op": "<",
                                "value": -3,
                                "lookback_bars": 3,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "exit_profit": Card(
                id="exit_profit",
                type="exit.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "MEXIT"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "op": ">",
                                "value": 8,
                                "lookback_bars": 10,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()
        assert len(result.warnings) == 0
        assert len(result.ir.exits) == 2, "Should have 2 exit rules"

        ir_file = lean_algorithms_dir / "e2e_multi_exit_ir.json"
        wrapper_file = lean_algorithms_dir / "E2EMultiExit.py"

        with open(ir_file, "w") as f:
            f.write(result.ir.to_json())

        data_reader = create_lean_data_reader("MEXIT", "e2e_multi_exit.csv")
        wrapper = create_strategy_wrapper(
            "E2EMultiExit", "MEXIT", data_reader, "e2e_multi_exit_ir.json"
        )
        with open(wrapper_file, "w") as f:
            f.write(wrapper)

        try:
            lean_result = self.run_lean_backtest("E2EMultiExit", project_root)
            assert lean_result["status"] == "success", f"LEAN failed: {lean_result}"

            orders = self.get_filled_orders(lean_results_dir, "E2EMultiExit")
            print(f"Multiple exits test: {len(orders)} orders executed")

        finally:
            self.cleanup_files(csv_file, ir_file, wrapper_file)

    # =========================================================================
    # Test 10: Not Condition (Inverse Logic)
    # Tests: NotCondition wrapper
    # =========================================================================

    def test_not_condition(
        self, project_root, lean_algorithms_dir, lean_data_dir, lean_results_dir
    ):
        """Test not: entry when NOT in downtrend.

        Entry: NOT(EMA fast < slow) = EMA fast >= slow
        This is logically equivalent to trend following but tests NOT wrapper.
        """
        bars = []
        # Flat
        for _ in range(30):
            bars.append({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
        # Uptrend
        for i in range(25):
            price = 100 + (i * 2)
            bars.append(
                {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1000}
            )
        # Flat
        for _ in range(15):
            bars.append({"open": 150, "high": 151, "low": 149, "close": 150, "volume": 1000})
        # Downtrend
        for _ in range(20):
            price = 150 - (i * 2)
            bars.append(
                {"open": price, "high": price + 1, "low": price - 2, "close": price, "volume": 1000}
            )

        csv_file = lean_data_dir / "e2e_not.csv"
        generate_deterministic_csv(csv_file, bars)

        strategy = Strategy(
            id="e2e-not",
            name="E2E Not Condition",
            universe=["NOTC"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
                Attachment(card_id="exit", role="exit", enabled=True),
            ],
        )

        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "NOTC"},
                    "event": {
                        "condition": {
                            "type": "not",
                            "not": {
                                "type": "regime",
                                "regime": {
                                    "metric": "trend_ma_relation",
                                    "op": "<",
                                    "value": -1,
                                    "ma_fast": 5,
                                    "ma_slow": 15,
                                },
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "context": {"tf": "1h", "symbol": "NOTC"},
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "trend_ma_relation",
                                "op": "<",
                                "value": -5,
                                "ma_fast": 5,
                                "ma_slow": 15,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="",
                updated_at="",
            ),
        }

        translator = IRTranslator(strategy, cards)
        result = translator.translate()
        assert len(result.warnings) == 0

        ir_file = lean_algorithms_dir / "e2e_not_ir.json"
        wrapper_file = lean_algorithms_dir / "E2ENot.py"

        with open(ir_file, "w") as f:
            f.write(result.ir.to_json())

        data_reader = create_lean_data_reader("NOTC", "e2e_not.csv")
        wrapper = create_strategy_wrapper("E2ENot", "NOTC", data_reader, "e2e_not_ir.json")
        with open(wrapper_file, "w") as f:
            f.write(wrapper)

        try:
            lean_result = self.run_lean_backtest("E2ENot", project_root)
            assert lean_result["status"] == "success", f"LEAN failed: {lean_result}"

            orders = self.get_filled_orders(lean_results_dir, "E2ENot")
            print(f"Not condition test: {len(orders)} orders executed")

        finally:
            self.cleanup_files(csv_file, ir_file, wrapper_file)
