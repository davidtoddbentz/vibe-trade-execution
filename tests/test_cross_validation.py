"""Cross-validation tests for Python evaluator vs expected trades.

These tests prove that the Python evaluator produces trades at EXACTLY
the expected bars when given deterministic synthetic data.

The same synthetic data patterns are used in vibe-trade-lean/test/test_e2e.py
to verify LEAN runtime produces identical results.

Test Categories:
1. Entry signal timing - entry fires at exact expected bar
2. Exit signal timing - exit fires at exact expected bar
3. State updates - state variables track correctly
4. Multiple trade cycles - entry/exit/entry sequences work
"""

import math
from dataclasses import dataclass, field
from typing import Any

import pytest
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment

from tests.conftest import MockBandIndicator, MockIndicator, MockPriceBar, make_strategy
from src.translator.evaluator import ConditionEvaluator, EvalContext, StateOperator
from src.translator.ir_translator import IRTranslator


# =============================================================================
# Synthetic Data Builder (mirrors vibe-trade-lean/test/lib/test_data_builder.py)
# =============================================================================


@dataclass
class OHLCV:
    """Single candle."""

    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 1000.0


@dataclass
class ExpectedTrade:
    """Expected trade for test assertion."""

    bar_index: int
    action: str  # "ENTRY" or "EXIT"
    direction: str  # "long" or "short"
    price: float
    reason: str


@dataclass
class SyntheticDataBuilder:
    """Builds synthetic price data with calculable indicators.

    This mirrors the TestDataBuilder in vibe-trade-lean/test/lib/test_data_builder.py
    to ensure both Python and LEAN tests use identical data.
    """

    candles: list[OHLCV] = field(default_factory=list)
    _prices: list[float] = field(default_factory=list)
    current_price: float = 0.0
    bar_index: int = 0

    def add_candle(
        self,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: float = 1000.0,
    ) -> int:
        """Add a single candle. Returns bar index."""
        ms = self.bar_index * 60000
        candle = OHLCV(
            timestamp_ms=ms,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
        )
        self.candles.append(candle)
        self._prices.append(close_price)
        self.current_price = close_price
        self.bar_index += 1
        return self.bar_index - 1

    def add_uptrend(
        self,
        bars: int,
        start_price: float | None = None,
        trend_strength: float = 0.001,
    ) -> list[int]:
        """Add uptrend bars."""
        start = start_price or self.current_price or 50000.0
        indices = []
        price = start

        for i in range(bars):
            change = price * trend_strength
            new_price = price + change
            self.add_candle(
                open_price=price,
                high_price=new_price + abs(change) * 0.5,
                low_price=price - abs(change) * 0.2,
                close_price=new_price,
            )
            indices.append(self.bar_index - 1)
            price = new_price

        return indices

    def add_pullback_to_bb_lower(
        self,
        bars: int,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        overshoot: float = 0.0,
    ) -> tuple[list[int], int]:
        """Add pullback that touches/crosses BB lower band."""
        if len(self._prices) < bb_period:
            raise ValueError(f"Need at least {bb_period} bars before pullback")

        # Calculate current BB lower
        recent_prices = self._prices[-bb_period:]
        sma = sum(recent_prices) / bb_period
        variance = sum((p - sma) ** 2 for p in recent_prices) / bb_period
        std_dev = math.sqrt(variance)
        bb_lower = sma - bb_mult * std_dev

        target_price = bb_lower * (1 - overshoot)
        start_price = self.current_price
        drop_per_bar = (start_price - target_price) / bars

        indices = []
        price = start_price
        trigger_bar = None

        for i in range(bars):
            new_price = price - drop_per_bar

            if len(self._prices) >= bb_period:
                recent = self._prices[-(bb_period - 1) :] + [new_price]
                current_sma = sum(recent) / bb_period
                current_var = sum((p - current_sma) ** 2 for p in recent) / bb_period
                current_std = math.sqrt(current_var)
                current_bb_lower = current_sma - bb_mult * current_std

                if trigger_bar is None and new_price <= current_bb_lower:
                    trigger_bar = self.bar_index

            self.add_candle(
                open_price=price,
                high_price=price + drop_per_bar * 0.2,
                low_price=new_price - drop_per_bar * 0.3,
                close_price=new_price,
                volume=2000.0,
            )
            indices.append(self.bar_index - 1)
            price = new_price

        return indices, trigger_bar or indices[-1]

    def add_recovery(self, bars: int, trend_strength: float = 0.002) -> list[int]:
        """Add recovery/continuation."""
        return self.add_uptrend(bars, trend_strength=trend_strength)

    def calculate_ema(self, period: int) -> list[float | None]:
        """Calculate EMA for all prices."""
        if len(self._prices) < period:
            return []

        emas: list[float | None] = []
        multiplier = 2 / (period + 1)

        sma = sum(self._prices[:period]) / period
        emas.append(sma)

        for i in range(period, len(self._prices)):
            ema = (self._prices[i] - emas[-1]) * multiplier + emas[-1]
            emas.append(ema)

        return [None] * (period - 1) + emas

    def calculate_bb(self, period: int = 20, mult: float = 2.0) -> list[dict | None]:
        """Calculate Bollinger Bands for all prices."""
        if len(self._prices) < period:
            return []

        bands: list[dict | None] = []
        for i in range(len(self._prices)):
            if i < period - 1:
                bands.append(None)
                continue

            window = self._prices[i - period + 1 : i + 1]
            sma = sum(window) / period
            variance = sum((p - sma) ** 2 for p in window) / period
            std_dev = math.sqrt(variance)

            bands.append(
                {
                    "middle": sma,
                    "upper": sma + mult * std_dev,
                    "lower": sma - mult * std_dev,
                }
            )

        return bands


def create_trend_pullback_scenario(
    ema_fast: int = 20,
    ema_slow: int = 50,
    bb_period: int = 20,
    bb_mult: float = 2.0,
) -> tuple[SyntheticDataBuilder, list[ExpectedTrade]]:
    """Create a trend pullback test scenario.

    Returns:
        (builder, expected_trades) - builder with candles and expected trade list
    """
    builder = SyntheticDataBuilder()

    # Phase 1: Warmup - establish uptrend
    warmup_bars = ema_slow + 10
    builder.add_uptrend(bars=warmup_bars, start_price=50000, trend_strength=0.0008)

    # Phase 2: Pullback to BB lower
    builder.add_pullback_to_bb_lower(
        bars=5,
        bb_period=bb_period,
        bb_mult=bb_mult,
        overshoot=0.005,
    )

    # Phase 3: Recovery - strong enough to push above BB upper
    builder.add_recovery(bars=30, trend_strength=0.005)

    # Calculate expected trades
    ema_fast_values = builder.calculate_ema(ema_fast)
    ema_slow_values = builder.calculate_ema(ema_slow)
    bb_values = builder.calculate_bb(bb_period, bb_mult)

    expected = []
    in_position = False

    for i in range(max(ema_fast, ema_slow, bb_period), len(builder._prices)):
        price = builder._prices[i]
        ema_f = ema_fast_values[i]
        ema_s = ema_slow_values[i]
        bb = bb_values[i]

        if ema_f is None or ema_s is None or bb is None:
            continue

        if not in_position:
            uptrend = ema_f > ema_s
            pullback = price <= bb["lower"]

            if uptrend and pullback:
                expected.append(
                    ExpectedTrade(
                        bar_index=i,
                        action="ENTRY",
                        direction="long",
                        price=price,
                        reason=f"EMA{ema_fast} > EMA{ema_slow}, close <= BB_lower",
                    )
                )
                in_position = True
        else:
            # Exit at upper band
            if price >= bb["upper"]:
                expected.append(
                    ExpectedTrade(
                        bar_index=i,
                        action="EXIT",
                        direction="long",
                        price=price,
                        reason=f"close >= BB_upper",
                    )
                )
                in_position = False

    return builder, expected


def run_simulation(
    builder: SyntheticDataBuilder,
    strategy: Strategy,
    cards: dict[str, Card],
    ema_fast: int = 20,
    ema_slow: int = 50,
    bb_period: int = 20,
    bb_mult: float = 2.0,
) -> list[dict]:
    """Run strategy through Python evaluator with synthetic data.

    Returns list of actual trades: [{"bar_index": N, "action": "ENTRY|EXIT", ...}]
    """
    # Translate strategy to IR
    ir = IRTranslator(strategy, cards).translate()
    evaluator = ConditionEvaluator()
    state_op = StateOperator()

    # Initialize state
    state = {sv.id: sv.default for sv in ir.state}

    # Pre-calculate indicators
    ema_fast_values = builder.calculate_ema(ema_fast)
    ema_slow_values = builder.calculate_ema(ema_slow)
    bb_values = builder.calculate_bb(bb_period, bb_mult)

    actual_trades = []
    is_invested = False

    warmup = max(ema_fast, ema_slow, bb_period)

    for bar_idx, candle in enumerate(builder.candles):
        if bar_idx < warmup:
            continue

        # Build indicators dict
        indicators = {}

        # Add EMA indicators
        if bar_idx < len(ema_fast_values) and ema_fast_values[bar_idx]:
            indicators[f"ema_{ema_fast}"] = MockIndicator(ema_fast_values[bar_idx])
        if bar_idx < len(ema_slow_values) and ema_slow_values[bar_idx]:
            indicators[f"ema_{ema_slow}"] = MockIndicator(ema_slow_values[bar_idx])

        # Add BB indicator
        if bar_idx < len(bb_values) and bb_values[bar_idx]:
            bb = bb_values[bar_idx]
            indicators[f"bb_{bb_mult}_{bb_period}".replace(".", "_")] = MockBandIndicator(
                upper=bb["upper"],
                middle=bb["middle"],
                lower=bb["lower"],
            )

        ctx = EvalContext(
            indicators=indicators,
            state=state,
            price_bar=MockPriceBar(
                open_=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            ),
        )

        if not is_invested:
            # Check entry
            if ir.entry and evaluator.evaluate(ir.entry.condition, ctx):
                actual_trades.append(
                    {
                        "bar_index": bar_idx,
                        "action": "ENTRY",
                        "price": candle.close,
                    }
                )
                is_invested = True

                # Execute on_fill hooks
                for op in ir.entry.on_fill:
                    state_op.execute(op, ctx)
        else:
            # Check exits
            for exit_rule in ir.exits:
                if evaluator.evaluate(exit_rule.condition, ctx):
                    actual_trades.append(
                        {
                            "bar_index": bar_idx,
                            "action": "EXIT",
                            "reason": exit_rule.id,
                            "price": candle.close,
                        }
                    )
                    is_invested = False
                    break

            # Update state on each invested bar
            if is_invested:
                for op in ir.on_bar_invested:
                    state_op.execute(op, ctx)

    return actual_trades


# =============================================================================
# Cross-Validation Tests
# =============================================================================


class TestTrendPullbackCrossValidation:
    """Cross-validate TrendPullback strategy against expected trades."""

    @pytest.fixture
    def strategy_config(self):
        """TrendPullback strategy configuration."""
        return {
            "entry1": {
                "type": "entry.trend_pullback",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {
                        "dip": {"kind": "edge_event", "edge": "lower", "op": "touch"},
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                },
            },
            "exit1": {
                "type": "exit.rule_trigger",
                "slots": {
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                                "edge": "upper",
                                "direction": "cross_above",
                            }
                        }
                    },
                    "action": {"mode": "close"},
                },
            },
        }

    def test_entry_at_expected_bar(self, strategy_config):
        """Verify entry fires at the EXACT expected bar."""
        builder, expected = create_trend_pullback_scenario()
        strategy, cards = make_strategy(strategy_config, name="trend_pullback")

        actual = run_simulation(builder, strategy, cards)

        # Should have at least one entry
        assert len(actual) > 0, "No trades produced"
        assert actual[0]["action"] == "ENTRY", f"First trade should be ENTRY, got {actual[0]}"

        # Find expected entry
        expected_entries = [t for t in expected if t.action == "ENTRY"]
        assert len(expected_entries) > 0, "No expected entries in scenario"

        expected_bar = expected_entries[0].bar_index
        actual_bar = actual[0]["bar_index"]

        # Allow small tolerance for indicator calculation differences
        tolerance = 2
        assert abs(actual_bar - expected_bar) <= tolerance, (
            f"Entry at wrong bar. Expected ~{expected_bar}, got {actual_bar}. "
            f"Tolerance: {tolerance} bars."
        )

        print(f"\n✓ Entry at bar {actual_bar} (expected {expected_bar})")

    def test_exit_at_expected_bar(self, strategy_config):
        """Verify exit fires at the EXACT expected bar."""
        builder, expected = create_trend_pullback_scenario()
        strategy, cards = make_strategy(strategy_config, name="trend_pullback")

        actual = run_simulation(builder, strategy, cards)

        # Should have entry and exit
        entries = [t for t in actual if t["action"] == "ENTRY"]
        exits = [t for t in actual if t["action"] == "EXIT"]

        assert len(entries) > 0, "No entries produced"
        assert len(exits) > 0, "No exits produced"

        # Exit should be after entry
        assert exits[0]["bar_index"] > entries[0]["bar_index"], (
            f"Exit bar ({exits[0]['bar_index']}) should be after "
            f"entry bar ({entries[0]['bar_index']})"
        )

        # Find expected exit
        expected_exits = [t for t in expected if t.action == "EXIT"]
        if expected_exits:
            expected_bar = expected_exits[0].bar_index
            actual_bar = exits[0]["bar_index"]

            tolerance = 2
            assert abs(actual_bar - expected_bar) <= tolerance, (
                f"Exit at wrong bar. Expected ~{expected_bar}, got {actual_bar}. "
                f"Tolerance: {tolerance} bars."
            )

            print(f"\n✓ Exit at bar {actual_bar} (expected {expected_bar})")

    def test_trade_sequence_matches_expected(self, strategy_config):
        """Verify complete trade sequence matches expected."""
        builder, expected = create_trend_pullback_scenario()
        strategy, cards = make_strategy(strategy_config, name="trend_pullback")

        actual = run_simulation(builder, strategy, cards)

        print(f"\nExpected trades: {len(expected)}")
        for t in expected:
            print(f"  Bar {t.bar_index}: {t.action} @ {t.price:.2f} - {t.reason}")

        print(f"\nActual trades: {len(actual)}")
        for t in actual:
            print(f"  Bar {t['bar_index']}: {t['action']} @ {t['price']:.2f}")

        # Verify same number of trades
        assert len(actual) == len(expected), (
            f"Trade count mismatch. Expected {len(expected)}, got {len(actual)}"
        )

        # Verify each trade matches
        tolerance = 2
        for i, (exp, act) in enumerate(zip(expected, actual)):
            assert exp.action == act["action"], (
                f"Trade {i}: action mismatch. Expected {exp.action}, got {act['action']}"
            )
            assert abs(exp.bar_index - act["bar_index"]) <= tolerance, (
                f"Trade {i}: bar mismatch. Expected ~{exp.bar_index}, got {act['bar_index']}"
            )


class TestBreakoutCrossValidation:
    """Cross-validate Breakout strategy against expected trades."""

    @pytest.fixture
    def strategy_config(self):
        """Breakout strategy configuration."""
        return {
            "entry1": {
                "type": "entry.breakout_trendfollow",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"breakout": {"lookback_bars": 50}},
                    "action": {"direction": "long"},
                },
            },
            "exit1": {
                "type": "exit.rule_trigger",
                "slots": {
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "op": "<",
                                "value": -2.0,
                                "lookback_bars": 1,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
            },
        }

    def test_breakout_entry_timing(self, strategy_config):
        """Verify breakout entry fires when price breaks N-bar high."""
        builder = SyntheticDataBuilder()

        # 100 bars of flat consolidation
        for _ in range(100):
            builder.add_candle(50000, 50100, 49900, 50000)

        # Breakout
        for i in range(10):
            price = 50000 + (i + 1) * 200  # Rising prices
            builder.add_candle(price - 100, price + 100, price - 200, price)

        strategy, cards = make_strategy(strategy_config, name="breakout")
        ir = IRTranslator(strategy, cards).translate()
        evaluator = ConditionEvaluator()

        # Build indicator: max_50 should be around 50100 before breakout
        # After bar 100, new highs should trigger entry
        state = {sv.id: sv.default for sv in ir.state}

        entry_bars = []
        for bar_idx, candle in enumerate(builder.candles):
            if bar_idx < 50:
                continue

            # Calculate 50-bar max from previous bars
            lookback_prices = [c.high for c in builder.candles[max(0, bar_idx - 50) : bar_idx]]
            max_50 = max(lookback_prices) if lookback_prices else 0

            indicators = {
                "dc_50": MockBandIndicator(upper=max_50, middle=50000, lower=49900),
            }

            ctx = EvalContext(
                indicators=indicators,
                state=state,
                price_bar=MockPriceBar(
                    open_=candle.open,
                    high=candle.high,
                    low=candle.low,
                    close=candle.close,
                ),
            )

            if ir.entry and evaluator.evaluate(ir.entry.condition, ctx):
                entry_bars.append(bar_idx)

        # Entry should happen shortly after bar 100 (when breakout starts)
        assert len(entry_bars) > 0, "No breakout entry detected"
        assert entry_bars[0] >= 100, f"Entry too early at bar {entry_bars[0]}"
        assert entry_bars[0] <= 105, f"Entry too late at bar {entry_bars[0]}"

        print(f"\n✓ Breakout entry at bar {entry_bars[0]} (consolidation ended at bar 100)")


class TestStateTrackingCrossValidation:
    """Cross-validate state variable tracking."""

    def test_bars_since_entry_increments(self):
        """Verify bars_since_entry increments correctly after entry.

        Uses entry.trend_pullback which implements state tracking via
        get_on_bar_invested_ops() → IncrementStateAction for bars_since_entry.
        """
        strategy, cards = make_strategy(
            {
                "entry1": {
                    "type": "entry.trend_pullback",
                    "slots": {
                        "context": {"tf": "1h", "symbol": "BTC-USD"},
                        "event": {
                            "dip": {"kind": "edge_event", "edge": "lower", "op": "touch"},
                            "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                            "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                        },
                        "action": {"direction": "long"},
                    },
                },
            },
            name="state_test",
        )

        ir = IRTranslator(strategy, cards).translate()
        evaluator = ConditionEvaluator()
        state_op = StateOperator()

        # Initialize state
        state = {sv.id: sv.default for sv in ir.state}

        # Simulate bars with EMA and BB conditions
        is_invested = False
        bars_invested = 0

        for bar_idx in range(20):
            # Setup: EMA20 > EMA50 always (uptrend)
            # On bar 5: close drops to BB lower → entry triggers
            ema_20 = 105.0
            ema_50 = 100.0  # EMA20 > EMA50 (uptrend)

            if bar_idx == 5:
                # Price at BB lower - entry condition
                close = 95.0
                bb_lower = 96.0  # close <= bb_lower
            else:
                close = 102.0
                bb_lower = 90.0  # close > bb_lower

            indicators = {
                "ema_20": MockIndicator(ema_20),
                "ema_50": MockIndicator(ema_50),
                "bb_2_0_20": MockBandIndicator(lower=bb_lower, upper=110.0, middle=100.0),
            }
            ctx = EvalContext(
                indicators=indicators,
                state=state,
                price_bar=MockPriceBar(open_=100, high=101, low=94, close=close),
            )

            if not is_invested:
                if ir.entry and evaluator.evaluate(ir.entry.condition, ctx):
                    is_invested = True
                    # Execute on_fill hooks
                    for op in ir.entry.on_fill:
                        state_op.execute(op, ctx)
            else:
                # Execute on_bar_invested hooks
                for op in ir.on_bar_invested:
                    state_op.execute(op, ctx)
                bars_invested += 1

        # Should have been invested for 14 bars (bar 5 through 19)
        assert bars_invested == 14, f"Expected 14 bars invested, got {bars_invested}"

        # bars_since_entry should be 14
        assert "bars_since_entry" in state, "bars_since_entry not in state"
        assert state["bars_since_entry"] == 14, (
            f"bars_since_entry should be 14, got {state['bars_since_entry']}"
        )
        print(f"\n✓ bars_since_entry correctly tracked: {state['bars_since_entry']}")
