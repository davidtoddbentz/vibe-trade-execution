"""Strategy simulation tests.

These tests verify semantic correctness by running translated IR through
the evaluator with deterministic price/indicator sequences and asserting
that signals fire at exactly the expected bars.

Each test scenario defines:
- A sequence of bars with indicator values
- Expected entry/exit signals at specific bar indices
"""

import pytest
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment

from src.translator.ir import (
    AllOfCondition,
    CompareCondition,
    CompareOp,
    Condition,
    EntryRule,
    ExpressionValue,
    IndicatorProperty,
    IndicatorPropertyValue,
    IndicatorValue,
    LiteralValue,
    PriceField,
    PriceValue,
    Resolution,
    SetHoldingsAction,
    StrategyIR,
    TimeValue,
    VolumeValue,
)
from src.translator.ir_translator import IRTranslator

# Import shared test infrastructure from conftest
from tests.conftest import (
    BarData,
    MockBar,
    MockBandIndicator,
    MockIndicator,
    Signal,
    SimulationScenario,
    assert_signals_match,
    run_simulation,
)


# =============================================================================
# EMA Crossover Scenarios
# =============================================================================


class TestEmaCrossoverSimulation:
    """Simulation tests for EMA Crossover strategies."""

    @pytest.fixture
    def ema_crossover_ir(self) -> StrategyIR:
        """Create EMA Crossover Long IR."""
        strategy = Strategy(
            id="ema-cross",
            name="EMA Crossover",
            universe=["BTC-USD"],
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
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "trend_ma_relation",
                                "op": ">",
                                "value": 0,
                                "ma_fast": 20,
                                "ma_slow": 50,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "trend_ma_relation",
                                "op": "<",
                                "value": 0,
                                "ma_fast": 20,
                                "ma_slow": 50,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }
        return IRTranslator(strategy, cards).translate()

    def test_bullish_crossover_triggers_entry(self, ema_crossover_ir):
        """Entry triggers exactly when EMA fast crosses above EMA slow."""
        scenario = SimulationScenario(
            name="Bullish crossover",
            description="Fast EMA crosses above slow EMA",
            bars=[
                # Bar 0: Fast below slow (no entry)
                BarData(
                    bar=MockBar(100, 101, 99, 100),
                    indicators={"ema_20": 98.0, "ema_50": 100.0},
                ),
                # Bar 1: Fast equals slow (no entry, need >)
                BarData(
                    bar=MockBar(100, 102, 99, 101),
                    indicators={"ema_20": 100.0, "ema_50": 100.0},
                ),
                # Bar 2: Fast above slow (ENTRY!)
                BarData(
                    bar=MockBar(101, 103, 100, 102),
                    indicators={"ema_20": 101.0, "ema_50": 100.0},
                ),
                # Bar 3: Still above (no new entry, already invested)
                BarData(
                    bar=MockBar(102, 104, 101, 103),
                    indicators={"ema_20": 102.0, "ema_50": 100.5},
                ),
            ],
            expected_signals=[Signal(bar_index=2, signal_type="entry")],
        )

        actual = run_simulation(ema_crossover_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_bearish_crossover_triggers_exit(self, ema_crossover_ir):
        """Exit triggers exactly when EMA fast crosses below EMA slow."""
        scenario = SimulationScenario(
            name="Bearish crossover exit",
            description="Fast EMA crosses below slow EMA after entry",
            bars=[
                # Bar 0: Entry condition met
                BarData(
                    bar=MockBar(100, 101, 99, 100),
                    indicators={"ema_20": 102.0, "ema_50": 100.0},
                ),
                # Bar 1: Still bullish
                BarData(
                    bar=MockBar(100, 105, 99, 104),
                    indicators={"ema_20": 103.0, "ema_50": 101.0},
                ),
                # Bar 2: Fast equals slow (no exit yet, need <)
                BarData(
                    bar=MockBar(104, 105, 100, 101),
                    indicators={"ema_20": 101.0, "ema_50": 101.0},
                ),
                # Bar 3: Fast below slow (EXIT!)
                BarData(
                    bar=MockBar(101, 102, 98, 99),
                    indicators={"ema_20": 99.0, "ema_50": 101.0},
                ),
            ],
            expected_signals=[
                Signal(bar_index=0, signal_type="entry"),
                Signal(bar_index=3, signal_type="exit"),
            ],
        )

        actual = run_simulation(ema_crossover_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_no_entry_when_fast_equals_slow(self, ema_crossover_ir):
        """No entry when EMAs are exactly equal (need > not >=)."""
        scenario = SimulationScenario(
            name="Equal EMAs",
            description="No entry when EMAs are equal",
            bars=[
                BarData(
                    bar=MockBar(100, 101, 99, 100),
                    indicators={"ema_20": 100.0, "ema_50": 100.0},
                ),
                BarData(
                    bar=MockBar(100, 101, 99, 100),
                    indicators={"ema_20": 100.0, "ema_50": 100.0},
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ema_crossover_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_multiple_entry_exit_cycles(self, ema_crossover_ir):
        """Test multiple entry/exit cycles."""
        scenario = SimulationScenario(
            name="Multiple cycles",
            description="Two complete entry/exit cycles",
            bars=[
                # Cycle 1: Entry
                BarData(
                    bar=MockBar(100, 101, 99, 100),
                    indicators={"ema_20": 102.0, "ema_50": 100.0},
                ),
                # Cycle 1: Exit
                BarData(
                    bar=MockBar(100, 101, 98, 98), indicators={"ema_20": 98.0, "ema_50": 100.0}
                ),
                # No signal (bearish, not invested)
                BarData(
                    bar=MockBar(98, 99, 97, 97), indicators={"ema_20": 96.0, "ema_50": 99.0}
                ),
                # Cycle 2: Entry
                BarData(
                    bar=MockBar(97, 102, 97, 101), indicators={"ema_20": 101.0, "ema_50": 99.0}
                ),
                # Cycle 2: Exit
                BarData(
                    bar=MockBar(101, 101, 95, 96), indicators={"ema_20": 97.0, "ema_50": 99.0}
                ),
            ],
            expected_signals=[
                Signal(bar_index=0, signal_type="entry"),
                Signal(bar_index=1, signal_type="exit"),
                Signal(bar_index=3, signal_type="entry"),
                Signal(bar_index=4, signal_type="exit"),
            ],
        )

        actual = run_simulation(ema_crossover_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


class TestEmaCrossoverShortSimulation:
    """Simulation tests for EMA Crossover Short."""

    @pytest.fixture
    def ema_crossover_short_ir(self) -> StrategyIR:
        """Create EMA Crossover Short IR."""
        strategy = Strategy(
            id="ema-cross-short",
            name="EMA Crossover Short",
            universe=["BTC-USD"],
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
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "trend_ma_relation",
                                "op": "<",
                                "value": 0,
                                "ma_fast": 20,
                                "ma_slow": 50,
                            },
                        }
                    },
                    "action": {"direction": "short"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "trend_ma_relation",
                                "op": ">",
                                "value": 0,
                                "ma_fast": 20,
                                "ma_slow": 50,
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }
        return IRTranslator(strategy, cards).translate()

    def test_bearish_crossover_triggers_short_entry(self, ema_crossover_short_ir):
        """Short entry triggers when EMA fast crosses below EMA slow."""
        scenario = SimulationScenario(
            name="Bearish crossover short",
            description="Fast EMA crosses below slow EMA triggers short",
            bars=[
                # Bar 0: Fast above slow (no entry)
                BarData(
                    bar=MockBar(100, 101, 99, 100),
                    indicators={"ema_20": 102.0, "ema_50": 100.0},
                ),
                # Bar 1: Fast below slow (SHORT ENTRY!)
                BarData(
                    bar=MockBar(100, 101, 98, 98),
                    indicators={"ema_20": 98.0, "ema_50": 100.0},
                ),
                # Bar 2: Bullish crossover (EXIT!)
                BarData(
                    bar=MockBar(98, 102, 98, 101),
                    indicators={"ema_20": 101.0, "ema_50": 100.0},
                ),
            ],
            expected_signals=[
                Signal(bar_index=1, signal_type="entry"),
                Signal(bar_index=2, signal_type="exit"),
            ],
        )

        actual = run_simulation(ema_crossover_short_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# Return Percentage Scenarios
# =============================================================================


class TestReturnThresholdSimulation:
    """Simulation tests for return threshold (ret_pct) strategies."""

    @pytest.fixture
    def ret_threshold_ir(self) -> StrategyIR:
        """Create return threshold entry IR."""
        strategy = Strategy(
            id="ret-threshold",
            name="Return Threshold",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )
        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "regime",
                            "regime": {
                                "metric": "ret_pct",
                                "op": "<",
                                "value": -5.0,
                                "lookback_bars": 14,
                            },
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }
        return IRTranslator(strategy, cards).translate()

    def test_entry_on_oversold(self, ret_threshold_ir):
        """Entry triggers when ROC < -5%.

        Note: LEAN's ROC returns decimals (-0.05 for -5%), which the IR
        multiplies by 100 to compare against percentage thresholds.
        """
        scenario = SimulationScenario(
            name="Oversold entry",
            description="ROC drops below -5% threshold",
            bars=[
                # Bar 0: ROC = -3% (above threshold, no entry)
                BarData(
                    bar=MockBar(100, 101, 99, 100),
                    indicators={"roc": -0.03},  # -3% as decimal
                ),
                # Bar 1: ROC = -5% (exactly at threshold, no entry - need <)
                BarData(
                    bar=MockBar(100, 100, 95, 96),
                    indicators={"roc": -0.05},  # -5% as decimal
                ),
                # Bar 2: ROC = -7% (below threshold, ENTRY!)
                BarData(
                    bar=MockBar(96, 96, 92, 93),
                    indicators={"roc": -0.07},  # -7% as decimal
                ),
            ],
            expected_signals=[Signal(bar_index=2, signal_type="entry")],
        )

        actual = run_simulation(ret_threshold_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_no_entry_above_threshold(self, ret_threshold_ir):
        """No entry when ROC stays above threshold."""
        scenario = SimulationScenario(
            name="Above threshold",
            description="ROC stays above -5%",
            bars=[
                BarData(bar=MockBar(100, 101, 99, 100), indicators={"roc": 0.02}),  # +2%
                BarData(bar=MockBar(100, 102, 99, 101), indicators={"roc": 0.01}),  # +1%
                BarData(bar=MockBar(101, 101, 98, 99), indicators={"roc": -0.02}),  # -2%
                BarData(bar=MockBar(99, 100, 97, 98), indicators={"roc": -0.04}),  # -4%
            ],
            expected_signals=[],
        )

        actual = run_simulation(ret_threshold_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# Breakout Strategy Scenarios
# =============================================================================


class TestBreakoutSimulation:
    """Simulation tests for breakout strategies."""

    @pytest.fixture
    def breakout_long_ir(self) -> StrategyIR:
        """Create breakout long IR."""
        strategy = Strategy(
            id="breakout",
            name="Breakout Long",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )
        cards = {
            "entry": Card(
                id="entry",
                type="entry.breakout_trendfollow",
                slots={
                    "event": {"breakout": {"lookback_bars": 20}},
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }
        return IRTranslator(strategy, cards).translate()

    def test_breakout_above_highest(self, breakout_long_ir):
        """Entry triggers when price breaks above 20-bar high."""
        # to_ir() uses Donchian Channel with upper/lower bands
        dc_id = "dc_20"
        scenario = SimulationScenario(
            name="Breakout entry",
            description="Price breaks above 20-bar high",
            bars=[
                # Bar 0: Below high
                BarData(
                    bar=MockBar(100, 102, 99, 101),
                    indicators={dc_id: {"upper": 105.0, "middle": 100.0, "lower": 95.0}},
                ),
                # Bar 1: Just below high
                BarData(
                    bar=MockBar(101, 104, 100, 103),
                    indicators={dc_id: {"upper": 105.0, "middle": 100.0, "lower": 95.0}},
                ),
                # Bar 2: Breakout! (close > upper)
                BarData(
                    bar=MockBar(103, 107, 102, 106),
                    indicators={dc_id: {"upper": 105.0, "middle": 100.0, "lower": 95.0}},
                ),
            ],
            expected_signals=[Signal(bar_index=2, signal_type="entry")],
        )

        actual = run_simulation(breakout_long_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_no_breakout_when_below(self, breakout_long_ir):
        """No entry when price stays below 20-bar high."""
        dc_id = "dc_20"
        scenario = SimulationScenario(
            name="No breakout",
            description="Price stays below 20-bar high",
            bars=[
                BarData(
                    bar=MockBar(100, 102, 99, 101),
                    indicators={dc_id: {"upper": 110.0, "middle": 102.5, "lower": 95.0}},
                ),
                BarData(
                    bar=MockBar(101, 104, 100, 103),
                    indicators={dc_id: {"upper": 110.0, "middle": 102.5, "lower": 95.0}},
                ),
                BarData(
                    bar=MockBar(103, 108, 102, 107),
                    indicators={dc_id: {"upper": 110.0, "middle": 102.5, "lower": 95.0}},
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(breakout_long_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# Trend Pullback (Band) Scenarios
# =============================================================================


class TestTrendPullbackSimulation:
    """Simulation tests for trend pullback strategies."""

    @pytest.fixture
    def trend_pullback_ir(self) -> StrategyIR:
        """Create trend pullback IR."""
        strategy = Strategy(
            id="pullback",
            name="Trend Pullback",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )
        cards = {
            "entry": Card(
                id="entry",
                type="entry.trend_pullback",
                slots={
                    "event": {
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }
        return IRTranslator(strategy, cards).translate()

    def test_pullback_entry_in_uptrend(self, trend_pullback_ir):
        """Entry triggers when in uptrend AND price touches lower band."""
        scenario = SimulationScenario(
            name="Pullback in uptrend",
            description="Uptrend + price at lower BB",
            bars=[
                # Bar 0: Uptrend but price in middle (no entry)
                BarData(
                    bar=MockBar(100, 101, 99, 100),
                    indicators={
                        "ema_20": 102.0,
                        "ema_50": 100.0,
                        "bb_2_0_20": {"upper": 105.0, "middle": 100.0, "lower": 95.0},
                    },
                ),
                # Bar 1: Uptrend + price at lower band (ENTRY!)
                BarData(
                    bar=MockBar(100, 100, 94, 94),
                    indicators={
                        "ema_20": 101.0,
                        "ema_50": 99.0,
                        "bb_2_0_20": {"upper": 105.0, "middle": 100.0, "lower": 95.0},
                    },
                ),
            ],
            expected_signals=[Signal(bar_index=1, signal_type="entry")],
        )

        actual = run_simulation(trend_pullback_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_no_entry_in_downtrend(self, trend_pullback_ir):
        """No entry when in downtrend even if price at lower band."""
        scenario = SimulationScenario(
            name="Pullback in downtrend",
            description="Downtrend + price at lower BB = no entry",
            bars=[
                # Downtrend (fast < slow) + price at lower band
                BarData(
                    bar=MockBar(100, 100, 94, 94),
                    indicators={
                        "ema_20": 98.0,  # Below slow = downtrend
                        "ema_50": 100.0,
                        "bb_2_0_20": {"upper": 105.0, "middle": 100.0, "lower": 95.0},
                    },
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(trend_pullback_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_no_entry_when_above_lower_band(self, trend_pullback_ir):
        """No entry when in uptrend but price above lower band."""
        scenario = SimulationScenario(
            name="Above lower band",
            description="Uptrend but price not at lower band",
            bars=[
                # Uptrend but price in middle of band
                BarData(
                    bar=MockBar(100, 102, 99, 101),
                    indicators={
                        "ema_20": 102.0,
                        "ema_50": 100.0,
                        "bb_2_0_20": {"upper": 105.0, "middle": 100.0, "lower": 95.0},
                    },
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(trend_pullback_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# Band Exit Scenarios
# =============================================================================


class TestBandExitSimulation:
    """Simulation tests for band exit strategies."""

    @pytest.fixture
    def band_exit_ir(self) -> StrategyIR:
        """Create trend pullback with band exit IR."""
        strategy = Strategy(
            id="pullback-band-exit",
            name="Pullback Band Exit",
            universe=["BTC-USD"],
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
                type="entry.trend_pullback",
                slots={
                    "event": {
                        "dip_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "trend_gate": {"fast": 20, "slow": 50, "op": ">"},
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
            "exit": Card(
                id="exit",
                type="exit.band_exit",
                slots={
                    "event": {
                        "exit_band": {"band": "bollinger", "length": 20, "mult": 2.0},
                        "exit_trigger": {"edge": "upper"},
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }
        return IRTranslator(strategy, cards).translate()

    def test_exit_at_upper_band(self, band_exit_ir):
        """Exit triggers when price reaches upper band."""
        # Band indicator IDs use naming convention: band_{type}_{length}_{mult}
        # e.g., bb_2_0_20 for bollinger length=20 mult=2.0
        scenario = SimulationScenario(
            name="Exit at upper band",
            description="Entry at lower band, exit at upper band",
            bars=[
                # Bar 0: Entry at lower band
                BarData(
                    bar=MockBar(100, 100, 94, 94),
                    indicators={
                        "ema_20": 101.0,
                        "ema_50": 99.0,
                        "bb_2_0_20": {"upper": 105.0, "middle": 100.0, "lower": 95.0},
                    },
                ),
                # Bar 1: Price rising, still below upper
                BarData(
                    bar=MockBar(94, 100, 94, 99),
                    indicators={
                        "ema_20": 101.0,
                        "ema_50": 99.0,
                        "bb_2_0_20": {"upper": 105.0, "middle": 100.0, "lower": 95.0},
                    },
                ),
                # Bar 2: Price at upper band (EXIT!)
                BarData(
                    bar=MockBar(99, 106, 99, 105),
                    indicators={
                        "ema_20": 101.0,
                        "ema_50": 99.0,
                        "bb_2_0_20": {"upper": 105.0, "middle": 100.0, "lower": 95.0},
                    },
                ),
            ],
            expected_signals=[
                Signal(bar_index=0, signal_type="entry"),
                Signal(bar_index=2, signal_type="exit"),
            ],
        )

        actual = run_simulation(band_exit_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# AllOf/AnyOf/Not Condition Scenarios
# =============================================================================


class TestCompositeConditionSimulation:
    """Simulation tests for composite conditions."""

    @pytest.fixture
    def allof_ir(self) -> StrategyIR:
        """Create strategy with AllOf condition."""
        strategy = Strategy(
            id="allof",
            name="AllOf Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )
        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
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
                                        "ma_fast": 20,
                                        "ma_slow": 50,
                                    },
                                },
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "ret_pct",
                                        "op": "<",
                                        "value": -2.0,
                                        "lookback_bars": 5,
                                    },
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }
        return IRTranslator(strategy, cards).translate()

    def test_allof_both_true(self, allof_ir):
        """Entry only when BOTH conditions are true."""
        scenario = SimulationScenario(
            name="AllOf both true",
            description="Both uptrend AND oversold",
            bars=[
                # Bar 0: Only uptrend (no entry) - ROC = -1% > -2%
                BarData(
                    bar=MockBar(100, 101, 99, 100),
                    indicators={
                        "ema_20": 102.0,
                        "ema_50": 100.0,
                        "roc": -0.01,
                    },  # -1% as decimal
                ),
                # Bar 1: Only oversold (no entry) - downtrend
                BarData(
                    bar=MockBar(100, 100, 96, 97),
                    indicators={
                        "ema_20": 98.0,
                        "ema_50": 100.0,
                        "roc": -0.03,
                    },  # -3% as decimal
                ),
                # Bar 2: BOTH uptrend AND oversold (ENTRY!)
                BarData(
                    bar=MockBar(97, 98, 95, 96),
                    indicators={
                        "ema_20": 101.0,
                        "ema_50": 100.0,
                        "roc": -0.03,
                    },  # -3% as decimal
                ),
            ],
            expected_signals=[Signal(bar_index=2, signal_type="entry")],
        )

        actual = run_simulation(allof_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    @pytest.fixture
    def anyof_ir(self) -> StrategyIR:
        """Create strategy with AnyOf condition."""
        strategy = Strategy(
            id="anyof",
            name="AnyOf Test",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )
        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "anyOf",
                            "anyOf": [
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "ret_pct",
                                        "op": "<",
                                        "value": -5.0,
                                        "lookback_bars": 10,
                                    },
                                },
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "ret_pct",
                                        "op": ">",
                                        "value": 5.0,
                                        "lookback_bars": 10,
                                    },
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }
        return IRTranslator(strategy, cards).translate()

    def test_anyof_first_true(self, anyof_ir):
        """Entry when first condition is true (oversold < -5%)."""
        scenario = SimulationScenario(
            name="AnyOf first true",
            description="Oversold condition triggers entry",
            bars=[
                BarData(
                    bar=MockBar(100, 100, 92, 93), indicators={"roc": -0.07}
                ),  # -7% < -5% (as decimal)
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(anyof_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_anyof_second_true(self, anyof_ir):
        """Entry when second condition is true (overbought > 5%)."""
        scenario = SimulationScenario(
            name="AnyOf second true",
            description="Overbought condition triggers entry",
            bars=[
                BarData(bar=MockBar(100, 108, 100, 107), indicators={"roc": 7.0}),  # 7% > 5%
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(anyof_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_anyof_neither_true(self, anyof_ir):
        """No entry when neither condition is true (-5% <= ROC <= 5%)."""
        scenario = SimulationScenario(
            name="AnyOf neither true",
            description="ROC in normal range",
            bars=[
                BarData(bar=MockBar(100, 102, 99, 101), indicators={"roc": 0.02}),  # +2% as decimal
                BarData(
                    bar=MockBar(101, 103, 100, 102), indicators={"roc": -0.01}
                ),  # -1% as decimal
            ],
            expected_signals=[],
        )

        actual = run_simulation(anyof_ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# Complex Multi-Condition Strategy Tests
# =============================================================================


class TestComplexMultiConditionStrategy:
    """Comprehensive tests for complex multi-condition strategies.

    These tests verify that real-world strategy patterns work correctly
    through complete trade lifecycles with realistic market scenarios.
    """

    @pytest.fixture
    def trend_pullback_band_strategy(self) -> StrategyIR:
        """Create a trend pullback strategy with band touch entry and band exit.

        Entry: Uptrend (EMA fast > slow) AND price touches lower Bollinger Band
        Exit: Price touches upper Bollinger Band

        This combines:
        - regime condition (trend_ma_relation)
        - band_event (touch lower band)
        - band_event exit (touch upper band)
        """
        strategy = Strategy(
            id="complex-pullback",
            name="Complex Trend Pullback",
            universe=["BTC-USD"],
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
                    "event": {
                        "condition": {
                            "type": "allOf",
                            "allOf": [
                                # Condition 1: Uptrend
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "trend_ma_relation",
                                        "op": ">",
                                        "value": 0,
                                        "ma_fast": 20,
                                        "ma_slow": 50,
                                    },
                                },
                                # Condition 2: Price touches lower BB
                                {
                                    "type": "band_event",
                                    "band_event": {
                                        "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                                        "kind": "edge_event",
                                        "event": "touch",
                                        "edge": "lower",
                                    },
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
            "exit": Card(
                id="exit",
                type="exit.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "band_event",
                            "band_event": {
                                "band": {"band": "bollinger", "length": 20, "mult": 2.0},
                                "kind": "edge_event",
                                "event": "touch",
                                "edge": "upper",
                            },
                        }
                    },
                    "action": {"mode": "close"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }
        return IRTranslator(strategy, cards).translate()

    def test_full_trade_lifecycle(self, trend_pullback_band_strategy):
        """Test complete trade lifecycle: no entry → entry → hold → exit.

        Simulates a realistic market scenario where:
        1. Price dips but no uptrend (no entry)
        2. Price dips in uptrend (entry)
        3. Price rises to upper band (exit)
        """
        ir = trend_pullback_band_strategy

        # BB: lower=90, middle=100, upper=110 (simplified)
        # ID format: band_{type}_{length}_{mult} with . replaced by _
        bb_bands = {"upper": 110.0, "middle": 100.0, "lower": 90.0}
        bb_id = "bb_2_0_20"

        scenario = SimulationScenario(
            name="Full trade lifecycle",
            description="Entry on dip in uptrend, exit at upper band",
            bars=[
                # Bar 0: Downtrend + CLOSE at band → NO ENTRY (wrong trend)
                # Touch uses CLOSE price, not LOW
                BarData(
                    bar=MockBar(95, 96, 85, 88),  # Close=88 <= lower band=90, but downtrend
                    indicators={"ema_20": 98.0, "ema_50": 100.0, bb_id: bb_bands},
                ),
                # Bar 1: Uptrend but CLOSE above band → NO ENTRY
                BarData(
                    bar=MockBar(100, 105, 98, 103),  # Close=103 > lower band=90
                    indicators={"ema_20": 102.0, "ema_50": 100.0, bb_id: bb_bands},
                ),
                # Bar 2: Uptrend + CLOSE at band → ENTRY!
                BarData(
                    bar=MockBar(95, 96, 85, 89),  # Close=89 <= lower band=90
                    indicators={"ema_20": 103.0, "ema_50": 100.0, bb_id: bb_bands},
                ),
                # Bar 3: Still invested, price rising
                BarData(
                    bar=MockBar(89, 102, 88, 100),  # Close=100, no exit yet
                    indicators={"ema_20": 104.0, "ema_50": 100.5, bb_id: bb_bands},
                ),
                # Bar 4: CLOSE at upper band → EXIT!
                BarData(
                    bar=MockBar(100, 115, 99, 112),  # Close=112 >= upper band=110
                    indicators={"ema_20": 105.0, "ema_50": 101.0, bb_id: bb_bands},
                ),
            ],
            expected_signals=[
                Signal(bar_index=2, signal_type="entry"),
                Signal(bar_index=4, signal_type="exit"),
            ],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_multiple_trade_cycles(self, trend_pullback_band_strategy):
        """Test multiple entry/exit cycles in the same scenario."""
        ir = trend_pullback_band_strategy

        bb_bands = {"upper": 110.0, "middle": 100.0, "lower": 90.0}
        bb_id = "bb_2_0_20"

        scenario = SimulationScenario(
            name="Multiple trade cycles",
            description="Two complete entry-exit cycles",
            bars=[
                # Cycle 1: Entry (CLOSE <= lower band in uptrend)
                BarData(
                    bar=MockBar(100, 101, 85, 88),  # Close=88 <= lower=90
                    indicators={"ema_20": 103.0, "ema_50": 100.0, bb_id: bb_bands},
                ),
                # Cycle 1: Exit (CLOSE >= upper band)
                BarData(
                    bar=MockBar(88, 115, 87, 112),  # Close=112 >= upper=110
                    indicators={"ema_20": 105.0, "ema_50": 101.0, bb_id: bb_bands},
                ),
                # Pause bar (no signal)
                BarData(
                    bar=MockBar(112, 113, 100, 102),  # Close=102, between bands
                    indicators={"ema_20": 104.0, "ema_50": 101.5, bb_id: bb_bands},
                ),
                # Cycle 2: Entry (CLOSE <= lower band in uptrend)
                BarData(
                    bar=MockBar(102, 103, 80, 85),  # Close=85 <= lower=90
                    indicators={"ema_20": 102.0, "ema_50": 100.0, bb_id: bb_bands},
                ),
                # Cycle 2: Exit (CLOSE >= upper band)
                BarData(
                    bar=MockBar(85, 120, 84, 115),  # Close=115 >= upper=110
                    indicators={"ema_20": 106.0, "ema_50": 102.0, bb_id: bb_bands},
                ),
            ],
            expected_signals=[
                Signal(bar_index=0, signal_type="entry"),
                Signal(bar_index=1, signal_type="exit"),
                Signal(bar_index=3, signal_type="entry"),
                Signal(bar_index=4, signal_type="exit"),
            ],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_boundary_conditions(self, trend_pullback_band_strategy):
        """Test exact boundary conditions (price exactly at band)."""
        ir = trend_pullback_band_strategy

        bb_bands = {"upper": 110.0, "middle": 100.0, "lower": 90.0}
        bb_id = "bb_2_0_20"

        scenario = SimulationScenario(
            name="Exact boundary touch",
            description="CLOSE exactly at band triggers entry/exit",
            bars=[
                # CLOSE exactly at lower band (should trigger entry)
                BarData(
                    bar=MockBar(95, 96, 88, 90.0),  # Close exactly 90 = lower band
                    indicators={"ema_20": 102.0, "ema_50": 100.0, bb_id: bb_bands},
                ),
                # CLOSE exactly at upper band (should trigger exit)
                BarData(
                    bar=MockBar(90, 115, 89, 110.0),  # Close exactly 110 = upper band
                    indicators={"ema_20": 103.0, "ema_50": 100.0, bb_id: bb_bands},
                ),
            ],
            expected_signals=[
                Signal(bar_index=0, signal_type="entry"),
                Signal(bar_index=1, signal_type="exit"),
            ],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_near_miss_no_entry(self, trend_pullback_band_strategy):
        """Test that near-miss conditions don't trigger signals."""
        ir = trend_pullback_band_strategy

        bb_bands = {"upper": 110.0, "middle": 100.0, "lower": 90.0}
        bb_id = "bb_2_0_20"

        scenario = SimulationScenario(
            name="Near miss - no trigger",
            description="CLOSE just above lower band should NOT trigger",
            bars=[
                # CLOSE just ABOVE lower band (should NOT trigger)
                BarData(
                    bar=MockBar(95, 96, 85, 90.01),  # Close=90.01 > lower band=90
                    indicators={"ema_20": 102.0, "ema_50": 100.0, bb_id: bb_bands},
                ),
                # More bars without entry
                BarData(
                    bar=MockBar(90, 95, 88, 94),  # Close=94 > lower band=90
                    indicators={"ema_20": 101.5, "ema_50": 100.0, bb_id: bb_bands},
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    @pytest.fixture
    def triple_condition_strategy(self) -> StrategyIR:
        """Create a strategy with 3 nested conditions.

        Entry: (Uptrend AND Oversold ROC) OR (Strong uptrend)
        This tests: anyOf containing allOf
        """
        strategy = Strategy(
            id="triple-condition",
            name="Triple Condition",
            universe=["BTC-USD"],
            status="ready",
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
            attachments=[
                Attachment(card_id="entry", role="entry", enabled=True),
            ],
        )
        cards = {
            "entry": Card(
                id="entry",
                type="entry.rule_trigger",
                slots={
                    "event": {
                        "condition": {
                            "type": "anyOf",
                            "anyOf": [
                                # Option 1: Uptrend + Oversold
                                {
                                    "type": "allOf",
                                    "allOf": [
                                        {
                                            "type": "regime",
                                            "regime": {
                                                "metric": "trend_ma_relation",
                                                "op": ">",
                                                "value": 0,
                                                "ma_fast": 20,
                                                "ma_slow": 50,
                                            },
                                        },
                                        {
                                            "type": "regime",
                                            "regime": {
                                                "metric": "ret_pct",
                                                "op": "<",
                                                "value": -5.0,
                                                "lookback_bars": 5,
                                            },
                                        },
                                    ],
                                },
                                # Option 2: Very strong uptrend (fast >> slow)
                                {
                                    "type": "regime",
                                    "regime": {
                                        "metric": "trend_ma_relation",
                                        "op": ">",
                                        "value": 10,  # Fast MA 10+ above slow MA
                                        "ma_fast": 20,
                                        "ma_slow": 50,
                                    },
                                },
                            ],
                        }
                    },
                    "action": {"direction": "long"},
                },
                schema_etag="test",
                created_at="2024-01-01T00:00:00Z",
                updated_at="2024-01-01T00:00:00Z",
            ),
        }
        return IRTranslator(strategy, cards).translate()

    def test_nested_anyof_allof_first_branch(self, triple_condition_strategy):
        """Test that first branch of anyOf(allOf(...), ...) triggers correctly."""
        ir = triple_condition_strategy

        scenario = SimulationScenario(
            name="Nested anyOf-allOf first branch",
            description="Entry via uptrend + oversold (allOf branch)",
            bars=[
                # Uptrend + Oversold → ENTRY via first anyOf branch
                BarData(
                    bar=MockBar(100, 101, 92, 93),
                    indicators={
                        "ema_20": 102.0,  # > slow (uptrend but diff only 2)
                        "ema_50": 100.0,
                        "roc": -0.06,  # < -5% (oversold) as decimal
                    },
                ),
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_nested_anyof_allof_second_branch(self, triple_condition_strategy):
        """Test that second branch of anyOf triggers correctly."""
        ir = triple_condition_strategy

        scenario = SimulationScenario(
            name="Nested anyOf-allOf second branch",
            description="Entry via strong uptrend alone",
            bars=[
                # Strong uptrend (diff > 10) → ENTRY via second anyOf branch
                BarData(
                    bar=MockBar(100, 105, 99, 104),
                    indicators={
                        "ema_20": 115.0,  # 15 above slow (> 10 threshold)
                        "ema_50": 100.0,
                        "roc": 3.0,  # NOT oversold, but doesn't matter
                    },
                ),
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_nested_anyof_allof_neither_branch(self, triple_condition_strategy):
        """Test that neither branch triggers when conditions not met."""
        ir = triple_condition_strategy

        scenario = SimulationScenario(
            name="Nested anyOf-allOf neither branch",
            description="No entry: uptrend but not oversold, and not strong uptrend",
            bars=[
                # Weak uptrend, not oversold → NO ENTRY
                BarData(
                    bar=MockBar(100, 102, 99, 101),
                    indicators={
                        "ema_20": 102.0,  # diff = 2 (not > 10 for strong)
                        "ema_50": 100.0,
                        "roc": -0.02,  # not < -5% (not oversold) as decimal
                    },
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# Helper Factory for Entry Strategies
# =============================================================================


def create_entry_strategy(entry_condition: Condition) -> StrategyIR:
    """Factory to create a simple entry-only strategy IR.

    Reduces boilerplate when testing individual condition types.
    The strategy enters at 100% on entry condition, no exit.

    Note: Indicators are mocked in the simulation, so we don't need to
    define them in the IR. Just ensure indicator IDs in conditions match
    the keys in BarData.indicators.
    """
    return StrategyIR(
        strategy_id="test-strategy-001",
        strategy_name="TestStrategy",
        symbol="SPY",
        resolution=Resolution.DAILY,
        entry=EntryRule(
            condition=entry_condition,
            action=SetHoldingsAction(allocation=1.0),
        ),
        exits=[],  # No exit for simple entry tests
        indicators=[],  # Mocked in simulation
        state=[],
    )


# =============================================================================
# Volume Metrics Simulation Tests
# =============================================================================


class TestVolumeSpikeSimulation:
    """Test volume spike detection via simulation."""

    @pytest.fixture
    def volume_spike_ir(self) -> StrategyIR:
        """Strategy that enters when volume > 2x average."""
        return create_entry_strategy(
            entry_condition=CompareCondition(
                left=VolumeValue(),
                op=CompareOp.GT,
                right=ExpressionValue(
                    op="*",
                    left=IndicatorValue(indicator_id="vol_sma_20"),
                    right=LiteralValue(value=2.0),
                ),
            ),
        )

    def test_volume_spike_triggers_entry(self, volume_spike_ir):
        """Test entry when volume spikes above threshold."""
        ir = volume_spike_ir

        scenario = SimulationScenario(
            name="Volume spike entry",
            description="Volume 2.5x average triggers entry",
            bars=[
                BarData(
                    bar=MockBar(100, 102, 99, 101, volume=2500.0),
                    indicators={"vol_sma_20": 1000.0},  # 2500 > 2000 (2x avg)
                ),
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_no_entry_on_normal_volume(self, volume_spike_ir):
        """Test no entry when volume is at normal levels."""
        ir = volume_spike_ir

        scenario = SimulationScenario(
            name="Normal volume no entry",
            description="Volume at 1.5x average does not trigger",
            bars=[
                BarData(
                    bar=MockBar(100, 102, 99, 101, volume=1500.0),
                    indicators={"vol_sma_20": 1000.0},  # 1500 < 2000 (2x avg)
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


class TestVolumeDipSimulation:
    """Test volume dip detection via simulation."""

    @pytest.fixture
    def volume_dip_ir(self) -> StrategyIR:
        """Strategy that enters when volume < 0.5x average."""
        return create_entry_strategy(
            entry_condition=CompareCondition(
                left=VolumeValue(),
                op=CompareOp.LT,
                right=ExpressionValue(
                    op="*",
                    left=IndicatorValue(indicator_id="vol_sma_20"),
                    right=LiteralValue(value=0.5),
                ),
            ),
        )

    def test_volume_dip_triggers_entry(self, volume_dip_ir):
        """Test entry when volume drops below threshold."""
        ir = volume_dip_ir

        scenario = SimulationScenario(
            name="Volume dip entry",
            description="Volume 0.3x average triggers entry",
            bars=[
                BarData(
                    bar=MockBar(100, 102, 99, 101, volume=300.0),
                    indicators={"vol_sma_20": 1000.0},  # 300 < 500 (0.5x avg)
                ),
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# Session Phase Simulation Tests
# =============================================================================


class TestSessionPhaseSimulation:
    """Test session phase (time-based) conditions via simulation."""

    @pytest.fixture
    def market_open_ir(self) -> StrategyIR:
        """Strategy that only enters during market open (hour 9-10)."""
        return create_entry_strategy(
            entry_condition=AllOfCondition(
                conditions=[
                    CompareCondition(
                        left=TimeValue(component="hour"),
                        op=CompareOp.GTE,
                        right=LiteralValue(value=9.0),
                    ),
                    CompareCondition(
                        left=TimeValue(component="hour"),
                        op=CompareOp.LT,
                        right=LiteralValue(value=10.0),
                    ),
                ]
            ),
        )

    def test_entry_during_market_open(self, market_open_ir):
        """Test entry when within market open hours."""
        ir = market_open_ir

        scenario = SimulationScenario(
            name="Market open entry",
            description="Entry at 9:30 AM",
            bars=[
                BarData(
                    bar=MockBar(100, 102, 99, 101),
                    indicators={},
                    hour=9,
                    minute=30,
                ),
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_no_entry_outside_market_open(self, market_open_ir):
        """Test no entry when outside market open hours."""
        ir = market_open_ir

        scenario = SimulationScenario(
            name="Outside market open no entry",
            description="No entry at 2:00 PM",
            bars=[
                BarData(
                    bar=MockBar(100, 102, 99, 101),
                    indicators={},
                    hour=14,
                    minute=0,
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# Price Level Simulation Tests
# =============================================================================


class TestPriceLevelTouchSimulation:
    """Test price level touch detection via simulation."""

    @pytest.fixture
    def level_touch_ir(self) -> StrategyIR:
        """Strategy that enters when price touches a level (150.0)."""
        level = 150.0
        return create_entry_strategy(
            entry_condition=AllOfCondition(
                conditions=[
                    CompareCondition(
                        left=PriceValue(field=PriceField.HIGH),
                        op=CompareOp.GTE,
                        right=LiteralValue(value=level),
                    ),
                    CompareCondition(
                        left=PriceValue(field=PriceField.LOW),
                        op=CompareOp.LTE,
                        right=LiteralValue(value=level),
                    ),
                ]
            ),
        )

    def test_price_touches_level(self, level_touch_ir):
        """Test entry when price bar spans the level."""
        ir = level_touch_ir

        scenario = SimulationScenario(
            name="Price touches level",
            description="Bar spans 148-152, touches 150",
            bars=[
                BarData(
                    bar=MockBar(149, 152, 148, 151),  # High >= 150, Low <= 150
                    indicators={},
                ),
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_price_misses_level_above(self, level_touch_ir):
        """Test no entry when price is entirely above level."""
        ir = level_touch_ir

        scenario = SimulationScenario(
            name="Price above level",
            description="Bar 151-155, above 150",
            bars=[
                BarData(
                    bar=MockBar(152, 155, 151, 154),  # Low > 150
                    indicators={},
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_price_misses_level_below(self, level_touch_ir):
        """Test no entry when price is entirely below level."""
        ir = level_touch_ir

        scenario = SimulationScenario(
            name="Price below level",
            description="Bar 145-149, below 150",
            bars=[
                BarData(
                    bar=MockBar(146, 149, 145, 148),  # High < 150
                    indicators={},
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


class TestPriceLevelCrossSimulation:
    """Test price level cross detection via simulation."""

    @pytest.fixture
    def level_cross_up_ir(self) -> StrategyIR:
        """Strategy that enters when close crosses above 150."""
        return create_entry_strategy(
            entry_condition=CompareCondition(
                left=PriceValue(field=PriceField.CLOSE),
                op=CompareOp.GT,
                right=LiteralValue(value=150.0),
            ),
        )

    def test_close_crosses_above_level(self, level_cross_up_ir):
        """Test entry when close is above level."""
        ir = level_cross_up_ir

        scenario = SimulationScenario(
            name="Close above level",
            description="Close at 152 > 150",
            bars=[
                BarData(
                    bar=MockBar(148, 153, 147, 152),
                    indicators={},
                ),
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_close_below_level(self, level_cross_up_ir):
        """Test no entry when close is below level."""
        ir = level_cross_up_ir

        scenario = SimulationScenario(
            name="Close below level",
            description="Close at 148 < 150",
            bars=[
                BarData(
                    bar=MockBar(146, 149, 145, 148),
                    indicators={},
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# Volatility Regime Simulation Tests
# =============================================================================


class TestVolatilityRegimeSimulation:
    """Test volatility regime conditions via simulation."""

    @pytest.fixture
    def low_vol_regime_ir(self) -> StrategyIR:
        """Strategy that enters in low volatility (BB width < 0.05)."""
        return create_entry_strategy(
            entry_condition=CompareCondition(
                left=IndicatorPropertyValue(
                    indicator_id="bb_20",
                    property=IndicatorProperty.BAND_WIDTH,
                ),
                op=CompareOp.LT,
                right=LiteralValue(value=0.05),
            ),
        )

    def test_entry_in_low_volatility(self, low_vol_regime_ir):
        """Test entry when volatility is low."""
        ir = low_vol_regime_ir

        # Mock BB with narrow bands (width = 0.03)
        # Width = (upper - lower) / middle = (102 - 98) / 100 = 0.04
        scenario = SimulationScenario(
            name="Low volatility entry",
            description="BB width < 0.05",
            bars=[
                BarData(
                    bar=MockBar(100, 101, 99, 100.5),
                    indicators={
                        "bb_20": {
                            "upper": 102.0,
                            "middle": 100.0,
                            "lower": 98.0,
                        }
                    },
                ),
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_no_entry_in_high_volatility(self, low_vol_regime_ir):
        """Test no entry when volatility is high."""
        ir = low_vol_regime_ir

        # Mock BB with wide bands (width = 0.10)
        # Width = (upper - lower) / middle = (105 - 95) / 100 = 0.10
        scenario = SimulationScenario(
            name="High volatility no entry",
            description="BB width > 0.05",
            bars=[
                BarData(
                    bar=MockBar(100, 101, 99, 100.5),
                    indicators={
                        "bb_20": {
                            "upper": 105.0,
                            "middle": 100.0,
                            "lower": 95.0,
                        }
                    },
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# Trend ADX Simulation Tests
# =============================================================================


class TestTrendADXSimulation:
    """Test trend strength via ADX conditions."""

    @pytest.fixture
    def strong_trend_ir(self) -> StrategyIR:
        """Strategy that enters when ADX > 25 (strong trend)."""
        return create_entry_strategy(
            entry_condition=CompareCondition(
                left=IndicatorValue(indicator_id="adx_14"),
                op=CompareOp.GT,
                right=LiteralValue(value=25.0),
            ),
        )

    def test_entry_in_strong_trend(self, strong_trend_ir):
        """Test entry when ADX indicates strong trend."""
        ir = strong_trend_ir

        scenario = SimulationScenario(
            name="Strong trend entry",
            description="ADX = 30 > 25",
            bars=[
                BarData(
                    bar=MockBar(100, 102, 99, 101),
                    indicators={"adx_14": 30.0},
                ),
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_no_entry_in_weak_trend(self, strong_trend_ir):
        """Test no entry when ADX indicates weak trend."""
        ir = strong_trend_ir

        scenario = SimulationScenario(
            name="Weak trend no entry",
            description="ADX = 18 < 25",
            bars=[
                BarData(
                    bar=MockBar(100, 102, 99, 101),
                    indicators={"adx_14": 18.0},
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# ATR Percentage Simulation Tests
# =============================================================================


class TestATRPercentageSimulation:
    """Test ATR as percentage of price conditions."""

    @pytest.fixture
    def high_atr_pct_ir(self) -> StrategyIR:
        """Strategy that enters when ATR% > 2% (high volatility)."""
        return create_entry_strategy(
            entry_condition=CompareCondition(
                left=ExpressionValue(
                    op="/",
                    left=IndicatorValue(indicator_id="atr_14"),
                    right=PriceValue(field=PriceField.CLOSE),
                ),
                op=CompareOp.GT,
                right=LiteralValue(value=0.02),  # 2%
            ),
        )

    def test_entry_on_high_atr_percentage(self, high_atr_pct_ir):
        """Test entry when ATR% exceeds threshold."""
        ir = high_atr_pct_ir

        # ATR = 3.0, Close = 100 → ATR% = 3%
        scenario = SimulationScenario(
            name="High ATR% entry",
            description="ATR/Close = 3/100 = 3% > 2%",
            bars=[
                BarData(
                    bar=MockBar(99, 102, 98, 100),
                    indicators={"atr_14": 3.0},
                ),
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_no_entry_on_low_atr_percentage(self, high_atr_pct_ir):
        """Test no entry when ATR% is below threshold."""
        ir = high_atr_pct_ir

        # ATR = 1.0, Close = 100 → ATR% = 1%
        scenario = SimulationScenario(
            name="Low ATR% no entry",
            description="ATR/Close = 1/100 = 1% < 2%",
            bars=[
                BarData(
                    bar=MockBar(99, 101, 99, 100),
                    indicators={"atr_14": 1.0},
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)


# =============================================================================
# Distance from VWAP Simulation Tests
# =============================================================================


class TestDistanceFromVWAPSimulation:
    """Test distance from VWAP percentage conditions."""

    @pytest.fixture
    def vwap_reversion_ir(self) -> StrategyIR:
        """Strategy that enters when price is > 2% below VWAP."""
        return create_entry_strategy(
            entry_condition=CompareCondition(
                left=ExpressionValue(
                    op="/",
                    left=ExpressionValue(
                        op="-",
                        left=IndicatorValue(indicator_id="vwap"),
                        right=PriceValue(field=PriceField.CLOSE),
                    ),
                    right=IndicatorValue(indicator_id="vwap"),
                ),
                op=CompareOp.GT,
                right=LiteralValue(value=0.02),  # 2% below
            ),
        )

    def test_entry_when_below_vwap(self, vwap_reversion_ir):
        """Test entry when price is significantly below VWAP."""
        ir = vwap_reversion_ir

        # VWAP = 100, Close = 97 → (100-97)/100 = 3% below
        scenario = SimulationScenario(
            name="Below VWAP entry",
            description="Price 3% below VWAP",
            bars=[
                BarData(
                    bar=MockBar(98, 99, 96, 97),
                    indicators={"vwap": 100.0},
                ),
            ],
            expected_signals=[Signal(bar_index=0, signal_type="entry")],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)

    def test_no_entry_near_vwap(self, vwap_reversion_ir):
        """Test no entry when price is near VWAP."""
        ir = vwap_reversion_ir

        # VWAP = 100, Close = 99 → (100-99)/100 = 1% below
        scenario = SimulationScenario(
            name="Near VWAP no entry",
            description="Price 1% below VWAP",
            bars=[
                BarData(
                    bar=MockBar(99, 100, 98, 99),
                    indicators={"vwap": 100.0},
                ),
            ],
            expected_signals=[],
        )

        actual = run_simulation(ir, scenario)
        assert_signals_match(scenario.expected_signals, actual, scenario.name)
