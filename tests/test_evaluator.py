"""Tests for the IR evaluator."""

import pytest

from src.translator.evaluator import (
    ActionExecutor,
    ConditionEvaluator,
    EvalContext,
    ExecContext,
    StateOperator,
    ValueResolver,
)
from src.translator.ir import (
    AllOfCondition,
    AnyOfCondition,
    BandField,
    CompareCondition,
    CompareOp,
    ExpressionValue,
    IncrementStateOp,
    IndicatorBandValue,
    IndicatorValue,
    LiquidateAction,
    LiteralValue,
    MaxStateOp,
    NotCondition,
    PriceField,
    PriceValue,
    SetHoldingsAction,
    SetStateOp,
    StateValue,
)

# =============================================================================
# Mock Objects for Testing
# =============================================================================


class MockIndicatorCurrent:
    """Mock for indicator.Current."""

    def __init__(self, value: float):
        self.Value = value


class MockIndicator:
    """Mock for a simple indicator (EMA, SMA, etc.)."""

    def __init__(self, value: float, is_ready: bool = True):
        self.Current = MockIndicatorCurrent(value)
        self.IsReady = is_ready


class MockBandIndicator:
    """Mock for band indicators (BB, KC)."""

    def __init__(self, upper: float, middle: float, lower: float, is_ready: bool = True):
        self.UpperBand = MockIndicator(upper)
        self.MiddleBand = MockIndicator(middle)
        self.LowerBand = MockIndicator(lower)
        self.IsReady = is_ready


class MockPriceBar:
    """Mock for LEAN price bar."""

    def __init__(self, open_: float, high: float, low: float, close: float):
        self.Open = open_
        self.High = high
        self.Low = low
        self.Close = close


class MockAlgorithm:
    """Mock for LEAN algorithm."""

    def __init__(self):
        self.holdings_calls = []
        self.liquidate_calls = []
        self.market_order_calls = []

    def SetHoldings(self, symbol, allocation):
        self.holdings_calls.append((symbol, allocation))

    def Liquidate(self, symbol):
        self.liquidate_calls.append(symbol)

    def MarketOrder(self, symbol, quantity):
        self.market_order_calls.append((symbol, quantity))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_context() -> EvalContext:
    """Create a simple evaluation context."""
    return EvalContext(
        indicators={
            "ema_fast": MockIndicator(105.0),
            "ema_slow": MockIndicator(100.0),
        },
        state={
            "entry_price": 100.0,
            "bars_since_entry": 5,
            "highest_since_entry": 110.0,
        },
        price_bar=MockPriceBar(open_=99.0, high=106.0, low=98.0, close=105.0),
    )


@pytest.fixture
def band_context() -> EvalContext:
    """Create context with band indicators."""
    return EvalContext(
        indicators={
            "bb": MockBandIndicator(upper=110.0, middle=100.0, lower=90.0),
            "kc": MockBandIndicator(upper=108.0, middle=100.0, lower=92.0),
        },
        state={},
        price_bar=MockPriceBar(open_=99.0, high=106.0, low=98.0, close=105.0),
    )


# =============================================================================
# ValueResolver Tests
# =============================================================================


class TestValueResolver:
    """Tests for ValueResolver."""

    def test_resolve_indicator_value(self, simple_context):
        """Test resolving an indicator's current value."""
        resolver = ValueResolver()
        ref = IndicatorValue(indicator_id="ema_fast")
        result = resolver.resolve(ref, simple_context)
        assert result == 105.0

    def test_resolve_indicator_band_upper(self, band_context):
        """Test resolving upper band value."""
        resolver = ValueResolver()
        ref = IndicatorBandValue(indicator_id="bb", band=BandField.UPPER)
        result = resolver.resolve(ref, band_context)
        assert result == 110.0

    def test_resolve_indicator_band_lower(self, band_context):
        """Test resolving lower band value."""
        resolver = ValueResolver()
        ref = IndicatorBandValue(indicator_id="bb", band=BandField.LOWER)
        result = resolver.resolve(ref, band_context)
        assert result == 90.0

    def test_resolve_price_close(self, simple_context):
        """Test resolving close price."""
        resolver = ValueResolver()
        ref = PriceValue(field=PriceField.CLOSE)
        result = resolver.resolve(ref, simple_context)
        assert result == 105.0

    def test_resolve_price_high(self, simple_context):
        """Test resolving high price."""
        resolver = ValueResolver()
        ref = PriceValue(field=PriceField.HIGH)
        result = resolver.resolve(ref, simple_context)
        assert result == 106.0

    def test_resolve_state_value(self, simple_context):
        """Test resolving state variable."""
        resolver = ValueResolver()
        ref = StateValue(state_id="entry_price")
        result = resolver.resolve(ref, simple_context)
        assert result == 100.0

    def test_resolve_literal_value(self, simple_context):
        """Test resolving literal value."""
        resolver = ValueResolver()
        ref = LiteralValue(value=42.5)
        result = resolver.resolve(ref, simple_context)
        assert result == 42.5

    def test_resolve_expression_add(self, simple_context):
        """Test resolving addition expression."""
        resolver = ValueResolver()
        ref = ExpressionValue(
            op="+",
            left=LiteralValue(value=10.0),
            right=LiteralValue(value=5.0),
        )
        result = resolver.resolve(ref, simple_context)
        assert result == 15.0

    def test_resolve_expression_multiply(self, simple_context):
        """Test resolving multiplication expression."""
        resolver = ValueResolver()
        ref = ExpressionValue(
            op="*",
            left=StateValue(state_id="entry_price"),
            right=LiteralValue(value=0.98),  # 2% stop loss
        )
        result = resolver.resolve(ref, simple_context)
        assert result == 98.0

    def test_resolve_nested_expression(self, simple_context):
        """Test resolving nested expression (entry_price - ATR * 2)."""
        resolver = ValueResolver()
        # Simulate entry_price - (atr_value * 2)
        # entry_price = 100.0
        ref = ExpressionValue(
            op="-",
            left=StateValue(state_id="entry_price"),
            right=ExpressionValue(
                op="*",
                left=LiteralValue(value=5.0),  # Mock ATR value
                right=LiteralValue(value=2.0),
            ),
        )
        result = resolver.resolve(ref, simple_context)
        assert result == 90.0  # 100 - (5 * 2)

    def test_resolve_division_by_zero_raises(self, simple_context):
        """Test that division by zero raises ValueError."""
        resolver = ValueResolver()
        ref = ExpressionValue(
            op="/",
            left=LiteralValue(value=10.0),
            right=LiteralValue(value=0.0),
        )
        with pytest.raises(ValueError, match="Division by zero"):
            resolver.resolve(ref, simple_context)

    def test_resolve_unknown_indicator_raises(self, simple_context):
        """Test that unknown indicator raises KeyError."""
        resolver = ValueResolver()
        ref = IndicatorValue(indicator_id="nonexistent")
        with pytest.raises(KeyError, match="Unknown indicator"):
            resolver.resolve(ref, simple_context)


# =============================================================================
# ConditionEvaluator Tests
# =============================================================================


class TestConditionEvaluator:
    """Tests for ConditionEvaluator."""

    def test_compare_greater_than_true(self, simple_context):
        """Test greater than comparison when true."""
        evaluator = ConditionEvaluator()
        condition = CompareCondition(
            left=IndicatorValue(indicator_id="ema_fast"),
            op=CompareOp.GT,
            right=IndicatorValue(indicator_id="ema_slow"),
        )
        result = evaluator.evaluate(condition, simple_context)
        assert result is True  # 105 > 100

    def test_compare_greater_than_false(self, simple_context):
        """Test greater than comparison when false."""
        evaluator = ConditionEvaluator()
        condition = CompareCondition(
            left=IndicatorValue(indicator_id="ema_slow"),
            op=CompareOp.GT,
            right=IndicatorValue(indicator_id="ema_fast"),
        )
        result = evaluator.evaluate(condition, simple_context)
        assert result is False  # 100 > 105

    def test_compare_less_than(self, simple_context):
        """Test less than comparison."""
        evaluator = ConditionEvaluator()
        condition = CompareCondition(
            left=PriceValue(field=PriceField.LOW),
            op=CompareOp.LT,
            right=LiteralValue(value=99.0),
        )
        result = evaluator.evaluate(condition, simple_context)
        assert result is True  # 98 < 99

    def test_all_of_condition_all_true(self, simple_context):
        """Test AllOf when all conditions are true."""
        evaluator = ConditionEvaluator()
        condition = AllOfCondition(
            conditions=[
                CompareCondition(
                    left=IndicatorValue(indicator_id="ema_fast"),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="ema_slow"),
                ),
                CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralValue(value=100.0),
                ),
            ]
        )
        result = evaluator.evaluate(condition, simple_context)
        assert result is True

    def test_all_of_condition_one_false(self, simple_context):
        """Test AllOf when one condition is false."""
        evaluator = ConditionEvaluator()
        condition = AllOfCondition(
            conditions=[
                CompareCondition(
                    left=IndicatorValue(indicator_id="ema_fast"),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="ema_slow"),
                ),
                CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralValue(value=200.0),  # False: 105 > 200
                ),
            ]
        )
        result = evaluator.evaluate(condition, simple_context)
        assert result is False

    def test_any_of_condition_one_true(self, simple_context):
        """Test AnyOf when one condition is true."""
        evaluator = ConditionEvaluator()
        condition = AnyOfCondition(
            conditions=[
                CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralValue(value=200.0),  # False
                ),
                CompareCondition(
                    left=IndicatorValue(indicator_id="ema_fast"),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="ema_slow"),  # True
                ),
            ]
        )
        result = evaluator.evaluate(condition, simple_context)
        assert result is True

    def test_any_of_condition_all_false(self, simple_context):
        """Test AnyOf when all conditions are false."""
        evaluator = ConditionEvaluator()
        condition = AnyOfCondition(
            conditions=[
                CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GT,
                    right=LiteralValue(value=200.0),
                ),
                CompareCondition(
                    left=IndicatorValue(indicator_id="ema_slow"),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="ema_fast"),
                ),
            ]
        )
        result = evaluator.evaluate(condition, simple_context)
        assert result is False

    def test_not_condition(self, simple_context):
        """Test Not condition."""
        evaluator = ConditionEvaluator()
        condition = NotCondition(
            condition=CompareCondition(
                left=IndicatorValue(indicator_id="ema_slow"),
                op=CompareOp.GT,
                right=IndicatorValue(indicator_id="ema_fast"),
            )
        )
        result = evaluator.evaluate(condition, simple_context)
        assert result is True  # not (100 > 105) = not False = True

    def test_price_below_band(self, band_context):
        """Test price below lower band condition."""
        # Modify context to have price below band
        band_context.price_bar = MockPriceBar(open_=88.0, high=89.0, low=85.0, close=87.0)

        evaluator = ConditionEvaluator()
        condition = CompareCondition(
            left=PriceValue(field=PriceField.CLOSE),
            op=CompareOp.LT,
            right=IndicatorBandValue(indicator_id="bb", band=BandField.LOWER),
        )
        result = evaluator.evaluate(condition, band_context)
        assert result is True  # 87 < 90


# =============================================================================
# ActionExecutor Tests
# =============================================================================


class TestActionExecutor:
    """Tests for ActionExecutor."""

    def test_set_holdings_action(self):
        """Test SetHoldings action execution."""
        mock_algo = MockAlgorithm()
        ctx = ExecContext(algorithm=mock_algo, symbol="BTC-USD")
        executor = ActionExecutor()

        action = SetHoldingsAction(allocation=0.95)
        executor.execute(action, ctx)

        assert len(mock_algo.holdings_calls) == 1
        assert mock_algo.holdings_calls[0] == ("BTC-USD", 0.95)

    def test_liquidate_action(self):
        """Test Liquidate action execution."""
        mock_algo = MockAlgorithm()
        ctx = ExecContext(algorithm=mock_algo, symbol="ETH-USD")
        executor = ActionExecutor()

        action = LiquidateAction()
        executor.execute(action, ctx)

        assert len(mock_algo.liquidate_calls) == 1
        assert mock_algo.liquidate_calls[0] == "ETH-USD"


# =============================================================================
# StateOperator Tests
# =============================================================================


class TestStateOperator:
    """Tests for StateOperator."""

    def test_set_state_op(self, simple_context):
        """Test SetStateOp."""
        operator = StateOperator()
        op = SetStateOp(
            state_id="entry_price",
            value=PriceValue(field=PriceField.CLOSE),
        )
        operator.execute(op, simple_context)
        assert simple_context.state["entry_price"] == 105.0

    def test_increment_state_op(self, simple_context):
        """Test IncrementStateOp."""
        operator = StateOperator()
        op = IncrementStateOp(state_id="bars_since_entry")
        operator.execute(op, simple_context)
        assert simple_context.state["bars_since_entry"] == 6  # Was 5, now 6

    def test_increment_state_op_from_none(self, simple_context):
        """Test IncrementStateOp when state is None."""
        simple_context.state["new_counter"] = None
        operator = StateOperator()
        op = IncrementStateOp(state_id="new_counter")
        operator.execute(op, simple_context)
        assert simple_context.state["new_counter"] == 1

    def test_max_state_op_new_higher(self, simple_context):
        """Test MaxStateOp when new value is higher."""
        operator = StateOperator()
        op = MaxStateOp(
            state_id="highest_since_entry",
            value=PriceValue(field=PriceField.HIGH),  # 106
        )
        # Current highest is 110, new is 106, should stay 110
        operator.execute(op, simple_context)
        assert simple_context.state["highest_since_entry"] == 110.0

    def test_max_state_op_current_higher(self, simple_context):
        """Test MaxStateOp when current value is higher."""
        # Set current to lower value
        simple_context.state["highest_since_entry"] = 102.0
        operator = StateOperator()
        op = MaxStateOp(
            state_id="highest_since_entry",
            value=PriceValue(field=PriceField.HIGH),  # 106
        )
        operator.execute(op, simple_context)
        assert simple_context.state["highest_since_entry"] == 106.0

    def test_max_state_op_from_none(self, simple_context):
        """Test MaxStateOp when current state is None."""
        simple_context.state["new_max"] = None
        operator = StateOperator()
        op = MaxStateOp(
            state_id="new_max",
            value=LiteralValue(value=50.0),
        )
        operator.execute(op, simple_context)
        assert simple_context.state["new_max"] == 50.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvaluatorIntegration:
    """Integration tests combining multiple evaluator components."""

    def test_ema_crossover_entry_condition(self, simple_context):
        """Test complete EMA crossover entry condition."""
        evaluator = ConditionEvaluator()

        # Entry: EMA fast > EMA slow AND price > EMA fast
        entry_condition = AllOfCondition(
            conditions=[
                CompareCondition(
                    left=IndicatorValue(indicator_id="ema_fast"),
                    op=CompareOp.GT,
                    right=IndicatorValue(indicator_id="ema_slow"),
                ),
                CompareCondition(
                    left=PriceValue(field=PriceField.CLOSE),
                    op=CompareOp.GTE,
                    right=IndicatorValue(indicator_id="ema_fast"),
                ),
            ]
        )
        result = evaluator.evaluate(entry_condition, simple_context)
        assert result is True  # 105 > 100 AND 105 >= 105

    def test_trailing_stop_exit_condition(self, simple_context):
        """Test trailing stop exit condition."""
        evaluator = ConditionEvaluator()

        # Exit: price < highest_since_entry * 0.95 (5% trailing stop)
        exit_condition = CompareCondition(
            left=PriceValue(field=PriceField.CLOSE),
            op=CompareOp.LT,
            right=ExpressionValue(
                op="*",
                left=StateValue(state_id="highest_since_entry"),
                right=LiteralValue(value=0.95),
            ),
        )
        # highest = 110, threshold = 104.5, close = 105
        result = evaluator.evaluate(exit_condition, simple_context)
        assert result is False  # 105 < 104.5 is False

        # Now test when price drops below threshold
        simple_context.price_bar = MockPriceBar(open_=104.0, high=104.5, low=103.0, close=103.0)
        result = evaluator.evaluate(exit_condition, simple_context)
        assert result is True  # 103 < 104.5 is True

    def test_full_trade_lifecycle(self, simple_context):
        """Test a complete trade lifecycle with state updates."""
        evaluator = ConditionEvaluator()
        state_op = StateOperator()

        # Initial state
        simple_context.state = {
            "entry_price": None,
            "highest_since_entry": None,
            "bars_since_entry": 0,
        }

        # Simulate entry
        entry_condition = CompareCondition(
            left=IndicatorValue(indicator_id="ema_fast"),
            op=CompareOp.GT,
            right=IndicatorValue(indicator_id="ema_slow"),
        )
        can_enter = evaluator.evaluate(entry_condition, simple_context)
        assert can_enter is True

        # On fill: set entry price and initialize tracking
        state_op.execute(
            SetStateOp(state_id="entry_price", value=PriceValue(field=PriceField.CLOSE)),
            simple_context,
        )
        state_op.execute(
            SetStateOp(state_id="highest_since_entry", value=PriceValue(field=PriceField.CLOSE)),
            simple_context,
        )
        assert simple_context.state["entry_price"] == 105.0
        assert simple_context.state["highest_since_entry"] == 105.0

        # On each bar: update highest and increment counter
        simple_context.price_bar = MockPriceBar(open_=105.0, high=108.0, low=104.0, close=107.0)
        state_op.execute(
            MaxStateOp(state_id="highest_since_entry", value=PriceValue(field=PriceField.HIGH)),
            simple_context,
        )
        state_op.execute(IncrementStateOp(state_id="bars_since_entry"), simple_context)

        assert simple_context.state["highest_since_entry"] == 108.0
        assert simple_context.state["bars_since_entry"] == 1

        # Check exit condition (trailing stop at 5%)
        exit_condition = CompareCondition(
            left=PriceValue(field=PriceField.CLOSE),
            op=CompareOp.LT,
            right=ExpressionValue(
                op="*",
                left=StateValue(state_id="highest_since_entry"),
                right=LiteralValue(value=0.95),
            ),
        )
        should_exit = evaluator.evaluate(exit_condition, simple_context)
        assert should_exit is False  # 107 < 102.6 is False

        # Price drops to trigger stop
        simple_context.price_bar = MockPriceBar(open_=102.0, high=102.5, low=100.0, close=101.0)
        should_exit = evaluator.evaluate(exit_condition, simple_context)
        assert should_exit is True  # 101 < 102.6 is True
