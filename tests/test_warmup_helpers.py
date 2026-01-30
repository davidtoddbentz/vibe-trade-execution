"""Tests for warmup calculation helper functions."""


from tests.conftest import calculate_trading_bar, calculate_warmup_bars


class TestCalculateWarmupBars:
    """Test warmup bar calculation based on QuantConnect WarmUpPeriod values."""

    def test_sma_warmup(self):
        """SMA has WarmUpPeriod = period, so first ready bar = period - 1."""
        indicators = [{"type": "SMA", "period": 20}]
        # SMA(20): WarmUpPeriod=20, first ready bar=19
        assert calculate_warmup_bars(indicators) == 19

    def test_ema_warmup(self):
        """EMA has WarmUpPeriod = period, so first ready bar = period - 1."""
        indicators = [{"type": "EMA", "period": 10}]
        # EMA(10): WarmUpPeriod=10, first ready bar=9
        assert calculate_warmup_bars(indicators) == 9

    def test_rsi_warmup(self):
        """RSI has WarmUpPeriod = period + 1, so first ready bar = period."""
        indicators = [{"type": "RSI", "period": 14}]
        # RSI(14): WarmUpPeriod=15, first ready bar=14
        assert calculate_warmup_bars(indicators) == 14

    def test_roc_warmup(self):
        """ROC has WarmUpPeriod = period + 1, so first ready bar = period."""
        indicators = [{"type": "ROC", "period": 10}]
        # ROC(10): WarmUpPeriod=11, first ready bar=10
        assert calculate_warmup_bars(indicators) == 10

    def test_adx_warmup(self):
        """ADX has WarmUpPeriod = period * 2, so first ready bar = 2*period - 1."""
        indicators = [{"type": "ADX", "period": 14}]
        # ADX(14): WarmUpPeriod=28, first ready bar=27
        assert calculate_warmup_bars(indicators) == 27

    def test_bollinger_warmup(self):
        """BollingerBands has WarmUpPeriod = period, so first ready bar = period - 1."""
        indicators = [{"type": "BollingerBands", "period": 20}]
        # BB(20): WarmUpPeriod=20, first ready bar=19
        assert calculate_warmup_bars(indicators) == 19

    def test_atr_warmup(self):
        """ATR has WarmUpPeriod = period, so first ready bar = period - 1."""
        indicators = [{"type": "ATR", "period": 14}]
        # ATR(14): WarmUpPeriod=14, first ready bar=13
        assert calculate_warmup_bars(indicators) == 13

    def test_case_insensitive(self):
        """Indicator type matching should be case insensitive."""
        indicators = [{"type": "rsi", "period": 14}]
        assert calculate_warmup_bars(indicators) == 14

    def test_multiple_indicators_uses_max(self):
        """Warmup should be the maximum across all indicators."""
        indicators = [
            {"type": "EMA", "period": 10},  # first ready bar = 9
            {"type": "RSI", "period": 14},  # first ready bar = 14
            {"type": "SMA", "period": 20},  # first ready bar = 19
        ]
        # Max is SMA(20) with first ready bar = 19
        assert calculate_warmup_bars(indicators) == 19

    def test_rsi_dominates_over_ema(self):
        """RSI(14) first ready bar (14) > EMA(10) first ready bar (9)."""
        indicators = [
            {"type": "EMA", "period": 10},  # first ready bar = 9
            {"type": "RSI", "period": 14},  # first ready bar = 14
        ]
        assert calculate_warmup_bars(indicators) == 14

    def test_empty_indicators(self):
        """No indicators means no warmup needed."""
        assert calculate_warmup_bars([]) == 0

    def test_missing_period(self):
        """Indicator without period should use 0."""
        indicators = [{"type": "SMA"}]
        # SMA with period=0: WarmUpPeriod=0, first ready bar=-1 -> 0
        assert calculate_warmup_bars(indicators) == 0


class TestCalculateTradingBar:
    """Test conversion from data bar to trading bar."""

    def test_sma_trading_bar(self):
        """Trading bar = data bar - warmup bars."""
        indicators = [{"type": "SMA", "period": 20}]
        # SMA(20): warmup=19, data bar 26 -> trading bar 7
        assert calculate_trading_bar(26, indicators) == 7

    def test_rsi_trading_bar(self):
        """RSI(14): warmup=14, data bar 14 -> trading bar 0."""
        indicators = [{"type": "RSI", "period": 14}]
        # RSI(14): warmup=14, data bar 14 -> trading bar 0
        assert calculate_trading_bar(14, indicators) == 0
        # data bar 15 -> trading bar 1
        assert calculate_trading_bar(15, indicators) == 1

    def test_no_indicators(self):
        """No indicators means data bar = trading bar."""
        assert calculate_trading_bar(5, []) == 5

    def test_multiple_indicators(self):
        """Trading bar uses max warmup across all indicators."""
        indicators = [
            {"type": "EMA", "period": 10},  # warmup = 9
            {"type": "RSI", "period": 14},  # warmup = 14
        ]
        # Max warmup = 14, data bar 20 -> trading bar 6
        assert calculate_trading_bar(20, indicators) == 6
