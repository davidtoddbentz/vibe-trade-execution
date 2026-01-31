"""Tests for buy-and-hold benchmark computation."""

from vibe_trade_shared.models.data import OHLCVBar

from src.service.backtest_service import _compute_benchmark


def _make_bar(timestamp_ms: int, close: float) -> OHLCVBar:
    """Create a minimal OHLCVBar for testing."""
    return OHLCVBar(t=timestamp_ms, o=close, h=close, l=close, c=close, v=100.0)


def test_benchmark_basic_return():
    """Buy-and-hold return: (200-100)/100 = 1.0 (100% return)."""
    bars = [
        _make_bar(1000, 100.0),
        _make_bar(2000, 150.0),
        _make_bar(3000, 200.0),
    ]
    bench_return, bench_dd, _ = _compute_benchmark(bars, start_timestamp_ms=1000)
    assert bench_return == 1.0
    assert bench_dd == 0.0  # No drawdown - monotonically increasing


def test_benchmark_with_drawdown():
    """Price: 100 -> 80 -> 120. DD = (80-100)/100 = -0.2."""
    bars = [
        _make_bar(1000, 100.0),
        _make_bar(2000, 80.0),
        _make_bar(3000, 120.0),
    ]
    bench_return, bench_dd, _ = _compute_benchmark(bars, start_timestamp_ms=1000)
    assert bench_return == 0.2  # (120-100)/100
    assert bench_dd == -0.2  # (80-100)/100


def test_benchmark_filters_warmup_bars():
    """Only bars at or after start_timestamp_ms should be considered."""
    bars = [
        _make_bar(500, 50.0),   # Warmup - should be excluded
        _make_bar(800, 80.0),   # Warmup - should be excluded
        _make_bar(1000, 100.0), # Trading period starts here
        _make_bar(2000, 150.0),
    ]
    bench_return, bench_dd, _ = _compute_benchmark(bars, start_timestamp_ms=1000)
    assert bench_return == 0.5  # (150-100)/100, NOT (150-50)/50


def test_benchmark_insufficient_data():
    """Less than 2 trading bars returns None."""
    bars = [_make_bar(1000, 100.0)]
    bench_return, bench_dd, _ = _compute_benchmark(bars, start_timestamp_ms=1000)
    assert bench_return is None
    assert bench_dd is None


def test_benchmark_empty_bars():
    """Empty bars returns None."""
    bench_return, bench_dd, _ = _compute_benchmark([], start_timestamp_ms=1000)
    assert bench_return is None
    assert bench_dd is None


def test_benchmark_zero_first_close():
    """Zero first close returns None (avoid division by zero)."""
    bars = [
        _make_bar(1000, 0.0),
        _make_bar(2000, 100.0),
    ]
    bench_return, bench_dd, _ = _compute_benchmark(bars, start_timestamp_ms=1000)
    assert bench_return is None
    assert bench_dd is None


def test_benchmark_negative_return():
    """Price drops: 100 -> 60. Return = -0.4."""
    bars = [
        _make_bar(1000, 100.0),
        _make_bar(2000, 80.0),
        _make_bar(3000, 60.0),
    ]
    bench_return, bench_dd, _ = _compute_benchmark(bars, start_timestamp_ms=1000)
    assert bench_return == -0.4  # (60-100)/100
    assert bench_dd == -0.4  # Max DD same as return since monotonically decreasing
