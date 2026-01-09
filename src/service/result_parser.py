"""Parse LEAN backtest results into our standard format."""

import json
from datetime import datetime
from pathlib import Path

from src.models.backtest_result import (
    BacktestResult,
    ChartSeries,
    EquityPoint,
    Order,
    OrderDirection,
    OrderStatus,
    PerformanceStatistics,
    Trade,
)


def parse_lean_results(results_dir: Path, strategy_id: str) -> BacktestResult:
    """Parse LEAN backtest results from a directory.

    LEAN produces several files:
    - <algorithm>-<id>.json - Main result file
    - log.txt - Runtime logs

    Args:
        results_dir: Path to the LEAN results directory
        strategy_id: ID of the strategy being backtested

    Returns:
        BacktestResult with parsed data
    """
    result = BacktestResult(
        backtest_id=f"bt-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        strategy_id=strategy_id,
        strategy_name="Unknown",
        symbol="BTC-USD",
        start_date=datetime.now(),
        end_date=datetime.now(),
        resolution="Hour",
        initial_cash=100000.0,
    )

    # Find the main result JSON file
    json_files = list(results_dir.glob("*.json"))
    result_json = None

    for f in json_files:
        if f.name not in ("config.json", "strategy_ir.json"):
            result_json = f
            break

    if result_json and result_json.exists():
        with open(result_json) as f:
            lean_result = json.load(f)
            result = _parse_lean_json(lean_result, result)

    # Parse logs
    log_file = results_dir / "log.txt"
    if log_file.exists():
        result.logs = _parse_log_file(log_file)

    # Parse custom output (our StrategyRuntime writes this)
    custom_output = results_dir / "strategy_output.json"
    if custom_output.exists():
        with open(custom_output) as f:
            custom = json.load(f)
            result = _merge_custom_output(custom, result)

    return result


def _parse_lean_json(lean: dict, result: BacktestResult) -> BacktestResult:
    """Parse the main LEAN result JSON."""
    # Statistics
    stats = lean.get("Statistics", {})
    runtime_stats = lean.get("RuntimeStatistics", {})

    result.statistics = PerformanceStatistics(
        total_return=_parse_pct(stats.get("Total Net Profit", "0%")),
        annual_return=_parse_pct(stats.get("Compounding Annual Return", "0%")),
        sharpe_ratio=_parse_float(stats.get("Sharpe Ratio", "0")),
        sortino_ratio=_parse_float(stats.get("Sortino Ratio", "0")),
        max_drawdown=_parse_pct(stats.get("Drawdown", "0%")),
        total_trades=int(stats.get("Total Trades", "0")),
        win_rate=_parse_pct(stats.get("Win Rate", "0%")),
        profit_factor=_parse_float(stats.get("Profit-Loss Ratio", "0")),
        average_win=_parse_pct(stats.get("Average Win", "0%")),
        average_loss=_parse_pct(stats.get("Average Loss", "0%")),
        net_profit=_parse_float(runtime_stats.get("Net Profit", "$0").replace("$", "").replace(",", "")),
    )

    # Calculate derived stats
    total = result.statistics.total_trades
    if total > 0:
        result.statistics.winning_trades = int(total * result.statistics.win_rate / 100)
        result.statistics.losing_trades = total - result.statistics.winning_trades

    # Final equity
    result.final_equity = _parse_float(
        runtime_stats.get("Equity", "$100,000").replace("$", "").replace(",", "")
    )

    # Orders
    orders = lean.get("Orders", {})
    for order_id, order_data in orders.items():
        result.orders.append(_parse_order(order_id, order_data))

    # Charts - equity curve
    charts = lean.get("Charts", {})
    if "Strategy Equity" in charts:
        equity_series = charts["Strategy Equity"].get("Series", {})
        if "Equity" in equity_series:
            values = equity_series["Equity"].get("Values", [])
            for point in values:
                result.equity_curve.append(
                    EquityPoint(
                        time=datetime.fromtimestamp(point.get("x", 0)),
                        equity=point.get("y", 0),
                        cash=0,  # Not in this series
                        holdings_value=0,
                        drawdown=0,
                        drawdown_duration=0,
                    )
                )

    # Trades from TotalPerformance
    perf = lean.get("TotalPerformance", {})
    if "ClosedTrades" in perf:
        for i, trade_data in enumerate(perf["ClosedTrades"]):
            result.trades.append(_parse_trade(i, trade_data))

    return result


def _parse_order(order_id: str, data: dict) -> Order:
    """Parse a LEAN order."""
    direction = OrderDirection.BUY if data.get("Direction") == "Buy" else OrderDirection.SELL
    status_map = {
        "Filled": OrderStatus.FILLED,
        "Canceled": OrderStatus.CANCELLED,
        "Invalid": OrderStatus.INVALID,
    }

    return Order(
        order_id=order_id,
        symbol=data.get("Symbol", {}).get("Value", ""),
        direction=direction,
        quantity=abs(data.get("Quantity", 0)),
        order_type=data.get("Type", "Market"),
        time=datetime.fromisoformat(data.get("Time", "2024-01-01T00:00:00")),
        status=status_map.get(data.get("Status"), OrderStatus.FILLED),
        fill_price=data.get("Price", 0),
        fill_quantity=abs(data.get("Quantity", 0)),
        fees=0,  # Fees in separate field
    )


def _parse_trade(idx: int, data: dict) -> Trade:
    """Parse a LEAN closed trade."""
    direction = (
        OrderDirection.BUY if data.get("Direction") == "Long" else OrderDirection.SELL
    )

    return Trade(
        trade_id=f"trade-{idx}",
        symbol=data.get("Symbol", ""),
        direction=direction,
        entry_time=datetime.fromisoformat(data.get("EntryTime", "2024-01-01T00:00:00")),
        entry_price=data.get("EntryPrice", 0),
        entry_quantity=abs(data.get("Quantity", 0)),
        exit_time=datetime.fromisoformat(data.get("ExitTime", "2024-01-01T00:00:00")),
        exit_price=data.get("ExitPrice", 0),
        exit_quantity=abs(data.get("Quantity", 0)),
        pnl=data.get("ProfitLoss", 0),
        pnl_percent=data.get("ProfitLossPercentage", 0) * 100,
        duration_bars=data.get("Duration", 0),
        fees=data.get("TotalFees", 0),
    )


def _parse_log_file(log_path: Path) -> list[str]:
    """Parse LEAN log file."""
    logs = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(line)
    return logs


def _merge_custom_output(custom: dict, result: BacktestResult) -> BacktestResult:
    """Merge our custom StrategyRuntime output."""
    # Custom output from StrategyRuntime.OnEndOfAlgorithm
    if "strategy_name" in custom:
        result.strategy_name = custom["strategy_name"]
    if "final_equity" in custom:
        result.final_equity = custom["final_equity"]
    if "trades" in custom:
        for i, t in enumerate(custom["trades"]):
            result.trades.append(
                Trade(
                    trade_id=f"custom-{i}",
                    symbol=t.get("symbol", "BTC-USD"),
                    direction=OrderDirection(t.get("direction", "buy")),
                    entry_time=datetime.fromisoformat(t["entry_time"]),
                    entry_price=t["entry_price"],
                    entry_quantity=t.get("quantity", 0),
                    exit_time=datetime.fromisoformat(t["exit_time"]) if t.get("exit_time") else None,
                    exit_price=t.get("exit_price"),
                    pnl=t.get("pnl"),
                    exit_reason=t.get("exit_reason"),
                )
            )
    if "equity_curve" in custom:
        for pt in custom["equity_curve"]:
            result.equity_curve.append(
                EquityPoint(
                    time=datetime.fromisoformat(pt["time"]),
                    equity=pt["equity"],
                    cash=pt.get("cash", 0),
                    holdings_value=pt.get("holdings", 0),
                    drawdown=pt.get("drawdown", 0),
                    drawdown_duration=0,
                )
            )

    return result


def _parse_pct(value: str) -> float:
    """Parse percentage string like '5.5%' to float."""
    if not value:
        return 0.0
    return float(value.replace("%", "").replace(",", ""))


def _parse_float(value: str) -> float:
    """Parse float string."""
    if not value:
        return 0.0
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return 0.0
