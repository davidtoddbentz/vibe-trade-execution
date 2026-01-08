"""
CLI for managing local LEAN strategies.

This helps you understand Docker controls and data flow.
"""

import argparse
import logging
from pathlib import Path

from .container_manager import ContainerManager

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def cmd_list(args):
    """List all running strategies."""
    manager = ContainerManager()
    strategies = manager.list_running_strategies()

    if not strategies:
        print("No running strategies")
        return

    print(f"\n{'Container ID':<15} {'Strategy ID':<20} {'Algorithm':<20} {'Status':<10}")
    print("-" * 70)
    for strategy in strategies:
        print(
            f"{strategy.container_id:<15} "
            f"{strategy.strategy_id:<20} "
            f"{strategy.algorithm_name:<20} "
            f"{strategy.status:<10}"
        )


def cmd_start(args):
    """Start a strategy locally."""
    if not args.algorithm:
        logger.error("Algorithm name required (use --algorithm)")
        return

    # For now, we'll read from Algorithms directory
    # Later, this will generate from strategy schema
    algorithm_file = (
        Path(__file__).parent.parent.parent / "lean" / "Algorithms" / f"{args.algorithm}.py"
    )

    if not algorithm_file.exists():
        logger.error(f"Algorithm not found: {algorithm_file}")
        return

    algorithm_code = algorithm_file.read_text()
    strategy_id = args.strategy_id or f"local-{args.algorithm.lower()}"

    manager = ContainerManager()
    try:
        info = manager.start_strategy(
            strategy_id=strategy_id,
            algorithm_name=args.algorithm,
            algorithm_code=algorithm_code,
        )
        print(f"✅ Started strategy: {strategy_id}")
        print(f"   Container: {info.container_id}")
        print(f"   Status: {info.status}")
    except Exception as e:
        logger.error(f"Failed to start strategy: {e}")


def cmd_stop(args):
    """Stop a running strategy."""
    manager = ContainerManager()
    if manager.stop_strategy(args.strategy_id):
        print(f"✅ Stopped strategy: {args.strategy_id}")
    else:
        print(f"❌ Failed to stop strategy: {args.strategy_id}")


def cmd_logs(args):
    """View logs from a strategy."""
    manager = ContainerManager()
    logs = manager.get_logs(args.strategy_id, tail=args.tail)
    print(logs)


def cmd_stats(args):
    """View resource stats for a strategy."""
    manager = ContainerManager()
    stats = manager.get_stats(args.strategy_id)

    if not stats:
        print(f"No stats available for {args.strategy_id}")
        return

    print(f"\nResource Usage for {args.strategy_id}:")
    print(f"  CPU: {stats.get('CPUPerc', 'N/A')}")
    print(f"  Memory: {stats.get('MemUsage', 'N/A')}")
    print(f"  Network I/O: {stats.get('NetIO', 'N/A')}")


def cmd_remove(args):
    """Remove a strategy container."""
    manager = ContainerManager()
    if manager.remove_strategy(args.strategy_id):
        print(f"✅ Removed strategy: {args.strategy_id}")
    else:
        print(f"❌ Failed to remove strategy: {args.strategy_id}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Manage local LEAN strategies")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    subparsers.add_parser("list", help="List running strategies")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start a strategy")
    start_parser.add_argument(
        "--algorithm", required=True, help="Algorithm name (e.g., SimpleMaCrossover)"
    )
    start_parser.add_argument("--strategy-id", help="Strategy ID (default: auto-generated)")

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop a strategy")
    stop_parser.add_argument("strategy_id", help="Strategy ID")

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="View strategy logs")
    logs_parser.add_argument("strategy_id", help="Strategy ID")
    logs_parser.add_argument("--tail", type=int, default=100, help="Number of lines to show")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="View strategy resource stats")
    stats_parser.add_argument("strategy_id", help="Strategy ID")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a strategy container")
    remove_parser.add_argument("strategy_id", help="Strategy ID")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "list": cmd_list,
        "start": cmd_start,
        "stop": cmd_stop,
        "logs": cmd_logs,
        "stats": cmd_stats,
        "remove": cmd_remove,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
