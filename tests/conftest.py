"""Shared test fixtures and helpers."""

import os
from typing import Iterator

import httpx
import pytest
from vibe_trade_shared.models import Card, Strategy
from vibe_trade_shared.models.strategy import Attachment


def calculate_warmup_bars(indicators: list[dict]) -> int:
    """Calculate the warmup period required for a set of indicators.

    Based on QuantConnect LEAN indicator WarmUpPeriod values:
    - SMA, EMA, ATR, BollingerBands, KeltnerChannels, DonchianChannels: WarmUpPeriod = period
    - RSI, ROC: WarmUpPeriod = period + 1
    - ADX: WarmUpPeriod = period * 2

    The warmup_bars is the 0-indexed data bar when indicators first become ready,
    which equals WarmUpPeriod - 1. In StrategyRuntime.py, bar_count only starts
    incrementing after _indicators_ready() returns True, so:
        trading_bar = data_bar - warmup_bars

    Args:
        indicators: List of indicator dicts with 'type' and 'period' keys

    Returns:
        Number of warmup bars (0-indexed first ready bar)
    """
    if not indicators:
        return 0

    max_warmup = 0
    for ind in indicators:
        ind_type = ind.get("type", "").upper()
        period = ind.get("period", 0)

        # Map indicator type to WarmUpPeriod formula
        if ind_type in ("RSI", "ROC"):
            # WarmUpPeriod = period + 1
            warmup_period = period + 1
        elif ind_type == "ADX":
            # WarmUpPeriod = period * 2
            warmup_period = period * 2
        else:
            # SMA, EMA, ATR, BB, KC, DC, etc: WarmUpPeriod = period
            warmup_period = period

        # First ready bar (0-indexed) = WarmUpPeriod - 1
        first_ready_bar = warmup_period - 1

        if first_ready_bar > max_warmup:
            max_warmup = first_ready_bar

    return max_warmup


def calculate_trading_bar(data_bar: int, indicators: list[dict]) -> int:
    """Convert a data bar index to a trading bar index.

    In StrategyRuntime.py, bar_count (trading bar) only starts incrementing
    after _indicators_ready() returns True. This function converts from
    the raw data bar index to the trading bar index.

    Args:
        data_bar: 0-indexed data bar
        indicators: List of indicator dicts with 'type' and 'period' keys

    Returns:
        Trading bar index (bar_count in StrategyRuntime.py)
    """
    warmup = calculate_warmup_bars(indicators)
    return data_bar - warmup


def make_strategy(
    cards_dict: dict[str, dict],
    name: str = "Test Strategy",
    symbol: str = "BTC-USD",
) -> tuple[Strategy, dict[str, Card]]:
    """Create a strategy with cards from a simplified dict specification.

    Args:
        cards_dict: Dict mapping card_id to card spec with keys:
            - type: Card type (e.g., "entry.rule_trigger")
            - slots: Card slots dict
            - name: Optional card name (defaults to card_id)
        name: Strategy name (defaults to "Test Strategy")
        symbol: Trading symbol (defaults to "BTC-USD")

    Returns:
        Tuple of (Strategy, dict of Cards)

    Example:
        strategy, cards = make_strategy({
            "entry_1": {
                "type": "entry.rule_trigger",
                "slots": {
                    "context": {"tf": "1h", "symbol": "BTC-USD"},
                    "event": {"condition": {...}},
                    "action": {"direction": "long", "position_policy": {"mode": "single"}}
                }
            }
        })
    """
    attachments = []
    cards = {}

    for card_id, card_spec in cards_dict.items():
        role = card_spec["type"].split(".")[0]  # entry, exit, gate, overlay
        attachments.append(Attachment(card_id=card_id, role=role, enabled=True, overrides={}))
        cards[card_id] = Card(
            id=card_id,
            type=card_spec["type"],
            name=card_spec.get("name", card_id),
            schema_etag="test",
            slots=card_spec["slots"],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T00:00:00Z",
        )

    # Generate strategy ID from name
    strategy_id = f"test-{name.lower().replace(' ', '-')}" if name != "Test Strategy" else "test-strategy"

    strategy = Strategy(
        id=strategy_id,
        name=name,
        universe=[symbol],
        attachments=attachments,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )

    return strategy, cards


# ---------------------------------------------------------------------------
# E2E Test Fixtures (for parallel execution)
# ---------------------------------------------------------------------------

def _get_worker_id() -> int:
    """Get pytest-xdist worker ID, or 0 if not using xdist."""
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    # Extract number from "gw0", "gw1", etc.
    try:
        return int(worker_id.replace("gw", ""))
    except ValueError:
        return 0


@pytest.fixture(scope="session")
def lean_container():
    """Create one LEAN container per worker (dynamic port allocation).
    
    First checks for an existing LEAN service (e.g., from make local-up)
    at standard ports. Falls back to creating a new container per worker.
    """
    # Check for existing LEAN service first (from make local-up or docker-compose)
    for port in [8083, 8081]:
        try:
            r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
            if r.status_code == 200:
                yield f"http://localhost:{port}/backtest"
                return
        except (httpx.ConnectError, httpx.TimeoutException):
            continue

    try:
        import docker
    except ImportError:
        pytest.skip("docker package not installed. Install with: uv sync --group dev")
    
    worker_id = _get_worker_id()
    client = docker.from_env()
    container_name = f"lean-test-{worker_id}"
    
    # Check if container already exists (from previous test run)
    try:
        existing = client.containers.get(container_name)
        if existing.status == "running":
            # Get the port
            ports = existing.attrs["NetworkSettings"]["Ports"]
            if "8080/tcp" in ports and ports["8080/tcp"]:
                port = ports["8080/tcp"][0]["HostPort"]
                # Verify it's healthy
                try:
                    r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
                    if r.status_code == 200:
                        yield f"http://localhost:{port}/backtest"
                        return
                except (httpx.ConnectError, httpx.TimeoutException):
                    # Container exists but not healthy, remove it
                    existing.remove(force=True)
    except docker.errors.NotFound:
        pass
    
    # Create new container with dynamic port allocation
    try:
        container = client.containers.run(
            "lean-backtest-service:latest",
            name=container_name,
            ports={"8080": None},  # Dynamic port allocation
            detach=True,
            remove=True,  # Auto-remove on stop
            environment={"PYTHONPATH": "/Lean"},
        )
        
        # Get assigned port
        container.reload()  # Refresh to get port assignment
        ports = container.attrs["NetworkSettings"]["Ports"]
        if "8080/tcp" not in ports or not ports["8080/tcp"]:
            container.remove(force=True)
            pytest.skip(f"Failed to get port for container {container_name}")
        
        port = ports["8080/tcp"][0]["HostPort"]
        url = f"http://localhost:{port}/backtest"
        
        # Wait for health (max 30 seconds)
        import time
        for attempt in range(30):
            try:
                r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
                if r.status_code == 200:
                    yield url
                    return
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError, httpx.RemoteProtocolError):
                if attempt < 29:  # Don't sleep on last attempt
                    time.sleep(1)
                continue
        
        # Not healthy, cleanup and skip
        try:
            container.remove(force=True)
        except:
            pass
        pytest.skip(f"Container {container_name} on port {port} not healthy after 30s. Check logs: docker logs {container_name}")
        
    except docker.errors.ImageNotFound:
        pytest.skip(
            "LEAN image 'lean-backtest-service:latest' not found. "
            "Build it with: cd ../vibe-trade-lean && make build"
        )
    except docker.errors.APIError as e:
        pytest.skip(f"Failed to create container: {e}")


@pytest.fixture(scope="session")
def lean_url(lean_container):
    """Session-scoped LEAN URL (uses dynamic container per worker)."""
    return lean_container


@pytest.fixture(scope="session")
def backtest_service(lean_url: str):
    """Session-scoped BacktestService (reused across tests).
    
    Tests should provide their own MockDataService per test for isolation.
    """
    from src.service.backtest_service import BacktestService
    
    return BacktestService(
        data_service=None,  # Tests provide their own MockDataService
        backtest_url=lean_url,
    )
