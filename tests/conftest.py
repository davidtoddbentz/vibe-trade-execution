"""Shared test fixtures and helpers."""

import os
import time

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
# E2E Test Fixtures (with auto-restart on container crash)
# ---------------------------------------------------------------------------

def _get_worker_id() -> int:
    """Get pytest-xdist worker ID, or 0 if not using xdist."""
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    try:
        return int(worker_id.replace("gw", ""))
    except ValueError:
        return 0


class _LeanContainerManager:
    """Manages a LEAN Docker container with auto-restart on crash.

    The LEAN .NET runtime accumulates memory across sequential subprocess runs,
    causing the container to crash after ~10-15 backtests. This manager detects
    crashes and transparently restarts the container with a new port.
    """

    def __init__(self):
        self._external_url: str | None = None
        self._client = None
        self._container = None
        self._container_name: str | None = None
        self._port: str | None = None

    def _check_external_service(self) -> str | None:
        """Check for a pre-existing LEAN service on standard ports."""
        for port in [8083, 8081]:
            try:
                r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
                if r.status_code == 200:
                    return f"http://localhost:{port}/backtest"
            except (httpx.ConnectError, httpx.TimeoutException):
                continue
        return None

    def _start_container(self) -> str:
        """Start (or restart) a managed Docker container. Returns backtest URL."""
        try:
            import docker
        except ImportError:
            pytest.skip("docker package not installed. Install with: uv sync --group dev")

        if self._client is None:
            self._client = docker.from_env()
            worker_id = _get_worker_id()
            self._container_name = f"lean-test-{worker_id}"

        # Remove old container if it exists (retry to handle Docker race conditions)
        for _attempt in range(3):
            try:
                old = self._client.containers.get(self._container_name)
                old.remove(force=True)
                time.sleep(3)
            except Exception:
                break  # Container doesn't exist, good to go

        try:
            self._container = self._client.containers.run(
                "lean-backtest-service:latest",
                name=self._container_name,
                ports={"8080": None},
                detach=True,
                environment={"PYTHONPATH": "/Lean"},
            )

            self._container.reload()
            ports = self._container.attrs["NetworkSettings"]["Ports"]
            if "8080/tcp" not in ports or not ports["8080/tcp"]:
                self._container.remove(force=True)
                pytest.skip(f"Failed to get port for {self._container_name}")

            self._port = ports["8080/tcp"][0]["HostPort"]
            url = f"http://localhost:{self._port}/backtest"

            for attempt in range(30):
                try:
                    r = httpx.get(
                        f"http://localhost:{self._port}/health", timeout=2.0
                    )
                    if r.status_code == 200:
                        return url
                except (
                    httpx.ConnectError,
                    httpx.TimeoutException,
                    httpx.ReadError,
                    httpx.RemoteProtocolError,
                ):
                    if attempt < 29:
                        time.sleep(1)
                    continue

            pytest.skip(f"Container {self._container_name} not healthy after 30s")

        except Exception as e:
            if "ImageNotFound" in type(e).__name__:
                pytest.skip(
                    "LEAN image 'lean-backtest-service:latest' not found. "
                    "Build it with: cd ../vibe-trade-lean && make build"
                )
            pytest.skip(f"Failed to create container: {e}")
        return ""  # unreachable

    def get_url(self) -> str:
        """Get a healthy LEAN backtest URL, restarting container if needed."""
        # If using an external service, check health
        if self._external_url is not None:
            try:
                port = self._external_url.rsplit(":", 1)[-1].split("/")[0]
                r = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
                if r.status_code == 200:
                    return self._external_url
            except (httpx.ConnectError, httpx.TimeoutException):
                # External service died -- fall through to managed container
                self._external_url = None

        # If we have a managed container, check health
        if self._port is not None:
            try:
                r = httpx.get(
                    f"http://localhost:{self._port}/health", timeout=2.0
                )
                if r.status_code == 200:
                    return f"http://localhost:{self._port}/backtest"
            except (httpx.ConnectError, httpx.TimeoutException):
                pass  # Container crashed -- restart below

        return self._start_container()

    def initialize(self) -> str:
        """Initial setup: check for external service, then start container."""
        ext = self._check_external_service()
        if ext:
            self._external_url = ext
            return ext
        return self._start_container()

    def cleanup(self):
        """Remove managed container."""
        if self._container is not None:
            try:
                self._container.remove(force=True)
            except Exception:
                pass


@pytest.fixture(scope="session")
def _lean_manager():
    """Session-scoped container manager (handles restarts)."""
    manager = _LeanContainerManager()
    manager.initialize()
    yield manager
    manager.cleanup()


@pytest.fixture
def lean_url(_lean_manager: _LeanContainerManager):
    """Per-test LEAN URL that auto-restarts the container if it crashed.

    Each test checks container health before use. If the container died
    (LEAN .NET runtime memory accumulation), it transparently restarts.
    """
    return _lean_manager.get_url()
