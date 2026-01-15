# vibe-trade-execution

Execution engine for trading strategies using QuantConnect LEAN.

## Overview

This service translates trading strategies created by `vibe-trade-agent` into executable LEAN algorithms. Strategies are defined as JSON schemas, translated to an intermediate representation (IR), and executed in isolated Docker containers.

### Translation Pipeline

```
MCP Schema (JSON) → IRTranslator → StrategyIR → JSON → LEAN StrategyRuntime
```

1. **Schema**: Strategy defined via `vibe-trade-mcp` with entry/exit/gate/overlay cards
2. **IR Translation**: `src/translator/` converts schema to typed IR dataclasses
3. **JSON Serialization**: IR exported as JSON for LEAN consumption
4. **LEAN Runtime**: `StrategyRuntime.py` interprets IR and executes trades

## Quick Start

### Prerequisites

- Docker installed and running
- Python 3.10+
- `uv` package manager

### Setup

```bash
# Install dependencies
make install

# Pull LEAN Docker image
make lean-setup
```

### Run a Backtest

```bash
# Run example moving average crossover
make lean-backtest ALGO=SimpleMaCrossover

# With custom parameters
make lean-backtest ALGO=SimpleMaCrossover START=20240101 END=20241231 CASH=100000
```

## Project Structure

```
vibe-trade-execution/
├── src/
│   ├── translator/           # Schema → IR translation
│   │   ├── ir.py             # IR dataclasses (conditions, actions, state)
│   │   ├── ir_translator.py  # Schema JSON → StrategyIR conversion
│   │   └── evaluator.py      # Python evaluator (for simulation tests)
│   ├── lean_runner/          # LEAN Docker container management
│   │   ├── engine.py         # Run backtests via Docker
│   │   ├── container_manager.py  # Live strategy container management
│   │   └── cli.py            # CLI for running strategies
│   └── service/              # (Future: execution service API)
├── lean/
│   ├── Algorithms/
│   │   ├── StrategyRuntime.py    # Core runtime - interprets IR JSON
│   │   ├── SimpleMaCrossover.py  # Example algorithm
│   │   └── EmaDeterministicTest.py  # Integration test fixture
│   ├── Data/                 # Market data files
│   ├── DataFeeds/            # Custom data feeds
│   └── Results/              # Backtest output (gitignored)
└── tests/                    # 239 tests
    ├── test_lean_e2e.py      # 5 E2E tests (actual LEAN execution)
    ├── test_lean_backtest_integration.py  # Integration tests
    ├── test_strategy_simulation.py  # Simulation tests (Python evaluator)
    └── test_*.py             # Unit/coverage tests
```

## Testing

The repo uses a three-tier testing strategy:

### 1. Unit Tests
Test individual components (IR types, translator, evaluator).

### 2. Simulation Tests
Test the Python evaluator with mock data - fast (~35s total):
```bash
uv run pytest tests/test_strategy_simulation.py -v
```

### 3. E2E LEAN Tests
Test actual LEAN execution with deterministic data (~6s per test):
```bash
uv run pytest tests/test_lean_e2e.py -v
```

### Run All Tests
```bash
make test
# or
uv run pytest tests/ -v
```

## Architecture Notes

### Dual Evaluation Implementations

The evaluation logic exists in two places:
- `src/translator/evaluator.py` - Python implementation for fast simulation tests
- `lean/Algorithms/StrategyRuntime.py` - LEAN implementation for actual execution

This is intentional: simulation tests run in seconds, E2E tests verify LEAN correctness.

### IR Schema Coverage

The IR supports full composability through `rule_trigger` archetypes + `ConditionSpec`:
- **6 condition types**: regime, allOf, anyOf, not, sequence, band_event
- **18 regime metrics**: trend, volatility, volume, price levels
- **5 band event types**: touch, cross_in, cross_out, distance, reentry
- **3 band indicators**: Bollinger, Keltner, Donchian

## Container Management

```bash
# List running strategies
make lean-list

# Start a strategy
make lean-start ALGO=SimpleMaCrossover ID=my-strategy

# Stop a strategy
make lean-stop ID=my-strategy

# View logs
make lean-logs ID=my-strategy
```

## Live Mode (Future)

For live trading with real-time data:
- Strategies consume data from GCP Pub/Sub topics
- Topic format: `vibe-trade-candles-{symbol}-1m`
- Requires custom data queue handler (not yet implemented)
- See `lean/DataFeeds/` for Pub/Sub integration components

## Development

```bash
# Install dev dependencies
make install

# Run linter
make lint

# Auto-fix lint issues
make lint-fix

# Format code
make format

# All checks (lint + format + test)
make check
```
