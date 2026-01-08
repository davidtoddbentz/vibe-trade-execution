.PHONY: install locally test lint format format-check check ci clean backtest lean-setup lean-test lean-backtest lean-list lean-start lean-stop lean-logs build-image

install:
	uv sync --all-groups

# Setup for local development: install deps, fix linting, and format code
locally: install lint-fix format
	@echo "✅ Local setup complete!"

# Run a backtest
# Usage: make backtest CONFIG=examples/strategy_bundle.json CSV=path/to/data.csv [CASH=100000] [PLOT=1]
backtest:
	@if [ -z "$(CONFIG)" ] || [ -z "$(CSV)" ]; then \
		echo "❌ Usage: make backtest CONFIG=<bundle.json> CSV=<data.csv> [CASH=<amount>] [PLOT=1]"; \
		echo "   Example: make backtest CONFIG=examples/strategy_bundle.json CSV=data/BTC-USD.csv"; \
		exit 1; \
	fi
	@CASH=$${CASH:-100000} && \
		if [ "$(PLOT)" = "1" ]; then \
			uv run python run_backtest.py --config $(CONFIG) --csv $(CSV) --cash $$CASH --plot; \
		else \
			uv run python run_backtest.py --config $(CONFIG) --csv $(CSV) --cash $$CASH; \
		fi

test:
	uv run python -m pytest tests/ -v

test-cov:
	uv run python -m pytest tests/ --cov=schema_bt --cov-report=term-missing --cov-report=html --cov-fail-under=60

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

check: lint format-check test-cov
	@echo "✅ All checks passed!"

ci: lint-fix format-check test-cov
	@echo "✅ CI checks passed!"

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov/ coverage.xml
	rm -rf *.egg-info build/ dist/

# LEAN Setup and Management
lean-setup:
	@echo "📦 Pulling LEAN Docker image..."
	docker pull quantconnect/lean:latest
	@echo "✅ LEAN setup complete!"

lean-test:
	@echo "🧪 Testing LEAN setup..."
	uv run python test_lean_setup.py

# Build custom LEAN image with Pub/Sub support
build-image:
	@echo "🔨 Building custom LEAN image with Pub/Sub..."
	docker build -t vibe-trade-lean:latest .
	@echo "✅ Custom image built: vibe-trade-lean:latest"

# Run a LEAN backtest
# Usage: make lean-backtest ALGO=SimpleMaCrossover [START=20240101] [END=20241231] [CASH=100000]
lean-backtest:
	@if [ -z "$(ALGO)" ]; then \
		echo "❌ Usage: make lean-backtest ALGO=<AlgorithmName> [START=YYYYMMDD] [END=YYYYMMDD] [CASH=amount]"; \
		echo "   Example: make lean-backtest ALGO=SimpleMaCrossover"; \
		exit 1; \
	fi
	@uv run python -c "from src.lean_runner.engine import LeanEngine; \
		import json; \
		engine = LeanEngine(); \
		result = engine.run_backtest( \
			algorithm_name='$(ALGO)', \
			start_date='$${START:-20240101}', \
			end_date='$${END:-20241231}', \
			cash=$${CASH:-100000.0} \
		); \
		print('Status:', result.get('status')); \
		if result.get('results_path'): \
			print('Results:', result.get('results_path')); \
		if result.get('error'): \
			print('Error:', result.get('error')); \
			exit(1)"

# List running strategies
lean-list:
	uv run python -m src.lean_runner.cli list

# Start a strategy locally
# Usage: make lean-start ALGO=SimpleMaCrossover [ID=my-strategy]
lean-start:
	@if [ -z "$(ALGO)" ]; then \
		echo "❌ Usage: make lean-start ALGO=<AlgorithmName> [ID=<strategy-id>]"; \
		exit 1; \
	fi
	@uv run python -m src.lean_runner.cli start --algorithm $(ALGO) $$([ -n "$(ID)" ] && echo "--strategy-id $(ID)")

# Stop a running strategy
# Usage: make lean-stop ID=my-strategy
lean-stop:
	@if [ -z "$(ID)" ]; then \
		echo "❌ Usage: make lean-stop ID=<strategy-id>"; \
		exit 1; \
	fi
	uv run python -m src.lean_runner.cli stop $(ID)

# View strategy logs
# Usage: make lean-logs ID=my-strategy [TAIL=100]
lean-logs:
	@if [ -z "$(ID)" ]; then \
		echo "❌ Usage: make lean-logs ID=<strategy-id> [TAIL=<lines>]"; \
		exit 1; \
	fi
	uv run python -m src.lean_runner.cli logs $(ID) --tail $${TAIL:-100}
