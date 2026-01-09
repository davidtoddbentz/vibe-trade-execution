.PHONY: install locally run test test-cov lint lint-fix format format-check check ci clean \
	backtest lean-setup lean-test lean-backtest lean-list lean-start lean-stop lean-logs \
	build-image docker-build docker-build-lean docker-push docker-build-push deploy force-revision

# Default variables
PROJECT_ID ?= vibe-trade-475704
REGION ?= us-central1
SERVICE_IMAGE ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/vibe-trade-execution/vibe-trade-execution:latest
LEAN_IMAGE ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/vibe-trade-lean/vibe-trade-lean:latest

install:
	@echo "📦 Installing dependencies..."
	@uv sync --all-groups
	@echo "✅ All dependencies installed successfully!"

# Setup for local development: install deps, fix linting, and format code
locally: install lint-fix format
	@echo "✅ Local setup complete!"

# Run the FastAPI server locally (loads .env automatically)
run:
	@bash -c '\
	if [ -f .env ]; then \
		export $$(grep -v "^#" .env | xargs); \
	fi; \
	echo "🚀 Starting execution service on port 8080..."; \
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8080 --reload'

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
	uv run python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=60

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

# =============================================================================
# Docker Build Commands
# =============================================================================

# Build execution service Docker image (loads GITHUB_TOKEN from .env or gh auth)
docker-build:
	@echo "🏗️  Building execution service image..."
	@echo "   Image: $(SERVICE_IMAGE)"
	@bash -c '\
		if [ -f .env ]; then \
			export $$(grep -v "^#" .env | xargs); \
		fi; \
		GITHUB_TOKEN=$${GITHUB_TOKEN:-$$(gh auth token 2>/dev/null || echo "")}; \
		if [ -z "$$GITHUB_TOKEN" ]; then \
			echo "⚠️  Warning: GITHUB_TOKEN not set. Add it to .env or run gh auth login"; \
		fi; \
		cd .. && DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
			--build-arg GITHUB_TOKEN="$$GITHUB_TOKEN" \
			-f vibe-trade-execution/Dockerfile \
			-t $(SERVICE_IMAGE) \
			.'
	@echo "✅ Build complete"

# Build custom LEAN image with Pub/Sub support (loads GITHUB_TOKEN from .env or gh auth)
docker-build-lean:
	@echo "🏗️  Building LEAN image..."
	@echo "   Image: $(LEAN_IMAGE)"
	@bash -c '\
		if [ -f .env ]; then \
			export $$(grep -v "^#" .env | xargs); \
		fi; \
		GITHUB_TOKEN=$${GITHUB_TOKEN:-$$(gh auth token 2>/dev/null || echo "")}; \
		if [ -z "$$GITHUB_TOKEN" ]; then \
			echo "⚠️  Warning: GITHUB_TOKEN not set. Add it to .env or run gh auth login"; \
		fi; \
		DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
			--build-arg GITHUB_TOKEN="$$GITHUB_TOKEN" \
			-f Dockerfile.lean \
			-t $(LEAN_IMAGE) \
			.'
	@echo "✅ Build complete"

# Legacy target
build-image: docker-build-lean

docker-push:
	@echo "📤 Pushing execution service image..."
	docker push $(SERVICE_IMAGE)
	@echo "✅ Push complete"

docker-push-lean:
	@echo "📤 Pushing LEAN image..."
	docker push $(LEAN_IMAGE)
	@echo "✅ Push complete"

docker-build-push: docker-build docker-push

docker-build-push-lean: docker-build-lean docker-push-lean

# =============================================================================
# Deployment Commands
# =============================================================================

# Full deployment: build, push, and update Cloud Run
deploy: docker-build-push force-revision
	@echo ""
	@echo "✅ Execution service deployment complete!"

# Deploy LEAN image only
deploy-lean: docker-build-push-lean
	@echo ""
	@echo "✅ LEAN image deployed!"

# Deploy both execution service and LEAN image
deploy-all: docker-build docker-build-lean docker-push docker-push-lean force-revision
	@echo ""
	@echo "✅ All images deployed!"

# Force Cloud Run to create a new revision with the latest image
force-revision:
	@echo "🔄 Forcing Cloud Run to use latest image..."
	@bash -c '\
		SERVICE_NAME=vibe-trade-execution && \
		echo "   Service: $$SERVICE_NAME" && \
		echo "   Region: $(REGION)" && \
		echo "   Image: $(SERVICE_IMAGE)" && \
		gcloud run services update $$SERVICE_NAME \
			--region=$(REGION) \
			--project=$(PROJECT_ID) \
			--image=$(SERVICE_IMAGE) \
			2>&1 | grep -E "(Deploying|revision|Service URL|Done)" || (echo "⚠️  Update may have failed or no changes needed" && exit 1)'

# =============================================================================
# LEAN Setup and Management
# =============================================================================

lean-setup:
	@echo "📦 Pulling LEAN Docker image..."
	docker pull quantconnect/lean:latest
	@echo "✅ LEAN setup complete!"

lean-test:
	@echo "🧪 Testing LEAN setup..."
	uv run python test_lean_setup.py

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

# =============================================================================
# Local Backtest Testing
# =============================================================================

# Local test data directory
LOCAL_TEST_DIR := $(shell pwd)/test_backtest

# Run a local backtest with the LEAN Docker image
backtest-local: docker-build-lean
	@echo "🧪 Running local backtest..."
	@mkdir -p $(LOCAL_TEST_DIR)/Data $(LOCAL_TEST_DIR)/Results $(LOCAL_TEST_DIR)/Algorithms
	@# Copy StrategyRuntime
	@cp lean/Algorithms/StrategyRuntime.py $(LOCAL_TEST_DIR)/Algorithms/
	@# Copy LEAN data files
	@cp -r lean/Data/* $(LOCAL_TEST_DIR)/Data/ 2>/dev/null || true
	@# Generate test strategy IR if not exists
	@if [ ! -f $(LOCAL_TEST_DIR)/Data/strategy_ir.json ]; then \
		echo '{"strategy_name":"Test Strategy","symbol":"BTC-USD","resolution":"Minute","indicators":[],"entry":null,"exits":[],"gates":[],"on_bar_invested":[],"on_bar":[]}' > $(LOCAL_TEST_DIR)/Data/strategy_ir.json; \
	fi
	@# Create config.json
	@echo '{"environment":"backtesting","algorithm-type-name":"StrategyRuntime","algorithm-language":"Python","algorithm-location":"/workspace/Algorithms/StrategyRuntime.py","data-folder":"/workspace/Data","results-destination-folder":"/workspace/Results","parameters":{"strategy_ir_path":"/workspace/Data/strategy_ir.json","start_date":"20260102","end_date":"20260103","initial_cash":"100000","data_folder":"/workspace/Data"},"log-handler":"QuantConnect.Logging.CompositeLogHandler","messaging-handler":"QuantConnect.Messaging.Messaging","job-queue-handler":"QuantConnect.Queues.JobQueue","api-handler":"QuantConnect.Api.Api","map-file-provider":"QuantConnect.Data.Auxiliary.LocalDiskMapFileProvider","factor-file-provider":"QuantConnect.Data.Auxiliary.LocalDiskFactorFileProvider","data-provider":"QuantConnect.Lean.Engine.DataFeeds.DefaultDataProvider","object-store":"QuantConnect.Lean.Engine.Storage.LocalObjectStore","data-aggregator":"QuantConnect.Lean.Engine.DataFeeds.AggregationManager"}' > $(LOCAL_TEST_DIR)/Algorithms/config.json
	@echo "📁 Test directory: $(LOCAL_TEST_DIR)"
	@echo "📊 Running LEAN..."
	@docker run --rm \
		-v $(LOCAL_TEST_DIR)/Data:/workspace/Data \
		-v $(LOCAL_TEST_DIR)/Results:/workspace/Results \
		-v $(LOCAL_TEST_DIR)/Algorithms:/workspace/Algorithms \
		$(LEAN_IMAGE) \
		dotnet /Lean/Launcher/bin/Debug/QuantConnect.Lean.Launcher.dll --config /workspace/Algorithms/config.json
	@echo ""
	@echo "✅ Backtest complete! Results in $(LOCAL_TEST_DIR)/Results/"

# Fetch test data for local backtest
backtest-fetch-data:
	@echo "📥 Fetching test data..."
	@mkdir -p $(LOCAL_TEST_DIR)/Data
	@cd .. && uv run python -c "from vibe_trade_data import DataFetcher, LeanDataExporter; from datetime import datetime; f = DataFetcher('batch-save'); c = f.fetch_candles('BTC-USD', datetime(2026,1,2), datetime(2026,1,3)); e = LeanDataExporter('$(LOCAL_TEST_DIR)/Data'); e.export_candles(c, 'BTC-USD'); print(f'Exported {len(c)} candles')"
	@echo "✅ Data fetched!"

# Clean local test directory
backtest-clean:
	@rm -rf $(LOCAL_TEST_DIR)
	@echo "🧹 Cleaned local test directory"
