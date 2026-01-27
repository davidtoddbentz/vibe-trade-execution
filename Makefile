.PHONY: install locally run test test-cov lint lint-fix format format-check check ci clean \
	test-e2e test-e2e-smoke test-e2e-conditions test-e2e-indicators test-e2e-exits test-e2e-gates test-e2e-shorts test-e2e-class \
	backtest-container-start backtest-container-stop backtest-container-logs \
	lean-list lean-start lean-stop lean-logs \
	build-image docker-build docker-build-lean docker-build-lean-service docker-push docker-build-push \
	deploy deploy-lean deploy-backtest-service force-revision

# Default variables
PROJECT_ID ?= vibe-trade-475704
REGION ?= us-central1
SERVICE_IMAGE ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/vibe-trade-execution/vibe-trade-execution:latest
LEAN_IMAGE ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/vibe-trade-lean/vibe-trade-lean:latest
BACKTEST_SERVICE_IMAGE ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/vibe-trade-lean/vibe-trade-backtest:latest

install:
	@echo "📦 Installing dependencies..."
	@uv sync --all-groups
	@echo "✅ All dependencies installed successfully!"

# Setup for local development: install deps, fix linting, and format code
locally: install lint-fix format
	@echo "✅ Local setup complete!"

# Run the FastAPI server locally (loads .env automatically)
# NOTE: Use port 8082 to avoid conflict with API service on 8080
# For full local stack, use `make execution-run` from repo root instead
run:
	@bash -c '\
	if [ -f .env ]; then \
		export $$(grep -v "^#" .env | xargs); \
	fi; \
	echo "🚀 Starting execution service on port 8082..."; \
	echo "   (API service uses 8080, UI expects execution on 8082)"; \
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8082 --reload'

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

# Note: All tests are now e2e tests (require LEAN running)
# Use `make test-e2e` to run all tests
test:
	uv run python -m pytest tests/e2e/ -v

test-cov:
	uv run python -m pytest tests/e2e/ --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=60

# =============================================================================
# E2E Test Commands (require LEAN backtest service running)
# =============================================================================

# Run all e2e tests (requires LEAN: make backtest-container-start)
test-e2e:
	uv run python -m pytest tests/e2e/ -v

# Quick smoke test - basic functionality (fast)
test-e2e-smoke:
	uv run python -m pytest tests/e2e/test_backtest_e2e.py -v -k "test_entry_on_exact_bar or test_crossover_entry"

# Test conditions (allOf, anyOf, not, compare operators)
test-e2e-conditions:
	uv run python -m pytest tests/e2e/test_backtest_e2e.py -v -k "Condition or Operators"

# Test indicators (EMA, BB, RSI, MACD, etc.)
test-e2e-indicators:
	uv run python -m pytest tests/e2e/test_backtest_e2e.py -v -k "EMA or Bollinger or RSI or MACD or ADX or ATR or SMA or Keltner or Donchian or VWAP"

# Test exits and state
test-e2e-exits:
	uv run python -m pytest tests/e2e/test_backtest_e2e.py -v -k "Exit or State or Stop"

# Test gates and overlays
test-e2e-gates:
	uv run python -m pytest tests/e2e/test_backtest_e2e.py -v -k "Gate or Overlay"

# Test short positions
test-e2e-shorts:
	uv run python -m pytest tests/e2e/test_backtest_e2e.py -v -k "Short"

# Run a specific test class (e.g., make test-e2e-class CLASS=TestBollingerBands)
test-e2e-class:
	@if [ -z "$(CLASS)" ]; then \
		echo "Usage: make test-e2e-class CLASS=<TestClassName>"; \
		exit 1; \
	fi
	uv run python -m pytest tests/e2e/test_backtest_e2e.py -v -k "$(CLASS)"

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

# Build custom LEAN image from vibe-trade-lean repo
docker-build-lean:
	@echo "🏗️  Building LEAN image from vibe-trade-lean..."
	@echo "   Image: $(LEAN_IMAGE)"
	@cd ../vibe-trade-lean && DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
		-f Dockerfile \
		-t $(LEAN_IMAGE) \
		.
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

# Deploy LEAN image only (for Cloud Run Jobs)
deploy-lean: docker-build-push-lean
	@echo ""
	@echo "✅ LEAN image deployed!"

# Build backtest service image (HTTP-based, for Cloud Run Service)
docker-build-lean-service:
	@echo "🏗️  Building backtest service image from vibe-trade-lean..."
	@echo "   Image: $(BACKTEST_SERVICE_IMAGE)"
	@GITHUB_TOKEN=$${GITHUB_TOKEN:-$$(gh auth token 2>/dev/null || echo "")}; \
	cd ../vibe-trade-lean && DOCKER_BUILDKIT=1 docker build --platform linux/amd64 \
		--build-arg GITHUB_TOKEN="$$GITHUB_TOKEN" \
		-f Dockerfile.service \
		-t $(BACKTEST_SERVICE_IMAGE) \
		.
	@echo "✅ Build complete"

docker-push-backtest-service:
	@echo "📤 Pushing backtest service image..."
	docker push $(BACKTEST_SERVICE_IMAGE)
	@echo "✅ Push complete"

# DEPRECATED: Use Terraform instead (vibe-trade-terraform/main.tf)
# The backtest service is now managed by Terraform for consistency.
# To deploy: cd ../vibe-trade-terraform && terraform apply
#
# This target remains for building/pushing the image only.
# Terraform manages the Cloud Run service configuration (memory, instances, etc.)
deploy-backtest-service: docker-build-lean-service docker-push-backtest-service
	@echo ""
	@echo "⚠️  DEPRECATED: Cloud Run service config is now managed by Terraform"
	@echo "   Image pushed successfully. To update the service:"
	@echo "   cd ../vibe-trade-terraform && terraform apply"
	@echo ""
	@echo "   Or to force Cloud Run to use the new image:"
	@echo "   gcloud run services update vibe-trade-backtest --image=$(BACKTEST_SERVICE_IMAGE) --region=$(REGION)"
	@echo ""

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
# Local Backtest Container
# =============================================================================

# Start local backtest service container (for development)
# The container runs on port 8081 and fetches data from GCS
backtest-container-start:
	@echo "🚀 Starting local backtest container..."
	@docker run -d --name vibe-trade-backtest-local -p 8081:8080 \
		-v ~/.config/gcloud/application_default_credentials.json:/tmp/keys/gcp-key.json:ro \
		-e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/gcp-key.json \
		$(BACKTEST_SERVICE_IMAGE)
	@echo "✅ Container started on http://localhost:8081"
	@echo "   Health check: curl http://localhost:8081/health"

# Stop local backtest container
backtest-container-stop:
	@echo "🛑 Stopping local backtest container..."
	@docker stop vibe-trade-backtest-local 2>/dev/null || true
	@docker rm vibe-trade-backtest-local 2>/dev/null || true
	@echo "✅ Container stopped"

# View local backtest container logs
backtest-container-logs:
	@docker logs -f vibe-trade-backtest-local

# =============================================================================
# Live Trading Management (via lean_runner)
# =============================================================================

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
