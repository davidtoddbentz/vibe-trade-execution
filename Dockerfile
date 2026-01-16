# Execution Service - Orchestrates backtests via Cloud Run Jobs
FROM python:3.11-slim

WORKDIR /workspace

# Install git (needed for Git-based dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install uv for linux/amd64
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN chmod +x /usr/local/bin/uv

# Copy the execution service and its local dependencies
COPY vibe-trade-shared/ ./vibe-trade-shared/
COPY vibe-trade-execution/ ./vibe-trade-execution/
WORKDIR /workspace/vibe-trade-execution

# Install dependencies
# GITHUB_TOKEN is required for private repos (vibe-trade-shared, vibe-trade-data)
ARG GITHUB_TOKEN
RUN if [ -n "$GITHUB_TOKEN" ]; then \
        git config --global url."https://$GITHUB_TOKEN@github.com/".insteadOf "https://github.com/"; \
    fi && \
    uv sync --no-dev --frozen --python 3.11 && \
    if [ -n "$GITHUB_TOKEN" ]; then \
        git config --global --unset url."https://$GITHUB_TOKEN@github.com/".insteadOf; \
    fi

# Expose port (Cloud Run uses PORT env var, default to 8080)
ENV PORT=8080
EXPOSE 8080

# Run the FastAPI server
CMD ["sh", "-c", ".venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
