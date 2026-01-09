# Custom LEAN Docker image with Pub/Sub support and Cloud Run Job capabilities
FROM quantconnect/lean:latest

# Install Python dependencies
# - google-cloud-pubsub: For real-time data streaming
# - google-cloud-storage: For GCS access in Cloud Run Jobs
# - fastavro: For parsing Avro candle data from GCS
# - pydantic: For data models
# Try multiple Python paths since LEAN may use different Python installations
RUN (python3 -m pip install --no-cache-dir \
        'google-cloud-pubsub>=2.18.0' \
        'google-cloud-storage>=2.14.0' \
        'pyarrow>=14.0.0' \
        'fastavro>=1.9.0' \
        'pydantic>=2.0.0' 2>/dev/null || \
     python -m pip install --no-cache-dir \
        'google-cloud-pubsub>=2.18.0' \
        'google-cloud-storage>=2.14.0' \
        'pyarrow>=14.0.0' \
        'fastavro>=1.9.0' \
        'pydantic>=2.0.0' 2>/dev/null || \
     pip install --no-cache-dir \
        'google-cloud-pubsub>=2.18.0' \
        'google-cloud-storage>=2.14.0' \
        'pyarrow>=14.0.0' \
        'fastavro>=1.9.0' \
        'pydantic>=2.0.0' 2>/dev/null) && \
    echo "Python dependencies installed"

# Copy the data module (fetcher, exporter, models) for market data handling
COPY src/data/ /Lean/src/data/
RUN touch /Lean/src/__init__.py

# Install .NET SDK for compiling C# handler (if not already present)
RUN dotnet --version || echo ".NET SDK check"

# Create directories for our custom data feeds
RUN mkdir -p /Lean/DataFeeds /Lean/CustomDataQueueHandler

# Copy C# data queue handler source
COPY lean/DataFeeds/PubSubDataQueueHandler.cs /Lean/CustomDataQueueHandler/

# Copy Python data feeds
COPY lean/DataFeeds/PubSubCandle.py /Lean/DataFeeds/
COPY lean/DataFeeds/PubSubDataQueueHandler.py /Lean/DataFeeds/
COPY lean/DataFeeds/__init__.py /Lean/DataFeeds/

# Copy C# project file
COPY lean/DataFeeds/PubSubDataQueueHandler.csproj /Lean/CustomDataQueueHandler/

# Compile C# handler (if dotnet is available)
# The compiled DLL will be placed where LEAN can find it
RUN if command -v dotnet >/dev/null 2>&1; then \
        echo "Compiling C# data queue handler..."; \
        cd /Lean/CustomDataQueueHandler && \
        dotnet restore && \
        dotnet build -c Release && \
        cp bin/Release/net10.0/PubSubDataQueueHandler.dll /Lean/Launcher/bin/Debug/ 2>/dev/null || \
        cp bin/Release/net*/PubSubDataQueueHandler.dll /Lean/Launcher/bin/Debug/ 2>/dev/null || \
        echo "⚠️  DLL not found after build"; \
        echo "✅ C# handler compilation attempted"; \
    else \
        echo "⚠️  dotnet not found - C# handler will need manual compilation"; \
    fi

# Copy StrategyRuntime for interpreting strategy IR at runtime
COPY lean/Algorithms/StrategyRuntime.py /Lean/Algorithm.Python/

# Copy backtest runner script for Cloud Run Jobs
COPY lean/run_backtest.py /Lean/

# Copy LEAN data files (symbol-properties, market-hours)
COPY lean/Data/ /Lean/Data/

# Note: For local Docker runs, the base image's entrypoint is preserved.
# For Cloud Run Jobs, use the run_backtest.py script as entrypoint:
#   CMD ["python3", "/Lean/run_backtest.py"]
#
# The base image uses /Lean/Launcher/bin/Debug as working directory.
# We don't change WORKDIR to preserve the base image's configuration for LEAN.

# Set up environment for Cloud Run Jobs
ENV PYTHONPATH="/Lean:${PYTHONPATH}"

# Override the base image's ENTRYPOINT for Cloud Run Jobs
# The base image has an ENTRYPOINT that runs LEAN directly,
# but we need to run our backtest script for Cloud Run Jobs
ENTRYPOINT []
CMD ["/opt/miniconda3/bin/python3", "/Lean/run_backtest.py"]

