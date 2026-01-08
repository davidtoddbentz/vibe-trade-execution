# Custom LEAN Docker image with Pub/Sub support
FROM quantconnect/lean:latest

# Install Python dependencies for Pub/Sub
# Try multiple Python paths since LEAN may use different Python installations
RUN (python3 -m pip install --no-cache-dir google-cloud-pubsub>=2.18.0 2>/dev/null || \
     python -m pip install --no-cache-dir google-cloud-pubsub>=2.18.0 2>/dev/null || \
     pip install --no-cache-dir google-cloud-pubsub>=2.18.0 2>/dev/null || \
     /usr/bin/python3 -m pip install --no-cache-dir google-cloud-pubsub>=2.18.0 2>/dev/null) && \
    echo "Pub/Sub library installation attempted"

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

# Note: We preserve the base image's entrypoint and working directory
# The base image uses /Lean/Launcher/bin/Debug as working directory
# We don't change WORKDIR to preserve the base image's configuration

