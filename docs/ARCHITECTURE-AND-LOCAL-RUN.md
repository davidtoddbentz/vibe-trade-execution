# Vibe Trade: Architecture & Local Run Reference

This doc summarizes all projects, how to run them locally, and how execution fits into the system (for the StrategyRuntime typed refactor work).

---

## 1. Projects Overview

| Project | Purpose | Port (local) | Key deps |
|--------|---------|--------------|----------|
| **vibe-trade-shared** | Pydantic models (IR, archetypes, Card, Strategy), Firestore repos | — | pydantic |
| **vibe-trade-mcp** | MCP server for AI agents; strategy/card tools; Firestore | 8080 | FastMCP, vibe-trade-shared |
| **vibe-trade-agent** | LangGraph agent that uses MCP tools | 2024 | MCP client |
| **vibe-trade-api** | FastAPI for users: strategies, sessions, backtest triggers | 8080 | Firebase Auth, vibe-trade-shared |
| **vibe-trade-execution** | **Orchestrates backtests**: translate strategy→IR, fetch data, call LEAN | **8082** | vibe-trade-shared, BigQuery, httpx |
| **vibe-trade-lean** | LEAN backtest **container**: FastAPI + subprocess LEAN; runs StrategyRuntime | **8083** (mapped from 8080) | pydantic, vibe-trade-shared (in Docker) |
| **vibe-trade-ingestion** | Coinbase candles → Pub/Sub | — | google-cloud-pubsub |
| **vibe-trade-ui** | Next.js (HeroUI, Firebase, Recharts) | 3000 | API + Execution |
| **vibe-trade-terraform** | GCP (Cloud Run, Artifact Registry, Firestore) | — | — |

- **Execution** and **LEAN** are the two sides of "run a backtest": Execution decides *what* to run and *with which data*; LEAN *executes* the strategy and returns trades/equity.

---

## 2. Running Locally

### One-time setup

- Docker (for emulators + LEAN container).
- Build LEAN backtest image:
  ```bash
  make lean-build
  ```
  (builds `lean-backtest-service` from `vibe-trade-lean/Dockerfile.service`)

### Start the stack

**Terminal 1 – Emulators + LEAN**

```bash
make local-up        # Firestore 8081, BigQuery 9050, Auth 9099, LEAN 8083
make local-seed      # Seed BQ + test strategy (run once)
```

**Terminal 2 – Execution (required for backtests)**

```bash
make execution-run   # Port 8082
```

**Terminal 3 – API (optional, for UI)**

```bash
make api-run         # Port 8080
```

**Terminal 4 – UI (optional)**

```bash
make ui-run          # Port 3000; uses API + Execution
```

### Env passed by root Makefile for execution

- `FIRESTORE_EMULATOR_HOST=localhost:8081`
- `BIGQUERY_EMULATOR_HOST=http://localhost:9050`
- `GOOGLE_CLOUD_PROJECT=test-project`
- `FIRESTORE_DATABASE=(default)`
- `BACKTEST_SERVICE_URL=http://localhost:8083/backtest` (not set in Makefile; execution defaults to `http://localhost:8083/backtest` when `BACKTEST_SERVICE_URL` is unset, so LEAN is used automatically)

Check: `make local-status` (health of emulators, LEAN, execution, API, UI).

---

## 3. Execution Architecture (Deep Dive)

Execution is the **backtest orchestrator**. It does not run LEAN itself; it prepares inputs and calls the LEAN HTTP service.

### 3.1 Flow (request → result)

```
POST /backtest (API or UI)
    → routes/backtest.py (fetch Strategy + Cards from Firestore)
    → BacktestService.run_backtest(request, strategy, cards)
        → 1) IR: strategy + cards → StrategyIR (IRTranslator)
        → 2) Data: DataService.get_ohlcv() or inline_bars (with warmup)
        → 3) Build LEANBacktestRequest (strategy_ir dict, data, config)
        → 4) HTTP POST to LEAN (e.g. http://localhost:8083/backtest)
    → LEAN returns LEANBacktestResponse
    → BacktestService maps to UI result (trades, statistics, equity_curve)
    → Persist to Firestore (BacktestRepository)
    → Return BacktestResponseModel
```

### 3.2 Key execution modules

| Path | Role |
|------|------|
| `src/main.py` | FastAPI app; mount backtest router |
| `src/routes/backtest.py` | POST /backtest (Firestore strategy/cards, run_backtest, persist) |
| `src/service/backtest_service.py` | **Core**: translate, data, build LEAN request, call LEAN, map response |
| `src/service/data_service.py` | Abstract: get_ohlcv(symbol, resolution, start, end) |
| `src/service/bigquery_data_service.py` | DataService impl (BQ or emulator) |
| `src/translator/ir_translator.py` | Strategy + Cards → **StrategyIR** (Pydantic) |
| `src/translator/ir.py` | Re-exports IR types from **vibe-trade-shared** |
| `src/models/lean_backtest.py` | BacktestConfig, LEANBacktestRequest, LEANBacktestResponse, EquityPoint, Trade |

- **IR ownership**: StrategyIR and all condition/ValueRef types live in **vibe-trade-shared**; execution's `translator/ir.py` only re-exports them.

### 3.3 What execution sends to LEAN

- **LEANBacktestRequest**
  - `strategy_ir`: dict (StrategyIR.model_dump())
  - `data`: BacktestDataInput (symbol, resolution, bars = list of OHLCVBar)
  - `config`: start_date, end_date, initial_cash, **trading_start_date** (user start; no trades before this)
  - `additional_data`: optional other symbols' bars

- Warmup: Execution computes `lean_start_date = request.start_date - warmup_bars * bar_duration`, sends that as `config.start_date` so LEAN has bars for indicators; `config.trading_start_date = request.start_date` so LEAN does not trade in the warmup window.

---

## 4. LEAN Side (vibe-trade-lean)

### 4.1 Two "layers"

1. **HTTP service** (`src/serve_backtest.py`)
   - FastAPI app; receives LEANBacktestRequest.
   - Writes `strategy_ir` to temp file `Data/strategy_ir.json`.
   - Writes OHLCV bars to CSV in `Data/`.
   - Copies algorithm files into a temp algo dir: `StrategyRuntime.py`, `typed_conditions.py`, `indicators/`, `conditions/`.
   - Calls **LEAN engine** via subprocess with `config.json` (algorithm-type-name: StrategyRuntime, parameters: strategy_ir_path, start_date, end_date, initial_cash, trading_start_date, data_folder).
   - Reads `strategy_output.json` (trades, equity_curve, statistics) and returns LEANBacktestResponse.

2. **StrategyRuntime** (`src/Algorithms/StrategyRuntime.py`)
   - Runs inside LEAN (Python algorithm).
   - Gets config via `GetParameter("strategy_ir_path")` (or `strategy_ir` string). Loads IR from file (or JSON string) with **json.loads** → **dict** (no Pydantic parsing yet).
   - Uses `self.ir.get(...)` throughout (~239 .get() calls).
   - Creates indicators from IR, evaluates entry/exit/gates via **conditions/registry** (which still delegates to methods on StrategyRuntime).
   - Writes results to `strategy_output.json` for serve_backtest to read.

### 4.2 Dependencies (LEAN)

- **In Docker**: Pydantic, FastAPI, uvicorn; **vibe-trade-shared** is in pyproject (used by tests; in container, algorithm code is copied, so shared may be in the image or models effectively inlined).
- **StrategyRuntime** today: imports `typed_conditions`, `indicators`, `conditions`; uses **dict** for IR.

### 4.3 Local dev detail

- `docker-compose.local.yml` mounts `vibe-trade-lean/src/Algorithms/StrategyRuntime.py` into the container so edits are used without rebuilding. The conditions/ and indicators/ packages are **not** mounted (they come from the image). So for Phase 6 you'll need to rebuild the image when changing `indicators/resolvers.py` (or add a mount).

---

## 5. How Projects Work Together (Backtest Path)

```
UI (3000) or API (8080)
    → POST /backtest (strategy_id, dates, etc.)
    → Execution (8082): Firestore (strategy, cards)
    → Execution: IRTranslator(strategy, cards) → StrategyIR
    → Execution: DataService or inline_bars → list[OHLCVBar]
    → Execution: HTTP POST to LEAN (8083/backtest) with IR + bars + config
    → LEAN serve_backtest: write IR + CSV, run LEAN subprocess
    → StrategyRuntime (inside LEAN): load IR (dict), run backtest, write strategy_output.json
    → serve_backtest: read strategy_output.json → LEANBacktestResponse
    → Execution: map to UI result, save to Firestore, return response
```

- **Shared library**: Execution and MCP/API use **vibe-trade-shared** for Strategy, Card, StrategyIR, and IR types. LEAN has access to shared in pyproject; at runtime the algorithm code is copied into a temp dir and run by LEAN's Python.

---

## 6. Testing (Execution & LEAN)

- **Execution**
  - E2E tests call BacktestService with optional `inline_bars` and `strategy_ir` (no Firestore/BQ).
  - LEAN must be reachable (local container or Cloud Run); default URL in tests is typically the same as local (e.g. 8083).
  - Run: `cd vibe-trade-execution && uv run pytest tests/ -v` (includes E2E); exclude E2E: `-m "not e2e"`.

- **LEAN**
  - Unit tests: `cd vibe-trade-lean && uv run pytest tests/ -v`.
  - E2E/Docker: `make test-e2e` (builds image, runs tests that drive the container).

---

## 7. Relevance for "StrategyRuntime Full Typed Refactor"

- **Execution** owns: IR translation (StrategyIR from shared), building LEAN request, and response mapping. No change to request/response contract needed for Phase 6; Execution keeps sending the same IR dict.
- **LEAN** owns: deserializing that IR and running the strategy. Phase 6 (typed value resolution) is **inside LEAN only**: add `indicators/resolvers.py` with typed `resolve_value(ValueRef, bar, ...)`, and start using it from StrategyRuntime instead of `_resolve_value` dict lookups. StrategyRuntime still loads IR as dict until Phase 11 (parse IR to Pydantic on load).
- **vibe-trade-shared** already defines all ValueRef and Condition types; LEAN can use them for typed resolution once resolvers and context are in place.

---

## 8. Quick Command Reference

| Goal | Command |
|------|---------|
| Build LEAN image | `make lean-build` (root) or `cd vibe-trade-lean && make build` (uses main Dockerfile; for service image use execution's docker-build-lean-service) |
| Start stack | `make local-up && make local-seed` |
| Run execution | `make execution-run` |
| Run API | `make api-run` |
| Run UI | `make ui-run` |
| Status | `make local-status` |
| Execution tests | `cd vibe-trade-execution && uv run pytest tests/ -v` |
| Execution tests (no E2E) | `cd vibe-trade-execution && uv run pytest tests/ -v -m "not e2e"` |
| LEAN unit tests | `cd vibe-trade-lean && uv run pytest tests/ -v` |

For local backtest, Execution expects LEAN at `http://localhost:8083/backtest` by default; that's what `make local-up` exposes (container port 8080 → host 8083).
