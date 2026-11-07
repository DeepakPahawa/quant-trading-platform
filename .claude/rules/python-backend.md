---
description: Conventions for the FastAPI + Polars + DuckDB backend.
paths:
  - "backend/**/*.py"
---

# Python Backend Conventions

## Language and typing
- Python 3.11+. Type-annotate every function signature, including return types.
- Pydantic v2 models for every FastAPI request and response body. Set `response_model=` on every route.
- Absolute package imports: `from backend.app.data.storage import DuckDBStorage`.

## Data processing
- Use **Polars** for all tabular transforms: bars, indicators, features, resampling.
- **Pandas is allowed only at the pytorch-forecasting boundary** (currently `backend/app/models/tft_model.py`). Convert to and from Polars at that boundary and nowhere else.
- Persist history as partitioned Parquet (`symbol=.../year_month=YYYY-MM/bars.parquet`) and query it through **DuckDB**. Reuse `DuckDBStorage` (`backend/app/data/storage.py`) instead of creating new connections.
- Cast to the canonical schema (`HISTORICAL_BAR_SCHEMA` in `backend/app/data/historical.py`) before storing. Deduplicate on `timestamp` and sort ascending.

## Structure
- Broker I/O lives in `backend/app/auth/` and `backend/app/data/`. Routers call into these modules. Models and features must not import the FYERS SDK.
- Business logic goes in `features/`, `models/`, or `backtest/`, not in router functions.
- Routers stay thin: validate input, call a service, map errors to `HTTPException`.

## Errors and logging
- Use `logger = logging.getLogger(__name__)`. Do not use `print` in library code.
- Log broker failures with context (symbol, range, resolution) and never with secrets (see `.claude/rules/secrets-and-broker-safety.md`).
- Do not swallow exceptions silently. If you catch and continue, log at `warning` or above and return an explicit status.

## Heavy work
- No model training or weight updates inside request handlers (see `.claude/rules/tft-modeling.md`).
- Long historical syncs should not block the event loop. Keep them in sync `def` routes (FastAPI runs them in a threadpool) or in a background task.

## Tests
- Every new public function gets a unit test in `backend/tests/`. Use fixtures for FYERS payloads; never call the live API.
- Use `httpx`/`TestClient` for route tests.
