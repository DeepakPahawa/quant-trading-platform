---
trigger: always_on
description: Invariants for Python, FastAPI, Polars, and database operations
---
# Python Backend Invariants
- Python Version: 3.11+. Dependency manager: `uv pip` or standard `pip`.
- Tabular Processing: Always use `polars` for rolling technical indicators and tick transformations. Do NOT use `pandas` in performance-critical code paths.
- Database: Use `duckdb` with persistent `.parquet` files for historical tick and candle archives.
- Security: Never hardcode credentials. Read `FYERS_CLIENT_ID`, `FYERS_SECRET_KEY`, `FYERS_PIN`, and `FYERS_TOTP_KEY` strictly from `.env`.
- Clean Code: Use Python type annotations (`typing`, `pydantic`) for all function signatures and FastAPI response schemas.