---
name: fyers-data-hub
description: Work on FYERS API v3 integration — TOTP/OAuth login, token caching, historical 1-minute bar sync into DuckDB/Parquet, symbol format, quotes, and the websocket streamer. Use for any change under backend/app/auth, backend/app/data, or backend/scripts that touches the broker.
---

# FYERS Data Hub

Read `docs/FYERS_AUTH_AND_DATA_GUIDE.md` for the endpoint reference, and `.agents/skills/fyers-data-hub/SKILL.md` for the original spec. This skill is the working checklist.

## Map of the code

| Concern | File |
|---|---|
| Config from env, OAuth code exchange, token cache | `backend/app/auth/fyers_auth.py` (`FyersAuthConfig`, `FyersAuthManager`, `get_valid_fyers_client`) |
| Historical bars, pagination, schema | `backend/app/data/historical.py` (`HistoricalBarFetcher`, `split_date_range`, `parse_candles_to_polars`, `HISTORICAL_BAR_SCHEMA`) |
| Parquet + DuckDB storage | `backend/app/data/storage.py` (`DuckDBStorage`, `store_bars`, `query_bars`) |
| Websocket streamer, 1-min aggregation | `backend/app/data/fyers_socket.py` (`BarAggregator`) |
| Quotes, depth, profile, status | `backend/app/data/market_overview.py` |
| REST routes | `backend/app/routers/market.py`, `stream.py`, `options.py` |

## Checklist for any change

1. **Symbols.** Use `EXCHANGE:SYMBOL-SERIES`. Equities need `-EQ` (`NSE:SBIN-EQ`). Indices need `-INDEX` (`NSE:NIFTY50-INDEX`). Validate the format at the route boundary and return 422 if it is wrong.
2. **Auth first.** Every broker call goes through `get_valid_fyers_client()`. Do not construct `fyersModel.FyersModel` directly in new code.
3. **Token lifecycle.** Tokens expire around 05:00 IST daily. A cached token is valid only if `get_profile()` succeeds. On failure, surface `authenticated: false` and stop. Do not loop on login.
4. **Historical pagination.** Split ranges with `split_date_range` into blocks of at most 100 days (the spec's limit; confirm against current FYERS docs if the limit changes). Merge the blocks, then dedupe and sort.
5. **Resolutions.** Normalize with `normalize_resolution`. Supported values are `1`, `2`, `3`, `5`, `10`, `15`, `20`, `30`, `60`, `120`, `240`, and `D`.
6. **Storage.** Write through `DuckDBStorage.store_bars`. It is idempotent, so re-syncing the same range is safe. Add a test that syncing twice leaves row counts unchanged.
7. **Websocket.** Subscribe to `SymbolUpdate` (channel 11) and `DepthUpdate` (channel 12). Reconnect with exponential backoff. Flush the in-memory tick buffer to storage about every 5 seconds. Make the flush thread-safe.
8. **Volume deltas.** `vol_traded_today` is cumulative. Compute per-tick deltas and reset them at the session boundary. Use `BarAggregator` as the reference.
9. **Secrets.** Never log the auth code, redirect URL, access token, PIN, or TOTP key. See `.claude/rules/secrets-and-broker-safety.md`.

## Tests to add or update

- Mock `fyersModel.FyersModel` and `data_ws`. Use recorded JSON payloads as fixtures under `backend/tests/fixtures/fyers/`.
- Pagination: a range of 250 days yields three blocks with no gaps or overlaps.
- Parsing: malformed candles are skipped, duplicates are dropped, and output is sorted ascending.
- Aggregation: a minute rollover emits exactly one completed bar, and cumulative-volume resets are handled.

## Verification

Run `pytest backend/tests/test_historical_ingest.py backend/tests/test_socket_pipeline.py -q`. Do not run `backend/scripts/authorize_and_fetch.py` yourself; it needs the user's PIN and TOTP. Ask the user to run it and paste back only the status line.
