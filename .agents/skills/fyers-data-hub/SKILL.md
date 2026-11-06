---
name: fyers-data-hub
description: Implements FYERS API v3 headless TOTP authentication, historical candle pagination into DuckDB, and WebSocket tick and depth streaming. Use when working on broker integration, auth, or tick collection.
---
# FYERS Data Hub Protocol

### Objective
Authenticate headlessly via TOTP, pull 1-minute historical OHLCV, and maintain persistent WebSocket streams for depth and ticks.

### Execution Directives
1. **Authentication (`backend/app/auth/fyers_auth.py`):**
   - Use `pyotp.TOTP(FYERS_TOTP_KEY).now()` to obtain 6-digit TOTP.
   - Execute OAuth flow headlessly via `requests.Session` against FYERS v3 login endpoints.
   - Cache `access_token` into `.token_cache.json` alongside the current timestamp.
   - Check cached token via `fyers.get_profile()` before attempting re-authentication.

2. **Historical Bar Collector (`backend/app/data/historical.py`):**
   - Chunk historical 1-minute queries into 100-day blocks to comply with FYERS API limits.
   - Store retrieved data directly into DuckDB/Parquet tables partitioned by `symbol` and `month`.

3. **WebSocket Streamer (`backend/app/data/fyers_socket.py`):**
   - Connect via `fyers_apiv3.FyersWebsocket.data_ws.FyersDataSocket`.
   - Subscribe to both `SymbolUpdate` and `DepthUpdate` (5-level depth).
   - Buffer incoming ticks in an in-memory queue and flush to DuckDB every 5 seconds.