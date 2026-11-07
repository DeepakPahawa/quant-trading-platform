---
description: Data-integrity invariants for NSE bars, sessions, timestamps and features. Prevents lookahead and timezone bugs.
paths:
  - "backend/app/data/**"
  - "backend/app/features/**"
  - "backend/app/models/**"
  - "backend/app/backtest/**"
---

# Market Data Integrity

## Time
- Store every timestamp as **epoch seconds, UTC**. Convert to IST only for session logic and display.
- The NSE equity session is **09:15–15:30 IST** (375 minutes, 75 five-minute bars). Session VWAP resets at 09:15 IST.
- Bars before 09:15 IST belong to the prior trading session (see `ensure_ist_session` in `backend/app/features/indicators.py`).
- Do not forward-fill across sessions, weekends, or market holidays. Treat gaps as gaps and let downstream code handle them explicitly.
- Handle half-day sessions and special sessions by reading the bar data, not by assuming 375 bars per day.

## Lookahead
- A feature stamped at time `t` may use only bars with timestamp `<= t`.
- Normalize with **rolling** statistics (for example a rolling z-score over a trailing window). Never use full-sample mean or std.
- A label (future return at horizon `h`) is the only place a forward shift is allowed. Keep labels in a separate column and never feed them as inputs.
- Every feature transform needs a truncation test: compute features on the full series and on the series cut at `t`, then assert the features at `<= t` are identical. (See the `quant-features` skill.)

## Volume and depth
- FYERS `vol_traded_today` is cumulative for the day. Derive per-bar volume from deltas and reset at the session boundary. Clamp negatives to zero and record the event.
- Index symbols (`-INDEX`) carry no meaningful volume or depth. Do not compute volume-ratio or depth features for them.
- Depth imbalance and micro-price are undefined when both top-of-book quantities are zero. Return null, not zero.

## Storage and dedupe
- Deduplicate on `(symbol, timestamp)`, keeping the latest write.
- Writes are idempotent. Re-running a sync must not duplicate bars.
- Partition by `symbol` and `year_month`. Do not write one flat file per symbol.
