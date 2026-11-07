---
name: react-vite-dashboard
description: Extend the React 19 + Vite + TypeScript dashboard in frontend/ — the typed API client, lightweight-charts views, the TFT forecast band, polling hooks, and the build/test gate. Use when adding a page, component, or API call under frontend/.
---

# React + Vite Dashboard

Rules: `.claude/rules/frontend-react.md`. Backend contract: `backend/app/routers/` and `backend/app/models/tft_model.py`. The scaffold already exists, so this skill is for extending it.

## Current layout

| Path | What lives there |
|---|---|
| `src/services/api.ts` | Typed client: `getHealth`, `getLoginUrl`, `exchangeAuthCode`, `getQuotes`, `getHistory`, `getForecast`. Non-2xx responses throw `ApiError`. |
| `src/types/api.ts` | Response types. Keep them in sync with the Pydantic models. |
| `src/lib/candles.ts` | `toCandlestickData`, `toVolumeData`. Pure functions, unit-tested. |
| `src/lib/format.ts` | `formatIst`, `formatInr`, `formatLogReturnPct` |
| `src/lib/forecast.ts` | `DISCLAIMER`, `bandPosition` |
| `src/lib/symbols.ts` | `DEFAULT_SYMBOLS` watchlist |
| `src/hooks/usePolling.ts` | Polls a loader, keeps the last good data, cancels on unmount or key change |
| `src/components/CandleChart.tsx` | Chart created once in a ref. Data goes through `setData`. |
| `src/components/ForecastPanel.tsx` | p10/p50/p90 band per horizon, buffering progress, disclaimer |
| `src/pages/` | `WatchlistPage` (`/`), `SymbolPage` (`/symbol/:symbol`), `LoginPage` (`/login`) |
| `src/App.tsx` | Router and nav. `main.tsx` mounts it. |

## Add a page or endpoint

1. Add the Pydantic response model on the backend first, with a test.
2. Mirror it in `src/types/api.ts`.
3. Add one function in `src/services/api.ts`. Components never call `fetch`.
4. Add a pure helper in `src/lib` if there is logic, and test it.
5. Build the component. Handle loading, error, `BUFFERING`, and `unauthenticated` states.
6. Register the route in `src/App.tsx`. Add a nav link if it is top-level.

## Gaps still open

- **Backtest page** (`/backtests`): needs a backend backtest endpoint first. None exists.
- **Forecasts table** (`/forecasts`): can be built from `getForecast` per symbol.
- **VWAP and bands on the chart**: compute them from the history candles or add a backend endpoint. Don't copy the formula loosely; match `compute_session_vwap`.
- **Live ticks**: needs a websocket client. Follow the rules: one socket per page, cleanup on unmount, and ticks kept in a ref with a throttled re-render.

## Verification gate

Run from `frontend/`:

```bash
pnpm typecheck && pnpm test && pnpm lint && pnpm build
```

All four should pass with no lint warnings. This is the proof the agent can give. A visual check needs the user to run `pnpm dev`, or a built preview in the browser pane. Check the 375 px layout and the dark theme.
