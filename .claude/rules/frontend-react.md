---
description: Conventions for the React + Vite + TypeScript dashboard.
paths:
  - "frontend/**"
---

# React / Vite / TypeScript Conventions

## Tooling
- React 18+, Vite, TypeScript with `"strict": true`. Package manager is **pnpm** (use `npm` only if the project already uses it).
- Verify with `pnpm build` (type-check plus bundle) and `pnpm vitest run`. Do not treat a running dev server as proof.
- Path alias `@/` maps to `src/`. Keep it in `vite.config.ts` and `tsconfig.json`.

## API access
- All backend calls go through `src/services/` (for example `src/services/api.ts`). Components never call `fetch` directly.
- Base URL comes from `import.meta.env.VITE_API_BASE_URL`, defaulting to `http://127.0.0.1:8000/api`. Do not hardcode it in components.
- Type every response. Keep the TypeScript interfaces in sync with the backend Pydantic models, and name them the same.
- Handle `{"status": "BUFFERING"}` and `{"status": "unauthenticated"}` as normal states, not errors. Show a clear message.

## Charts
- Use `lightweight-charts` for candles, VWAP, and volume.
- Create each chart **once** in a `useEffect`, store it in a `useRef`, and push data with `series.setData()` or `series.update()`. Never recreate the chart inside a render or a reactive loop.
- Timestamps are UTC epoch seconds. Format labels in IST with a custom `timeFormatter` or `localization` config so the x-axis reads in IST.
- Render the forecast as a band: p10 and p90 as an area or two lines, p50 as a line. Label the horizon in minutes.

## Streaming and state
- High-frequency ticks do not go through React state on every message. Keep them in a ref or an external store, and let a throttled timer (for example 250 ms) trigger re-renders.
- Use a single WebSocket per page, opened in an effect and closed in its cleanup. Reconnect with backoff.
- Keep server state (quotes, history, forecasts) in one data-fetching layer. Do not duplicate it across components.

## Security and UX
- Never store FYERS credentials, PINs, TOTP keys, or access tokens in the frontend or in `localStorage`. The backend owns the session.
- Show a standing disclaimer on forecast views: model output for research, not investment advice.
- Do not rely on color alone for bullish/bearish. Pair color with a text label or an arrow.
- Support light and dark themes through CSS variables, and make the layout usable at 375 px width.
