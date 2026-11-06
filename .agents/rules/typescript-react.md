---
trigger: always_on
description: Invariants for React, TypeScript, and UI chart rendering
---
# React & Frontend Invariants
- Runtime: React 18+ with Vite and TypeScript (strict mode enabled).
- Package Manager: Use `pnpm` (or `npm`) consistently.
- Charting: Use `lightweight-charts` (TradingView) for candlestick and technical series. Never re-instantiate the chart instance inside reactive render loops; use `useRef` and push updates via `.update()`.
- State & Streaming: Manage WebSocket updates outside high-frequency React component re-renders using event listeners or refs to prevent frame drops.