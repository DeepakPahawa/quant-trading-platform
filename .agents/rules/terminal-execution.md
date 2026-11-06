---
trigger: always_on
description: Core terminal execution boundaries and headless testing protocol
---
# Terminal & Process Rules
- Never launch long-running blocking servers (e.g., `uvicorn`, `npm run dev`, `vite`) in the interactive terminal foreground.
- To verify backend endpoints and WebSocket connections, write self-terminating test scripts using `pytest` or `asyncio.run(test_client())`.
- To verify React/Vite builds, run `pnpm build` or headless test runners (`vitest run`).
- Keep command outputs concise: pipe verbose outputs through grep or head if inspecting large logs.