---
description: Terminal and process rules — never block the agent on long-running servers; verify with self-terminating commands.
---

# Terminal & Process Rules

- Never launch long-running blocking processes in the agent's foreground: `uvicorn`, `vite`, `pnpm dev`, `npm run dev`, file watchers. Ask the user to run them, or start them detached with a timeout.
- Verify backend endpoints and WebSocket behavior with self-terminating checks: `pytest`, or `httpx` against the app in-process (`TestClient`), or a short `asyncio.run(...)` script.
- Verify frontend with `pnpm build` (type-check plus bundle) and `pnpm vitest run`. Do not rely on a dev server to prove a change works.
- Keep command output small. Pipe large logs through `tail`, `head`, or `grep`.
- Never use `git stash` / `git stash pop` in this worktree setup. Use a temporary commit or `git stash push -u -m "<tag>"` as described in the session instructions.
- The `.venv` lives at the repo root. Activate it, or call `.venv/bin/python` / `.venv/bin/pytest` directly.
