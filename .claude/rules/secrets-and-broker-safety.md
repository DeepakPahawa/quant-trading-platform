---
description: Secrets handling and broker safety for FYERS integration. Always applies.
---

# Secrets & Broker Safety

## Secrets
- FYERS credentials come only from environment variables: `FYERS_CLIENT_ID`, `FYERS_SECRET_KEY`, `FYERS_PIN`, `FYERS_TOTP_KEY`, `FYERS_REDIRECT_URI`, `FYERS_USER_ID`, and optionally `FYERS_ACCESS_TOKEN`. Load them from `.env` (repo root or `backend/.env`).
- Never hardcode, commit, or echo these values. Never print `access_token`, `auth_code`, the redirect URL containing `auth_code`, the PIN, or the TOTP secret in logs, exceptions, API responses, or test output.
- `.env`, `.token_cache.json`, `*.parquet`, `*.duckdb` are gitignored. Keep it that way. Do not add them to commits.
- `.env.example` may contain placeholders only (`YOUR_...`). Never put real values in it.
- Redact secrets in any example or doc you write. Use `YOUR_APP_ID-100`-style placeholders.
- The frontend must never receive or store a FYERS secret, PIN, TOTP key, or access token. It talks to the backend, and the backend holds the session. Do not put tokens in `localStorage`.

## Broker safety
- This platform is for data, research, and simulated trading. Do not write code that places, modifies, or cancels real orders unless the user explicitly asks for execution work in this session.
- Any future order-placement path must sit behind an explicit config flag that defaults to paper or dry-run mode, and it must be confirmed with the user before enabling.
- Do not call live FYERS endpoints from tests. Mock `fyersModel` and the websocket with fixtures. If a test must hit the live API, mark it and skip it unless credentials are present.
- Scripts in `backend/scripts/` that perform the OAuth flow are for the user to run interactively. Do not run them yourself; they need the user's PIN and TOTP.
- The OAuth auth code is single-use and short-lived. Do not retry an exchange with a code that already failed without asking the user for a fresh one.
