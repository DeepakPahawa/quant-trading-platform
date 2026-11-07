# Quant Trading Platform — Project Memory

Indian equities (NSE) research platform. **Primary goal: swing-trade stock suggestions.**

1. **Swing research (primary)**: daily bars for the Nifty 50 and Nifty 500 universes. A LightGBM-style ranker (scikit-learn `HistGradientBoostingRegressor`) scores every stock each day. The top-N are suggested for a 3–10 (or 3–20) trading-day hold. Evaluated walk-forward, against the Nifty 50, after delivery costs.
2. **Intraday TFT (secondary, experimental)**: 1-minute bars, TFT quantile forecasts. Not suited to swing trading. Kept for later intraday work.
3. **UI**: a React + Vite + TypeScript dashboard for watchlist quotes, candles, and forecast bands. No swing suggestions view yet.

Read this file first. Path-specific rules live in `.claude/rules/`, and task recipes live in `.claude/skills/`.

## Layout

```
backend/app/auth/            FYERS OAuth + TOTP login, token cache
backend/app/data/            historical fetch, socket streamer, DuckDB/Parquet storage, market overview
backend/app/features/        indicators (VWAP, RSI, ATR, RV), microstructure, tft_features (shared TFT feature pipeline)
backend/app/models/          tft_model (inference), tft_training (training utilities), volatility (QuantLib)
backend/app/backtest/        engine, strategies, Indian tax/cost model
backend/app/routers/         FastAPI routers under /api (market, forecast, stream, options, backtest)
backend/scripts/             CLI tools: train_tft (training), auth and inspection scripts
backend/tests/               pytest suite (pytest.ini -> testpaths backend/tests)
backend/data/                LOCAL ONLY (gitignored): DuckDB, Parquet archive, trained model runs
docs/                        FYERS auth and data guide (API reference)
.agents/                     original domain specs. Claude Code does not auto-load these; .claude/ holds the Claude-facing versions.
frontend/                    React 19 + Vite + TypeScript dashboard (strict)
  src/services/api.ts        typed client for every backend endpoint
  src/lib/                   candle mapping, IST/INR formatting, forecast band math, symbol list
  src/hooks/usePolling.ts    polling with cancel-on-unmount
  src/components/            CandleChart (lightweight-charts), ForecastPanel
  src/pages/                 Watchlist, Symbol (chart + forecast), Login
```

## Commands

Run these from the repo root. Do not run long-lived servers in the agent's foreground (see `.claude/rules/terminal-execution.md`).

```bash
# Backend setup (one time). The venv is Python 3.11 (created with uv).
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r backend/requirements.txt

# Backend tests (self-terminating)
pytest -q

# Swing: put nifty50.csv / nifty500.csv in backend/data/universe/, then:
python -m backend.scripts.backfill_bars --universe nifty50 --start 2018-01-01 --resolution D
python -m backend.scripts.run_swing_research --universe nifty50 --horizon 10 --top-n 5
python -m backend.scripts.swing_suggest --universe nifty50 --horizon 10 --top-n 10

# Train the TFT on archived bars (reads DuckDB only, never calls FYERS)
python -m backend.scripts.train_tft --symbols NSE:SBIN-EQ,NSE:RELIANCE-EQ \
  --start 2025-01-01 --end 2026-06-30 --val-start 2026-01-01 --test-start 2026-04-01

# Serve a trained run: set TFT_MODEL_DIR=<run dir printed by train_tft> in .env

# Backend server (user runs this in their own terminal)
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

# Frontend (run from frontend/)
pnpm install
pnpm dev                 # user runs this
pnpm typecheck && pnpm test && pnpm lint && pnpm build   # agent verification
```

## Stack

- Python 3.11, FastAPI, Pydantic v2, Polars 2.x (primary), DuckDB + Parquet (storage), `fyers-apiv3`, `pyotp`
- PyTorch 2.12, pytorch-forecasting 1.8 (TFT architecture), lightning, QuantLib, SciPy
- React 19, Vite 8, TypeScript 7 (strict), pnpm, `lightweight-charts` 5, `react-router-dom`, Vitest + Testing Library

## Non-negotiables

1. **No secrets in code, logs, tests, or the frontend.** Credentials come from `.env` only.
2. **No lookahead.** Features at time `t` use only data available at or before `t`. Normalization is fit on the training period only.
3. **Time-ordered splits only**, with purging between train and validation. Never shuffle a time series.
4. **No live orders** unless the user explicitly asks for execution work. Default to paper or backtest mode.
5. **Store time as epoch seconds (UTC). Interpret market sessions in IST (09:15–15:30).**
6. **Never commit.** The user commits changes themselves. Leave work in the working tree.

## Swing pipeline (daily, primary)

- Universe: `backend/data/universe/nifty50.csv`, `nifty500.csv` (NSE constituent CSVs, `Symbol` column). Benchmark: `NSE:NIFTY50-INDEX`.
- Daily archive: `backend/data/parquet_daily/`. Kept separate from the 1-minute archive, which has no resolution column.
- Features: `backend/app/features/daily_features.py`. All causal. Cross-sectional ranks use only the same day's values.
- Labels: `backend/app/models/swing_dataset.py`. Entry at the **next day's open**, exit at the **close H trading days later**. Target is the cross-sectional percentile of the excess return over the Nifty 50. Labels are null across data gaps.
- Model and walk-forward: `backend/app/models/swing_model.py`. Training uses only labels that matured before each test start (purging).
- Backtest: `backend/app/backtest/swing_backtest.py`. Equal-weight top-N, rebalanced every H days, delivery costs.
- Scripts: `backfill_bars --universe nifty50 --resolution D`, `run_swing_research` (walk-forward report), `swing_suggest` (latest-day ranking).

Evidence to keep in mind: on 10 null synthetic universes (pure random walks), the top-3 strategy's excess return averaged +7.7% with a standard deviation of 21.8%. Small samples produce large swings. Trust a result only with 100+ trades.

## Model contract (TFT)

- Input: `[batch, 60, 8]` in `ALL_FEATURES` order (`backend/app/features/tft_features.py`): six continuous features, then `minute_of_day` (IST minutes since 09:15, clipped to 0..375) and `day_of_week`.
- Output: `[batch, 15, 3]` quantile log-returns. Step k is the return from the last observed bar to bar t+k. Index 4 is t+5m and index 14 is t+15m.
- Targets during training are the same cumulative returns, from the anchor bar.
- Column order is mapped to pytorch-forecasting's `x_reals` by name, never by position.
- Normalization (per-feature mean/std and target scale) lives in the state dict. Training and serving apply it the same way.
- A run is `model.pt` + `sidecar.json`. Loading refuses a feature-schema mismatch.

## Resolved (since the last review)

- `minute_of_day` was synthesized with `linspace`. It now comes from real IST session time.
- RSI had two implementations. The TFT pipeline now uses Wilder's `compute_rsi` from `indicators.py`.
- `prepare_tft_feature_tensor` computed `minute_of_day` from UTC hours (09:15 IST looked like 03:45). Fixed by the shared pipeline.
- Column order: the model placed continuous features where pytorch-forecasting expects known time covariates. Now mapped by name, with a regression test.
- `vwap_diff` used a cumulative VWAP that never reset. It now resets at 09:15 IST each session.
- `.gitignore` now excludes `backend/data/`, `*.ckpt`, `*.pt`, `frontend/node_modules/`, and `frontend/dist/`.
- The training script existed only as a plan. `backend/scripts/train_tft.py` now exists and has been run end-to-end on synthetic data.

## Known issues (open)

0. **Swing cost assumptions need verifying.** `DeliveryCostModel` (`backend/app/backtest/swing_backtest.py`) assumes ₹0 brokerage on delivery and a ₹16 DP charge per sell. The rates are the delivery STT (0.1% both sides), stamp duty (0.015% buy), and exchange, SEBI, and GST. Confirm each against circulars and your broker before trusting net results.
0a. **The 1-minute archive mixes resolutions if you ever store daily bars there.** Daily bars now go to `parquet_daily`. Don't write `resolution="D"` into the minute archive.
0b. **Swing research is not yet validated on real data.** The pipeline is tested on synthetic data only. Run `run_swing_research` on the real universes and check the trade count and the warning before drawing conclusions.
1. **Depth features are constant during training.** Historical bars carry no depth, so `depth_imbalance` and `micro_price_spread` are 0.0 in training. Only four features carry signal. Archive depth snapshots if these features matter.
2. **The model has no demonstrated edge.** A smoke run on a synthetic random walk shows p50 no better than the zero-return baseline, which is correct for random data. Real evaluation needs real archived bars and a walk-forward run before any claim.
3. **`/api/market/tft/{symbol}` still uses a loose spread.** It computes `micro_price - ltp` and calls that `micro_price_spread`. The spec is `(ask - bid) / mid`. Training uses 0.0, so live and training inputs disagree for this field.
4. **`.agents/skills/event-backtester/SKILL.md` lists options sell STT as 0.125%.** The current rate appears to be 0.1% on premium since Oct 2024. Verify against current circulars before hard-coding costs.
5. **Polars 2.x changed APIs.** `min_periods` was renamed to `min_samples`. `requirements.txt` has `polars>=0.20`, which now resolves to 2.x. Pin a major version.
6. **Not built yet:** the backtest endpoint and page, the forecasts table page, VWAP and volatility bands on the chart, and a websocket-driven live view.
7. **The UI has not been viewed in a browser.** The build and tests pass, but nothing has been checked visually at 375 px or in dark mode.

## Working agreements

- Keep changes scoped. Don't refactor unrelated modules while implementing a feature.
- Every new route gets a Pydantic response model and at least one test.
- Every new feature transform gets a no-lookahead test (see `quant-features` skill).
- Before claiming something works, run the relevant test or build and report the result.
- Never run `git commit` or `git push`. The user commits.
