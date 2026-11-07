---
description: Modeling rules for the TFT forecaster — feature schema, time splits, quantile outputs, checkpoints.
paths:
  - "backend/app/models/**"
  - "backend/app/features/**"
  - "backend/scripts/**"
  - "backend/tests/test_tft*"
---

# TFT Modeling Rules

## Feature schema (fixed contract)
- Feature order is `ALL_FEATURES` in `backend/app/features/tft_features.py`: `log_return`, `volume_ratio`, `rsi_14`, `vwap_diff`, `depth_imbalance`, `micro_price_spread`, then time covariates `minute_of_day`, `day_of_week`.
- Time covariates come from **actual bar timestamps in IST**. `minute_of_day` is minutes since 09:15 IST, clipped to 0..375. Never synthesize them.
- Only `build_tft_feature_frame` computes features. Training and live inference both call it, so they cannot drift.
- Input is `[batch, 60, 8]`. Changing the window, feature order, or normalization invalidates every saved run.
- A run's `sidecar.json` records the feature list and sequence length. `TFTPriceForecaster.load_run` refuses a mismatch.
- pytorch-forecasting orders reals known-first. Map columns by name using `hparams.x_reals`. Never rely on position.

## Targets and horizons
- Horizons are **1-minute bars**. Index k-1 of the output is the return from the last observed bar to bar t+k. `t+5m` is index 4, `t+15m` is index 14.
- Training target for anchor bar `a` at step k is `log(close[a+k] / close[a])`. `build_samples` in `backend/app/models/tft_training.py` builds it.
- Use quantiles `[0.1, 0.5, 0.9]` with `QuantileLoss`. The p10–p90 band is the uncertainty output.
- Depth features are constant 0.0 in training, because depth is not archived. Say so in any result.

## Splits and evaluation
- Split by time: train on the earliest part, validate on the next, and test on the most recent. Use **walk-forward** folds for model selection. Never use random shuffling.
- Keep a **purge gap** of at least the max horizon between train and validation windows so labels do not overlap.
- Report on the held-out test period:
  - pinball (quantile) loss for each quantile
  - empirical coverage of the p10–p90 band (target ≈ 80%)
  - directional accuracy of the p50 sign against realized return
  - comparison against a naive baseline (zero return, or last-value persistence)
- Do not claim the model is profitable from forecast accuracy alone. Profitability requires a backtest with costs (see `indian-market-backtest` skill).

## Inference
- Use `model.eval()` and `torch.no_grad()`.
- If the buffer has fewer than 60 bars, return `{"status": "BUFFERING"}` and skip the model.
- Inference never trains and never updates weights. Training runs in a separate script with a logged config.
- Round outputs for display only. Keep full precision in the API.

## Checkpoints and reproducibility
- Runs are written to `backend/data/models/tft/<UTC timestamp>/` as `model.pt` (state dict, including normalization buffers) plus `sidecar.json` (symbols, data range, splits, seed, git commit, training history, test metrics).
- `backend/data/` is gitignored, and so are `*.pt` and `*.ckpt`.
- Seeds for torch and numpy are set from `--seed` and recorded in the sidecar.
- Serve a run by setting `TFT_MODEL_DIR`. Without it, the service uses untrained weights and logs a warning.
