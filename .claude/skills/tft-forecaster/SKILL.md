---
name: tft-forecaster
description: Build, train, evaluate, and serve the Temporal Fusion Transformer price forecaster — feature pipeline shared by training and serving, purged time splits, quantile (p10/p50/p90) cumulative-return targets, run directories with sidecars, and the forecast endpoint. Use when touching backend/app/models, backend/scripts/train_tft.py, or the forecast route.
---

# TFT Forecaster

Spec: `.agents/skills/tft-forecaster/SKILL.md`. Modeling rules: `.claude/rules/tft-modeling.md`. This skill is the procedure.

## Pipeline (as built)

```
DuckDB/Parquet bars
  -> build_tft_feature_frame          backend/app/features/tft_features.py   (shared with live inference)
  -> build_samples                    backend/app/models/tft_training.py     (60-bar windows, 15-step targets)
  -> split_samples                    purged, time-ordered
  -> fit_normalization                train split only
  -> train_model                      quantile loss, Adam, early stopping, best-weights restore
  -> evaluate_forecaster              pinball, coverage, directional accuracy, zero baseline
  -> save_run                         backend/data/models/tft/<ts>/{model.pt, sidecar.json}
  -> serve: TFT_MODEL_DIR -> get_tft_model -> predict_realtime -> GET /api/forecast/{symbol}
```

## Train

```bash
.venv/bin/python -m backend.scripts.train_tft \
  --symbols NSE:SBIN-EQ,NSE:RELIANCE-EQ \
  --start 2025-01-01 --end 2026-06-30 \
  --val-start 2026-01-01 --test-start 2026-04-01 \
  --epochs 20 --patience 4 --seed 42
```

- Reads archived bars only. Sync them first with `/api/history/{symbol}?resolution=1&sync=true`, which is the one place a broker call happens.
- Prints the run directory and test metrics. Read them before claiming anything.
- Exit codes: 1 for no samples or an empty split, 2 for no symbols.

## Procedure checklist

1. **Features.** Use `build_tft_feature_frame`. Never copy the formulas into a training script. Run the truncation test in `backend/tests/test_tft_features.py` after any feature change.
2. **Targets.** Step k is `log(close[a+k]/close[a])` for anchor `a`. Windows with non-finite values are dropped.
3. **Splits.** Train targets must end before the validation start. Validation anchors must start after it and end before the test start. Test is used once, at the end. Don't tune on it.
4. **Normalization.** Fit on the train split only. It is stored in the state dict, so serving uses the same numbers.
5. **Training.** Start from the default architecture (`hidden_size=16`, `attention_head_size=2`, `dropout=0.1`, `hidden_continuous_size=8`). Change it only with a validation result to justify it.
6. **Evaluation.** Report `h5` and `h15`: pinball per quantile, p10–p90 coverage (about 0.80 is calibrated), directional accuracy, and MAE against the zero-return baseline. If the model does not beat the zero baseline, say so plainly.
7. **Serving.** `get_tft_model()` loads `TFT_MODEL_DIR`. `predict_realtime` returns BUFFERING under 60 bars. The forecast route is `GET /api/forecast/{symbol}`, which reads archived bars only and never calls FYERS.

## Tests that guard this

- `test_tft_inference.py`: forward shapes, quantile ordering, time covariates required for 6-column input, minute-of-day column mapping, BUFFERING.
- `test_tft_features.py`: IST minute-of-day, session VWAP reset, RSI shared with indicators, no lookahead by truncation.
- `test_tft_training.py`: sample targets, purged split, normalization on train only, a short training run, save/load round-trip, schema mismatch rejected.
- `test_forecast_route.py`: BUFFERING for unknown or short archives, READY with ordered quantiles.

## Known limits (state these in any result)

- Depth features are constant in training, so the model uses four real signals.
- Windows can cross overnight gaps. Session boundaries are not yet an input to sampling.
- Live features are computed on a 240-bar lookback, while training computes them on the full history. The rolling RSI can differ slightly.
