---
name: tft-forecaster
description: Manages Temporal Fusion Transformer (TFT) multi-horizon price forecasting using PyTorch Forecasting, preparing sequence tensors, and generating quantile predictions (p10, p50, p90). Use when working with deep learning time-series models.
---
# TFT Multi-Horizon Forecasting Protocol

### Objective
Generate multi-horizon quantile return forecasts ($t+5m$, $t+15m$) from multivariate input sequences.

### Execution Directives
1. **Tensor Preparation (`backend/app/models/tft_model.py`):**
   - Input shape: `[batch_size, 60_bars, num_features]`.
   - Feature Vector: `[log_return, volume_ratio, rsi_14, vwap_diff, depth_imbalance, micro_price_spread]`.
   - Time Covariates: Normalized minute-of-day ($0$ to $375$) and day-of-week.

2. **Inference Execution:**
   - Execute forward pass in `torch.no_grad()` / `eval()` mode.
   - Extract quantile outputs: $p_{10}$ (lower band), $p_{50}$ (median), $p_{90}$ (upper band).
   - If feature buffer has fewer than 60 bars, bypass inference and return `{"status": "BUFFERING"}`.

### Constraints
- Do not trigger full model retraining or weight updates inside API request handlers.