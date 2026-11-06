---
name: quant-feature-store
description: Calculates microstructure metrics, top-5 order depth imbalances, volume-weighted micro-price, VWAP bands, RSI, and ATR using Polars. Use when building indicators, feature pipelines, or order book transforms.
---
# Quant Feature Store Protocol

### Objective
Transform raw 1-minute bars and Level-2 tick snapshots into normalized quantitative feature tensors using Polars.

### Execution Directives
1. **Microstructure Features (`backend/app/features/microstructure.py`):**
   - Top-5 Bid-Ask Imbalance:
     `depth_imbalance = (sum(bid_quantities) - sum(ask_quantities)) / (sum(bid_quantities) + sum(ask_quantities))`
   - Micro-Price:
     `micro_price = (best_bid * ask_qty_1 + best_ask * bid_qty_1) / (bid_qty_1 + ask_qty_1)`
   - Spread Metric: `(best_ask - best_bid) / mid_price`

2. **Technicals Pipeline (`backend/app/features/indicators.py`):**
   - Session VWAP: Calculate cumulative volume-weighted average price reset at 09:15 IST daily.
   - Standard Deviation Bands: Compute rolling ±1σ and ±2σ bands around Session VWAP.
   - Realized Volatility: Rolling annualized standard deviation of 5-minute log returns.

### Constraints
- Never compute global normalization across the entire time-series; use rolling historical z-scores to prevent lookahead bias.