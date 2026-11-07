---
name: quant-features
description: Implement or review quant features with Polars — session VWAP and bands, RSI, ATR, realized volatility, volume ratio, depth imbalance, micro-price, spread — and prove they have no lookahead. Use when adding or changing anything in backend/app/features or the feature tensor in market_overview.py.
---

# Quant Features

Spec: `.agents/skills/quant-feature-store/SKILL.md`. Data rules: `.claude/rules/market-data-integrity.md`.

## Where things live

- `backend/app/features/indicators.py`: `ensure_ist_session`, `compute_session_vwap`, RSI, ATR, realized volatility. Config in `TechnicalIndicatorConfig`.
- `backend/app/features/microstructure.py`: depth imbalance, micro-price, normalized spread. Result type `MicrostructureMetrics`.
- `backend/app/data/market_overview.py`: `prepare_tft_feature_tensor` assembles the model input. Volume ratio and RSI are computed here too (see the duplicate-RSI issue in `CLAUDE.md`).

## Definitions (keep these exact)

- Session VWAP: cumulative `sum(typical_price * volume) / sum(volume)` from 09:15 IST, reset daily. Typical price is `(high + low + close) / 3`.
- Bands: rolling ±1σ and ±2σ around VWAP using a trailing window (default 20).
- RSI(14): Wilder smoothing. Seed with a simple mean of the first 14 values, then `avg = (prev_avg * 13 + x) / 14`. Use this, not a rolling mean. Fix `market_overview.py` to match.
- ATR(14): Wilder smoothing of True Range.
- Realized volatility: rolling std of 5-minute log returns times `sqrt(18900)` (75 bars × 252 days). Use the constant in `ANNUALIZATION_FACTOR_5MIN`.
- Volume ratio: `volume / rolling_mean(volume, 20)`. Use the trailing mean, and treat a zero mean as null, not 1.0.
- Depth imbalance (top 5): `(sum(bid_qty) - sum(ask_qty)) / (sum(bid_qty) + sum(ask_qty))`. Return null when the denominator is zero.
- Micro-price: `(best_bid * ask_qty_1 + best_ask * bid_qty_1) / (bid_qty_1 + ask_qty_1)`.
- Spread: `(best_ask - best_bid) / mid`.

## Rules when writing a feature

1. **Polars only**, with lazy expressions where practical. No row-by-row Python loops over bars.
2. **Trailing windows only.** Use `rolling_*` with `min_samples` set, and never `center=True`.
3. **Null, not fill.** Use `null` for insufficient history and for undefined values. Don't fill with 0 or 1 silently. Callers decide how to handle nulls.
4. **Session-aware.** Anything that accumulates over a day resets at 09:15 IST via `session_id`.
5. **Vectorized normalization** uses a rolling z-score, `(x - rolling_mean) / rolling_std`, never a full-sample statistic.

## The truncation test (required for every feature)

```python
def test_no_lookahead(bars):
    full = compute_features(bars)
    for t in [100, 500, 1500]:
        cut = compute_features(bars.head(t + 1))
        # Features at rows 0..t must be identical whether or not future bars exist
        assert_frame_equal(full.head(t + 1), cut, check_exact=False, rtol=1e-12)
```

This test fails if any feature peeks forward. Put it in `backend/tests/` next to the feature module and run it with `pytest`.

## Other tests

- VWAP resets at 09:15 IST across a session boundary, including a bar at 09:14 that belongs to the prior session.
- RSI on a constant series is 50 or null by definition. Document which one you chose.
- Micro-price equals mid when bid and ask quantities are equal.
- Index symbols produce null volume features rather than zeros.
