"""Feature pipeline tests: IST time covariates, shared RSI, and no-lookahead."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from backend.app.data.market_overview import prepare_tft_feature_tensor
from backend.app.features.indicators import compute_rsi
from backend.app.features.tft_features import ALL_FEATURES, build_tft_feature_frame


def make_bars(n: int, start_utc: datetime, seed: int = 7) -> pl.DataFrame:
    """Deterministic 1-minute bars with a random-walk close."""
    rng = np.random.default_rng(seed)
    close = 2500.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.5
    low = np.minimum(open_, close) - 0.5
    volume = rng.integers(1000, 5000, n)
    t0 = int(start_utc.timestamp())
    return pl.DataFrame({
        "timestamp": np.arange(n, dtype=np.int64) * 60 + t0,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume.astype(np.int64),
        "open_interest": np.zeros(n, dtype=np.int64),
    })


# 2026-03-02 03:45 UTC is 09:15 IST, the NSE open.
OPEN_UTC = datetime(2026, 3, 2, 3, 45, tzinfo=timezone.utc)


def test_minute_of_day_is_ist_session_time():
    bars = make_bars(60, OPEN_UTC)
    feats = build_tft_feature_frame(bars)
    minutes = feats["minute_of_day"].to_list()
    assert minutes[0] == 0.0, "09:15 IST must map to minute 0 (not the UTC hour)"
    assert minutes[-1] == 59.0


def test_minute_of_day_clips_to_session_length():
    # Start 2 hours before the open so the window spans pre-open bars that clip to 0.
    bars = make_bars(60, datetime(2026, 3, 2, 1, 45, tzinfo=timezone.utc))
    minutes = build_tft_feature_frame(bars)["minute_of_day"].to_list()
    assert min(minutes) == 0.0
    assert max(minutes) <= 375.0


def test_all_features_present_and_first_log_return_null():
    feats = build_tft_feature_frame(make_bars(40, OPEN_UTC))
    for name in ALL_FEATURES:
        assert name in feats.columns
    assert feats["log_return"][0] is None


def test_session_vwap_resets_each_session():
    # Two sessions: bars from 09:15 IST for 10 minutes, then 1 day later.
    day1 = make_bars(10, OPEN_UTC)
    day2 = make_bars(10, OPEN_UTC.replace(day=3))
    bars = pl.concat([day1, day2])
    feats = build_tft_feature_frame(bars)
    # The first bar of session 2 has vwap_diff computed against its own VWAP only.
    first_day2 = feats.filter(pl.col("session_id") == feats["session_id"][-1]).head(1)
    close = float(first_day2["close"][0])
    vwap_diff = float(first_day2["vwap_diff"][0])
    assert abs(vwap_diff) < 1e-12, f"single-bar session must have vwap == close (got diff {vwap_diff}, close {close})"


def test_no_lookahead_features_are_causal():
    """Truncation test: features at bar t must not change when future bars are appended."""
    full_bars = make_bars(200, OPEN_UTC)
    full = build_tft_feature_frame(full_bars)
    for t in (30, 90, 150):
        cut = build_tft_feature_frame(full_bars.head(t + 1))
        for name in ALL_FEATURES:
            a = full[name].head(t + 1).to_numpy().astype(float)
            b = cut[name].to_numpy().astype(float)
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=f"{name} at t={t}")


def test_rsi_is_wilder_implementation_shared_with_indicators():
    """Regression: the TFT pipeline must use indicators.compute_rsi, not a second RSI."""
    bars = make_bars(120, OPEN_UTC)
    via_pipeline = build_tft_feature_frame(bars)["rsi_14"].to_numpy()
    via_indicators = compute_rsi(bars.sort("timestamp"), period=14, close_col="close", output_col="r")["r"].to_numpy()
    np.testing.assert_allclose(via_pipeline, via_indicators, rtol=1e-12, atol=1e-12)


def test_live_tensor_has_no_nulls_and_full_width():
    bars = make_bars(60, OPEN_UTC)
    df = prepare_tft_feature_tensor(bars, sequence_length=60)
    assert len(df) == 60
    assert df.select(ALL_FEATURES).null_count().sum_horizontal().item() == 0


@pytest.mark.parametrize("n", [60, 61, 120])
def test_live_tensor_length_matches_request(n: int):
    df = prepare_tft_feature_tensor(make_bars(n, OPEN_UTC), sequence_length=60)
    assert len(df) == 60
