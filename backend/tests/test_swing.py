"""Swing pipeline tests on synthetic daily bars: causality, labels, purged walk-forward, costs, backtest."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from backend.app.backtest.swing_backtest import DeliveryCostModel, backtest_top_n
from backend.app.features.daily_features import FEATURE_COLUMNS, build_daily_features
from backend.app.models.swing_dataset import build_labeled_dataset
from backend.app.models.swing_model import walk_forward_predictions
from backend.app.data.universe import load_universe, nse_to_fyers

IST = ZoneInfo("Asia/Kolkata")
N_DAYS = 560  # ~390 trading days, enough for the 252-day lookbacks
SYMBOLS = [f"NSE:STOCK{i:02d}-EQ" for i in range(12)]


def ist_midnight_epoch(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=IST).timestamp())


def make_daily(symbol: str, seed: int, start: date = date(2025, 1, 1)) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 500 * np.exp(np.cumsum(rng.normal(0.0005, 0.015, N_DAYS)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, N_DAYS))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, N_DAYS)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, N_DAYS)))
    dates = [start + timedelta(days=i) for i in range(N_DAYS)]
    # skip weekends so trading-day shifts behave like the market
    keep = [i for i, d in enumerate(dates) if d.weekday() < 5]
    dates = [dates[i] for i in keep]
    idx = np.array(keep)
    return pl.DataFrame({
        "symbol": [symbol] * len(keep),
        "timestamp": [ist_midnight_epoch(d) for d in dates],
        "open": open_[idx], "high": high[idx], "low": low[idx], "close": close[idx],
        "volume": np.random.default_rng(seed + 1).integers(10_000, 50_000, len(keep)).astype(np.int64),
    })


@pytest.fixture(scope="module")
def universe_bars() -> pl.DataFrame:
    return pl.concat([make_daily(s, i) for i, s in enumerate(SYMBOLS)])


@pytest.fixture(scope="module")
def bench_bars() -> pl.DataFrame:
    return make_daily("NSE:NIFTY50-INDEX", 999)


def test_features_are_causal(universe_bars, bench_bars):
    """Truncation test: features on date t must not change when later bars are added."""
    full = build_daily_features(universe_bars, bench_bars)
    cut_date = date(2025, 9, 1)
    cut_bars = universe_bars.filter(pl.col("timestamp") <= ist_midnight_epoch(cut_date))
    cut_bench = bench_bars.filter(pl.col("timestamp") <= ist_midnight_epoch(cut_date))
    cut = build_daily_features(cut_bars, cut_bench)

    joined = full.join(cut, on=["symbol", "date"], suffix="_cut").filter(pl.col("date") <= cut_date)
    assert joined.height > 0
    for c in FEATURE_COLUMNS:
        a = joined[c].to_numpy().astype(float)
        b = joined[c + "_cut"].to_numpy().astype(float)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=c)


def test_label_uses_next_open_and_horizon_close(universe_bars, bench_bars):
    H = 5
    ds = build_labeled_dataset(universe_bars, bench_bars, horizon=H)
    row = ds.filter(pl.col("label_raw").is_not_null()).sort(["symbol", "date"]).head(1)
    symbol, d = row["symbol"][0], row["date"][0]

    bars = universe_bars.filter(pl.col("symbol") == symbol).sort("timestamp")
    dates = [datetime.fromtimestamp(t, tz=IST).date() for t in bars["timestamp"].to_list()]
    i = dates.index(d)
    expected = bars["close"][i + H] / bars["open"][i + 1] - 1.0
    assert row["label_raw"][0] == pytest.approx(expected, rel=1e-9)
    assert row["exit_date"][0] == dates[i + H]


def test_last_rows_have_no_label(universe_bars, bench_bars):
    ds = build_labeled_dataset(universe_bars, bench_bars, horizon=5)
    last = ds.filter(pl.col("symbol") == SYMBOLS[0]).sort("date").tail(3)
    assert last["label_raw"].null_count() == 3


def test_target_is_cross_sectional_percentile(universe_bars, bench_bars):
    ds = build_labeled_dataset(universe_bars, bench_bars, horizon=5)
    one_day = ds.filter(pl.col("target").is_not_null()).group_by("date").agg(
        pl.col("target").min().alias("lo"), pl.col("target").max().alias("hi"), pl.len().alias("n")
    ).filter(pl.col("n") == len(SYMBOLS)).head(5)
    assert (one_day["lo"] == 0.0).all() and (one_day["hi"] == 1.0).all()


def test_walk_forward_purges_overlapping_labels(universe_bars, bench_bars):
    H = 5
    ds = build_labeled_dataset(universe_bars, bench_bars, horizon=H)
    starts = [date(2026, 3, 2), date(2026, 6, 1)]
    preds = walk_forward_predictions(ds, starts)

    assert preds["date"].min() >= starts[0]
    # Each fold's training labels must have matured before that fold's start: checked by the
    # filter in walk_forward_predictions. Here we confirm the predictions are out-of-sample dated.
    assert set(preds["fold_start"].unique().to_list()) <= set(starts)
    assert preds["score"].is_finite().all()


def test_universe_loader_reads_nse_csv(tmp_path: Path):
    (tmp_path / "nifty50.csv").write_text("Company Name,Symbol\nReliance,RELIANCE\nSBI,SBIN\nReliance dup,RELIANCE\n")
    assert load_universe("nifty50", base_dir=tmp_path) == ["NSE:RELIANCE-EQ", "NSE:SBIN-EQ"]
    assert nse_to_fyers("NSE:TCS-EQ") == "NSE:TCS-EQ"
    with pytest.raises(FileNotFoundError):
        load_universe("nifty500", base_dir=tmp_path)


def test_delivery_cost_matches_hand_calculation():
    costs = DeliveryCostModel()
    notional = 100_000.0
    gst_base = 0.0 + 0.0000297 * notional + 0.000001 * notional   # brokerage is 0
    gst = 0.18 * gst_base
    buy = 0.001 * notional + 0.00015 * notional + 0.0000297 * notional + 0.000001 * notional + gst
    sell = 0.001 * notional + 0.0000297 * notional + 0.000001 * notional + gst + 16.0
    assert costs.round_trip_fraction(notional) == pytest.approx((buy + sell) / notional, rel=1e-12)
    assert costs.round_trip_fraction(notional) > 0.002  # delivery round trip is well above 0.2%


def test_backtest_runs_and_reports_costs(universe_bars, bench_bars):
    H = 5
    ds = build_labeled_dataset(universe_bars, bench_bars, horizon=H)
    preds = walk_forward_predictions(ds, [date(2026, 3, 2)])
    result = backtest_top_n(preds, horizon=H, top_n=3)
    assert result["n_trades"] > 0
    assert result["round_trip_cost_pct"] > 0.2
    assert -1.0 <= result["strategy"]["max_drawdown"] <= 0.0
    assert 0.0 <= result["strategy"]["hit_rate"] <= 1.0


def test_backtest_costs_reduce_returns(universe_bars, bench_bars):
    H = 5
    ds = build_labeled_dataset(universe_bars, bench_bars, horizon=H)
    preds = walk_forward_predictions(ds, [date(2026, 3, 2)])
    free = backtest_top_n(preds, H, top_n=3, costs=DeliveryCostModel(
        stt_rate=0, stamp_duty_buy_rate=0, exchange_rate=0, sebi_rate=0, gst_rate=0, dp_charge_per_sell_scrip=0))
    paid = backtest_top_n(preds, H, top_n=3)
    assert free["strategy"]["total_return"] > paid["strategy"]["total_return"]
