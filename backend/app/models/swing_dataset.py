"""Labelled daily dataset for swing-trade ranking.

Timing convention (what a real swing trade can do):
- A signal is computed from the close of day `t`.
- Entry is at the OPEN of day t+1.
- Exit is at the CLOSE of day t+H (H trading days later).

Labels:
- label_raw:     close[t+H] / open[t+1] - 1
- bench_label:   the same return for the Nifty 50 over the same dates
- label_excess:  label_raw - bench_label
- target:        cross-sectional percentile (0..1) of label_excess among stocks on date t

Labels are null when the future bars are missing or the calendar gap is implausibly long
for H trading days (a data gap, not a real hold).
"""

import polars as pl

from backend.app.features.daily_features import (
    FEATURE_COLUMNS,
    build_daily_features,
    to_daily_frame,
)


def _per_symbol(expr: pl.Expr) -> pl.Expr:
    return expr.over("symbol")


def build_labeled_dataset(
    bars: pl.DataFrame,
    benchmark: pl.DataFrame,
    horizon: int,
) -> pl.DataFrame:
    """Features plus labels for every (symbol, date). Rows near the end have null labels."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1 trading day")

    features = build_daily_features(bars, benchmark)
    daily = to_daily_frame(bars)

    labels = daily.select(
        "symbol",
        "date",
        _per_symbol(pl.col("open").shift(-1)).alias("entry_open"),
        _per_symbol(pl.col("close").shift(-horizon)).alias("exit_close"),
        _per_symbol(pl.col("date").shift(-horizon)).alias("exit_date"),
    )

    bench = to_daily_frame(benchmark).select(
        "date",
        pl.col("open").shift(-1).alias("bench_entry_open"),
        pl.col("close").shift(-horizon).alias("bench_exit_close"),
    )

    labels = (
        labels.join(bench, on="date", how="left")
        .with_columns(
            (pl.col("exit_close") / pl.col("entry_open") - 1.0).alias("label_raw"),
            (pl.col("bench_exit_close") / pl.col("bench_entry_open") - 1.0).alias("bench_label"),
        )
        .with_columns((pl.col("label_raw") - pl.col("bench_label")).alias("label_excess"))
    )

    # Guard against data gaps: H trading days should not span far more calendar days than that.
    max_calendar_days = horizon * 2 + 7
    gap_ok = (pl.col("exit_date") - pl.col("date")).dt.total_days() <= max_calendar_days
    labels = labels.with_columns(
        pl.when(gap_ok).then(pl.col(c)).otherwise(None).alias(c)
        for c in ("label_raw", "bench_label", "label_excess")
    ).with_columns(
        pl.when(gap_ok).then(pl.col("exit_date")).otherwise(None).alias("exit_date")
    )

    df = features.join(
        labels.select(["symbol", "date", "exit_date", "label_raw", "bench_label", "label_excess"]),
        on=["symbol", "date"],
        how="left",
    )

    # Regression target: percentile of the excess return among the day's stocks.
    rank = pl.col("label_excess").rank(method="average").over("date")
    count = pl.col("label_excess").count().over("date")
    df = df.with_columns(
        pl.when(pl.col("label_excess").is_null() | (count <= 1))
        .then(None)
        .otherwise((rank - 1.0) / (count - 1.0))
        .alias("target")
    )
    return df


def feature_columns() -> list[str]:
    return list(FEATURE_COLUMNS)
