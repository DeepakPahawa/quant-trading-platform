"""Daily swing-trading features for a universe of stocks.

Every feature at date `t` uses only bars dated `t` or earlier. Cross-sectional ranks use
only the values on date `t`, across the universe, so they are causal too.

Input bars: one row per (symbol, bar), with `timestamp` (epoch seconds), `open`, `high`,
`low`, `close`, `volume`. Benchmark bars use the same schema, for the Nifty 50 index.
"""

from typing import Final, List

import polars as pl

# Raw per-stock features.
RAW_FEATURES: Final[List[str]] = [
    "ret_5",
    "ret_20",
    "ret_60",
    "ema20_dist",
    "ema50_dist",
    "ema200_dist",
    "rsi_14",
    "atr_pct_14",
    "vol_20",
    "volume_ratio_20",
    "high_52w_prox",
    "breakout_20",
    "rel_ret_5",
    "rel_ret_20",
]

# Cross-sectional ranks (0..1 within each day's universe) of the features that matter most.
RANKED_FEATURES: Final[List[str]] = [
    "ret_5",
    "ret_20",
    "ret_60",
    "rsi_14",
    "volume_ratio_20",
    "rel_ret_20",
    "high_52w_prox",
    "vol_20",
]
RANK_PREFIX: Final[str] = "cs_"

FEATURE_COLUMNS: Final[List[str]] = RAW_FEATURES + [RANK_PREFIX + c for c in RANKED_FEATURES]

TRADING_DAYS_PER_YEAR: Final[int] = 252


def to_daily_frame(bars: pl.DataFrame) -> pl.DataFrame:
    """Add the IST trading `date`, drop duplicate (symbol, date) rows, sort by symbol and date."""
    df = bars.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s")
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Asia/Kolkata")
        .dt.date()
        .alias("date")
    )
    return (
        df.unique(subset=["symbol", "date"], keep="last")
        .sort(["symbol", "date"])
        .select(["symbol", "date", "open", "high", "low", "close", "volume"])
    )


def _per_symbol(expr: pl.Expr) -> pl.Expr:
    return expr.over("symbol")


def add_price_features(df: pl.DataFrame) -> pl.DataFrame:
    """Per-stock features. `df` must be sorted by (symbol, date) with no duplicates."""
    close = pl.col("close")
    prev_close = _per_symbol(close.shift(1))
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    log_ret = (close / prev_close).log()
    gain = pl.when(close > prev_close).then(close - prev_close).otherwise(0.0)
    loss = pl.when(close < prev_close).then(prev_close - close).otherwise(0.0)

    df = df.with_columns(
        _per_symbol(close.pct_change(5)).alias("ret_5"),
        _per_symbol(close.pct_change(20)).alias("ret_20"),
        _per_symbol(close.pct_change(60)).alias("ret_60"),
        _per_symbol(close.ewm_mean(span=20, adjust=False)).alias("_ema20"),
        _per_symbol(close.ewm_mean(span=50, adjust=False)).alias("_ema50"),
        _per_symbol(close.ewm_mean(span=200, adjust=False)).alias("_ema200"),
        _per_symbol(tr.ewm_mean(alpha=1 / 14, adjust=False)).alias("_atr14"),
        _per_symbol(gain.ewm_mean(alpha=1 / 14, adjust=False)).alias("_avg_gain"),
        _per_symbol(loss.ewm_mean(alpha=1 / 14, adjust=False)).alias("_avg_loss"),
        _per_symbol(log_ret.rolling_std(window_size=20)).alias("_std20"),
        _per_symbol(pl.col("volume").rolling_mean(window_size=20)).alias("_vol_mean20"),
        _per_symbol(pl.col("high").rolling_max(window_size=252)).alias("_high252"),
        _per_symbol(pl.col("high").rolling_max(window_size=20).shift(1)).alias("_high20_prev"),
    )

    df = df.with_columns(
        (pl.col("close") / pl.col("_ema20") - 1).alias("ema20_dist"),
        (pl.col("close") / pl.col("_ema50") - 1).alias("ema50_dist"),
        (pl.col("close") / pl.col("_ema200") - 1).alias("ema200_dist"),
        (pl.col("_atr14") / pl.col("close")).alias("atr_pct_14"),
        (100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / pl.col("_avg_loss"))).alias("rsi_14"),
        (pl.col("_std20") * float(TRADING_DAYS_PER_YEAR) ** 0.5).alias("vol_20"),
        pl.when(pl.col("_vol_mean20") > 0)
        .then(pl.col("volume") / pl.col("_vol_mean20"))
        .otherwise(None)
        .alias("volume_ratio_20"),
        (pl.col("close") / pl.col("_high252") - 1).alias("high_52w_prox"),
        (pl.col("close") > pl.col("_high20_prev")).cast(pl.Float64).alias("breakout_20"),
    )
    return df.drop([c for c in df.columns if c.startswith("_")])


def add_benchmark_relative(df: pl.DataFrame, benchmark: pl.DataFrame) -> pl.DataFrame:
    """Join benchmark returns by date and compute stock-minus-benchmark returns."""
    bench = (
        to_daily_frame(benchmark)
        .select(
            "date",
            pl.col("close").pct_change(5).alias("bench_ret_5"),
            pl.col("close").pct_change(20).alias("bench_ret_20"),
        )
    )
    return (
        df.join(bench, on="date", how="left")
        .with_columns(
            (pl.col("ret_5") - pl.col("bench_ret_5")).alias("rel_ret_5"),
            (pl.col("ret_20") - pl.col("bench_ret_20")).alias("rel_ret_20"),
        )
        .drop(["bench_ret_5", "bench_ret_20"])
    )


def add_cross_sectional_ranks(df: pl.DataFrame) -> pl.DataFrame:
    """Percentile rank (0..1) of each ranked feature among stocks on the same date."""
    exprs = []
    for col in RANKED_FEATURES:
        rank = pl.col(col).rank(method="average").over("date")
        count = pl.col(col).count().over("date")
        exprs.append(
            pl.when(count > 1)
            .then((rank - 1.0) / (count - 1.0))
            .otherwise(0.5)
            .alias(RANK_PREFIX + col)
        )
    return df.with_columns(exprs)


def build_daily_features(bars: pl.DataFrame, benchmark: pl.DataFrame) -> pl.DataFrame:
    """Full feature table: one row per (symbol, date) with FEATURE_COLUMNS.

    Rows are kept even when a feature is null (early history). Callers decide what to drop.
    """
    df = to_daily_frame(bars)
    df = add_price_features(df)
    df = add_benchmark_relative(df, benchmark)
    df = add_cross_sectional_ranks(df)
    return df.select(["symbol", "date", "open", "close", *FEATURE_COLUMNS])
