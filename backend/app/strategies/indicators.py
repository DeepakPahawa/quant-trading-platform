"""Causal Polars indicator expressions for daily strategies.

Every expression is evaluated per symbol (`.over("symbol")`) on a frame sorted by (symbol, date).
A value at row `t` uses only rows at or before `t`. Rolling windows return null until full.
"""

import polars as pl

SYMBOL = "symbol"


def sma(col: str, window: int) -> pl.Expr:
    return pl.col(col).rolling_mean(window_size=window).over(SYMBOL)


def wilder_rsi(period: int, col: str = "close") -> pl.Expr:
    """Wilder RSI. RSI(2) with alpha=1/2 is the Connors short-term RSI."""
    delta = pl.col(col) - pl.col(col).shift(1).over(SYMBOL)
    gain = pl.when(delta > 0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0.0)
    avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period + 1).over(SYMBOL)
    avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period + 1).over(SYMBOL)
    return (
        pl.when(avg_loss == 0)
        .then(pl.when(avg_gain == 0).then(50.0).otherwise(100.0))
        .otherwise(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))
    )


def atr(period: int = 14) -> pl.Expr:
    """Wilder average true range, in price units."""
    prev_close = pl.col("close").shift(1).over(SYMBOL)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return tr.ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period).over(SYMBOL)


def prior_high(window: int) -> pl.Expr:
    """Highest high of the previous `window` bars, excluding today."""
    return pl.col("high").rolling_max(window_size=window).shift(1).over(SYMBOL)


def prior_low(window: int) -> pl.Expr:
    """Lowest low of the previous `window` bars, excluding today."""
    return pl.col("low").rolling_min(window_size=window).shift(1).over(SYMBOL)


def prior_avg_volume(window: int) -> pl.Expr:
    """Average volume of the previous `window` bars, excluding today."""
    return pl.col("volume").rolling_mean(window_size=window).shift(1).over(SYMBOL)


def pct_return(window: int, col: str = "close") -> pl.Expr:
    return pl.col(col).pct_change(window).over(SYMBOL)


def add_regime(df: pl.DataFrame, bench: pl.DataFrame, window: int = 200) -> pl.DataFrame:
    """Add `regime_ok`: benchmark close above its own `window`-day SMA on that date (false when unknown)."""
    b = bench.sort("date").select(
        "date",
        (pl.col("close") > pl.col("close").rolling_mean(window_size=window)).alias("regime_ok"),
    )
    return (
        df.join(b, on="date", how="left")
        .with_columns(pl.col("regime_ok").fill_null(False))
        .sort([SYMBOL, "date"])
    )


def add_bench_return(df: pl.DataFrame, bench: pl.DataFrame, window: int, name: str) -> pl.DataFrame:
    """Add `name`: the benchmark's `window`-day return on each date."""
    b = bench.sort("date").select("date", pl.col("close").pct_change(window).alias(name))
    return df.join(b, on="date", how="left").sort([SYMBOL, "date"])
