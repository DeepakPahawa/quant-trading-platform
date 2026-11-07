"""Standard strategy signal generators using Polars for high-performance backtesting."""

from typing import Any, Dict, Optional
import polars as pl

from backend.app.features.indicators import compute_rsi, compute_session_vwap


def generate_sma_signals(
    df: pl.DataFrame,
    fast_period: int = 9,
    slow_period: int = 21,
) -> pl.DataFrame:
    """Generate SMA crossover signals.

    BUY when fast SMA crosses above slow SMA.
    SELL when fast SMA crosses below slow SMA.

    Returns:
        DataFrame with columns 'timestamp', 'signal' ('BUY', 'SELL', 'FLAT').
    """
    if len(df) < max(fast_period, slow_period):
        return pl.DataFrame({"timestamp": [], "signal": []}, schema={"timestamp": pl.Int64, "signal": pl.Utf8})

    df_calc = df.sort("timestamp").with_columns([
        pl.col("close").rolling_mean(window_size=fast_period).alias("sma_fast"),
        pl.col("close").rolling_mean(window_size=slow_period).alias("sma_slow"),
    ])

    df_calc = df_calc.with_columns([
        pl.col("sma_fast").shift(1).alias("prev_fast"),
        pl.col("sma_slow").shift(1).alias("prev_slow"),
    ])

    signal_expr = (
        pl.when((pl.col("sma_fast") > pl.col("sma_slow")) & (pl.col("prev_fast") <= pl.col("prev_slow")))
        .then(pl.lit("BUY"))
        .when((pl.col("sma_fast") < pl.col("sma_slow")) & (pl.col("prev_fast") >= pl.col("prev_slow")))
        .then(pl.lit("SELL"))
        .otherwise(pl.lit("FLAT"))
        .alias("signal")
    )

    result = df_calc.with_columns(signal_expr).select(["timestamp", "signal"])
    return result.filter(pl.col("signal") != "FLAT")


def generate_rsi_signals(
    df: pl.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pl.DataFrame:
    """Generate RSI mean-reversion signals.

    BUY when RSI crosses above oversold (e.g. 30).
    SELL when RSI crosses below overbought (e.g. 70).
    """
    if len(df) < period + 5:
        return pl.DataFrame({"timestamp": [], "signal": []}, schema={"timestamp": pl.Int64, "signal": pl.Utf8})

    df_calc = compute_rsi(df, period=period, price_col="close", rsi_col="rsi")
    df_calc = df_calc.with_columns(pl.col("rsi").shift(1).alias("prev_rsi"))

    signal_expr = (
        pl.when((pl.col("rsi") > oversold) & (pl.col("prev_rsi") <= oversold))
        .then(pl.lit("BUY"))
        .when((pl.col("rsi") < overbought) & (pl.col("prev_rsi") >= overbought))
        .then(pl.lit("SELL"))
        .otherwise(pl.lit("FLAT"))
        .alias("signal")
    )

    result = df_calc.with_columns(signal_expr).select(["timestamp", "signal"])
    return result.filter(pl.col("signal") != "FLAT")


def generate_vwap_signals(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Generate VWAP band breakout/reversion signals."""
    if len(df) < 20:
        return pl.DataFrame({"timestamp": [], "signal": []}, schema={"timestamp": pl.Int64, "signal": pl.Utf8})

    df_calc = compute_session_vwap(df)
    df_calc = df_calc.with_columns(pl.col("close").shift(1).alias("prev_close"))

    # Buy when rebounding off lower band (-2 sigma), sell when hitting upper band (+2 sigma)
    signal_expr = (
        pl.when((pl.col("close") > pl.col("vwap_lower_2sigma")) & (pl.col("prev_close") <= pl.col("vwap_lower_2sigma")))
        .then(pl.lit("BUY"))
        .when((pl.col("close") < pl.col("vwap_upper_2sigma")) & (pl.col("prev_close") >= pl.col("vwap_upper_2sigma")))
        .then(pl.lit("SELL"))
        .otherwise(pl.lit("FLAT"))
        .alias("signal")
    )

    result = df_calc.with_columns(signal_expr).select(["timestamp", "signal"])
    return result.filter(pl.col("signal") != "FLAT")


def build_strategy_signals(
    strategy_name: str,
    df: pl.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> pl.DataFrame:
    """Dispatch strategy name to corresponding Polars signal generator."""
    p = params or {}
    strat = strategy_name.lower().strip()

    if strat in ("sma", "sma_cross", "sma_crossover"):
        fast_p = int(p.get("fast_period", 9))
        slow_p = int(p.get("slow_period", 21))
        return generate_sma_signals(df, fast_period=fast_p, slow_period=slow_p)
    elif strat in ("rsi", "rsi_momentum", "rsi_reversion"):
        period = int(p.get("period", 14))
        oversold = float(p.get("oversold", 30.0))
        overbought = float(p.get("overbought", 70.0))
        return generate_rsi_signals(df, period=period, oversold=oversold, overbought=overbought)
    elif strat in ("vwap", "vwap_bands", "vwap_reversion"):
        return generate_vwap_signals(df)
    else:
        # Default empty signals if unrecognized or custom
        return pl.DataFrame({"timestamp": [], "signal": []}, schema={"timestamp": pl.Int64, "signal": pl.Utf8})

