"""Shared TFT feature pipeline.

Training and real-time inference both call `build_tft_feature_frame`, so the
feature definitions cannot drift between the two paths.

All features are causal: the row at time `t` uses only bars with timestamp <= t.
"""

from typing import Final, List, Union

import polars as pl

from backend.app.features.indicators import compute_rsi, ensure_ist_session

CONTINUOUS_FEATURES: Final[List[str]] = [
    "log_return",
    "volume_ratio",
    "rsi_14",
    "vwap_diff",
    "depth_imbalance",
    "micro_price_spread",
]
TIME_COVARIATES: Final[List[str]] = ["minute_of_day", "day_of_week"]
ALL_FEATURES: Final[List[str]] = CONTINUOUS_FEATURES + TIME_COVARIATES

# NSE cash session: 09:15 to 15:30 IST = 375 minutes.
SESSION_OPEN_MINUTE: Final[int] = 9 * 60 + 15
SESSION_LENGTH_MINUTES: Final[int] = 375
VOLUME_WINDOW: Final[int] = 20
RSI_PERIOD: Final[int] = 14


def build_tft_feature_frame(
    bars: pl.DataFrame,
    depth_imbalance: Union[float, None] = None,
    micro_price_spread: Union[float, None] = None,
) -> pl.DataFrame:
    """Compute the TFT feature columns for every bar in `bars`.

    Args:
        bars: Polars frame with columns `timestamp` (epoch seconds, UTC), `close`, `volume`.
              Must hold a single symbol.
        depth_imbalance: Constant value used when `bars` has no `depth_imbalance` column.
            Historical bars carry no depth, so training passes None and gets 0.0.
        micro_price_spread: Same as `depth_imbalance`, for `micro_price_spread`.

    Returns:
        Frame sorted by timestamp with `timestamp`, `close`, `session_id`, and every
        column in `ALL_FEATURES`. The first bar has a null `log_return` (no prior close).
    """
    df = bars.sort("timestamp")
    df = ensure_ist_session(df)  # adds datetime_ist and session_id

    # 1. Log return from the previous bar close.
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")
    )

    # 2. Volume ratio against a trailing mean. min_periods=1 avoids a 20-bar warmup
    #    so a 60-bar live buffer is enough. A zero mean yields null, not a fake 1.0.
    vol_mean = pl.col("volume").rolling_mean(window_size=VOLUME_WINDOW, min_samples=1)
    df = df.with_columns(
        pl.when(vol_mean > 0)
        .then(pl.col("volume") / vol_mean)
        .otherwise(None)
        .alias("volume_ratio")
    )

    # 3. RSI(14), Wilder smoothing (shared implementation in indicators.py).
    df = compute_rsi(df, period=RSI_PERIOD, close_col="close", output_col="rsi_14")

    # 4. Session VWAP: cumulative from 09:15 IST, reset every session. Causal.
    pv = (pl.col("close") * pl.col("volume")).cum_sum().over("session_id")
    vol_cum = pl.col("volume").cum_sum().over("session_id")
    vwap = pl.when(vol_cum > 0).then(pv / vol_cum).otherwise(None)
    df = df.with_columns(((pl.col("close") - vwap) / vwap).alias("vwap_diff"))

    # 5. Microstructure covariates: per-row column if present, else a constant.
    df = _attach_constant_or_column(df, "depth_imbalance", depth_imbalance)
    df = _attach_constant_or_column(df, "micro_price_spread", micro_price_spread)

    # 6. Time covariates from IST session time (not synthesized).
    minute_in_day = pl.col("datetime_ist").dt.hour().cast(pl.Int32) * 60 + pl.col(
        "datetime_ist"
    ).dt.minute().cast(pl.Int32)
    df = df.with_columns(
        (minute_in_day - SESSION_OPEN_MINUTE)
        .clip(0, SESSION_LENGTH_MINUTES)
        .cast(pl.Float64)
        .alias("minute_of_day"),
        pl.col("datetime_ist").dt.weekday().cast(pl.Float64).alias("day_of_week"),
    )

    return df.select(["timestamp", "close", "session_id", *ALL_FEATURES])


def _attach_constant_or_column(df: pl.DataFrame, name: str, value: Union[float, None]) -> pl.DataFrame:
    if name in df.columns:
        return df.with_columns(pl.col(name).cast(pl.Float64))
    fill = 0.0 if value is None else float(value)
    return df.with_columns(pl.lit(fill, dtype=pl.Float64).alias(name))
