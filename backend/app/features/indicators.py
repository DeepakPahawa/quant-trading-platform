"""Technical Indicators Pipeline using Polars.

Provides vectorized calculations for:
1. Session VWAP: Cumulative volume-weighted average price reset daily at 09:15 IST.
   Includes rolling ±1σ and ±2σ standard deviation bands.
2. 14-period Relative Strength Index (RSI) using Wilder's exponential smoothing.
3. 14-period Average True Range (ATR).
4. 20-period Realized Volatility of 5-minute log returns (annualized for Indian market hours).
"""

import datetime
import logging
from typing import Optional, Union

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Indian market trading hours: 09:15 to 15:30 IST = 375 minutes = 75 5-minute bars / day.
# 252 trading days/year -> 75 * 252 = 18,900 5-minute bars/year.
ANNUALIZATION_FACTOR_5MIN = 18900.0


class TechnicalIndicatorConfig(BaseModel):
    """Configuration parameters for technical indicators."""
    rsi_period: int = Field(default=14, ge=2, description="Lookback period for RSI")
    atr_period: int = Field(default=14, ge=1, description="Lookback period for ATR")
    rv_period: int = Field(default=20, ge=2, description="Rolling window for Realized Volatility")
    vwap_rolling_window: int = Field(default=20, ge=2, description="Rolling window for VWAP std bands")
    annualization_factor: float = Field(
        default=ANNUALIZATION_FACTOR_5MIN,
        description="Annualization factor for 5-minute returns",
    )


def ensure_ist_session(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    session_col: str = "session_id",
) -> pl.DataFrame:
    """Ensure timestamp column is converted to IST datetime and assign daily trading session ID.

    Trading session starts at 09:15:00 IST. Bars occurring at or after 09:15:00 belong
    to that calendar day's session. Any candle prior to 09:15:00 belongs to the prior session.
    """
    dtype = df.schema[timestamp_col]

    if dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32, pl.Float64):
        # Infer seconds vs milliseconds vs microseconds
        # 1e11 cutoff separates seconds (< 1e11) from milliseconds (> 1e11)
        sample_val = df[timestamp_col][0] if len(df) > 0 else 0
        if sample_val is not None and sample_val > 1e14:
            ts_expr = pl.from_epoch(pl.col(timestamp_col), time_unit="us")
        elif sample_val is not None and sample_val > 1e11:
            ts_expr = pl.from_epoch(pl.col(timestamp_col), time_unit="ms")
        else:
            ts_expr = pl.from_epoch(pl.col(timestamp_col), time_unit="s")
        dt_expr = ts_expr.dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Kolkata")
    elif dtype == pl.String:
        dt_expr = pl.col(timestamp_col).str.to_datetime().dt.convert_time_zone("Asia/Kolkata")
    elif isinstance(dtype, pl.Datetime):
        if dtype.time_zone is None:
            dt_expr = pl.col(timestamp_col).dt.replace_time_zone("Asia/Kolkata")
        else:
            dt_expr = pl.col(timestamp_col).dt.convert_time_zone("Asia/Kolkata")
    elif dtype == pl.Date:
        dt_expr = pl.col(timestamp_col).cast(pl.Datetime).dt.replace_time_zone("Asia/Kolkata")
    else:
        dt_expr = pl.col(timestamp_col).cast(pl.Datetime).dt.replace_time_zone("Asia/Kolkata")

    # Session ID assignment: if time >= 09:15:00 IST -> current date, else previous date
    session_expr = (
        pl.when(
            (pl.col("datetime_ist").dt.hour() > 9)
            | (
                (pl.col("datetime_ist").dt.hour() == 9)
                & (pl.col("datetime_ist").dt.minute() >= 15)
            )
        )
        .then(pl.col("datetime_ist").dt.date())
        .otherwise(pl.col("datetime_ist").dt.date() - pl.duration(days=1))
        .alias(session_col)
    )

    return (
        df.with_columns(dt_expr.alias("datetime_ist"))
        .with_columns(session_expr)
    )


def compute_session_vwap(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    volume_col: str = "volume",
    price_col: Optional[str] = None,
    rolling_window: int = 20,
    prefix: str = "vwap",
) -> pl.DataFrame:
    """Compute Session VWAP reset daily at 09:15 IST with rolling ±1σ and ±2σ bands.

    Computes:
    - vwap: Cumulative volume-weighted average price reset at 09:15 IST daily.
    - vwap_std: Rolling standard deviation of (close - vwap) over `rolling_window`.
    - vwap_upper_1: vwap + 1.0 * vwap_std
    - vwap_lower_1: vwap - 1.0 * vwap_std
    - vwap_upper_2: vwap + 2.0 * vwap_std
    - vwap_lower_2: vwap - 2.0 * vwap_std
    - session_vwap_std: Cumulative volume-weighted standard deviation from session open.
    """
    if df.is_empty():
        return df

    # Prepare IST timestamp & daily session grouping
    df_session = ensure_ist_session(df, timestamp_col=timestamp_col)

    # Determine typical price: (high + low + close) / 3 if available, else price_col or close
    schema = df_session.schema
    if price_col and price_col in schema:
        tp_expr = pl.col(price_col)
    elif "high" in schema and "low" in schema and "close" in schema:
        tp_expr = (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    elif "close" in schema:
        tp_expr = pl.col("close")
    else:
        raise ValueError("DataFrame must contain 'high', 'low', 'close' or specified 'price_col'")

    vol_expr = pl.col(volume_col).cast(pl.Float64)
    tp_expr = tp_expr.cast(pl.Float64).alias("_typical_price")

    df_prepared = df_session.with_columns([tp_expr, vol_expr.alias("_vol")])

    # Cumulative session sums over session_id
    cum_vol_expr = pl.col("_vol").cum_sum().over("session_id")
    cum_pv_expr = (pl.col("_typical_price") * pl.col("_vol")).cum_sum().over("session_id")
    cum_p2v_expr = ((pl.col("_typical_price") ** 2) * pl.col("_vol")).cum_sum().over("session_id")

    df_vwap = df_prepared.with_columns([
        cum_vol_expr.alias("_cum_vol"),
        cum_pv_expr.alias("_cum_pv"),
        cum_p2v_expr.alias("_cum_p2v"),
    ])

    # Primary VWAP calculation
    vwap_expr = (
        pl.when(pl.col("_cum_vol") > 0.0)
        .then(pl.col("_cum_pv") / pl.col("_cum_vol"))
        .otherwise(pl.col("_typical_price"))
        .alias(prefix)
    )

    df_vwap = df_vwap.with_columns(vwap_expr)

    # 1. Session volume-weighted standard deviation
    # Var = (sum(P^2 * V) / sum(V)) - VWAP^2
    session_var_expr = (
        pl.when(pl.col("_cum_vol") > 0.0)
        .then(
            pl.max_horizontal(
                (pl.col("_cum_p2v") / pl.col("_cum_vol")) - (pl.col(prefix) ** 2),
                pl.lit(0.0),
            )
        )
        .otherwise(0.0)
    )
    session_std_expr = session_var_expr.sqrt().alias("session_vwap_std")

    # 2. Rolling standard deviation of (close - VWAP) over rolling_window
    close_series = pl.col("close") if "close" in schema else pl.col("_typical_price")
    rolling_std_expr = (
        (close_series - pl.col(prefix))
        .rolling_std(window_size=rolling_window, min_samples=1)
        .over("session_id")
        .fill_null(0.0)
        .alias(f"{prefix}_std")
    )

    df_vwap = df_vwap.with_columns([session_std_expr, rolling_std_expr])

    # Rolling ±1σ and ±2σ standard deviation bands
    bands_exprs = [
        (pl.col(prefix) + 1.0 * pl.col(f"{prefix}_std")).alias(f"{prefix}_upper_1"),
        (pl.col(prefix) - 1.0 * pl.col(f"{prefix}_std")).alias(f"{prefix}_lower_1"),
        (pl.col(prefix) + 2.0 * pl.col(f"{prefix}_std")).alias(f"{prefix}_upper_2"),
        (pl.col(prefix) - 2.0 * pl.col(f"{prefix}_std")).alias(f"{prefix}_lower_2"),
    ]

    result = df_vwap.with_columns(bands_exprs).drop(["_typical_price", "_vol", "_cum_vol", "_cum_pv", "_cum_p2v"])
    return result


def compute_rsi(
    df: pl.DataFrame,
    period: int = 14,
    close_col: str = "close",
    output_col: str = "rsi",
) -> pl.DataFrame:
    """Compute Relative Strength Index (RSI) using Wilder's exponential smoothing.

    Wilder's smoothing uses alpha = 1 / period (span = 2 * period - 1).
    RSI = 100 - (100 / (1 + avg_gain / avg_loss)).
    """
    if df.is_empty() or close_col not in df.schema:
        return df

    delta = pl.col(close_col).cast(pl.Float64).diff()
    gain = pl.when(delta > 0.0).then(delta).otherwise(0.0)
    loss = pl.when(delta < 0.0).then(-delta).otherwise(0.0)

    alpha = 1.0 / float(period)
    avg_gain = gain.ewm_mean(alpha=alpha, adjust=False)
    avg_loss = loss.ewm_mean(alpha=alpha, adjust=False)

    rs = avg_gain / avg_loss

    rsi_expr = (
        pl.when(avg_loss == 0.0)
        .then(pl.when(avg_gain == 0.0).then(50.0).otherwise(100.0))
        .otherwise(100.0 - (100.0 / (1.0 + rs)))
        .fill_null(50.0)
        .alias(output_col)
    )

    return df.with_columns(rsi_expr)


def compute_atr(
    df: pl.DataFrame,
    period: int = 14,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    output_col: str = "atr",
) -> pl.DataFrame:
    """Compute 14-period Average True Range (ATR) using Wilder's smoothing.

    True Range = max(high - low, abs(high - close_prev), abs(low - close_prev)).
    ATR = ewm_mean(TR, alpha=1/period, adjust=False).
    """
    if df.is_empty():
        return df

    high = pl.col(high_col).cast(pl.Float64)
    low = pl.col(low_col).cast(pl.Float64)
    close = pl.col(close_col).cast(pl.Float64)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pl.max_horizontal([tr1, tr2, tr3]).fill_null(tr1)

    alpha = 1.0 / float(period)
    atr_expr = tr.ewm_mean(alpha=alpha, adjust=False).alias(output_col)

    return df.with_columns(atr_expr)


def resample_to_5min(
    df: pl.DataFrame,
    timestamp_col: str = "timestamp",
    sort: bool = True,
) -> pl.DataFrame:
    """Resample 1-minute OHLCV DataFrame into 5-minute bars using Polars dynamic grouping."""
    if df.is_empty():
        return df

    df_ist = ensure_ist_session(df, timestamp_col=timestamp_col)
    if sort:
        df_ist = df_ist.sort("datetime_ist")

    agg_exprs = [
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
    ]

    if "open_interest" in df.schema:
        agg_exprs.append(pl.col("open_interest").last().alias("open_interest"))

    df_5m = (
        df_ist.group_by_dynamic("datetime_ist", every="5m")
        .agg(agg_exprs)
    )

    return df_5m


def compute_realized_volatility(
    df: pl.DataFrame,
    period: int = 20,
    close_col: str = "close",
    annualize: bool = True,
    annualization_factor: float = ANNUALIZATION_FACTOR_5MIN,
    output_col: str = "realized_vol_5m",
) -> pl.DataFrame:
    """Compute 20-period Realized Volatility of 5-minute log returns.

    Formula:
      r_t = ln(close_t / close_{t-1})
      RV_raw = rolling_std(r_t, window=period)
      RV_ann = RV_raw * sqrt(annualization_factor)

    For Indian markets, annualization_factor = 75 bars/day * 252 days = 18,900.
    """
    if df.is_empty() or close_col not in df.schema:
        return df

    close = pl.col(close_col).cast(pl.Float64)
    log_return = (close / close.shift(1)).log()

    rolling_std = log_return.rolling_std(window_size=period, min_samples=2)

    if annualize:
        ann_mult = float(np.sqrt(annualization_factor))
        rv_expr = (rolling_std * ann_mult).fill_null(0.0).alias(output_col)
    else:
        rv_expr = rolling_std.fill_null(0.0).alias(output_col)

    return df.with_columns(rv_expr)


def calculate_session_indicators(
    df: pl.DataFrame,
    config: Optional[TechnicalIndicatorConfig] = None,
    timestamp_col: str = "timestamp",
) -> pl.DataFrame:
    """Unified pipeline computing Session VWAP, Bands, RSI, ATR, and 5-min Realized Volatility.

    If input has 1-minute bars, it calculates 5-min Realized Volatility and maps it back
    or resamples cleanly.
    """
    if df.is_empty():
        return df

    cfg = config or TechnicalIndicatorConfig()

    # 1. VWAP + Bands reset at 09:15 IST
    df_out = compute_session_vwap(
        df,
        timestamp_col=timestamp_col,
        rolling_window=cfg.vwap_rolling_window,
    )

    # 2. 14-period RSI
    df_out = compute_rsi(
        df_out,
        period=cfg.rsi_period,
        close_col="close",
        output_col="rsi_14",
    )

    # 3. 14-period ATR
    df_out = compute_atr(
        df_out,
        period=cfg.atr_period,
        high_col="high",
        low_col="low",
        close_col="close",
        output_col="atr_14",
    )

    # 4. 20-period Realized Volatility of 5-min log returns
    # Resample to 5m, compute RV, and join back on datetime_ist using backward asof join
    df_5m = resample_to_5min(df_out, timestamp_col="datetime_ist")
    df_5m_rv = compute_realized_volatility(
        df_5m,
        period=cfg.rv_period,
        annualize=True,
        annualization_factor=cfg.annualization_factor,
        output_col="realized_vol_5m",
    )

    if "realized_vol_5m" in df_5m_rv.schema:
        # Join backward on datetime_ist so every 1-min bar receives the latest 5-min RV
        df_5m_subset = df_5m_rv.select(["datetime_ist", "realized_vol_5m"]).sort("datetime_ist")
        df_out = (
            df_out.sort("datetime_ist")
            .join_asof(
                df_5m_subset,
                on="datetime_ist",
                strategy="backward",
            )
            .fill_null(0.0)
        )

    return df_out

