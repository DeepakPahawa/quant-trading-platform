"""Market Overview Data Fetcher and TFT Feature Pipeline.

Fetches and unifies:
1. Profile Details (get_profile)
2. Transaction Info (funds, holdings, positions, orderbook, tradebook)
3. Real-time Quotes & 5-Depth Order Book (quotes, quotes_v2)
4. Market Status & Trading Date (market_status)
5. Historical 1-minute OHLCV Bars (history)
6. TFT Tensor Preparation: Extracts the required 60-bar multivariate feature vector:
   [log_return, volume_ratio, rsi_14, vwap_diff, depth_imbalance, micro_price_spread]
"""

from datetime import datetime, timedelta, timezone
import logging
import math
from typing import Any, Dict, List, Optional, Union

from fyers_apiv3 import fyersModel
import polars as pl
from pydantic import BaseModel, Field

from backend.app.features.microstructure import calculate_microstructure_metrics

logger = logging.getLogger(__name__)


class ProfileSummary(BaseModel):
    fy_id: str = Field(..., description="FYERS Client User ID")
    name: str = Field(..., description="Client Account Name")
    email: Optional[str] = Field(default=None, description="Email ID")
    mobile: Optional[str] = Field(default=None, description="Mobile number")
    pan: Optional[str] = Field(default=None, description="PAN number")
    totp_enabled: bool = Field(default=False, description="Whether TOTP 2FA is active")


class TransactionSummary(BaseModel):
    available_margin: float = Field(default=0.0, description="Available trading margin in INR")
    used_margin: float = Field(default=0.0, description="Margin utilized by current positions")
    total_balance: float = Field(default=0.0, description="Total balance in account")
    holdings_count: int = Field(default=0, description="Number of demat equity holdings")
    positions_count: int = Field(default=0, description="Number of open intraday/F&O positions")
    orders_count: int = Field(default=0, description="Total orders placed today")
    trades_count: int = Field(default=0, description="Total trades executed today")
    holdings: List[Dict[str, Any]] = Field(default_factory=list)
    positions: List[Dict[str, Any]] = Field(default_factory=list)
    orders: List[Dict[str, Any]] = Field(default_factory=list)
    trades: List[Dict[str, Any]] = Field(default_factory=list)


class MarketStatusSummary(BaseModel):
    market_date: str = Field(..., description="Current trade date (YYYY-MM-DD)")
    market_status: str = Field(..., description="Market state (e.g. OPEN, CLOSED, HOLIDAY)")
    exchanges: Dict[str, str] = Field(default_factory=dict, description="Status per exchange segment (NSE, BSE, MCX)")


class StockQuoteSummary(BaseModel):
    symbol: str
    ltp: float = Field(..., description="Last Traded Price")
    open: float
    high: float
    low: float
    prev_close: float
    change: float
    change_pct: float
    volume: int
    depth_imbalance: Optional[float] = None
    micro_price: Optional[float] = None
    normalized_spread: Optional[float] = None


class MarketDataOverview(BaseModel):
    profile: Optional[ProfileSummary] = None
    transactions: Optional[TransactionSummary] = None
    market_status: Optional[MarketStatusSummary] = None
    quotes: List[StockQuoteSummary] = Field(default_factory=list)
    historical_bars_count: int = Field(default=0)


def fetch_fyers_profile(client: fyersModel.FyersModel) -> Dict[str, Any]:
    """Fetch user profile details from FYERS API."""
    res = client.get_profile()
    if isinstance(res, dict) and (res.get("s") == "ok" or res.get("code") == 200):
        data = res.get("data") or res
        return {
            "s": "ok",
            "data": {
                "fy_id": data.get("fy_id", ""),
                "name": data.get("name", ""),
                "email": data.get("email_id", ""),
                "mobile": data.get("mobile_number", ""),
                "pan": data.get("pan", ""),
                "totp_enabled": bool(data.get("totp", False)),
            },
            "raw": res,
        }
    return {"s": "error", "message": str(res)}


def fetch_fyers_transactions(client: fyersModel.FyersModel) -> Dict[str, Any]:
    """Fetch unified account transaction data: funds, holdings, positions, orders, trades."""
    funds_res = client.funds()
    holdings_res = client.holdings()
    positions_res = client.positions()
    orderbook_res = client.orderbook()
    tradebook_res = client.tradebook()

    # Extract funds margin
    funds_data = funds_res.get("fund_limit", []) if isinstance(funds_res, dict) else []
    avail_margin = 0.0
    used_margin = 0.0
    total_balance = 0.0

    if isinstance(funds_data, list):
        for item in funds_data:
            title = item.get("title", "")
            val = float(item.get("equityAmount", 0.0))
            if "Available Margin" in title or "Available" in title:
                avail_margin = val
            elif "Utilized" in title or "Used" in title:
                used_margin = val
            elif "Total Balance" in title:
                total_balance = val

    holdings_list = (holdings_res.get("holdings") or []) if isinstance(holdings_res, dict) else []
    positions_list = (positions_res.get("netPositions") or []) if isinstance(positions_res, dict) else []
    orders_list = (orderbook_res.get("orderBook") or []) if isinstance(orderbook_res, dict) else []
    trades_list = (tradebook_res.get("tradeBook") or []) if isinstance(tradebook_res, dict) else []

    return {
        "s": "ok",
        "data": {
            "available_margin": avail_margin,
            "used_margin": used_margin,
            "total_balance": total_balance,
            "holdings_count": len(holdings_list),
            "positions_count": len(positions_list),
            "orders_count": len(orders_list),
            "trades_count": len(trades_list),
            "holdings": holdings_list,
            "positions": positions_list,
            "orders": orders_list,
            "trades": trades_list,
        },
    }


def fetch_fyers_market_status(client: fyersModel.FyersModel) -> Dict[str, Any]:
    """Fetch current market operational state and exchange trade date."""
    res = client.market_status()
    now_ist = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    market_date = now_ist.strftime("%Y-%m-%d")

    exchanges: Dict[str, str] = {}
    overall_status = "CLOSED"

    if isinstance(res, dict) and "marketStatus" in res:
        status_list = res.get("marketStatus", [])
        for item in status_list:
            exch = item.get("exchange", "")
            seg = item.get("segment", "")
            st = item.get("status", "")
            exchanges[f"{exch}_{seg}"] = st
            if st == "OPEN":
                overall_status = "OPEN"
    elif isinstance(res, dict) and "s" in res:
        overall_status = res.get("s", "UNKNOWN")

    return {
        "s": "ok",
        "data": {
            "market_date": market_date,
            "market_status": overall_status,
            "exchanges": exchanges,
        },
        "raw": res,
    }


def fetch_fyers_quotes(
    client: fyersModel.FyersModel,
    symbols: List[str],
) -> List[Dict[str, Any]]:
    """Fetch real-time stock quotes, Level-2 depth, and compute microstructure metrics."""
    if not symbols:
        return []

    symbols_param = ",".join(symbols)
    res = client.quotes({"symbols": symbols_param})

    quotes_out: List[Dict[str, Any]] = []
    if isinstance(res, dict) and "d" in res:
        for item in res.get("d", []):
            v = item.get("v", {})
            sym = v.get("symbol") or item.get("n", "")

            # Extract depth and calculate microstructure metrics if cmd exists
            cmd = v.get("cmd", {})
            bids = cmd.get("bids", [])
            asks = cmd.get("asks", [])

            depth_payload = {
                "symbol": sym,
                "bids": bids,
                "asks": asks,
                "bid_price": v.get("bid", 0.0),
                "ask_price": v.get("ask", 0.0),
            }
            metrics = calculate_microstructure_metrics(depth_payload)

            quotes_out.append({
                "symbol": sym,
                "ltp": float(v.get("lp", 0.0)),
                "open": float(v.get("open_price", 0.0)),
                "high": float(v.get("high_price", 0.0)),
                "low": float(v.get("low_price", 0.0)),
                "prev_close": float(v.get("prev_close_price", 0.0)),
                "change": float(v.get("ch", 0.0)),
                "change_pct": float(v.get("chp", 0.0)),
                "volume": int(v.get("volume", 0)),
                "depth_imbalance": metrics.depth_imbalance,
                "micro_price": metrics.micro_price,
                "normalized_spread": metrics.normalized_spread,
                "raw_depth": cmd,
            })
    return quotes_out


def fetch_fyers_historical_bars(
    client: fyersModel.FyersModel,
    symbol: str,
    resolution: str = "1",
    days: int = 5,
) -> pl.DataFrame:
    """Fetch 1-minute historical candles from FYERS into a typed Polars DataFrame."""
    now = datetime.now()
    start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    payload = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start_date,
        "range_to": end_date,
        "cont_flag": "1",
    }
    res = client.history(payload)

    if isinstance(res, dict) and "candles" in res and res["candles"]:
        data = res["candles"]
        # Format: [timestamp, open, high, low, close, volume]
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        rows = [row[:6] for row in data]
        df = pl.DataFrame(rows, schema=cols, orient="row")
        return df.sort("timestamp")
    return pl.DataFrame(schema=["timestamp", "open", "high", "low", "close", "volume"])


def prepare_tft_feature_tensor(
    bars_df: pl.DataFrame,
    depth_imbalance: float = 0.0,
    micro_price_spread: float = 0.0,
    sequence_length: int = 60,
) -> pl.DataFrame:
    """Prepare Multivariate Feature Vector for TFT Model according to the TFT Skill Protocol.
    
    TFT Protocol Specification:
    - Input shape: [batch_size, 60_bars, num_features]
    - Feature Vector: [log_return, volume_ratio, rsi_14, vwap_diff, depth_imbalance, micro_price_spread]
    - Time Covariates: Normalized minute-of-day (0 to 375) and day-of-week.
    """
    if len(bars_df) < sequence_length:
        logger.warning("Fewer than %d bars available (%d). Cannot feed complete sequence.", sequence_length, len(bars_df))

    # 1. Log return: ln(close / close_{t-1})
    df = bars_df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().fill_null(0.0).alias("log_return")
    )

    # 2. Volume Ratio: volume / rolling_mean(volume, 20)
    df = df.with_columns(
        (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20).fill_null(1.0))
        .fill_nan(1.0)
        .alias("volume_ratio")
    )

    # 3. RSI 14
    change = pl.col("close") - pl.col("close").shift(1)
    gain = pl.when(change > 0).then(change).otherwise(0.0)
    loss = pl.when(change < 0).then(-change).otherwise(0.0)

    avg_gain = gain.rolling_mean(window_size=14).fill_null(0.0)
    avg_loss = loss.rolling_mean(window_size=14).fill_null(0.0)
    rs = pl.when(avg_loss == 0).then(100.0).otherwise(avg_gain / (avg_loss + 1e-9))
    rsi_14 = (100.0 - (100.0 / (1.0 + rs))).fill_null(50.0).alias("rsi_14")
    df = df.with_columns(rsi_14)

    # 4. VWAP diff: (close - cumulative_vwap) / cumulative_vwap
    pv = pl.col("close") * pl.col("volume")
    cum_pv = pv.cum_sum()
    cum_vol = pl.col("volume").cum_sum()
    vwap = pl.when(cum_vol > 0).then(cum_pv / cum_vol).otherwise(pl.col("close"))
    vwap_diff = ((pl.col("close") - vwap) / vwap).fill_null(0.0).alias("vwap_diff")
    df = df.with_columns(vwap_diff)

    # 5. Microstructure covariates: depth_imbalance & micro_price_spread
    df = df.with_columns([
        pl.lit(depth_imbalance).alias("depth_imbalance"),
        pl.lit(micro_price_spread).alias("micro_price_spread"),
    ])

    # 6. Time Covariates: Minute of day (0 to 375 for 09:15 to 15:30 IST) & Day of week
    df = df.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("dt")
    ).with_columns([
        # Normalized minute of trading session (0 to 375)
        ((pl.col("dt").dt.hour().cast(pl.Int32) - 9) * 60 + (pl.col("dt").dt.minute().cast(pl.Int32) - 15))
        .clip(0, 375)
        .alias("minute_of_day"),
        pl.col("dt").dt.weekday().cast(pl.Int32).alias("day_of_week"),
    ])

    # Select the last sequence_length (60) bars
    features = df.tail(sequence_length)
    return features

