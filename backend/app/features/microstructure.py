"""Microstructure Metrics Feature Pipeline using Polars and Vectorized Calculations.

Computes:
1. Top-5 Bid-Ask Depth Imbalance:
   (sum(BidQty) - sum(AskQty)) / (sum(BidQty) + sum(AskQty))
2. Volume-Weighted Micro-Price:
   (best_bid * ask_qty_1 + best_ask * bid_qty_1) / (bid_qty_1 + ask_qty_1)
3. Normalized Bid-Ask Spread:
   (best_ask - best_bid) / mid_price
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import polars as pl
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OrderBookLevel(BaseModel):
    """Single level in an order book (price and quantity)."""
    price: float = Field(..., description="Order price at this level")
    quantity: float = Field(..., description="Available quantity at this level")
    orders: Optional[int] = Field(default=None, description="Number of orders at this level")


class MicrostructureMetrics(BaseModel):
    """Pydantic schema containing microstructure indicators computed from order book depth."""
    symbol: Optional[str] = Field(default=None, description="Trading instrument symbol")
    timestamp: Optional[float] = Field(default=None, description="Epoch timestamp of snapshot")
    best_bid: float = Field(..., description="Best bid price (Level 1)")
    best_ask: float = Field(..., description="Best ask price (Level 1)")
    mid_price: float = Field(..., description="Arithmetic mean of best bid and best ask")
    depth_imbalance: float = Field(..., description="Top-5 Bid-Ask Depth Imbalance [-1.0, 1.0]")
    micro_price: float = Field(..., description="Volume-Weighted Micro-Price")
    normalized_spread: float = Field(..., description="Bid-ask spread normalized by mid price")
    total_bid_qty: float = Field(default=0.0, description="Sum of top-5 bid quantities")
    total_ask_qty: float = Field(default=0.0, description="Sum of top-5 ask quantities")
    bid_qty_1: float = Field(default=0.0, description="Best bid level quantity")
    ask_qty_1: float = Field(default=0.0, description="Best ask level quantity")


def compute_depth_imbalance(
    bid_quantities: Sequence[Union[int, float]],
    ask_quantities: Sequence[Union[int, float]],
) -> float:
    """Compute Top-5 Bid-Ask Depth Imbalance.
    
    Formula:
      (sum(BidQty) - sum(AskQty)) / (sum(BidQty) + sum(AskQty))

    Returns:
      float in range [-1.0, 1.0]. Returns 0.0 if total volume is 0.
    """
    total_bids = float(sum(bid_quantities[:5]))
    total_asks = float(sum(ask_quantities[:5]))
    total_volume = total_bids + total_asks

    if total_volume <= 0.0:
        return 0.0

    return (total_bids - total_asks) / total_volume


def compute_micro_price(
    best_bid: float,
    best_ask: float,
    bid_qty_1: float,
    ask_qty_1: float,
) -> float:
    """Compute Volume-Weighted Micro-Price.

    Formula:
      (best_bid * ask_qty_1 + best_ask * bid_qty_1) / (bid_qty_1 + ask_qty_1)

    When bid quantity is high, the micro-price is pulled upwards towards the ask.
    When ask quantity is high, the micro-price is pulled downwards towards the bid.
    Falls back to arithmetic mid_price if level 1 quantities sum to 0.
    """
    sum_l1_qty = bid_qty_1 + ask_qty_1
    if sum_l1_qty <= 0.0:
        if best_bid > 0.0 and best_ask > 0.0:
            return (best_bid + best_ask) / 2.0
        return max(best_bid, best_ask, 0.0)

    return (best_bid * ask_qty_1 + best_ask * bid_qty_1) / sum_l1_qty


def compute_normalized_spread(
    best_bid: float,
    best_ask: float,
    mid_price: Optional[float] = None,
) -> float:
    """Compute Normalized Bid-Ask Spread.

    Formula:
      (best_ask - best_bid) / mid_price

    where mid_price = (best_bid + best_ask) / 2.0.
    Returns 0.0 if mid_price <= 0.
    """
    if mid_price is None:
        mid_price = (best_bid + best_ask) / 2.0

    if mid_price <= 0.0:
        return 0.0

    spread = best_ask - best_bid
    return spread / mid_price


def extract_depth_levels(
    payload: Dict[str, Any],
) -> Tuple[List[OrderBookLevel], List[OrderBookLevel]]:
    """Extract up to 5 bid and ask levels from FYERS or generic depth payloads.
    
    Supports:
    1. FYERS WebSocket flat keys:
       - bid_price1..5, bid_size1..5 (or bid_qty1..5), bid_order1..5
       - ask_price1..5, ask_size1..5 (or ask_qty1..5), ask_order1..5
    2. Nested dictionaries/lists:
       - bids: [{"price": ..., "quantity"|"size"|"qty": ...}, ...]
       - asks: [{"price": ..., "quantity"|"size"|"qty": ...}, ...]
       - bids: [[price, qty], ...], asks: [[price, qty], ...]
    """
    bids: List[OrderBookLevel] = []
    asks: List[OrderBookLevel] = []

    # Check nested bids/asks first
    if "bids" in payload and isinstance(payload["bids"], (list, tuple)):
        for item in payload["bids"][:5]:
            if isinstance(item, dict):
                p = float(item.get("price", 0.0))
                q = float(item.get("quantity") or item.get("qty") or item.get("size") or 0.0)
                o = item.get("orders") or item.get("order")
                bids.append(OrderBookLevel(price=p, quantity=q, orders=int(o) if o is not None else None))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                bids.append(OrderBookLevel(price=float(item[0]), quantity=float(item[1])))

    if "asks" in payload and isinstance(payload["asks"], (list, tuple)):
        for item in payload["asks"][:5]:
            if isinstance(item, dict):
                p = float(item.get("price", 0.0))
                q = float(item.get("quantity") or item.get("qty") or item.get("size") or 0.0)
                o = item.get("orders") or item.get("order")
                asks.append(OrderBookLevel(price=p, quantity=q, orders=int(o) if o is not None else None))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                asks.append(OrderBookLevel(price=float(item[0]), quantity=float(item[1])))

    # If nested levels were found, return them
    if bids or asks:
        return bids, asks

    # Parse FYERS flat numbered keys (1 to 5)
    for i in range(1, 6):
        bp = payload.get(f"bid_price{i}") or payload.get(f"bidPrice{i}") or payload.get(f"bp{i}")
        bq = (
            payload.get(f"bid_size{i}")
            or payload.get(f"bidSize{i}")
            or payload.get(f"bid_qty{i}")
            or payload.get(f"bidQty{i}")
            or payload.get(f"bq{i}")
        )
        bo = payload.get(f"bid_order{i}") or payload.get(f"bidOrder{i}")
        if bp is not None or bq is not None:
            bids.append(
                OrderBookLevel(
                    price=float(bp or 0.0),
                    quantity=float(bq or 0.0),
                    orders=int(bo) if bo is not None else None,
                )
            )

        ap = payload.get(f"ask_price{i}") or payload.get(f"askPrice{i}") or payload.get(f"ap{i}")
        aq = (
            payload.get(f"ask_size{i}")
            or payload.get(f"askSize{i}")
            or payload.get(f"ask_qty{i}")
            or payload.get(f"askQty{i}")
            or payload.get(f"aq{i}")
        )
        ao = payload.get(f"ask_order{i}") or payload.get(f"askOrder{i}")
        if ap is not None or aq is not None:
            asks.append(
                OrderBookLevel(
                    price=float(ap or 0.0),
                    quantity=float(aq or 0.0),
                    orders=int(ao) if ao is not None else None,
                )
            )

    return bids, asks


def calculate_microstructure_metrics(payload: Dict[str, Any]) -> MicrostructureMetrics:
    """Calculate all microstructure metrics from a depth payload dictionary.

    Args:
        payload: FYERS or standard depth dictionary.

    Returns:
        Typed MicrostructureMetrics instance.
    """
    symbol = payload.get("symbol")
    timestamp = payload.get("timestamp") or payload.get("exch_feed_time") or payload.get("last_traded_time")
    if timestamp is not None:
        try:
            timestamp = float(timestamp)
        except (ValueError, TypeError):
            timestamp = None

    bids, asks = extract_depth_levels(payload)

    best_bid = bids[0].price if bids else float(payload.get("bid_price", 0.0))
    best_ask = asks[0].price if asks else float(payload.get("ask_price", 0.0))

    bid_qty_1 = bids[0].quantity if bids else float(payload.get("bid_size", 0.0))
    ask_qty_1 = asks[0].quantity if asks else float(payload.get("ask_size", 0.0))

    bid_quantities = [b.quantity for b in bids]
    ask_quantities = [a.quantity for a in asks]

    total_bid_qty = float(sum(bid_quantities))
    total_ask_qty = float(sum(ask_quantities))

    # Mid price
    mid_price = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else max(best_bid, best_ask, 0.0)

    # Metrics
    depth_imbalance = compute_depth_imbalance(bid_quantities, ask_quantities)
    micro_price = compute_micro_price(best_bid, best_ask, bid_qty_1, ask_qty_1)
    normalized_spread = compute_normalized_spread(best_bid, best_ask, mid_price)

    return MicrostructureMetrics(
        symbol=symbol,
        timestamp=timestamp,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid_price,
        depth_imbalance=depth_imbalance,
        micro_price=micro_price,
        normalized_spread=normalized_spread,
        total_bid_qty=total_bid_qty,
        total_ask_qty=total_ask_qty,
        bid_qty_1=bid_qty_1,
        ask_qty_1=ask_qty_1,
    )


def calculate_microstructure_polars(df: pl.DataFrame) -> pl.DataFrame:
    """Compute vectorized microstructure features on a Polars DataFrame.

    Requires Level-1 and Level-2..5 depth columns if available.
    Supports either FYERS style (bid_size1..5, ask_size1..5, bid_price1, ask_price1)
    or (bid_qty1..5, ask_qty1..5, etc.).
    """
    cols = df.columns

    # Resolve bid qty columns
    bid_qty_cols = []
    ask_qty_cols = []
    for i in range(1, 6):
        if f"bid_size{i}" in cols:
            bid_qty_cols.append(pl.col(f"bid_size{i}"))
        elif f"bid_qty{i}" in cols:
            bid_qty_cols.append(pl.col(f"bid_qty{i}"))

        if f"ask_size{i}" in cols:
            ask_qty_cols.append(pl.col(f"ask_size{i}"))
        elif f"ask_qty{i}" in cols:
            ask_qty_cols.append(pl.col(f"ask_qty{i}"))

    # Resolve best prices
    best_bid_expr = (
        pl.col("bid_price1") if "bid_price1" in cols
        else pl.col("bid_price") if "bid_price" in cols
        else pl.lit(0.0)
    )
    best_ask_expr = (
        pl.col("ask_price1") if "ask_price1" in cols
        else pl.col("ask_price") if "ask_price" in cols
        else pl.lit(0.0)
    )

    # Level 1 quantities
    bid_qty_1_expr = (
        pl.col("bid_size1") if "bid_size1" in cols
        else pl.col("bid_qty1") if "bid_qty1" in cols
        else pl.col("bid_size") if "bid_size" in cols
        else pl.lit(0.0)
    )
    ask_qty_1_expr = (
        pl.col("ask_size1") if "ask_size1" in cols
        else pl.col("ask_qty1") if "ask_qty1" in cols
        else pl.col("ask_size") if "ask_size" in cols
        else pl.lit(0.0)
    )

    total_bid_expr = pl.sum_horizontal(bid_qty_cols) if bid_qty_cols else bid_qty_1_expr
    total_ask_expr = pl.sum_horizontal(ask_qty_cols) if ask_qty_cols else ask_qty_1_expr
    total_vol_expr = total_bid_expr + total_ask_expr

    # Depth Imbalance
    imbalance_expr = (
        pl.when(total_vol_expr > 0)
        .then((total_bid_expr - total_ask_expr) / total_vol_expr)
        .otherwise(0.0)
        .alias("depth_imbalance")
    )

    # Mid price
    mid_price_expr = ((best_bid_expr + best_ask_expr) / 2.0).alias("mid_price")

    # Volume-weighted Micro-price
    sum_l1_expr = bid_qty_1_expr + ask_qty_1_expr
    micro_price_expr = (
        pl.when(sum_l1_expr > 0)
        .then((best_bid_expr * ask_qty_1_expr + best_ask_expr * bid_qty_1_expr) / sum_l1_expr)
        .otherwise((best_bid_expr + best_ask_expr) / 2.0)
        .alias("micro_price")
    )

    # Normalized spread
    spread_expr = (
        pl.when((best_bid_expr + best_ask_expr) > 0)
        .then((best_ask_expr - best_bid_expr) / ((best_bid_expr + best_ask_expr) / 2.0))
        .otherwise(0.0)
        .alias("normalized_spread")
    )

    return df.with_columns([
        total_bid_expr.alias("total_bid_qty"),
        total_ask_expr.alias("total_ask_qty"),
        mid_price_expr,
        imbalance_expr,
        micro_price_expr,
        spread_expr,
    ])
