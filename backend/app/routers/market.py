"""Market data REST endpoints for historical bars and symbol discovery."""

import logging
from typing import Any, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.data.historical import HistoricalBarFetcher, normalize_resolution
from backend.app.data.storage import DuckDBStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Market Data"])

# Module-level storage singleton
_storage_instance: Optional[DuckDBStorage] = None


def get_storage() -> DuckDBStorage:
    """Dependency provider for DuckDBStorage instance."""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = DuckDBStorage()
    return _storage_instance


class BarRecord(BaseModel):
    """Single OHLCV + OI bar."""
    timestamp: int = Field(..., description="Epoch seconds timestamp")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Volume traded")
    open_interest: int = Field(default=0, description="Open interest")


class HistoryResponse(BaseModel):
    """Schema for historical candle data response."""
    status: str = Field(default="ok", description="Status code ('ok', 'error')")
    symbol: str = Field(..., description="Product symbol")
    resolution: str = Field(..., description="Normalized resolution ('1', 'D', etc.)")
    count: int = Field(..., description="Number of candles returned")
    candles: List[List[Union[int, float]]] = Field(
        default_factory=list,
        description="List of [timestamp, open, high, low, close, volume, open_interest]"
    )


class SymbolsResponse(BaseModel):
    """Schema for stored symbols listing."""
    status: str = Field(default="ok")
    symbols: List[str] = Field(default_factory=list)
    count: int = Field(...)


@router.get("/history/{symbol:path}", response_model=HistoryResponse)
def get_historical_candles(
    symbol: str,
    resolution: str = Query(default="1", description="Candle resolution: '1', '5', '15', 'D', etc."),
    range_from: Optional[str] = Query(default=None, description="Start date 'YYYY-MM-DD' or epoch timestamp"),
    range_to: Optional[str] = Query(default=None, description="End date 'YYYY-MM-DD' or epoch timestamp"),
    sync: bool = Query(default=False, description="Whether to fetch and sync from FYERS if missing locally"),
    storage: DuckDBStorage = Depends(get_storage),
) -> HistoryResponse:
    """
    Retrieve historical OHLCV + OI bar data for a given product symbol.
    
    Checks DuckDB local parquet storage first. If empty and sync is requested,
    queries the FYERS API v3 and stores results into the local archive.
    """
    norm_res = normalize_resolution(resolution)

    # 1. Query local DuckDB storage
    df = storage.query_bars(
        symbol=symbol,
        range_from=range_from,
        range_to=range_to,
        resolution=norm_res,
    )

    # 2. If storage has no data and sync is enabled, attempt live broker ingestion
    if df.is_empty() and sync:
        try:
            fetcher = HistoricalBarFetcher()
            fetched_df = fetcher.fetch_bars(
                symbol=symbol,
                resolution=norm_res,
                range_from=range_from,
                range_to=range_to,
            )
            if not fetched_df.is_empty():
                storage.store_bars(fetched_df, symbol=symbol)
                df = storage.query_bars(
                    symbol=symbol,
                    range_from=range_from,
                    range_to=range_to,
                    resolution=norm_res,
                )
        except Exception as e:
            logger.warning("Live sync failed for %s: %s", symbol, e)

    candles: List[List[Union[int, float]]] = [list(r) for r in df.iter_rows()]

    return HistoryResponse(
        status="ok",
        symbol=symbol,
        resolution=norm_res,
        count=len(candles),
        candles=candles,
    )


@router.get("/symbols", response_model=SymbolsResponse)
def list_available_symbols(
    storage: DuckDBStorage = Depends(get_storage),
) -> SymbolsResponse:
    """List all symbols currently cached in local DuckDB Parquet storage."""
    symbols = storage.get_available_symbols()
    return SymbolsResponse(status="ok", symbols=symbols, count=len(symbols))

