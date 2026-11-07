"""Market data REST endpoints for historical bars and symbol discovery."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.auth.fyers_auth import FyersAuthManager, get_valid_fyers_client
from backend.app.data.historical import HistoricalBarFetcher, normalize_resolution
from backend.app.data.market_overview import (
    fetch_fyers_historical_bars,
    fetch_fyers_market_status,
    fetch_fyers_profile,
    fetch_fyers_quotes,
    fetch_fyers_transactions,
    prepare_tft_feature_tensor,
)
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


class LoginUrlResponse(BaseModel):
    login_url: str = Field(..., description="Official FYERS OAuth authorization URL")


class AuthExchangeRequest(BaseModel):
    auth_code_or_url: Optional[str] = Field(
        default=None,
        description="Authorization code or redirected URL from FYERS login. If omitted, read from FYERS_AUTH_CODE in .env"
    )


class AuthExchangeResponse(BaseModel):
    status: str = Field(default="ok")
    authenticated: bool = Field(...)
    profile: Optional[Dict[str, Any]] = None
    message: str = Field(default="")


@router.get("/auth/login-url", response_model=LoginUrlResponse)
def get_login_url() -> LoginUrlResponse:
    """Generate the official FYERS API v3 OAuth login URL."""
    auth_manager = FyersAuthManager()
    url = auth_manager.get_login_url()
    return LoginUrlResponse(login_url=url)


@router.post("/auth/exchange", response_model=AuthExchangeResponse)
def exchange_token(request: Optional[AuthExchangeRequest] = None) -> AuthExchangeResponse:
    """Exchange browser redirect auth_code or URL for an access token.
    
    If no code is provided in the body, it falls back to FYERS_AUTH_CODE from .env.
    """
    # 1. Determine auth_code_or_url
    code_input: Optional[str] = None
    if request and request.auth_code_or_url and request.auth_code_or_url.strip():
        code_input = request.auth_code_or_url.strip()
    else:
        # Reload .env to ensure freshly edited tokens are picked up
        for env_path in [Path("backend/.env"), Path(".env"), Path(__file__).resolve().parent.parent.parent / ".env"]:
            if env_path.exists():
                load_dotenv(env_path, override=True)
                break
        code_input = (
            os.getenv("FYERS_AUTH_CODE")
            or os.getenv("AUTH_CODE")
            or os.getenv("FYERS_AUTHCODE")
            or os.getenv("AUTHCODE")
        )
        if code_input:
            code_input = code_input.strip()

    if not code_input:
        return AuthExchangeResponse(
            status="error",
            authenticated=False,
            profile=None,
            message="No auth_code provided in request body and FYERS_AUTH_CODE not found in .env",
        )

    auth_manager = FyersAuthManager()
    try:
        auth_manager.exchange_auth_code(code_input)
        client = auth_manager.get_client()
        profile_res = client.get_profile()
        profile_data = profile_res.get("data") if isinstance(profile_res, dict) else profile_res
        return AuthExchangeResponse(
            status="ok",
            authenticated=True,
            profile=profile_data,
            message="Successfully authenticated and acquired FYERS access token.",
        )
    except Exception as e:
        logger.error("Failed to exchange auth code: %s", e)
        return AuthExchangeResponse(
            status="error",
            authenticated=False,
            profile=None,
            message=str(e),
        )


@router.get("/market/quote/{symbol:path}")
def get_live_quote(symbol: str) -> Dict[str, Any]:
    """Fetch real-time Level-2 quote and compute microstructure metrics straight from FYERS API."""
    try:
        client = get_valid_fyers_client()
        quotes = fetch_fyers_quotes(client, [symbol])
        if not quotes:
            raise HTTPException(status_code=404, detail=f"No quote data returned for symbol {symbol}")
        return {"status": "ok", "symbol": symbol, "data": quotes[0]}
    except Exception as e:
        logger.error("Failed fetching live quote for %s: %s", symbol, e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


@router.get("/market/tft/{symbol:path}")
def get_tft_feature_sequence(
    symbol: str,
    days: int = Query(default=2, ge=1, le=10, description="Days of 1-minute historical candles to fetch"),
) -> Dict[str, Any]:
    """Fetch live 1-minute historical bars and generate 60-bar multivariate feature tensor for TFT model."""
    try:
        client = get_valid_fyers_client()
        quotes = fetch_fyers_quotes(client, [symbol])
        depth_imbalance = quotes[0].get("depth_imbalance", 0.0) if quotes else 0.0
        micro_price = quotes[0].get("micro_price", 0.0) if quotes else 0.0
        ltp = quotes[0].get("ltp", 0.0) if quotes else 0.0
        micro_spread = micro_price - ltp

        df_bars = fetch_fyers_historical_bars(client, symbol=symbol, resolution="1", days=days)
        if df_bars.is_empty():
            raise HTTPException(status_code=404, detail=f"No historical candles available for symbol {symbol}")

        tft_df = prepare_tft_feature_tensor(
            df_bars,
            depth_imbalance=depth_imbalance,
            micro_price_spread=micro_spread,
            sequence_length=60,
        )

        return {
            "status": "ok",
            "symbol": symbol,
            "bars_retrieved": len(df_bars),
            "sequence_length": len(tft_df),
            "features": tft_df.to_dicts(),
        }
    except Exception as e:
        logger.error("Failed preparing TFT tensor for %s: %s", symbol, e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


@router.get("/market/quotes")
def get_multiple_quotes(
    symbols: str = Query(
        default="NSE:SBIN-EQ,NSE:RELIANCE-EQ",
        description="Comma-separated list of symbols (e.g. 'NSE:SBIN-EQ,NSE:ICICIBANK-EQ,NSE:TCS-EQ')"
    ),
) -> Dict[str, Any]:
    """Fetch real-time Level-2 quotes and microstructure metrics for a list of stocks."""
    try:
        client = get_valid_fyers_client()
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
        if not sym_list:
            raise HTTPException(status_code=400, detail="No symbols provided")
        quotes = fetch_fyers_quotes(client, sym_list)
        return {"status": "ok", "count": len(quotes), "data": quotes}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed fetching multiple quotes: %s", e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


@router.get("/market/overview")
def get_market_overview(
    symbols: str = Query(default="NSE:SBIN-EQ,NSE:RELIANCE-EQ,NSE:NIFTY50-INDEX"),
) -> Dict[str, Any]:
    """Unified endpoint returning profile, funds, positions, market status, and quotes."""
    try:
        client = get_valid_fyers_client()
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]

        profile = fetch_fyers_profile(client)
        transactions = fetch_fyers_transactions(client)
        market_status = fetch_fyers_market_status(client)
        quotes = fetch_fyers_quotes(client, sym_list)

        return {
            "status": "ok",
            "profile": profile.get("data"),
            "transactions": transactions.get("data"),
            "market_status": market_status.get("data"),
            "quotes": quotes,
        }
    except Exception as e:
        logger.error("Failed fetching market overview: %s", e)
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(e))


