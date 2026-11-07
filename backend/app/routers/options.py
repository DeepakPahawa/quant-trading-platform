"""Options Chain and Volatility Surface API Router."""

import datetime
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.auth.fyers_auth import get_valid_fyers_client
from backend.app.models.volatility import (
    OptionQuote,
    SVICurvePoint,
    SVIParameters,
    VolatilitySurfaceResult,
    calibrate_option_surface,
    generate_synthetic_option_chain,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/options", tags=["Options & Volatility"])


def normalize_index_symbol(symbol: str) -> str:
    """Normalize common short tickers to standard FYERS exchange symbols."""
    sym = symbol.strip().upper()
    mapping = {
        "NIFTY": "NSE:NIFTY50-INDEX",
        "NIFTY50": "NSE:NIFTY50-INDEX",
        "BANKNIFTY": "NSE:BANKNIFTY-INDEX",
        "FINNIFTY": "NSE:FINNIFTY-INDEX",
        "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
        "SENSEX": "BSE:SENSEX-INDEX",
    }
    return mapping.get(sym, sym)


def get_default_underlying_price(symbol: str) -> float:
    """Provide realistic reference spot prices for Indian benchmark underlyings."""
    sym = normalize_index_symbol(symbol)
    if "BANKNIFTY" in sym:
        return 51800.0
    elif "FINNIFTY" in sym:
        return 23500.0
    elif "MIDCPNIFTY" in sym:
        return 12500.0
    elif "NIFTY50" in sym or "NIFTY-INDEX" in sym or "NIFTY" in sym:
        return 24200.0
    elif "SENSEX" in sym:
        return 79500.0
    elif "SBIN" in sym:
        return 820.0
    elif "RELIANCE" in sym:
        return 2750.0
    elif "HDFCBANK" in sym:
        return 1750.0
    elif "ICICIBANK" in sym:
        return 1260.0
    return 2000.0


class OptionsChainResponse(BaseModel):
    """Unified response schema for options chain and calibrated SVI surface."""
    status: str = Field(default="ok", description="Status code ('ok', 'error')")
    symbol: str = Field(..., description="Underlying asset symbol")
    data_source: str = Field(default="broker", description="Source of quotes ('broker' or 'synthetic')")
    spot: float = Field(..., description="Underlying spot price (S)")
    forward: float = Field(..., description="Forward price (F)")
    time_to_expiry_days: float = Field(..., description="Calendar days to expiration")
    time_to_expiry_years: float = Field(..., description="Year fraction T = days / 365.0")
    risk_free_rate: float = Field(..., description="Annualized risk-free rate")
    svi_parameters: SVIParameters = Field(..., description="Calibrated SVI smile parameters")
    curve_points: List[SVICurvePoint] = Field(default_factory=list, description="Fitted SVI smile curve points")
    quotes: List[OptionQuote] = Field(default_factory=list, description="Option chain contracts with Greeks & IV")


@router.get("/chain/{symbol:path}", response_model=OptionsChainResponse)
def get_options_chain(
    symbol: str,
    strikecount: int = Query(default=10, ge=1, le=50, description="Number of strikes above and below ATM"),
    expiry: Optional[str] = Query(default=None, description="Target expiry date YYYY-MM-DD or epoch timestamp"),
    risk_free_rate: float = Query(default=0.07, ge=0.0, le=0.5, description="Annualized risk-free rate (e.g. 0.07)"),
    discard_illiquid: bool = Query(default=True, description="Discard quotes where spread > 15% mid-price"),
    force_synthetic: bool = Query(default=False, description="Force synthetic generator (useful for offline testing)"),
) -> OptionsChainResponse:
    """Retrieve full options chain with Black-76 Greeks and calibrated SVI Volatility Surface.

    1. Inverts market mid-prices to Black-76 Implied Volatility (IV).
    2. Calculates QuantLib Delta, Gamma, daily Theta, and Vega.
    3. Fits Gatheral's SVI curve across log-moneyness k = ln(K / F).
    4. Filters out illiquid strikes with bid-ask spread > 15% of mid-price.
    """
    clean_sym = normalize_index_symbol(symbol)
    quotes_raw: List[Dict[str, Any]] = []
    spot_price = 0.0
    forward_price = 0.0
    tte_days = 14.0
    data_source = "synthetic"

    # Attempt to query live broker if not explicitly requesting synthetic data
    if not force_synthetic:
        try:
            client = get_valid_fyers_client()
            req_data: Dict[str, Any] = {
                "symbol": clean_sym,
                "strikecount": strikecount,
            }
            if expiry:
                try:
                    req_data["timestamp"] = int(expiry)
                except ValueError:
                    pass

            broker_res = client.optionchain(data=req_data)
            if (
                isinstance(broker_res, dict)
                and broker_res.get("s") == "ok"
                and "data" in broker_res
                and isinstance(broker_res["data"], dict)
            ):
                b_data = broker_res["data"]
                spot_price = float(b_data.get("spot_price") or b_data.get("underlyingValue") or 0.0)
                raw_chain = b_data.get("optionsChain") or []

                for item in raw_chain:
                    quotes_raw.append({
                        "strike": float(item.get("strike_price") or item.get("strikePrice") or item.get("strike") or 0.0),
                        "option_type": str(item.get("option_type") or item.get("optionType") or "CE").upper(),
                        "bid": float(item.get("bid") or item.get("bid_price") or item.get("bPrice") or 0.0),
                        "ask": float(item.get("ask") or item.get("ask_price") or item.get("aPrice") or 0.0),
                        "ltp": float(item.get("ltp") or item.get("last_price") or 0.0),
                        "oi": int(item.get("oi") or item.get("open_interest") or 0),
                        "volume": int(item.get("volume") or 0),
                    })

                if quotes_raw:
                    data_source = "broker"
                    # Approximate forward if not provided
                    tte_years = tte_days / 365.0
                    forward_price = spot_price * (1.0 + risk_free_rate * tte_years)

        except Exception as err:
            logger.info("Broker option chain query failed for %s (%s); utilizing synthetic generator", clean_sym, err)

    # Fallback to high-fidelity synthetic options chain if live broker is offline or empty
    if not quotes_raw:
        data_source = "synthetic"
        spot_price = get_default_underlying_price(clean_sym)
        strike_interval = 100.0 if "BANK" in clean_sym or "SENSEX" in clean_sym else 50.0
        synthetic_data = generate_synthetic_option_chain(
            symbol=clean_sym,
            spot=spot_price,
            time_to_expiry_days=tte_days,
            strikecount=strikecount,
            strike_interval=strike_interval,
            risk_free_rate=risk_free_rate,
        )
        quotes_raw = synthetic_data["quotes"]
        forward_price = synthetic_data["forward"]

    # Calibrate SVI surface and compute Greeks
    surface_result = calibrate_option_surface(
        symbol=clean_sym,
        spot=spot_price,
        forward=forward_price,
        time_to_expiry_days=tte_days,
        quotes_raw=quotes_raw,
        risk_free_rate=risk_free_rate,
        discard_illiquid=discard_illiquid,
    )

    return OptionsChainResponse(
        status="ok",
        symbol=clean_sym,
        data_source=data_source,
        spot=surface_result.spot,
        forward=surface_result.forward,
        time_to_expiry_days=surface_result.time_to_expiry_days,
        time_to_expiry_years=surface_result.time_to_expiry_years,
        risk_free_rate=surface_result.risk_free_rate,
        svi_parameters=surface_result.svi_parameters,
        curve_points=surface_result.curve_points,
        quotes=surface_result.quotes,
    )
