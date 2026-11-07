"""Options Volatility Modeling and SVI Volatility Surface Calibration.

Implements:
1. QuantLib Black-76 / European Option Implied Volatility (IV) inversion.
2. Greeks Calculation: Delta (Δ), Gamma (Γ), Theta (Θ), Vega (ν) via QuantLib.
3. SVI (Stochastic Volatility Inspired) curve calibration across log-moneyness:
   k = ln(K / F) for near-month expiry contracts.
4. Liquidity filtering: Discard option quotes where (ask - bid) / mid_price > 15%.
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import least_squares
import QuantLib as ql

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Schemas
# ---------------------------------------------------------------------------

class GreeksResult(BaseModel):
    """Greeks and implied volatility calculated from Black-76 / European engine."""
    iv: float = Field(..., ge=0.0, description="Annualized Implied Volatility (e.g., 0.15 for 15%)")
    delta: float = Field(..., description="Option Delta (sensitivity to underlying price)")
    gamma: float = Field(..., ge=0.0, description="Option Gamma (rate of change of Delta)")
    theta: float = Field(..., description="Daily Theta decay (loss per calendar day)")
    theta_annual: float = Field(..., description="Annualized Theta")
    vega: float = Field(..., ge=0.0, description="Vega per 1 percentage point shift in IV")
    vega_total: float = Field(..., ge=0.0, description="Vega per 1.0 (100%) change in IV")


class OptionQuote(BaseModel):
    """Single strike option contract representation."""
    strike: float = Field(..., description="Strike price (K)")
    option_type: str = Field(..., description="'CE' for Call or 'PE' for Put")
    market_price: float = Field(..., description="Last traded price or mid price")
    bid: Optional[float] = Field(default=None, description="Best bid price")
    ask: Optional[float] = Field(default=None, description="Best ask price")
    mid_price: float = Field(..., description="Arithmetic mid price: (bid + ask) / 2")
    spread: float = Field(..., description="Bid-ask spread (ask - bid)")
    spread_pct: float = Field(..., description="Bid-ask spread normalized by mid price")
    is_liquid: bool = Field(..., description="True if spread_pct <= 0.15 (15% threshold)")
    log_moneyness: Optional[float] = Field(default=None, description="Log moneyness k = ln(K / F)")
    iv: Optional[float] = Field(default=None, description="Black-76 Implied Volatility")
    svi_iv: Optional[float] = Field(default=None, description="Calibrated SVI model Implied Volatility")
    delta: Optional[float] = Field(default=None, description="QuantLib Delta")
    gamma: Optional[float] = Field(default=None, description="QuantLib Gamma")
    theta: Optional[float] = Field(default=None, description="QuantLib Daily Theta")
    vega: Optional[float] = Field(default=None, description="QuantLib Vega (per 1% IV)")
    open_interest: Optional[int] = Field(default=0, description="Open interest")
    volume: Optional[int] = Field(default=0, description="Contract volume")


class SVIParameters(BaseModel):
    """Raw SVI model parameters: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))."""
    a: float = Field(..., description="Overall level of variance (vertical shift)")
    b: float = Field(..., ge=0.0, description="Angle between asymptotes (wing slope)")
    rho: float = Field(..., ge=-1.0, le=1.0, description="Orientation / skew (-1 to 1)")
    m: float = Field(..., description="Horizontal translation (ATM vertex offset)")
    sigma: float = Field(..., gt=0.0, description="Smoothness / vertex curvature parameter")
    time_to_expiry: float = Field(..., gt=0.0, description="Time to expiry in years")
    rmse: float = Field(default=0.0, description="Root Mean Squared Error of IV fit")
    is_arbitrage_free: bool = Field(
        default=True,
        description="True if a + b * sigma * sqrt(1 - rho^2) >= 0 (no negative total variance)",
    )


class SVICurvePoint(BaseModel):
    """Point on the fitted SVI volatility smile."""
    strike: float = Field(..., description="Strike price")
    log_moneyness: float = Field(..., description="Log moneyness k = ln(K / F)")
    total_variance: float = Field(..., description="SVI total variance w(k)")
    implied_volatility: float = Field(..., description="SVI annualized implied volatility sigma(k)")


class VolatilitySurfaceResult(BaseModel):
    """Calibration output containing fitted SVI curve and computed Greeks."""
    symbol: str = Field(..., description="Underlying product symbol")
    spot: float = Field(..., description="Underlying spot price (S)")
    forward: float = Field(..., description="Underlying forward price (F)")
    time_to_expiry_days: float = Field(..., description="Time to expiry in calendar days")
    time_to_expiry_years: float = Field(..., description="Time to expiry in years (T)")
    risk_free_rate: float = Field(default=0.07, description="Risk-free discount rate")
    svi_parameters: SVIParameters = Field(..., description="Calibrated SVI model parameters")
    curve_points: List[SVICurvePoint] = Field(default_factory=list, description="Smooth fitted SVI smile curve")
    quotes: List[OptionQuote] = Field(default_factory=list, description="Evaluated option chain contracts")


# ---------------------------------------------------------------------------
# QuantLib Black-76 IV & Greeks Engine
# ---------------------------------------------------------------------------

def normalize_option_type(option_type: str) -> int:
    """Map string option type to QuantLib Option Type."""
    upper = option_type.upper().strip()
    if upper in ("CE", "CALL", "C"):
        return ql.Option.Call
    elif upper in ("PE", "PUT", "P"):
        return ql.Option.Put
    raise ValueError(f"Unknown option type '{option_type}'. Expected 'CE' or 'PE'.")


def solve_implied_volatility_black76(
    option_type: str,
    strike: float,
    forward: float,
    market_price: float,
    time_to_expiry: float,
    risk_free_rate: float = 0.07,
    initial_guess: float = 0.20,
) -> Optional[float]:
    """Invert option market price to Black-76 Implied Volatility (IV) using QuantLib.

    Args:
        option_type: 'CE' (Call) or 'PE' (Put).
        strike: Option strike price K.
        forward: Forward price of underlying F.
        market_price: Observed option market price (e.g. mid-price).
        time_to_expiry: Time to expiry in years T (e.g., days / 365.0).
        risk_free_rate: Annualized risk-free interest rate r.
        initial_guess: Initial IV estimate (default 0.20 = 20%).

    Returns:
        Annualized Implied Volatility (float >= 0.0) or None if arbitrage violated.
    """
    if time_to_expiry <= 1e-6 or market_price <= 0.0 or strike <= 0.0 or forward <= 0.0:
        return None

    ql_type = normalize_option_type(option_type)
    discount = math.exp(-risk_free_rate * time_to_expiry)

    # Intrinsic value check under Black-76
    if ql_type == ql.Option.Call:
        intrinsic = max(0.0, (forward - strike) * discount)
        upper_bound = forward * discount
    else:
        intrinsic = max(0.0, (strike - forward) * discount)
        upper_bound = strike * discount

    if market_price <= intrinsic + 1e-6 or market_price >= upper_bound - 1e-6:
        # Market price outside theoretical no-arbitrage bounds
        return None

    try:
        std_dev_guess = initial_guess * math.sqrt(time_to_expiry)
        implied_std_dev = ql.blackFormulaImpliedStdDev(
            ql_type,
            float(strike),
            float(forward),
            float(market_price),
            float(discount),
            0.0,
            float(std_dev_guess),
            1e-6,
            100,
        )
        iv = implied_std_dev / math.sqrt(time_to_expiry)
        return float(iv) if iv >= 0.0 else None
    except (RuntimeError, ValueError) as err:
        logger.debug("QuantLib IV solver failed for strike %s: %s", strike, err)
        return None


def calculate_greeks_black76(
    option_type: str,
    strike: float,
    forward: float,
    iv: float,
    time_to_expiry: float,
    risk_free_rate: float = 0.07,
) -> GreeksResult:
    """Calculate Delta, Gamma, Theta, and Vega using QuantLib BlackCalculator.

    Args:
        option_type: 'CE' or 'PE'.
        strike: Option strike price.
        forward: Forward price.
        iv: Implied volatility (annualized, e.g., 0.15).
        time_to_expiry: Time to expiry in years.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Typed GreeksResult with Delta, Gamma, daily Theta, and 1% Vega.
    """
    ql_type = normalize_option_type(option_type)
    discount = math.exp(-risk_free_rate * max(time_to_expiry, 1e-6))
    payoff = ql.PlainVanillaPayoff(ql_type, strike)
    std_dev = max(iv, 1e-6) * math.sqrt(max(time_to_expiry, 1e-6))

    calc = ql.BlackCalculator(payoff, forward, std_dev, discount)

    delta = float(calc.delta(forward))
    gamma = float(calc.gamma(forward))
    theta_annual = float(calc.theta(forward, time_to_expiry))
    # Daily theta: loss per calendar day
    theta_daily = float(theta_annual / 365.0)

    # QuantLib vega is d(Price) / d(vol) where vol is 1.0 (100%)
    vega_total = float(calc.vega(time_to_expiry))
    # Standard market quoting: Vega per 1% change in IV (i.e. / 100)
    vega_pct = float(vega_total / 100.0)

    return GreeksResult(
        iv=max(0.0, float(iv)),
        delta=delta,
        gamma=max(0.0, gamma),
        theta=theta_daily,
        theta_annual=theta_annual,
        vega=max(0.0, vega_pct),
        vega_total=max(0.0, vega_total),
    )


# ---------------------------------------------------------------------------
# SVI (Stochastic Volatility Inspired) Model
# ---------------------------------------------------------------------------

def raw_svi_total_variance(
    k: Union[float, np.ndarray],
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> Union[float, np.ndarray]:
    """Gatheral Raw SVI formulation:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """
    diff = k - m
    return a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))


def raw_svi_implied_vol(
    k: Union[float, np.ndarray],
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
    time_to_expiry: float,
) -> Union[float, np.ndarray]:
    """Calculate implied volatility sigma(k) = sqrt(max(w(k), 1e-8) / T)."""
    w = raw_svi_total_variance(k, a, b, rho, m, sigma)
    # Guarantee non-negative total variance
    w_safe = np.maximum(w, 1e-8)
    return np.sqrt(w_safe / max(time_to_expiry, 1e-6))


class SVICalibrator:
    """Calibrate SVI parameters across log-moneyness for an options expiry slice."""

    def __init__(self, time_to_expiry: float, forward: float):
        self.time_to_expiry = max(time_to_expiry, 1e-6)
        self.forward = forward

    def fit(
        self,
        strikes: np.ndarray,
        market_ivs: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> SVIParameters:
        """Fit raw SVI parameters against strikes and market IVs.

        Args:
            strikes: Array of strike prices K.
            market_ivs: Array of market annualized IVs.
            weights: Optional fitting weights per strike.

        Returns:
            Calibrated SVIParameters instance.
        """
        if len(strikes) < 3:
            raise ValueError(f"At least 3 option quotes required to fit SVI, got {len(strikes)}")

        k = np.log(strikes / self.forward)
        w_market = (market_ivs ** 2) * self.time_to_expiry

        if weights is None:
            w_weights = np.ones_like(k)
        else:
            w_weights = np.asarray(weights, dtype=float)

        min_w = float(np.min(w_market))
        max_w = float(np.max(w_market))

        # Initial parameter estimates
        x0 = [
            max(min_w * 0.5, 1e-4),  # a: floor
            0.1,                     # b: wing slope
            -0.3,                    # rho: negative skew typical for indices
            0.0,                     # m: ATM center
            0.1,                     # sigma: smile curvature
        ]

        # Bounds: a in [-1, 2], b in [0, 5], rho in [-0.999, 0.999], m in [-2, 2], sigma in [1e-4, 2]
        bounds = (
            [-1.0, 0.0, -0.999, -2.0, 1e-4],
            [max_w * 2.0 + 1.0, 5.0, 0.999, 2.0, 2.0],
        )

        def objective(params: np.ndarray) -> np.ndarray:
            a, b, rho, m, sig = params
            pred_w = raw_svi_total_variance(k, a, b, rho, m, sig)
            residuals = (pred_w - w_market) * w_weights

            # Arbitrage penalty if minimum total variance < 0: a + b * sigma * sqrt(1 - rho^2) < 0
            min_tv = a + b * sig * np.sqrt(1.0 - rho ** 2)
            if min_tv < 0.0:
                residuals = residuals + (abs(min_tv) * 100.0)

            return residuals

        res = least_squares(objective, x0=x0, bounds=bounds, ftol=1e-8, xtol=1e-8, max_nfev=2000)

        a, b, rho, m, sig = res.x
        is_arb_free = bool(a + b * sig * np.sqrt(1.0 - rho ** 2) >= 0.0)

        # Compute RMSE on implied volatility
        pred_iv = raw_svi_implied_vol(k, a, b, rho, m, sig, self.time_to_expiry)
        rmse = float(np.sqrt(np.mean((pred_iv - market_ivs) ** 2)))

        return SVIParameters(
            a=float(a),
            b=float(b),
            rho=float(rho),
            m=float(m),
            sigma=float(sig),
            time_to_expiry=self.time_to_expiry,
            rmse=rmse,
            is_arbitrage_free=is_arb_free,
        )


# ---------------------------------------------------------------------------
# High-Level Surface Calibration Pipeline
# ---------------------------------------------------------------------------

def calibrate_option_surface(
    symbol: str,
    spot: float,
    forward: float,
    time_to_expiry_days: float,
    quotes_raw: List[Dict[str, Any]],
    risk_free_rate: float = 0.07,
    discard_illiquid: bool = True,
    max_spread_pct: float = 0.15,
) -> VolatilitySurfaceResult:
    """Full pipeline: Invert market IVs, filter illiquid quotes, and calibrate SVI surface.

    Args:
        symbol: Product symbol (e.g., 'NSE:NIFTY50-INDEX').
        spot: Current spot price.
        forward: Current forward price (if unknown, spot * exp(r * T)).
        time_to_expiry_days: Days until option expiry.
        quotes_raw: List of raw quote dictionaries containing strike, option_type, bid, ask, ltp.
        risk_free_rate: Annualized risk-free rate.
        discard_illiquid: If True, exclude quotes where spread > 15% mid-price from SVI fit.
        max_spread_pct: Maximum allowed spread ratio (default 0.15 = 15%).

    Returns:
        VolatilitySurfaceResult containing evaluated quotes, Greeks, and SVI curve.
    """
    time_to_expiry_years = max(time_to_expiry_days / 365.0, 1e-4)
    if forward <= 0.0:
        forward = spot * math.exp(risk_free_rate * time_to_expiry_years)

    processed_quotes: List[OptionQuote] = []
    fitting_strikes: List[float] = []
    fitting_ivs: List[float] = []

    for item in quotes_raw:
        strike = float(item["strike"])
        option_type = str(item.get("option_type") or item.get("type", "CE")).upper()
        bid = float(item.get("bid") or item.get("bid_price") or 0.0)
        ask = float(item.get("ask") or item.get("ask_price") or 0.0)
        ltp = float(item.get("ltp") or item.get("price") or item.get("market_price") or 0.0)

        # Mid price calculation
        if bid > 0.0 and ask > 0.0:
            mid = (bid + ask) / 2.0
            spread = ask - bid
        else:
            mid = ltp if ltp > 0.0 else max(bid, ask, 0.0)
            spread = 0.0

        spread_pct = (spread / mid) if mid > 0.0 else 0.0
        is_liquid = bool(spread_pct <= max_spread_pct) if (bid > 0.0 and ask > 0.0) else True
        k = math.log(strike / forward)

        # Solve for Black-76 IV using QuantLib
        iv = solve_implied_volatility_black76(
            option_type=option_type,
            strike=strike,
            forward=forward,
            market_price=mid if mid > 0.0 else ltp,
            time_to_expiry=time_to_expiry_years,
            risk_free_rate=risk_free_rate,
        )

        delta = gamma = theta = vega = None
        if iv is not None and iv > 0.0:
            greeks = calculate_greeks_black76(
                option_type=option_type,
                strike=strike,
                forward=forward,
                iv=iv,
                time_to_expiry=time_to_expiry_years,
                risk_free_rate=risk_free_rate,
            )
            delta = greeks.delta
            gamma = greeks.gamma
            theta = greeks.theta
            vega = greeks.vega

            # Collect eligible points for SVI calibration
            # Discard illiquid quotes (spread > 15%) per skill protocol
            if not discard_illiquid or is_liquid:
                fitting_strikes.append(strike)
                fitting_ivs.append(iv)

        quote = OptionQuote(
            strike=strike,
            option_type=option_type,
            market_price=ltp if ltp > 0 else mid,
            bid=bid,
            ask=ask,
            mid_price=mid,
            spread=spread,
            spread_pct=spread_pct,
            is_liquid=is_liquid,
            log_moneyness=k,
            iv=iv,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            open_interest=int(item.get("oi") or item.get("open_interest") or 0),
            volume=int(item.get("volume") or 0),
        )
        processed_quotes.append(quote)

    # Calibrate SVI
    if len(fitting_strikes) >= 3:
        calibrator = SVICalibrator(time_to_expiry=time_to_expiry_years, forward=forward)
        svi_params = calibrator.fit(np.array(fitting_strikes), np.array(fitting_ivs))
    else:
        # Fallback default SVI params if insufficient quotes
        svi_params = SVIParameters(
            a=0.04 * time_to_expiry_years,
            b=0.1,
            rho=-0.3,
            m=0.0,
            sigma=0.1,
            time_to_expiry=time_to_expiry_years,
            rmse=0.0,
            is_arbitrage_free=True,
        )

    # Populate SVI IV on each quote
    for q in processed_quotes:
        if q.log_moneyness is not None:
            q.svi_iv = float(
                raw_svi_implied_vol(
                    q.log_moneyness,
                    svi_params.a,
                    svi_params.b,
                    svi_params.rho,
                    svi_params.m,
                    svi_params.sigma,
                    time_to_expiry_years,
                )
            )

    # Generate smooth SVI curve across log-moneyness grid for visualization
    min_strike = min(q.strike for q in processed_quotes) if processed_quotes else forward * 0.8
    max_strike = max(q.strike for q in processed_quotes) if processed_quotes else forward * 1.2
    grid_strikes = np.linspace(min_strike, max_strike, 50)
    grid_k = np.log(grid_strikes / forward)

    curve_points: List[SVICurvePoint] = []
    for strike_val, k_val in zip(grid_strikes, grid_k):
        tot_var = float(
            raw_svi_total_variance(
                k_val,
                svi_params.a,
                svi_params.b,
                svi_params.rho,
                svi_params.m,
                svi_params.sigma,
            )
        )
        fitted_iv = float(
            raw_svi_implied_vol(
                k_val,
                svi_params.a,
                svi_params.b,
                svi_params.rho,
                svi_params.m,
                svi_params.sigma,
                time_to_expiry_years,
            )
        )
        curve_points.append(
            SVICurvePoint(
                strike=float(strike_val),
                log_moneyness=float(k_val),
                total_variance=tot_var,
                implied_volatility=fitted_iv,
            )
        )

    return VolatilitySurfaceResult(
        symbol=symbol,
        spot=spot,
        forward=forward,
        time_to_expiry_days=time_to_expiry_days,
        time_to_expiry_years=time_to_expiry_years,
        risk_free_rate=risk_free_rate,
        svi_parameters=svi_params,
        curve_points=curve_points,
        quotes=processed_quotes,
    )


# ---------------------------------------------------------------------------
# Synthetic Option Chain Generator
# ---------------------------------------------------------------------------

def generate_synthetic_option_chain(
    symbol: str = "NSE:NIFTY50-INDEX",
    spot: float = 24000.0,
    time_to_expiry_days: float = 14.0,
    strikecount: int = 10,
    strike_interval: float = 100.0,
    base_iv: float = 0.14,
    skew: float = -0.25,
    risk_free_rate: float = 0.07,
) -> Dict[str, Any]:
    """Generate realistic synthetic options chain data for offline testing & UI development."""
    tte = time_to_expiry_days / 365.0
    discount = math.exp(-risk_free_rate * tte)
    forward = spot * math.exp(risk_free_rate * tte)

    atm_strike = round(spot / strike_interval) * strike_interval
    strikes = [
        atm_strike + i * strike_interval
        for i in range(-strikecount, strikecount + 1)
    ]

    quotes_raw = []
    for strike in strikes:
        k = math.log(strike / forward)
        # Volatility smile with negative skew: IV = base_iv + skew * k + 0.5 * k^2
        strike_iv = max(0.05, base_iv + skew * k + 0.6 * (k ** 2))
        std_dev = strike_iv * math.sqrt(tte)

        # Theoretical Call & Put prices
        call_payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
        put_payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike)

        calc_call = ql.BlackCalculator(call_payoff, forward, std_dev, discount)
        calc_put = ql.BlackCalculator(put_payoff, forward, std_dev, discount)

        call_mid = max(0.5, float(calc_call.value()))
        put_mid = max(0.5, float(calc_put.value()))

        # Spread: 1% to 3% for near ATM, wider for deep OTM
        spread_factor = 0.02 + 0.05 * abs(k)
        call_spread = max(0.5, call_mid * spread_factor)
        put_spread = max(0.5, put_mid * spread_factor)

        # Call
        quotes_raw.append({
            "strike": strike,
            "option_type": "CE",
            "bid": max(0.05, round(call_mid - call_spread / 2.0, 2)),
            "ask": round(call_mid + call_spread / 2.0, 2),
            "ltp": round(call_mid, 2),
            "oi": int(100000 / (1.0 + 5.0 * abs(k))),
            "volume": int(50000 / (1.0 + 5.0 * abs(k))),
        })

        # Put
        quotes_raw.append({
            "strike": strike,
            "option_type": "PE",
            "bid": max(0.05, round(put_mid - put_spread / 2.0, 2)),
            "ask": round(put_mid + put_spread / 2.0, 2),
            "ltp": round(put_mid, 2),
            "oi": int(100000 / (1.0 + 5.0 * abs(k))),
            "volume": int(50000 / (1.0 + 5.0 * abs(k))),
        })

    return {
        "symbol": symbol,
        "spot": spot,
        "forward": forward,
        "time_to_expiry_days": time_to_expiry_days,
        "quotes": quotes_raw,
    }
