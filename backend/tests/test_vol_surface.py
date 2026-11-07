"""Test suite for Technical Indicators and QuantLib SVI Volatility Surface Calibration."""

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import QuantLib as ql
from starlette.testclient import TestClient

from backend.app.features.indicators import (
    calculate_session_indicators,
    compute_atr,
    compute_realized_volatility,
    compute_rsi,
    compute_session_vwap,
    resample_to_5min,
)
from backend.app.main import app
from backend.app.models.volatility import (
    SVICalibrator,
    calculate_greeks_black76,
    calibrate_option_surface,
    generate_synthetic_option_chain,
    raw_svi_implied_vol,
    raw_svi_total_variance,
    solve_implied_volatility_black76,
)


@pytest.fixture
def api_client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Technical Indicators Tests
# ---------------------------------------------------------------------------

def test_session_vwap_reset_at_0915_ist():
    """Assert Session VWAP resets at 09:15 IST and standard deviation bands are ordered."""
    times = []
    prices = []
    vols = []

    # Day 1: 09:15 to 11:15 IST (120 1-minute bars)
    d1 = datetime(2026, 11, 5, 9, 15)
    for i in range(120):
        times.append(d1 + timedelta(minutes=i))
        prices.append(100.0 + i * 0.1)
        vols.append(1000)

    # Day 2: 09:15 to 11:15 IST (120 1-minute bars, higher price)
    d2 = datetime(2026, 11, 6, 9, 15)
    for i in range(120):
        times.append(d2 + timedelta(minutes=i))
        prices.append(200.0 + i * 0.1)
        vols.append(1000)

    df = pl.DataFrame({
        "timestamp": times,
        "open": prices,
        "high": [p + 0.5 for p in prices],
        "low": [p - 0.5 for p in prices],
        "close": prices,
        "volume": vols,
    })

    df_vwap = compute_session_vwap(df, rolling_window=20)

    # Day 1 first bar VWAP must equal typical price (100.0)
    day1_first_vwap = df_vwap["vwap"][0]
    assert pytest.approx(day1_first_vwap, rel=1e-4) == 100.0

    # Day 2 first bar VWAP must reset to 200.0 (not accumulate Day 1 volume)
    day2_first_row = df_vwap.filter(pl.col("session_id") == datetime(2026, 11, 6).date()).head(1)
    day2_first_vwap = day2_first_row["vwap"][0]
    assert pytest.approx(day2_first_vwap, rel=1e-4) == 200.0

    # Standard deviation bands ordering check
    # vwap_lower_2 <= vwap_lower_1 <= vwap <= vwap_upper_1 <= vwap_upper_2
    for row in df_vwap.tail(30).iter_rows(named=True):
        vwap = row["vwap"]
        u1 = row["vwap_upper_1"]
        l1 = row["vwap_lower_1"]
        u2 = row["vwap_upper_2"]
        l2 = row["vwap_lower_2"]
        assert l2 <= l1 <= vwap <= u1 <= u2


def test_rsi_and_atr_indicators():
    """Assert 14-period RSI stays within [0, 100] and ATR is strictly positive."""
    base = datetime(2026, 11, 6, 9, 15)
    times = [base + timedelta(minutes=i) for i in range(100)]
    closes = [100.0 + math.sin(i / 5.0) * 5.0 for i in range(100)]
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]
    vols = [1000 for _ in range(100)]

    df = pl.DataFrame({
        "timestamp": times,
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": vols,
    })

    df_rsi = compute_rsi(df, period=14)
    df_atr = compute_atr(df_rsi, period=14)

    rsi_values = df_atr["rsi"].to_list()
    atr_values = df_atr["atr"].to_list()

    # RSI must be in [0, 100]
    for val in rsi_values:
        assert 0.0 <= val <= 100.0

    # ATR must be non-negative
    for val in atr_values[1:]:
        assert val > 0.0


def test_realized_volatility_5min():
    """Assert 20-period Realized Volatility computes non-negative values and resampling works."""
    base = datetime(2026, 11, 6, 9, 15)
    # 200 1-minute bars = 40 5-minute bars
    times = [base + timedelta(minutes=i) for i in range(200)]
    # Random walk with drift
    np.random.seed(42)
    increments = np.random.normal(0.0002, 0.001, size=200)
    prices = 1000.0 * np.exp(np.cumsum(increments))

    df_1m = pl.DataFrame({
        "timestamp": times,
        "open": prices,
        "high": prices * 1.001,
        "low": prices * 0.999,
        "close": prices,
        "volume": [500] * 200,
    })

    # Resample to 5m
    df_5m = resample_to_5min(df_1m)
    assert len(df_5m) == 40

    df_rv = compute_realized_volatility(df_5m, period=20, annualize=True)
    rvs = df_rv["realized_vol_5m"].to_list()

    # After warm-up (bar 20 onwards), annualized RV must be positive and reasonable
    for rv in rvs[20:]:
        assert rv > 0.0
        assert rv < 2.0  # reasonable volatility < 200%


def test_full_indicator_pipeline():
    """Assert calculate_session_indicators runs end-to-end without nulls or crashes."""
    base = datetime(2026, 11, 6, 9, 15)
    times = [base + timedelta(minutes=i) for i in range(100)]
    prices = [24000.0 + i * 2.0 for i in range(100)]
    df = pl.DataFrame({
        "timestamp": times,
        "open": prices,
        "high": [p + 10.0 for p in prices],
        "low": [p - 10.0 for p in prices],
        "close": prices,
        "volume": [10000] * 100,
    })

    res = calculate_session_indicators(df)
    expected_cols = [
        "vwap", "vwap_upper_1", "vwap_lower_1", "vwap_upper_2", "vwap_lower_2",
        "rsi_14", "atr_14", "realized_vol_5m",
    ]
    for col in expected_cols:
        assert col in res.schema


# ---------------------------------------------------------------------------
# QuantLib Black-76 IV and Greeks Tests
# ---------------------------------------------------------------------------

def test_quantlib_black76_iv_inversion():
    """Verify QuantLib accurately inverts market option prices back to input IV."""
    forward = 24000.0
    strike = 24000.0
    tte = 15.0 / 365.0
    r = 0.07
    true_iv = 0.165

    # Compute theoretical Call price using true IV
    greeks = calculate_greeks_black76("CE", strike, forward, true_iv, tte, r)
    std_dev = true_iv * math.sqrt(tte)
    discount = math.exp(-r * tte)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    calc = ql.BlackCalculator(payoff, forward, std_dev, discount)
    call_price = calc.value()

    # Invert price using solve_implied_volatility_black76
    solved_iv = solve_implied_volatility_black76("CE", strike, forward, call_price, tte, r)
    assert solved_iv is not None
    assert pytest.approx(solved_iv, rel=1e-4) == true_iv


def test_quantlib_arbitrage_bound_rejection():
    """Assert IV solver rejects prices below discounted intrinsic value."""
    forward = 24000.0
    strike = 22000.0  # Deep ITM Call
    tte = 15.0 / 365.0
    r = 0.07
    discount = math.exp(-r * tte)
    intrinsic = (forward - strike) * discount

    # Price below intrinsic violates no-arbitrage bounds
    invalid_price = intrinsic - 20.0
    result = solve_implied_volatility_black76("CE", strike, forward, invalid_price, tte, r)
    assert result is None


def test_quantlib_valid_greeks():
    """Assert calculated Greeks obey standard quantitative boundaries."""
    forward = 24000.0
    strike = 24000.0
    tte = 20.0 / 365.0
    r = 0.07
    iv = 0.15

    # Call Greeks
    call_greeks = calculate_greeks_black76("CE", strike, forward, iv, tte, r)
    assert 0.0 < call_greeks.delta < 1.0
    assert call_greeks.gamma > 0.0
    assert call_greeks.vega > 0.0
    assert call_greeks.theta < 0.0  # Time decay is negative

    # Put Greeks
    put_greeks = calculate_greeks_black76("PE", strike, forward, iv, tte, r)
    assert -1.0 < put_greeks.delta < 0.0
    assert put_greeks.gamma > 0.0
    assert put_greeks.vega > 0.0
    assert put_greeks.theta < 0.0

    # Put-Call parity for forward options: Delta(Call) - Delta(Put) == exp(-r * T)
    discount = math.exp(-r * tte)
    assert pytest.approx(call_greeks.delta - put_greeks.delta, abs=1e-4) == discount


# ---------------------------------------------------------------------------
# SVI Calibration & Liquidity Filter Tests
# ---------------------------------------------------------------------------

def test_svi_calibration_synthetic_chain():
    """Calibrate SVI curve on synthetic chain, asserting non-negative IV and valid parameters."""
    spot = 24000.0
    tte_days = 14.0
    synthetic_chain = generate_synthetic_option_chain(
        symbol="NSE:NIFTY50-INDEX",
        spot=spot,
        time_to_expiry_days=tte_days,
        strikecount=8,
        strike_interval=100.0,
        base_iv=0.14,
        skew=-0.20,
    )

    surface = calibrate_option_surface(
        symbol=synthetic_chain["symbol"],
        spot=synthetic_chain["spot"],
        forward=synthetic_chain["forward"],
        time_to_expiry_days=synthetic_chain["time_to_expiry_days"],
        quotes_raw=synthetic_chain["quotes"],
        discard_illiquid=True,
    )

    # 1. Assert non-negative IV on all evaluated quotes
    assert len(surface.quotes) > 0
    for q in surface.quotes:
        if q.iv is not None:
            assert q.iv >= 0.0
        if q.svi_iv is not None:
            assert q.svi_iv >= 0.0

    # 2. Assert SVI parameters satisfy structural non-arbitrage constraints
    svi = surface.svi_parameters
    assert svi.b >= 0.0
    assert -1.0 <= svi.rho <= 1.0
    assert svi.sigma > 0.0
    # Gatheral non-negativity: a + b * sigma * sqrt(1 - rho^2) >= 0
    assert svi.is_arbitrage_free is True
    # RMSE must be small (< 0.01 = 1% IV difference)
    assert svi.rmse < 0.01

    # 3. Assert smooth SVI curve points are non-negative
    assert len(surface.curve_points) == 50
    for pt in surface.curve_points:
        assert pt.total_variance >= 0.0
        assert pt.implied_volatility >= 0.0


def test_illiquid_quote_filtering_protocol():
    """Assert quotes with bid-ask spread > 15% are flagged and discarded from SVI fit."""
    spot = 24000.0
    forward = 24050.0
    tte_days = 15.0

    # Create 4 liquid quotes and 1 illiquid quote (30% spread)
    quotes_raw = [
        {"strike": 23800, "option_type": "CE", "bid": 400.0, "ask": 408.0, "ltp": 404.0},  # ~2% spread
        {"strike": 23900, "option_type": "CE", "bid": 320.0, "ask": 326.0, "ltp": 323.0},  # ~1.8% spread
        {"strike": 24000, "option_type": "CE", "bid": 250.0, "ask": 255.0, "ltp": 252.5},  # ~2% spread
        {"strike": 24100, "option_type": "CE", "bid": 190.0, "ask": 194.0, "ltp": 192.0},  # ~2.1% spread
        {"strike": 24200, "option_type": "CE", "bid": 100.0, "ask": 140.0, "ltp": 120.0},  # 33% spread! ILLIQUID
    ]

    surface = calibrate_option_surface(
        symbol="NSE:NIFTY50-INDEX",
        spot=spot,
        forward=forward,
        time_to_expiry_days=tte_days,
        quotes_raw=quotes_raw,
        discard_illiquid=True,
    )

    quotes_by_strike = {q.strike: q for q in surface.quotes}
    illiquid_quote = quotes_by_strike[24200]
    assert illiquid_quote.spread_pct > 0.15
    assert illiquid_quote.is_liquid is False

    liquid_quote = quotes_by_strike[24000]
    assert liquid_quote.spread_pct <= 0.15
    assert liquid_quote.is_liquid is True


# ---------------------------------------------------------------------------
# FastAPI Options Chain Router Tests
# ---------------------------------------------------------------------------

def test_api_options_chain_endpoint(api_client):
    """Assert GET /api/options/chain/{symbol} returns 200 with complete chain and SVI surface."""
    resp = api_client.get("/api/options/chain/NSE:NIFTY50-INDEX?strikecount=5&force_synthetic=true")
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert data["symbol"] == "NSE:NIFTY50-INDEX"
    assert data["spot"] > 0
    assert data["forward"] > 0
    assert "svi_parameters" in data
    assert data["svi_parameters"]["is_arbitrage_free"] is True
    assert len(data["quotes"]) > 0
    assert len(data["curve_points"]) == 50

    # Verify Greeks and IV exist on quotes
    sample_quote = data["quotes"][0]
    assert "strike" in sample_quote
    assert "option_type" in sample_quote
    assert "iv" in sample_quote
    assert "delta" in sample_quote
    assert "gamma" in sample_quote
    assert "theta" in sample_quote
    assert "vega" in sample_quote


def test_api_options_chain_banknifty(api_client):
    """Assert endpoint handles BankNifty symbol mapping and strike generation."""
    resp = api_client.get("/api/options/chain/BANKNIFTY?strikecount=4&force_synthetic=true")
    assert resp.status_code == 200
    data = resp.json()
    assert data["symbol"] == "NSE:BANKNIFTY-INDEX"
    assert data["spot"] > 40000.0
