"""Diagnostic and Data Fetcher Script for FYERS API v3 and TFT Feature Preparation."""

from datetime import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Ensure backend root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

# Load .env
load_dotenv(root_dir / "backend" / ".env")

from fyers_apiv3 import fyersModel
from backend.app.auth.fyers_auth import FyersAuthManager
from backend.app.data.market_overview import (
    fetch_fyers_historical_bars,
    fetch_fyers_market_status,
    fetch_fyers_profile,
    fetch_fyers_quotes,
    fetch_fyers_transactions,
    prepare_tft_feature_tensor,
)


def inspect_env_readiness() -> Dict[str, Any]:
    """Check configuration and authentication readiness."""
    client_id = os.getenv("FYERS_CLIENT_ID", "")
    secret_key = os.getenv("FYERS_SECRET_KEY", "")
    pin = os.getenv("FYERS_PIN", "")
    totp_key = os.getenv("FYERS_TOTP_KEY", "")
    user_id = os.getenv("FYERS_USER_ID", "")
    redirect_uri = os.getenv("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri/index.html")
    access_token = os.getenv("FYERS_ACCESS_TOKEN", "")

    has_client_id = bool(client_id and client_id != "your_client_id")
    has_secret_key = bool(secret_key and secret_key != "your_secret_key")
    has_pin = bool(pin and pin != "1234")
    has_totp = bool(totp_key and totp_key != "your_totp_secret_key")
    has_user_id = bool(user_id)
    has_access_token = bool(access_token)

    auth_url = ""
    if has_client_id and has_secret_key:
        try:
            session = fyersModel.SessionModel(
                client_id=client_id,
                secret_key=secret_key,
                redirect_uri=redirect_uri,
                response_type="code",
                grant_type="authorization_code",
            )
            auth_url = session.generate_authcode()
        except Exception:
            pass

    return {
        "client_id": client_id,
        "has_client_id": has_client_id,
        "has_secret_key": has_secret_key,
        "has_pin": has_pin,
        "has_totp": has_totp,
        "has_user_id": has_user_id,
        "has_access_token": has_access_token,
        "auth_url": auth_url,
    }


def main():
    print("=" * 70)
    print("FYERS API v3 - Live Data Fetch & TFT Model Ingestion Pipeline")
    print("=" * 70)

    env_status = inspect_env_readiness()
    print("\n[1] Environment Status:")
    print(f"  - FYERS_CLIENT_ID: {'CONFIGURED' if env_status['has_client_id'] else 'MISSING'}")
    print(f"  - FYERS_SECRET_KEY: {'CONFIGURED' if env_status['has_secret_key'] else 'MISSING'}")
    print(f"  - FYERS_USER_ID: {'CONFIGURED' if env_status['has_user_id'] else 'NOT SET (Needed for automated 2FA)'}")
    print(f"  - FYERS_PIN: {'CONFIGURED' if env_status['has_pin'] else 'DEFAULT (1234 - Needs actual trading PIN)'}")
    print(f"  - FYERS_TOTP_KEY: {'CONFIGURED' if env_status['has_totp'] else 'DEFAULT (Placeholder - Needs 2FA secret)'}")

    # Attempt to authenticate
    auth_manager = FyersAuthManager()
    client = None

    try:
        client = auth_manager.get_client()
        print("\n>>> Authentication SUCCESSFUL! FYERS client active.")
    except Exception as ex:
        print(f"\n>>> Authentication Pending: {ex}")

    if client is not None:
        # 1. Profile Details
        print("\n[2] Fetching Profile Details...")
        profile = fetch_fyers_profile(client)
        print("Profile Output:")
        print(json.dumps(profile["data"], indent=2))

        # 2. Transaction Info
        print("\n[3] Fetching Transaction & Account Info...")
        tx = fetch_fyers_transactions(client)
        print("Transaction Output:")
        print(json.dumps(tx["data"], indent=2))

        # 3. Market Status & Date
        print("\n[4] Fetching Market Status & Date...")
        status = fetch_fyers_market_status(client)
        print("Market Status Output:")
        print(json.dumps(status["data"], indent=2))

        # 4. Quotes & Microstructure Depth
        print("\n[5] Fetching Real-Time Quotes & 5-Depth Metrics...")
        symbols = ["NSE:SBIN-EQ", "NSE:RELIANCE-EQ", "NSE:NIFTY50-INDEX"]
        quotes = fetch_fyers_quotes(client, symbols)
        print(f"Quotes Output ({len(quotes)} instruments):")
        for q in quotes:
            print(f"  Symbol: {q['symbol']} | LTP: {q['ltp']} | Spread: {q['normalized_spread']} | Imbalance: {q['depth_imbalance']} | Micro-Price: {q['micro_price']}")

        # 5. Historical Bars & TFT Tensor Preparation
        print("\n[6] Fetching 1-Minute Historical Bars for TFT Model...")
        df_bars = fetch_fyers_historical_bars(client, symbol="NSE:SBIN-EQ", resolution="1", days=2)
        print(f"Retrieved {len(df_bars)} historical 1-minute bars.")
        if len(df_bars) >= 60:
            tft_tensor = prepare_tft_feature_tensor(df_bars)
            print("\nPrepared TFT Input Sequence (Last 5 of 60 bars):")
            print(tft_tensor.select(["timestamp", "close", "log_return", "volume_ratio", "rsi_14", "vwap_diff", "minute_of_day"]).tail(5))
    else:
        print("\n" + "=" * 70)
        print("EXPLANATION OF DATA RETURNED BY FYERS & TFT INGESTION SPECIFICATION")
        print("=" * 70)

        print("\nWhen authenticated, FYERS API v3 returns the following data structures:\n")

        print("1. PROFILE DETAILS (client.get_profile()):")
        sample_profile = {
            "s": "ok",
            "code": 200,
            "message": "",
            "data": {
                "fy_id": "XX12345",
                "name": "Trader Name",
                "email_id": "user@example.com",
                "mobile_number": "98XXXXXXXX",
                "pan": "ABCDE1234F",
                "totp": True,
                "pin_change_date": "2024-01-01 10:00:00",
                "pwd_change_date": "2024-01-01 10:00:00"
            }
        }
        print(json.dumps(sample_profile, indent=2))

        print("\n2. TRANSACTION & ACCOUNT INFO (client.funds(), holdings(), positions(), orderbook(), tradebook()):")
        sample_tx = {
            "funds": {
                "available_margin": 125000.50,
                "used_margin": 24500.00,
                "total_balance": 149500.50
            },
            "holdings": [
                {
                    "symbol": "NSE:SBIN-EQ",
                    "quantity": 100,
                    "costPrice": 580.0,
                    "marketVal": 60500.0,
                    "pl": 2500.0,
                    "ltp": 605.0
                }
            ],
            "positions": [
                {
                    "symbol": "NSE:RELIANCE-EQ",
                    "buyQty": 50,
                    "sellQty": 0,
                    "netQty": 50,
                    "netAvg": 2980.50,
                    "realized_profit": 0.0,
                    "unrealized_profit": 450.0
                }
            ],
            "orderbook": [
                {
                    "id": "1234567890",
                    "symbol": "NSE:SBIN-EQ",
                    "qty": 100,
                    "type": 2,
                    "side": 1,
                    "status": 2,
                    "orderDateTime": "07-Nov-2024 09:30:15"
                }
            ],
            "tradebook": [
                {
                    "id": "987654321",
                    "symbol": "NSE:SBIN-EQ",
                    "tradePrice": 605.0,
                    "tradeQty": 100,
                    "tradeTime": "09:30:15"
                }
            ]
        }
        print(json.dumps(sample_tx, indent=2))

        print("\n3. REAL-TIME QUOTES & 5-DEPTH ORDER BOOK (client.quotes()):")
        sample_quotes = {
            "symbol": "NSE:SBIN-EQ",
            "ltp": 605.50,
            "open": 600.00,
            "high": 608.20,
            "low": 599.10,
            "prev_close": 598.75,
            "volume": 8421030,
            "bids_5_depth": [
                {"price": 605.45, "volume": 1200, "orders": 5},
                {"price": 605.40, "volume": 3500, "orders": 12},
                {"price": 605.35, "volume": 5000, "orders": 18},
                {"price": 605.30, "volume": 2200, "orders": 7},
                {"price": 605.25, "volume": 4100, "orders": 14}
            ],
            "asks_5_depth": [
                {"price": 605.50, "volume": 900, "orders": 3},
                {"price": 605.55, "volume": 2100, "orders": 8},
                {"price": 605.60, "volume": 4300, "orders": 15},
                {"price": 605.65, "volume": 1800, "orders": 6},
                {"price": 605.70, "volume": 3900, "orders": 11}
            ],
            "computed_microstructure_metrics": {
                "depth_imbalance": 0.230769,
                "micro_price": 605.4714,
                "normalized_spread": 0.0000826
            }
        }
        print(json.dumps(sample_quotes, indent=2))

        print("\n4. MARKET STATUS & DATE (client.market_status()):")
        sample_status = {
            "market_date": "2026-11-07",
            "market_status": "CLOSED",
            "exchanges": {
                "NSE_CAPITAL": "CLOSED",
                "NSE_FNO": "CLOSED",
                "BSE_CAPITAL": "CLOSED",
                "MCX_COMMODITY": "OPEN"
            }
        }
        print(json.dumps(sample_status, indent=2))

        print("\n5. TFT MODEL MULTIVARIATE FEATURE VECTOR (tft-forecaster input tensor):")
        print("Shape: [batch_size, 60_bars, 6_features + 2_time_covariates]")
        print("Features:")
        print("  1. log_return: ln(close_t / close_{t-1})")
        print("  2. volume_ratio: volume / rolling_20_volume_mean")
        print("  3. rsi_14: 14-period Relative Strength Index")
        print("  4. vwap_diff: (close - VWAP) / VWAP")
        print("  5. depth_imbalance: (sum(BidQty) - sum(AskQty)) / (sum(BidQty) + sum(AskQty))")
        print("  6. micro_price_spread: micro_price - mid_price")
        print("Time Covariates:")
        print("  7. minute_of_day: 0 to 375 (normalized for IST 09:15-15:30 session)")
        print("  8. day_of_week: 1 to 5 (Monday to Friday)")

        if env_status["auth_url"]:
            print("\n" + "=" * 70)
            print("HOW TO CONNECT LIVE RIGHT NOW:")
            print("=" * 70)
            print("\nOption 1: Complete headless authentication credentials in backend/.env:")
            print("  FYERS_USER_ID=<your 7-char FYERS user id, e.g. AB12345>")
            print("  FYERS_PIN=<your 4-digit trading PIN>")
            print("  FYERS_TOTP_KEY=<your 32-char TOTP secret key from FYERS 2FA settings>")
            print("\nOption 2: Generate token via browser using this FYERS OAuth link:")
            print(f"  {env_status['auth_url']}")
            print("  Once you log in, copy the 'auth_code' from the redirected URL.")


if __name__ == "__main__":
    main()

