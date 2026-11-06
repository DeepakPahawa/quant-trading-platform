"""Single-Command FYERS Browser Authorization & Live SBIN Data Fetcher.

Usage:
  # Interactive mode (opens browser, prompts for redirect URL):
  .venv/bin/python backend/scripts/authorize_and_fetch.py

  # Direct mode (pass auth_code or redirect URL as argument):
  .venv/bin/python backend/scripts/authorize_and_fetch.py "<auth_code_or_url>"
"""

import json
import os
import sys
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

# Path configuration
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
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
from backend.app.data.storage import DuckDBStorage


def fetch_and_display_stocks_live(client: fyersModel.FyersModel, symbols: list = None):
    """Fetch live data straight from FYERS API for SBIN and ICICI Bank."""
    if symbols is None:
        symbols = ["NSE:SBIN-EQ", "NSE:ICICIBANK-EQ"]

    print("\n" + "=" * 70)
    print("LIVE MARKET DATA FOR " + " & ".join(symbols))
    print("=" * 70)

    # 1. Market Status
    print("\n[1] Market Operating Status & Trade Date:")
    status = fetch_fyers_market_status(client)
    print(f"  - Exchange Trade Date: {status['data'].get('market_date')}")
    print(f"  - Overall Status: {status['data'].get('market_status')}")
    exchs = status['data'].get('exchanges', {})
    for exch, st in exchs.items():
        print(f"    * {exch}: {st}")

    # 2. User Profile & Account
    print("\n[2] Broker Account Profile:")
    prof = fetch_fyers_profile(client)
    if prof.get("s") == "ok":
        pdata = prof["data"]
        print(f"  - Client ID: {pdata.get('fy_id')}")
        print(f"  - Account Holder: {pdata.get('name')}")
        print(f"  - Email: {pdata.get('email')}")
        print(f"  - Mobile: {pdata.get('mobile')}")
        print(f"  - 2FA TOTP Active: {pdata.get('totp_enabled')}")

    # 3. Available Funds & Margins
    print("\n[3] Trading Account Funds & Margins:")
    tx = fetch_fyers_transactions(client)
    tdata = tx["data"]
    print(f"  - Available Margin: ₹{tdata.get('available_margin', 0.0):,.2f}")
    print(f"  - Utilized Margin: ₹{tdata.get('used_margin', 0.0):,.2f}")
    print(f"  - Total Balance: ₹{tdata.get('total_balance', 0.0):,.2f}")
    print(f"  - Total Demat Holdings: {tdata.get('holdings_count', 0)}")
    print(f"  - Open Positions: {tdata.get('positions_count', 0)}")

    # 4. Live Quotes & Top-5 Order Book Depth
    print(f"\n[4] Real-Time Level-2 Quotes & 5-Depth Metrics:")
    quotes = fetch_fyers_quotes(client, symbols)
    for q in quotes:
        sym = q['symbol']
        print(f"\n  --- {sym} ---")
        print(f"  - Last Traded Price (LTP): ₹{q['ltp']:.2f}")
        print(f"  - Change: ₹{q['change']:+.2f} ({q['change_pct']:+.2f}%)")
        print(f"  - Day Range: Low ₹{q['low']:.2f} — High ₹{q['high']:.2f} (Open ₹{q['open']:.2f}, Prev Close ₹{q['prev_close']:.2f})")
        print(f"  - Total Cumulative Traded Volume: {q['volume']:,} shares")

        print("  - Microstructure Metrics:")
        print(f"    * Top-5 Bid-Ask Depth Imbalance: {q['depth_imbalance']:.6f} (Range [-1.0, 1.0])")
        print(f"    * Volume-Weighted Micro-Price: ₹{q['micro_price']:.4f}")
        print(f"    * Normalized Bid-Ask Spread: {q['normalized_spread']:.8f}")

    # 5. Live 1-Minute Historical Bars & TFT Sequences
    storage = DuckDBStorage()
    for sym in symbols:
        print(f"\n[5] Fetching 1-Minute Historical Bars for {sym}...")
        df_bars = fetch_fyers_historical_bars(client, symbol=sym, resolution="1", days=3)
        print(f"  Retrieved {len(df_bars)} historical 1-minute bars from FYERS API.")

        if not df_bars.is_empty():
            storage.store_bars(df_bars, symbol=sym)
            print(f"  Successfully archived {len(df_bars)} bars into local DuckDB / Parquet storage.")

            if len(df_bars) >= 60:
                print(f"\n[6] Preparing Multivariate Input Sequence for Temporal Fusion Transformer (TFT) - {sym}:")
                match_q = next((q for q in quotes if q['symbol'] == sym), None)
                depth_imb = match_q.get("depth_imbalance", 0.0) if match_q else 0.0
                micro_spread = (match_q.get("micro_price", 0.0) - match_q.get("ltp", 0.0)) if match_q else 0.0

                tft_df = prepare_tft_feature_tensor(
                    df_bars,
                    depth_imbalance=depth_imb,
                    micro_price_spread=micro_spread,
                    sequence_length=60,
                )
                print(f"  Constructed TFT sequence: shape={tft_df.shape} (60 historical bars lookback)")
                print(f"\n  Last 3 Bars of Multivariate Feature Tensor ({sym}):")
                subset = tft_df.select([
                    "timestamp", "close", "log_return", "volume_ratio", "rsi_14", "vwap_diff", "depth_imbalance", "minute_of_day"
                ]).tail(3)
                print(subset)

    print("\n" + "=" * 70)
    print("SUCCESS: Live FYERS API data fetched and processed successfully.")
    print("=" * 70)


def main():
    auth_manager = FyersAuthManager()

    # Step 1: Check if already authenticated with valid token
    cached_token = auth_manager.load_cached_token()
    if cached_token:
        try:
            print("Found existing token. Testing connection...")
            client = auth_manager.get_client()
            prof = client.get_profile()
            if isinstance(prof, dict) and (prof.get("s") == "ok" or prof.get("code") == 200):
                print("Token is ACTIVE and VALID! Fetching live SBIN data directly...")
                fetch_and_display_sbin_live(client)
                return
        except Exception:
            print("Existing token is expired or invalid. Re-authorizing...")

    # Step 2: Handle authorization
    code_input = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg in ("--help", "-h"):
            login_url = auth_manager.get_login_url()
            print("\nUsage:")
            print("  .venv/bin/python backend/scripts/authorize_and_fetch.py [auth_code_or_url]")
            print("\nAuthorization URL:")
            print(f"  {login_url}\n")
            return
        code_input = arg

    # Check if auth code is configured in backend/.env
    if not code_input:
        env_code = (
            os.getenv("FYERS_AUTH_CODE")
            or os.getenv("AUTH_CODE")
            or os.getenv("FYERS_AUTHCODE")
            or os.getenv("AUTHCODE")
        )
        if env_code and env_code.strip():
            print("Found auth code configured in backend/.env! Proceeding with exchange...")
            code_input = env_code.strip()

    if not code_input:
        login_url = auth_manager.get_login_url()
        print("\n" + "=" * 70)
        print("FYERS ONE-TIME BROWSER AUTHORIZATION")
        print("=" * 70)
        print("\n1. Opening FYERS login page in your default browser...")
        print(f"   URL: {login_url}\n")

        try:
            webbrowser.open(login_url)
        except Exception:
            pass

        print("2. Log in using your FYERS credentials (XD17919), OTP, and PIN.")
        print("3. When complete, FYERS redirects to a page starting with:")
        print("   https://trade.fyers.in/api-login/redirect-uri/index.html?auth_code=...\n")

        try:
            code_input = input("Enter the redirected URL or auth_code: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. You can run this command anytime with your auth_code or URL:")
            print("  .venv/bin/python backend/scripts/authorize_and_fetch.py \"<url_or_code>\"")
            return

    if not code_input:
        print("No authorization code provided. Aborting.")
        return

    # Step 3: Exchange code for access token
    print("\nExchanging authorization code for FYERS access token...")
    try:
        access_token = auth_manager.exchange_auth_code(code_input)
        print("Successfully acquired access token and saved to .token_cache.json!")
    except Exception as e:
        print(f"Error exchanging authorization code: {e}")
        return

    # Step 4: Fetch live SBIN data
    client = auth_manager.get_client()
    fetch_and_display_sbin_live(client)


if __name__ == "__main__":
    main()
