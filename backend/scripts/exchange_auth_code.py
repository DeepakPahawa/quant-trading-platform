"""Exchange FYERS Auth Code for an Access Token and cache it."""

import json
import os
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
load_dotenv(root_dir / "backend" / ".env")

from fyers_apiv3 import fyersModel
from backend.app.auth.fyers_auth import FyersAuthManager


def exchange_code(auth_code_or_url: str):
    client_id = os.getenv("FYERS_CLIENT_ID")
    secret_key = os.getenv("FYERS_SECRET_KEY")
    redirect_uri = os.getenv("FYERS_REDIRECT_URI", "https://trade.fyers.in/api-login/redirect-uri/index.html")

    if not client_id or not secret_key:
        print("Error: FYERS_CLIENT_ID or FYERS_SECRET_KEY missing in backend/.env")
        return

    # Extract auth_code if full URL was pasted
    code = auth_code_or_url.strip()
    if "auth_code=" in code:
        parsed = urlparse(code)
        params = parse_qs(parsed.query)
        if "auth_code" in params:
            code = params["auth_code"][0]

    print(f"Exchanging auth_code with FYERS API v3...")
    session = fyersModel.SessionModel(
        client_id=client_id,
        secret_key=secret_key,
        redirect_uri=redirect_uri,
        response_type="code",
        grant_type="authorization_code",
    )
    session.set_token(code)
    response = session.generate_token()

    if isinstance(response, dict) and "access_token" in response:
        token = response["access_token"]
        manager = FyersAuthManager()
        manager.save_cached_token(token)
        print("SUCCESS! Access token acquired and cached to .token_cache.json")
    else:
        print("Failed to exchange code. Response:", response)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python backend/scripts/exchange_auth_code.py <auth_code_or_full_redirect_url>")
    else:
        exchange_code(sys.argv[1])

