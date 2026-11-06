"""FYERS API v3 Headless Authentication and Token Manager."""

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from fyers_apiv3 import fyersModel
from pydantic import BaseModel, Field
import pyotp
import requests

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class FyersAuthConfig(BaseModel):
    """Configuration schema for FYERS v3 credentials."""
    client_id: str = Field(..., description="App ID (e.g. ABCDE12345-100)")
    secret_key: str = Field(..., description="App Secret Key")
    pin: str = Field(..., description="4-digit trading PIN")
    totp_key: str = Field(..., description="TOTP Secret Key")
    redirect_uri: str = Field(
        default="https://trade.fyers.in/api-login/redirect-uri/index.html",
        description="Registered Redirect URI"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="FYERS User ID (e.g. AB12345). Extracted from client_id if not explicitly set."
    )

    @classmethod
    def from_env(cls) -> "FyersAuthConfig":
        """Instantiate configuration strictly from environment variables."""
        client_id = os.getenv("FYERS_CLIENT_ID")
        secret_key = os.getenv("FYERS_SECRET_KEY")
        pin = os.getenv("FYERS_PIN")
        totp_key = os.getenv("FYERS_TOTP_KEY")
        redirect_uri = os.getenv(
            "FYERS_REDIRECT_URI",
            "https://trade.fyers.in/api-login/redirect-uri/index.html"
        )
        user_id = os.getenv("FYERS_USER_ID")

        missing = []
        if not client_id:
            missing.append("FYERS_CLIENT_ID")
        if not secret_key:
            missing.append("FYERS_SECRET_KEY")
        if not pin:
            missing.append("FYERS_PIN")
        if not totp_key:
            missing.append("FYERS_TOTP_KEY")

        if missing:
            raise ValueError(
                f"Missing required FYERS environment variables: {', '.join(missing)}. "
                "Ensure they are defined in your .env file."
            )

        extracted_user_id = user_id or (client_id.split("-")[0] if "-" in client_id else client_id)

        return cls(
            client_id=client_id,
            secret_key=secret_key,
            pin=pin,
            totp_key=totp_key,
            redirect_uri=redirect_uri,
            user_id=extracted_user_id,
        )


class FyersAuthManager:
    """Manages FYERS API v3 token caching, verification, and headless authentication."""

    DEFAULT_CACHE_FILENAME = ".token_cache.json"

    def __init__(
        self,
        config: Optional[FyersAuthConfig] = None,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.cache_path = cache_path or self._locate_cache_file()

    def _locate_cache_file(self) -> Path:
        """Find or determine the path to .token_cache.json at project root or current dir."""
        # Try finding git root or workspace root
        current_path = Path.cwd()
        if (current_path / self.DEFAULT_CACHE_FILENAME).exists():
            return current_path / self.DEFAULT_CACHE_FILENAME
        
        # Check parent directories for .git or backend
        candidate = current_path
        for _ in range(3):
            if (candidate / "backend").exists() or (candidate / ".git").exists():
                return candidate / self.DEFAULT_CACHE_FILENAME
            candidate = candidate.parent

        return current_path / self.DEFAULT_CACHE_FILENAME

    def _get_config(self) -> FyersAuthConfig:
        if self.config is None:
            self.config = FyersAuthConfig.from_env()
        return self.config

    def load_cached_token(self) -> Optional[str]:
        """Load cached access token if file exists and token is not expired."""
        if not self.cache_path.exists():
            return None

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            access_token = cache_data.get("access_token")
            expires_at = cache_data.get("expires_at", 0)

            # Check if token is still within validity window (buffer of 60 seconds)
            if access_token and time.time() < (expires_at - 60):
                logger.info("Found unexpired access token in cache: %s", self.cache_path)
                return access_token

            logger.info("Cached token is expired or invalid.")
            return None
        except Exception as e:
            logger.warning("Failed to read token cache: %s", e)
            return None

    def save_cached_token(self, access_token: str, expires_in_seconds: int = 86400) -> None:
        """Save access token to cache file with expiration timestamp."""
        now = time.time()
        payload = {
            "access_token": access_token,
            "cached_at": now,
            "expires_at": now + expires_in_seconds,
            "cache_path": str(self.cache_path),
        }
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info("Saved access token to cache: %s", self.cache_path)
        except Exception as e:
            logger.error("Failed to save token to cache: %s", e)

    def verify_token_health(self, client_id: str, access_token: str) -> Optional[Dict[str, Any]]:
        """Verify token health via fyers.get_profile(). Returns profile if valid, None otherwise."""
        try:
            fyers = fyersModel.FyersModel(
                client_id=client_id,
                token=access_token,
                is_async=False,
                log_level="ERROR"
            )
            profile_response = fyers.get_profile()
            if isinstance(profile_response, dict):
                # FYERS returns {'s': 'ok', 'code': 200, 'data': {...}} on success
                if profile_response.get("s") == "ok" or profile_response.get("code") == 200:
                    logger.info("FYERS token is valid and active.")
                    return profile_response
            logger.warning("Token verification check failed: %s", profile_response)
            return None
        except Exception as e:
            logger.warning("Error verifying token with get_profile(): %s", e)
            return None

    def authenticate_headless(self) -> str:
        """Perform headless authentication against FYERS v3 login endpoints using TOTP."""
        config = self._get_config()
        session = requests.Session()
        session.headers.update({
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/115.0.0.0 Safari/537.36"
            ),
        })

        logger.info("Initiating headless login for user: %s", config.user_id)

        # Step 1: Send Login OTP
        fy_id_encoded = base64.b64encode(config.user_id.encode()).decode()
        payload_otp = {
            "fy_id": fy_id_encoded,
            "app_id": "2",
        }
        res_otp = session.post(
            "https://api-t2.fyers.in/vagator/v2/send_login_otp_v2",
            json=payload_otp,
            timeout=15,
        )
        if res_otp.status_code != 200:
            raise RuntimeError(f"Step 1 failed (send_login_otp_v2): HTTP {res_otp.status_code} - {res_otp.text}")

        data_otp = res_otp.json()
        request_key = (
            data_otp.get("sotp")
            or data_otp.get("request_key")
            or data_otp.get("data", {}).get("request_key")
        )
        if not request_key:
            raise RuntimeError(f"Step 1 failed: request_key missing in response: {data_otp}")

        # Step 2: Verify TOTP
        totp = pyotp.TOTP(config.totp_key).now()
        payload_totp = {
            "request_key": request_key,
            "otp": str(totp),
        }
        res_totp = session.post(
            "https://api-t2.fyers.in/vagator/v2/verify_otp",
            json=payload_totp,
            timeout=15,
        )
        if res_totp.status_code != 200:
            raise RuntimeError(f"Step 2 failed (verify_otp): HTTP {res_totp.status_code} - {res_totp.text}")

        data_totp = res_totp.json()
        request_key_2 = data_totp.get("request_key") or data_totp.get("data", {}).get("request_key")
        if not request_key_2:
            raise RuntimeError(f"Step 2 failed: request_key missing after TOTP verification: {data_totp}")

        # Step 3: Verify PIN
        payload_pin = {
            "request_key": request_key_2,
            "identity_type": "pin",
            "identifier": str(config.pin),
        }
        res_pin = session.post(
            "https://api-t2.fyers.in/vagator/v2/verify_pin",
            json=payload_pin,
            timeout=15,
        )
        if res_pin.status_code != 200:
            raise RuntimeError(f"Step 3 failed (verify_pin): HTTP {res_pin.status_code} - {res_pin.text}")

        data_pin = res_pin.json()
        access_token_vague = (
            data_pin.get("data", {}).get("access_token")
            or data_pin.get("access_token")
        )
        if not access_token_vague:
            raise RuntimeError(f"Step 3 failed: access_token missing from PIN verification: {data_pin}")

        # Step 4: Get Auth Code via Thirdparty Code Endpoint
        app_id_prefix = config.client_id.split("-")[0] if "-" in config.client_id else config.client_id
        app_type = config.client_id.split("-")[1] if "-" in config.client_id else "100"

        auth_headers = {
            "authorization": f"Bearer {access_token_vague}",
            "content-type": "application/json",
        }
        payload_thirdparty = {
            "fyers_id": config.user_id,
            "app_id": app_id_prefix,
            "redirect_uri": config.redirect_uri,
            "appType": app_type,
            "code_challenge": "",
            "state": "quant_auth_state",
            "scope": "",
            "nonce": "",
            "response_type": "code",
            "create_cookie": True,
        }
        res_tp = session.post(
            "https://api.fyers.in/api/v2/user/thirdparty_code",
            json=payload_thirdparty,
            headers=auth_headers,
            timeout=15,
        )
        if res_tp.status_code != 200:
            raise RuntimeError(f"Step 4 failed (thirdparty_code): HTTP {res_tp.status_code} - {res_tp.text}")

        data_tp = res_tp.json()
        url_to_parse = data_tp.get("Url")
        if not url_to_parse:
            raise RuntimeError(f"Step 4 failed: Redirection Url not found in response: {data_tp}")

        parsed_url = urlparse(url_to_parse)
        query_params = parse_qs(parsed_url.query)
        if "auth_code" not in query_params:
            raise RuntimeError(f"Step 4 failed: auth_code not found in redirection URL: {url_to_parse}")

        auth_code = query_params["auth_code"][0]
        logger.info("Successfully retrieved auth_code from FYERS.")

        # Step 5: Exchange Auth Code for Access Token
        session_model = fyersModel.SessionModel(
            client_id=config.client_id,
            secret_key=config.secret_key,
            redirect_uri=config.redirect_uri,
            response_type="code",
            grant_type="authorization_code",
        )
        session_model.set_token(auth_code)
        token_response = session_model.generate_token()

        if not isinstance(token_response, dict) or "access_token" not in token_response:
            raise RuntimeError(f"Step 5 failed: Failed to exchange auth_code for token: {token_response}")

        final_token = token_response["access_token"]
        logger.info("Successfully acquired FYERS access token.")

        # Cache the new token
        self.save_cached_token(final_token)
        return final_token

    def get_client(self, force_reauth: bool = False) -> fyersModel.FyersModel:
        """Get an authenticated FyersModel client instance.
        
        Checks cache and validates with get_profile() first;
        if invalid or expired, executes headless re-authentication.
        """
        config = self._get_config()
        token: Optional[str] = None

        if not force_reauth:
            cached = self.load_cached_token()
            if cached:
                # Verify token health
                profile = self.verify_token_health(config.client_id, cached)
                if profile is not None:
                    return fyersModel.FyersModel(
                        client_id=config.client_id,
                        token=cached,
                        is_async=False,
                    )
                logger.info("Cached token health check failed. Re-authenticating...")

        # Authenticate headlessly
        token = self.authenticate_headless()
        return fyersModel.FyersModel(
            client_id=config.client_id,
            token=token,
            is_async=False,
        )


def get_valid_fyers_client(
    cache_path: Optional[Path] = None,
    force_reauth: bool = False,
) -> fyersModel.FyersModel:
    """Convenience helper to retrieve a validated FYERS API v3 client."""
    manager = FyersAuthManager(cache_path=cache_path)
    return manager.get_client(force_reauth=force_reauth)

