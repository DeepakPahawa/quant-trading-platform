"""Headless verification probe for FYERS API v3 authentication."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.auth.fyers_auth import get_valid_fyers_client


def run_probe() -> int:
    """Run verification probe and assert a non-null client object."""
    print("Executing FYERS auth probe...")

    has_real_credentials = all(
        os.getenv(k) for k in ["FYERS_CLIENT_ID", "FYERS_SECRET_KEY", "FYERS_PIN", "FYERS_TOTP_KEY"]
    )

    if has_real_credentials:
        print("Detected live FYERS credentials in environment. Executing live authentication...")
        client = get_valid_fyers_client()
        assert client is not None, "Live client object must not be None"
        print("Live client successfully instantiated and validated.")
        return 0

    print("No live FYERS credentials found; executing simulated headless auth flow probe...")
    test_env = {
        "FYERS_CLIENT_ID": "TEST12345-100",
        "FYERS_SECRET_KEY": "test_secret_key",
        "FYERS_PIN": "1234",
        "FYERS_TOTP_KEY": "JBSWY3DPEHPK3PXP",
        "FYERS_REDIRECT_URI": "https://trade.fyers.in/api-login/redirect-uri/index.html",
        "FYERS_USER_ID": "TEST12345",
    }

    with patch.dict(os.environ, test_env):
        with patch("requests.Session") as mock_session_cls, \
             patch("backend.app.auth.fyers_auth.fyersModel.SessionModel") as mock_sm_cls, \
             patch("backend.app.auth.fyers_auth.fyersModel.FyersModel") as mock_fm_cls:

            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session

            # Step 1: send_login_otp_v2
            mock_resp_otp = MagicMock(status_code=200)
            mock_resp_otp.json.return_value = {"s": "ok", "request_key": "req_key_step1"}

            # Step 2: verify_otp
            mock_resp_totp = MagicMock(status_code=200)
            mock_resp_totp.json.return_value = {"s": "ok", "request_key": "req_key_step2"}

            # Step 3: verify_pin
            mock_resp_pin = MagicMock(status_code=200)
            mock_resp_pin.json.return_value = {
                "s": "ok",
                "data": {"access_token": "vagator_bearer_token_xyz"}
            }

            # Step 4: thirdparty_code
            mock_resp_tp = MagicMock(status_code=200)
            mock_resp_tp.json.return_value = {
                "s": "ok",
                "Url": "https://trade.fyers.in/api-login/redirect-uri/index.html?auth_code=test_auth_code_123"
            }

            mock_session.post.side_effect = [
                mock_resp_otp,
                mock_resp_totp,
                mock_resp_pin,
                mock_resp_tp,
            ]

            # Step 5: SessionModel token exchange
            mock_sm = MagicMock()
            mock_sm.generate_token.return_value = {
                "s": "ok",
                "code": 200,
                "access_token": "fyers_test_access_token_999"
            }
            mock_sm_cls.return_value = mock_sm

            # Mock client returned by FyersModel
            mock_client_instance = MagicMock()
            mock_client_instance.get_profile.return_value = {
                "s": "ok",
                "code": 200,
                "data": {"fy_id": "TEST12345", "name": "Quant Tester"}
            }
            mock_fm_cls.return_value = mock_client_instance

            test_cache_path = PROJECT_ROOT / ".test_token_cache.json"
            if test_cache_path.exists():
                test_cache_path.unlink()

            try:
                # 1. Force re-auth to test the 5-step headless pipeline
                client = get_valid_fyers_client(cache_path=test_cache_path, force_reauth=True)
                assert client is not None, "Assert failed: client is None"
                print(f"Step 1 check passed: non-null client obtained -> {client}")

                # 2. Check cache creation
                assert test_cache_path.exists(), "Assert failed: cache file was not created"
                print("Step 2 check passed: token cache file created successfully.")

                # 3. Test token re-use from cache with health verification
                cached_client = get_valid_fyers_client(cache_path=test_cache_path)
                assert cached_client is not None, "Assert failed: cached client is None"
                print(f"Step 3 check passed: cached client loaded and validated -> {cached_client}")

            finally:
                if test_cache_path.exists():
                    test_cache_path.unlink()

    print("Verification complete: FYERS auth probe succeeded with exit code 0.")
    return 0


if __name__ == "__main__":
    sys.exit(run_probe())

