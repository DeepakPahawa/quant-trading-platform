"""Unit tests for FYERS headless auth logic, token caching, and health verification."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.app.auth.fyers_auth import FyersAuthConfig, FyersAuthManager, get_valid_fyers_client


def test_auth_config_from_env_missing():
    """Verify that missing environment variables raise a ValueError."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="Missing required FYERS environment variables"):
            FyersAuthConfig.from_env()


def test_auth_config_from_env_valid():
    """Verify valid environment parsing and user_id inference."""
    env = {
        "FYERS_CLIENT_ID": "USER123-100",
        "FYERS_SECRET_KEY": "secret_abc",
        "FYERS_PIN": "1234",
        "FYERS_TOTP_KEY": "JBSWY3DPEHPK3PXP",
    }
    with patch.dict("os.environ", env, clear=True):
        config = FyersAuthConfig.from_env()
        assert config.client_id == "USER123-100"
        assert config.secret_key == "secret_abc"
        assert config.pin == "1234"
        assert config.totp_key == "JBSWY3DPEHPK3PXP"
        assert config.user_id == "USER123"


def test_token_cache_save_and_load(tmp_path: Path):
    """Verify saving to cache and loading valid unexpired token."""
    cache_file = tmp_path / ".token_cache.json"
    manager = FyersAuthManager(cache_path=cache_file)

    # Initial load: no file
    assert manager.load_cached_token() is None

    # Save token
    manager.save_cached_token("test_access_token_123", expires_in_seconds=3600)
    assert cache_file.exists()

    # Load valid token
    loaded_token = manager.load_cached_token()
    assert loaded_token == "test_access_token_123"


def test_token_cache_expired(tmp_path: Path):
    """Verify that an expired cached token is ignored."""
    cache_file = tmp_path / ".token_cache.json"
    manager = FyersAuthManager(cache_path=cache_file)

    # Expired token in cache (expires_at in past)
    with open(cache_file, "w") as f:
        json.dump({
            "access_token": "expired_token",
            "cached_at": time.time() - 100000,
            "expires_at": time.time() - 50000,
        }, f)

    assert manager.load_cached_token() is None


def test_token_health_check_valid():
    """Verify token validation succeeds with ok response."""
    manager = FyersAuthManager()
    with patch("backend.app.auth.fyers_auth.fyersModel.FyersModel") as mock_fm_cls:
        mock_fm = MagicMock()
        mock_fm.get_profile.return_value = {"s": "ok", "code": 200, "data": {"fy_id": "AB12345"}}
        mock_fm_cls.return_value = mock_fm

        res = manager.verify_token_health("CLIENT-100", "some_token")
        assert res is not None
        assert res["s"] == "ok"


def test_token_health_check_invalid():
    """Verify token validation fails when get_profile returns error."""
    manager = FyersAuthManager()
    with patch("backend.app.auth.fyers_auth.fyersModel.FyersModel") as mock_fm_cls:
        mock_fm = MagicMock()
        mock_fm.get_profile.return_value = {"s": "error", "code": 401, "message": "Token expired"}
        mock_fm_cls.return_value = mock_fm

        res = manager.verify_token_health("CLIENT-100", "bad_token")
        assert res is None

