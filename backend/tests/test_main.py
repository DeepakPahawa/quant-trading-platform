"""Unit tests for FastAPI endpoints and health probe."""

import pytest
from unittest.mock import MagicMock, patch
from starlette.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_cors_headers(client):
    """Verify CORS headers for frontend origin http://localhost:5173."""
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_health_check_authenticated(client):
    """Test /api/health when FYERS client is successfully authenticated."""
    mock_fyers = MagicMock()
    mock_fyers.get_profile.return_value = {
        "s": "ok",
        "code": 200,
        "data": {
            "fy_id": "AB12345",
            "name": "Quant Trader",
            "email_id": "trader@quant.local",
        },
    }

    with patch("backend.app.main.get_valid_fyers_client", return_value=mock_fyers):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["authenticated"] is True
        assert data["profile"]["fy_id"] == "AB12345"
        assert "timestamp" in data


def test_health_check_unauthenticated(client):
    """Test /api/health when FYERS credentials are not configured or auth fails."""
    with patch("backend.app.main.get_valid_fyers_client", side_effect=ValueError("Missing required FYERS environment variables")):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unauthenticated"
        assert data["authenticated"] is False
        assert data["profile"] is None
        assert "Missing required FYERS" in data["error"]

