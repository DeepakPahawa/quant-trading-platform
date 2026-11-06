"""FastAPI Main Application for Quant Trading Platform."""

import logging
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.app.auth.fyers_auth import get_valid_fyers_client
from backend.app.routers.market import router as market_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Quant Trading Platform - FYERS Data Hub API",
    version="1.0.0",
    description="High-performance backend for quant trading, data streaming, and automated execution.",
)

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market_router)


class HealthResponse(BaseModel):
    """Pydantic schema for /api/health endpoint response."""
    status: str = Field(..., description="Service status ('ok', 'unauthenticated', 'error')")
    authenticated: bool = Field(..., description="Whether FYERS client is successfully authenticated")
    profile: Optional[Dict[str, Any]] = Field(default=None, description="FYERS profile metadata if authenticated")
    error: Optional[str] = Field(default=None, description="Error message if unauthenticated")
    timestamp: float = Field(default_factory=time.time, description="Current epoch timestamp")


@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint exposing broker authentication status and profile metadata."""
    try:
        client = get_valid_fyers_client()
        profile_response = client.get_profile()

        if isinstance(profile_response, dict) and (
            profile_response.get("s") == "ok" or profile_response.get("code") == 200
        ):
            return HealthResponse(
                status="ok",
                authenticated=True,
                profile=profile_response.get("data") or profile_response,
                timestamp=time.time(),
            )

        return HealthResponse(
            status="unauthenticated",
            authenticated=False,
            profile=None,
            error=str(profile_response),
            timestamp=time.time(),
        )
    except Exception as e:
        logger.warning("Health check detected unauthenticated state: %s", e)
        return HealthResponse(
            status="unauthenticated",
            authenticated=False,
            profile=None,
            error=str(e),
            timestamp=time.time(),
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)

