"""FastAPI application entry point.

Startup sequence
----------------
1. Build the in-memory sample data index (scans the sample data directory).
2. Mount static files so motor spectrogram PNGs are accessible to the frontend.
3. Register all API routers under /api and the WebSocket router under /ws.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import alerts, inference, vehicles, websocket
from app.services.sample_data_loader import build_index, get_index_stats, get_sample_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated on_event startup/shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the sample index at startup."""
    logger.info("Starting up – building sample data index …")
    build_index()
    stats = get_index_stats()
    logger.info(
        "Index ready: %d battery samples, %d motor samples",
        stats["battery_total"],
        stats["motor_total"],
    )
    yield
    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EV Anomaly Detection Dashboard API",
    description=(
        "Backend API for the autonomous EV motor-reducer and battery "
        "anomaly detection dashboard. Uses real sample data from IONIQ, "
        "KONA, and NIRO vehicles."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS – allow all origins for development (restrict in production)
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files – serve motor spectrogram PNGs
# ---------------------------------------------------------------------------

_sample_root = get_sample_root()
if _sample_root.exists():
    # Mount the entire 샘플데이터 directory tree so frontend can display
    # spectrogram images at /static/samples/<relative-path>
    _static_root = _sample_root.parent  # 샘플데이터/
    app.mount(
        "/static/samples",
        StaticFiles(directory=str(_sample_root)),
        name="samples",
    )
    logger.info("Static files mounted at /static/samples → %s", _sample_root)
else:
    logger.warning("Sample root not found, static files not mounted: %s", _sample_root)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

# Inference endpoints (POST upload + GET demo)
app.include_router(
    inference.router,
    prefix="/api/inference",
    tags=["Inference"],
)
app.include_router(
    inference.router,
    prefix="/api/demo",
    tags=["Demo"],
)

# Alerts
app.include_router(
    alerts.router,
    prefix="/api/alerts",
    tags=["Alerts"],
)

# Vehicles
app.include_router(
    vehicles.router,
    prefix="/api/vehicles",
    tags=["Vehicles"],
)

# WebSocket
app.include_router(
    websocket.router,
    prefix="/ws",
    tags=["WebSocket"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Return service health and sample data index statistics."""
    stats = get_index_stats()
    return {
        "status": "ok",
        "index": stats,
    }


@app.get("/", tags=["System"])
async def root() -> Dict[str, str]:
    return {
        "message": "EV Anomaly Detection Dashboard API",
        "docs": "/docs",
        "health": "/health",
    }
