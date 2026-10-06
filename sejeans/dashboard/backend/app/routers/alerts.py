"""Alerts router.

GET /api/alerts  – return recent alerts with optional filtering
DELETE /api/alerts  – clear all alerts (utility for testing)
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from app.schemas import AlertListResponse
from app.services import alert_store

router = APIRouter()


@router.get("", response_model=AlertListResponse)
async def get_alerts(
    level: Optional[str] = Query(
        None,
        description="Filter by level: info | warning | critical",
    ),
    component: Optional[str] = Query(
        None,
        description="Filter by component: motor | battery",
    ),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts to return"),
):
    """Return recent alerts, newest first.

    All parameters are optional; omitting them returns all recent alerts up to
    the specified limit.
    """
    alerts = alert_store.get_alerts(level=level, component=component, limit=limit)
    return AlertListResponse(alerts=alerts, total=len(alerts))


@router.delete("")
async def clear_alerts():
    """Clear the in-memory alert store (useful for testing/reset)."""
    alert_store.clear()
    return {"message": "Alert store cleared"}
