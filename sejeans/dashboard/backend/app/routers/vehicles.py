"""Vehicles router.

GET /api/vehicles  – return the list of monitored vehicles with current status
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Query

from app.schemas import Vehicle, VehicleListResponse

router = APIRouter()

# Static fleet definition – in a real system this would come from a database
_FLEET: List[Vehicle] = [
    Vehicle(
        vehicle_id="MOB-X001",
        car_model="IONIQ",
        status="normal",
        last_seen=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ),
    Vehicle(
        vehicle_id="MOB-X002",
        car_model="IONIQ",
        status="warning",
        last_seen=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ),
    Vehicle(
        vehicle_id="MOB-X003",
        car_model="IONIQ",
        status="critical",
        last_seen=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ),
    Vehicle(
        vehicle_id="MOB-Y001",
        car_model="KONA",
        status="normal",
        last_seen=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ),
    Vehicle(
        vehicle_id="MOB-Y002",
        car_model="KONA",
        status="normal",
        last_seen=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ),
    Vehicle(
        vehicle_id="MOB-Z001",
        car_model="NIRO",
        status="warning",
        last_seen=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    ),
]


def _refresh_timestamps() -> List[Vehicle]:
    """Return the fleet with freshly randomised last_seen timestamps."""
    now = datetime.now(timezone.utc)
    refreshed = []
    for v in _FLEET:
        # Simulate vehicles last seen between 0 and 5 minutes ago
        offset_seconds = random.randint(0, 300)
        last_seen = (now - timedelta(seconds=offset_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
        refreshed.append(Vehicle(
            vehicle_id=v.vehicle_id,
            car_model=v.car_model,
            status=v.status,
            last_seen=last_seen,
        ))
    return refreshed


@router.get("", response_model=VehicleListResponse)
async def list_vehicles(
    car_model: Optional[str] = Query(None, description="Filter by model: IONIQ | KONA | NIRO"),
    status: Optional[str] = Query(None, description="Filter by status: normal | warning | critical"),
):
    """Return the list of vehicles in the monitored fleet."""
    vehicles = _refresh_timestamps()
    if car_model:
        vehicles = [v for v in vehicles if v.car_model.upper() == car_model.upper()]
    if status:
        vehicles = [v for v in vehicles if v.status.lower() == status.lower()]
    return VehicleListResponse(vehicles=vehicles)
