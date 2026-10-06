"""In-memory alert store.

Alerts are auto-generated from InferenceResponse objects and kept in a
bounded deque so the store does not grow indefinitely during long demo
sessions.
"""
from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Deque, List, Optional

from app.schemas import Alert, InferenceResponse

_MAX_ALERTS = 200

_store: Deque[Alert] = deque(maxlen=_MAX_ALERTS)


def _level_from_status(status: str) -> str:
    return {"normal": "info", "caution": "warning", "warning": "warning",
            "defect": "critical", "critical": "critical"}.get(status.lower(), "info")


def ingest_inference(response: InferenceResponse) -> None:
    """Create alert(s) from an inference result and add them to the store.

    Normal-status results are stored as info-level so the alert timeline
    remains populated even during healthy operation.
    """
    ts = response.timestamp

    # Motor alert
    motor = response.motor
    motor_level = _level_from_status(motor.status)
    if motor.fault_type != "NORMAL" or motor.anomaly_score > 5:
        motor_msg = (
            f"Motor anomaly detected: {motor.fault_type} "
            f"(score={motor.anomaly_score:.1f})"
            if motor.fault_type != "NORMAL"
            else f"Motor operating normally (score={motor.anomaly_score:.1f})"
        )
        _store.append(Alert(
            id=str(uuid.uuid4()),
            timestamp=ts,
            vehicle_id=response.vehicle_id,
            car_model=response.car_model,
            level=motor_level,
            component="motor",
            message=motor_msg,
            fault_type=motor.fault_type,
        ))

    # Battery alert
    bat = response.battery
    bat_level = _level_from_status(bat.status)
    if bat.classification != "NORMAL" or bat.soh_proxy < 90:
        bat_msg = (
            f"Battery {bat.classification}: SOH={bat.soh_proxy:.0f}%, "
            f"{len(bat.faulty_cell_indices)} fault cell(s)"
            if bat.classification != "NORMAL"
            else f"Battery healthy: SOH={bat.soh_proxy:.0f}%"
        )
        _store.append(Alert(
            id=str(uuid.uuid4()),
            timestamp=ts,
            vehicle_id=response.vehicle_id,
            car_model=response.car_model,
            level=bat_level,
            component="battery",
            message=bat_msg,
            fault_type=bat.classification,
        ))


def get_alerts(
    level: Optional[str] = None,
    component: Optional[str] = None,
    limit: int = 50,
) -> List[Alert]:
    """Return recent alerts, newest first, with optional filters."""
    alerts = list(reversed(_store))
    if level:
        alerts = [a for a in alerts if a.level == level.lower()]
    if component:
        alerts = [a for a in alerts if a.component == component.lower()]
    return alerts[:limit]


def clear() -> None:
    _store.clear()
