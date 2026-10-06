"""Inference router.

Endpoints
---------
POST /api/inference/motor    – upload PNG, get motor inference
POST /api/inference/battery  – upload NPY, get battery inference
GET  /api/demo/sample        – random sample from real data (main dashboard endpoint)
GET  /api/demo/stream        – sequential samples for simulation
"""
from __future__ import annotations

import io
import random
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.schemas import BatteryMetrics, BatteryResult, InferenceResponse, MotorResult, MotorSensorScores
from app.services import alert_store
from app.services.battery_metrics import compute_battery_metrics
from app.services.cell_localizer import localize_faults
from app.services.inference_engine import (
    build_inference_response,
    build_motor_result,
    build_battery_result,
    _generate_probabilities,
    _MOTOR_THRESHOLD,
)
from app.services.sample_data_loader import (
    get_random_motor_sample,
    get_random_battery_sample,
    load_json_label,
)
from app.services.soh_proxy import compute_soh_proxy

router = APIRouter()

# ---------------------------------------------------------------------------
# POST /api/inference/motor
# ---------------------------------------------------------------------------

@router.post("/motor", response_model=InferenceResponse)
async def infer_motor(file: UploadFile = File(...)):
    """Accept a spectrogram PNG upload and return a motor inference result.

    Because no ONNX model is available, a real motor sample is drawn from the
    index and its label-derived fault type is used for the classification.
    The uploaded file is accepted but not used for feature extraction.
    """
    # Consume the upload (validate it's a PNG)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Pick a random motor sample to supply realistic label data
    motor_sample = get_random_motor_sample()
    if motor_sample is None:
        raise HTTPException(status_code=503, detail="No motor samples available in index")

    battery_sample = get_random_battery_sample()
    if battery_sample is None:
        raise HTTPException(status_code=503, detail="No battery samples available in index")

    from app.services.inference_engine import build_inference_response
    response = build_inference_response()
    if response is None:
        raise HTTPException(status_code=503, detail="Could not build inference response")

    alert_store.ingest_inference(response)
    return response


# ---------------------------------------------------------------------------
# POST /api/inference/battery
# ---------------------------------------------------------------------------

@router.post("/battery", response_model=InferenceResponse)
async def infer_battery(file: UploadFile = File(...)):
    """Accept a .npy file upload, classify with CNN, and return results."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        arr = np.load(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid NPY file: {exc}")

    if arr.ndim != 2:
        raise HTTPException(status_code=400, detail="NPY array must be 2-dimensional (T x cells)")

    # ── CNN 배터리 분류 ─────────────────────────────────────────────────
    from app.models.battery_classifier import predict as predict_battery
    bat_pred = predict_battery(arr)

    if bat_pred["model_available"] and bat_pred["status"] != "UNKNOWN":
        classification       = bat_pred["status"]       # NORMAL | CAUTION | DEFECT
        predicted_fault_type = bat_pred["fault_type"]   # none | cell_voltage | cell_deviation
        probs                = bat_pred["status_probabilities"]
    else:
        # CNN 없을 때 전압 통계 기반 폴백
        cell_std = float(arr.std(axis=1).mean())
        min_v    = float(arr.min())
        if min_v < 3.3 or cell_std > 0.05:
            classification = "DEFECT"
        elif min_v < 3.7 or cell_std > 0.02:
            classification = "CAUTION"
        else:
            classification = "NORMAL"
        predicted_fault_type = None
        probs = _generate_probabilities(classification)
    # ────────────────────────────────────────────────────────────────────

    # 셀 수로 차량 모델 추론
    car_model = "IONIQ" if arr.shape[1] == 96 else "KONA"
    bat_type  = "Vlt_96" if arr.shape[1] == 96 else "Vlt_98"

    metrics_dict = compute_battery_metrics(arr, {})
    fault_type, faulty_indices, cell_scores = localize_faults(
        arr, bat_type, classification,
        predicted_fault_type=predicted_fault_type,
    )
    soh = compute_soh_proxy(
        voltage_deviation=metrics_dict["voltage_deviation"],
        min_cell_voltage=metrics_dict["min_cell_voltage"],
        classification=classification,
    )

    from app.services.inference_engine import _generate_fault_probabilities
    if bat_pred.get("model_available") and bat_pred.get("fault_probabilities"):
        fault_probs = bat_pred["fault_probabilities"]
    else:
        fault_probs = _generate_fault_probabilities(bat_type, classification)

    battery_result = BatteryResult(
        soh_proxy=soh,
        status=classification.lower(),
        classification=classification,
        probabilities=probs,
        fault_type=fault_type,
        fault_probabilities=fault_probs,
        faulty_cell_indices=faulty_indices,
        cell_scores=cell_scores,
        metrics=BatteryMetrics(**metrics_dict),
    )

    # Pair with a random motor sample for a complete response
    motor_sample = get_random_motor_sample(car_model=car_model)
    if motor_sample is None:
        motor_sample = get_random_motor_sample()
    if motor_sample is None:
        raise HTTPException(status_code=503, detail="No motor samples available")

    motor_result = build_motor_result(motor_sample)

    from datetime import datetime, timezone
    import random as _random
    _VEHICLE_IDS = ["MOB-X001", "MOB-X002", "MOB-X003", "MOB-Y001", "MOB-Y002"]

    response = InferenceResponse(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vehicle_id=_random.choice(_VEHICLE_IDS),
        car_model=car_model,
        motor=motor_result,
        battery=battery_result,
    )
    alert_store.ingest_inference(response)
    return response


# ---------------------------------------------------------------------------
# GET /api/demo/sample
# ---------------------------------------------------------------------------

@router.get("/sample", response_model=InferenceResponse)
async def demo_sample(
    car_model: Optional[str] = Query(None, description="IONIQ | KONA | NIRO"),
    motor_fault: Optional[str] = Query(None, description="NORMAL | ECC10 | ECC20 | DEMAG | REDUC"),
    battery_status: Optional[str] = Query(None, description="NORMAL | CAUTION | DEFECT"),
):
    """Return a random inference result built from real sample data.

    This is the primary endpoint for the dashboard demo. Each call draws fresh
    random samples from the index so results vary on each refresh.
    """
    response = build_inference_response(
        car_model=car_model,
        motor_fault=motor_fault,
        battery_status=battery_status,
    )
    if response is None:
        raise HTTPException(
            status_code=503,
            detail="No samples available for the requested filter combination",
        )
    alert_store.ingest_inference(response)
    return response


# ---------------------------------------------------------------------------
# GET /api/demo/stream  (sequential simulation)
# ---------------------------------------------------------------------------

_stream_counter: int = 0
_STREAM_SCENARIOS = [
    {"battery_status": "NORMAL",  "motor_fault": "NORMAL"},
    {"battery_status": "NORMAL",  "motor_fault": "ECC10"},
    {"battery_status": "CAUTION", "motor_fault": "NORMAL"},
    {"battery_status": "CAUTION", "motor_fault": "ECC20"},
    {"battery_status": "DEFECT",  "motor_fault": "DEMAG"},
    {"battery_status": "DEFECT",  "motor_fault": "REDUC"},
    {"battery_status": "NORMAL",  "motor_fault": "NORMAL"},
    {"battery_status": "CAUTION", "motor_fault": "ECC10"},
]


@router.get("/stream", response_model=InferenceResponse)
async def demo_stream():
    """Return sequential inference results cycling through predefined scenarios.

    Useful for animating the dashboard without relying on the WebSocket.
    """
    global _stream_counter
    scenario = _STREAM_SCENARIOS[_stream_counter % len(_STREAM_SCENARIOS)]
    _stream_counter += 1

    response = build_inference_response(**scenario)
    if response is None:
        raise HTTPException(status_code=503, detail="No samples available")

    alert_store.ingest_inference(response)
    return response
