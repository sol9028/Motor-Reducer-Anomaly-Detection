"""Inference engine.

Combines real NPY / PNG sample data with CNN models and rule-based logic
to produce InferenceResponse objects.

Motor  : ONNX motor_localizer  → fault type classification + bbox
Battery: ONNX battery_classifier → status + fault type classification
         (falls back to label-derived values if ONNX not available)
"""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.schemas import (
    BatteryMetrics,
    BatteryResult,
    InferenceResponse,
    MotorResult,
    MotorSensorScores,
    PredictedBBox,
)
from app.services.battery_metrics import compute_battery_metrics
from app.services.cell_localizer import localize_faults
from app.services.sample_data_loader import (
    BatterySample,
    MotorSample,
    get_random_battery_sample,
    get_random_motor_sample,
    load_json_label,
    load_npy,
)
from app.services.soh_proxy import compute_soh_proxy

# ---------------------------------------------------------------------------
# Score generation helpers
# ---------------------------------------------------------------------------

# Motor anomaly score ranges per fault type
_MOTOR_SCORE_RANGES: Dict[str, Tuple[float, float]] = {
    "NORMAL": (0.5, 8.0),
    "ECC10":  (12.0, 35.0),
    "ECC20":  (20.0, 55.0),
    "DEMAG":  (30.0, 70.0),
    "REDUC":  (25.0, 65.0),
}

_MOTOR_THRESHOLD = 10.0

# Battery classification probability templates
_BAT_PROBS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "NORMAL":  {"normal": (0.80, 0.98), "caution": (0.01, 0.15), "defect": (0.00, 0.05)},
    "CAUTION": {"normal": (0.05, 0.25), "caution": (0.50, 0.75), "defect": (0.05, 0.30)},
    "DEFECT":  {"normal": (0.00, 0.08), "caution": (0.05, 0.20), "defect": (0.72, 0.95)},
}

# Vehicle ID pool used by the demo
_VEHICLE_IDS = [
    "MOB-X001", "MOB-X002", "MOB-X003",
    "MOB-Y001", "MOB-Y002",
    "MOB-Z001",
]


def _rand_in(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)


def _generate_motor_sensor_scores(fault_type: str) -> MotorSensorScores:
    """Generate per-channel sensor scores (0-100) consistent with fault type."""
    if fault_type == "NORMAL":
        return MotorSensorScores(
            current_u=_rand_in(2, 15),
            vib_motor=_rand_in(2, 15),
            vib_tm=_rand_in(2, 15),
        )
    elif fault_type == "ECC10":
        return MotorSensorScores(
            current_u=_rand_in(30, 55),
            vib_motor=_rand_in(10, 30),
            vib_tm=_rand_in(10, 30),
        )
    elif fault_type == "ECC20":
        return MotorSensorScores(
            current_u=_rand_in(55, 85),
            vib_motor=_rand_in(25, 55),
            vib_tm=_rand_in(25, 55),
        )
    elif fault_type == "DEMAG":
        return MotorSensorScores(
            current_u=_rand_in(40, 70),
            vib_motor=_rand_in(50, 80),
            vib_tm=_rand_in(30, 60),
        )
    else:  # REDUC
        return MotorSensorScores(
            current_u=_rand_in(20, 45),
            vib_motor=_rand_in(40, 70),
            vib_tm=_rand_in(55, 85),
        )


def _status_from_motor_score(score: float) -> str:
    if score < _MOTOR_THRESHOLD:
        return "normal"
    elif score < 30.0:
        return "caution"
    return "warning"


def _battery_status_from_class(classification: str) -> str:
    return {"NORMAL": "normal", "CAUTION": "caution", "DEFECT": "warning"}.get(
        classification.upper(), "normal"
    )


def _generate_probabilities(classification: str) -> Dict[str, float]:
    """Generate realistic status classification probabilities."""
    ranges = _BAT_PROBS.get(classification.upper(), _BAT_PROBS["NORMAL"])
    raw = {k: random.uniform(lo, hi) for k, (lo, hi) in ranges.items()}
    total = sum(raw.values())
    return {k: round(v / total, 4) for k, v in raw.items()}


def _generate_fault_probabilities(bat_type: str, classification: str) -> Dict[str, float]:
    """Generate fallback fault-type probabilities when CNN is unavailable."""
    if classification == "NORMAL":
        return {"none": 1.0, "cell_voltage": 0.0, "cell_deviation": 0.0}
    if bat_type.startswith("Dev"):
        none_p = round(random.uniform(0.01, 0.05), 4)
        cv_p   = round(random.uniform(0.01, 0.10), 4)
        cd_p   = round(1.0 - none_p - cv_p, 4)
    else:
        none_p = round(random.uniform(0.01, 0.05), 4)
        cd_p   = round(random.uniform(0.01, 0.10), 4)
        cv_p   = round(1.0 - none_p - cd_p, 4)
    return {"none": none_p, "cell_voltage": cv_p, "cell_deviation": cd_p}


def _spectrogram_url(motor_sample: MotorSample) -> Optional[str]:
    """Return a URL path for the Current_U spectrogram PNG, or None."""
    from app.services.sample_data_loader import get_sample_root

    png = motor_sample.png_paths.get("Current_U")
    if png is None and motor_sample.png_paths:
        png = next(iter(motor_sample.png_paths.values()))
    if png is None:
        return None
    # Static mount: /static/samples → Sample/ directory
    # So URL should be /static/samples/<path relative to Sample/>
    try:
        sample_root = get_sample_root()
        rel = png.resolve().relative_to(sample_root)
        return "/static/samples/" + rel.as_posix()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core inference builders
# ---------------------------------------------------------------------------

def _model_sensor_scores(motor_sample: MotorSample, primary_result: dict) -> MotorSensorScores:
    """채널별 이상 기여도(0~100)를 계산한다.

    Current_U: primary_result 재사용 (이미 추론 완료, 중복 실행 방지)
    Vib_Motor / Vib_TM: 각 채널 PNG를 모델에 추가 실행
    """
    from app.models.motor_localizer import predict

    def score_from_result(res: dict) -> float:
        if not res.get("model_available"):
            return 0.0
        if res.get("anomaly_score") is not None:
            return round(res["anomaly_score"], 1)
        p_normal = res.get("probabilities", {}).get("NORMAL", 1.0)
        return round((1.0 - p_normal) * 100, 1)

    def channel_score(png_path) -> float:
        if png_path is None:
            return 0.0
        try:
            return score_from_result(predict(str(png_path)))
        except Exception:
            return 0.0

    # Current_U: primary 추론 결과 재사용 (추가 모델 실행 없음)
    cu_score = score_from_result(primary_result)

    return MotorSensorScores(
        current_u=cu_score,
        vib_motor=channel_score(motor_sample.png_paths.get("Vib_Motor")),
        vib_tm=channel_score(motor_sample.png_paths.get("Vib_TM")),
    )


def build_motor_result(motor_sample: MotorSample) -> MotorResult:
    spec_url = _spectrogram_url(motor_sample)

    anomaly_score = None
    fault_type    = None
    sensor_scores = None
    localization  = None

    # ── CNN 모델로 예측 ───────────────────────────────────────────────
    if spec_url is not None:
        try:
            from app.models.motor_localizer import predict
            from app.services.sample_data_loader import get_sample_root
            rel_path = spec_url.removeprefix("/static/samples/")
            png_path = str(get_sample_root() / rel_path)
            result   = predict(png_path)

            if result["model_available"]:
                # 이상 점수: 모델 score_head 출력 우선, 없으면 (1 - P(NORMAL)) * 100
                if result.get("anomaly_score") is not None:
                    anomaly_score = result["anomaly_score"]
                else:
                    p_normal      = result["probabilities"].get("NORMAL", 1.0)
                    anomaly_score = round((1.0 - p_normal) * 100, 1)

                fault_type    = result["predicted_class"]
                sensor_scores = _model_sensor_scores(motor_sample, result)

                if result["has_bbox"]:
                    bbox = result["bbox"]
                    localization = PredictedBBox(
                        x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                        predicted_class=result["predicted_class"],
                        probabilities=result["probabilities"],
                        has_bbox=True,
                        model_available=True,
                    )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("Motor model error: %s", exc)
    # ──────────────────────────────────────────────────────────────────

    # 폴백: 모델 없거나 실패한 경우 라벨 기반
    if anomaly_score is None:
        _ft = motor_sample.fault_type
        lo, hi = _MOTOR_SCORE_RANGES.get(_ft, (5.0, 20.0))
        anomaly_score = _rand_in(lo, hi)
        fault_type    = _ft
        sensor_scores = _generate_motor_sensor_scores(_ft)

    status = _status_from_motor_score(anomaly_score)

    return MotorResult(
        anomaly_score=anomaly_score,
        threshold=_MOTOR_THRESHOLD,
        status=status,
        fault_type=fault_type,
        sensor_scores=sensor_scores,
        spectrogram_url=spec_url,
        localization=localization,
    )


def build_battery_result(battery_sample: BatterySample) -> Optional[BatteryResult]:
    arr = load_npy(battery_sample.npy_path)
    if arr is None or arr.ndim != 2:
        return None

    label = load_json_label(battery_sample.json_path)
    metadata: Dict[str, Any] = label.get("metadata", {})

    # Compute real metrics from raw voltages
    metrics_dict = compute_battery_metrics(arr, metadata)

    # ── CNN 배터리 분류 ────────────────────────────────────────────────
    predicted_fault_type: Optional[str] = None
    probs: Optional[Dict[str, float]] = None
    fault_probs: Optional[Dict[str, float]] = None

    try:
        from app.models.battery_classifier import predict as predict_battery
        bat_pred = predict_battery(arr)
        if bat_pred["model_available"] and bat_pred["status"] != "UNKNOWN":
            classification       = bat_pred["status"]
            predicted_fault_type = bat_pred["fault_type"]
            probs                = bat_pred["status_probabilities"]
            fault_probs          = bat_pred["fault_probabilities"]
        else:
            classification = battery_sample.status.upper()
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).debug("Battery classifier error: %s", exc)
        classification = battery_sample.status.upper()
    # ──────────────────────────────────────────────────────────────────

    if probs is None:
        probs = _generate_probabilities(classification)
    if fault_probs is None:
        fault_probs = _generate_fault_probabilities(battery_sample.bat_type, classification)

    fault_type, faulty_indices, cell_scores = localize_faults(
        arr, battery_sample.bat_type, classification,
        predicted_fault_type=predicted_fault_type,
    )

    soh = compute_soh_proxy(
        voltage_deviation=metrics_dict["voltage_deviation"],
        min_cell_voltage=metrics_dict["min_cell_voltage"],
        classification=classification,
    )

    return BatteryResult(
        soh_proxy=soh,
        status=_battery_status_from_class(classification),
        classification=classification,
        probabilities=probs,
        fault_type=fault_type,
        fault_probabilities=fault_probs,
        faulty_cell_indices=faulty_indices,
        cell_scores=cell_scores,
        metrics=BatteryMetrics(**metrics_dict),
    )


def build_inference_response(
    car_model: Optional[str] = None,
    motor_fault: Optional[str] = None,
    battery_status: Optional[str] = None,
) -> Optional[InferenceResponse]:
    """Build a full InferenceResponse using random samples from the index.

    All three optional filters can be used to bias sample selection.
    """
    motor_sample = get_random_motor_sample(car_model=car_model, fault_type=motor_fault)
    battery_sample = get_random_battery_sample(car_model=car_model, status=battery_status)

    if motor_sample is None or battery_sample is None:
        return None

    motor_result = build_motor_result(motor_sample)
    battery_result = build_battery_result(battery_sample)
    if battery_result is None:
        return None

    # Derive car_model from the motor sample (always present)
    resolved_car_model = motor_sample.car_model

    return InferenceResponse(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        vehicle_id=random.choice(_VEHICLE_IDS),
        car_model=resolved_car_model,
        motor=motor_result,
        battery=battery_result,
    )
