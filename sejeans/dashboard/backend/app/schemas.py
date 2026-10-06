from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Motor sub-schema
# ---------------------------------------------------------------------------

class MotorSensorScores(BaseModel):
    current_u: float
    vib_motor: float
    vib_tm: float


class PredictedBBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    predicted_class: str
    probabilities: Dict[str, float]
    has_bbox: bool
    model_available: bool


class MotorResult(BaseModel):
    anomaly_score: float
    threshold: float
    status: str                 # "normal" | "caution" | "warning"
    fault_type: str             # NORMAL | ECC10 | ECC20 | DEMAG | REDUC
    sensor_scores: MotorSensorScores
    spectrogram_url: Optional[str] = None
    localization: Optional[PredictedBBox] = None  # 모델 예측 bbox


# ---------------------------------------------------------------------------
# Battery sub-schema
# ---------------------------------------------------------------------------

class BatteryMetrics(BaseModel):
    min_cell_voltage: float
    voltage_deviation: float
    temperature: Optional[float]
    cell_count: int
    cell_mean_voltages: List[float]


class BatteryResult(BaseModel):
    soh_proxy: float
    status: str                            # "normal" | "caution" | "defect"
    classification: str                    # NORMAL | CAUTION | DEFECT
    probabilities: Dict[str, float]        # {"normal": ..., "caution": ..., "defect": ...}
    fault_type: str                        # "none" | "cell_voltage" | "cell_deviation"
    fault_probabilities: Dict[str, float]  # {"none": ..., "cell_voltage": ..., "cell_deviation": ...}
    faulty_cell_indices: List[int]
    cell_scores: List[float]
    metrics: BatteryMetrics


# ---------------------------------------------------------------------------
# Top-level inference response
# ---------------------------------------------------------------------------

class InferenceResponse(BaseModel):
    timestamp: str
    vehicle_id: str
    car_model: str
    motor: MotorResult
    battery: BatteryResult


# ---------------------------------------------------------------------------
# Alert schema
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    id: str
    timestamp: str
    vehicle_id: str
    car_model: str
    level: str          # "info" | "warning" | "critical"
    component: str      # "motor" | "battery"
    message: str
    fault_type: str


class AlertListResponse(BaseModel):
    alerts: List[Alert]
    total: int


# ---------------------------------------------------------------------------
# Vehicle schema
# ---------------------------------------------------------------------------

class Vehicle(BaseModel):
    vehicle_id: str
    car_model: str
    status: str          # "normal" | "warning" | "critical"
    last_seen: str


class VehicleListResponse(BaseModel):
    vehicles: List[Vehicle]
