"""Battery classifier — ONNX Runtime inference wrapper.

Loads battery_classifier.onnx (trained by scripts/train_battery_classifier.py)
and exposes a single `predict(npy_array)` function that returns:
  - status        : str   "NORMAL" | "CAUTION" | "DEFECT"
  - fault_type    : str   "none" | "cell_voltage" | "cell_deviation"
  - status_probabilities : dict  {cls: prob}
  - fault_probabilities  : dict  {cls: prob}
  - model_available      : bool
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── 경로 ───────────────────────────────────────────────────────────────
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "weights"
_ONNX_PATH   = _WEIGHTS_DIR / "battery_classifier.onnx"
_META_PATH   = _WEIGHTS_DIR / "battery_classifier_meta.json"

# ── 기본값 (메타 파일이 없을 때) ──────────────────────────────────────
_STATUS_CLASSES  = ["CAUTION", "DEFECT", "NORMAL"]
_FAULT_CLASSES   = ["cell_deviation", "cell_voltage", "none"]
_FIXED_TIMESTEPS = 20
_FIXED_CELLS     = 98
_V_MIN           = 3.0
_V_MAX           = 4.2

# ── 모듈 상태 ─────────────────────────────────────────────────────────
_session = None
_loaded  = False


def _try_load() -> bool:
    global _session, _STATUS_CLASSES, _FAULT_CLASSES
    global _FIXED_TIMESTEPS, _FIXED_CELLS, _V_MIN, _V_MAX, _loaded

    if _loaded:
        return True
    if not _ONNX_PATH.exists():
        logger.info(
            "battery_classifier.onnx not found – battery CNN disabled. "
            "Run scripts/train_battery_classifier.py to enable it."
        )
        return False

    try:
        import onnxruntime as ort
        _session = ort.InferenceSession(
            str(_ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )
        if _META_PATH.exists():
            with open(_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            _STATUS_CLASSES  = meta["status_classes"]
            _FAULT_CLASSES   = meta["fault_classes"]
            _FIXED_TIMESTEPS = meta.get("fixed_timesteps", 20)
            _FIXED_CELLS     = meta.get("fixed_cells", 98)
            _V_MIN           = meta.get("v_min", 3.0)
            _V_MAX           = meta.get("v_max", 4.2)

        _loaded = True
        logger.info(
            "Battery classifier loaded: %s | status=%s | fault=%s",
            _ONNX_PATH.name, _STATUS_CLASSES, _FAULT_CLASSES,
        )
        return True

    except Exception as exc:
        logger.warning("Failed to load battery classifier: %s", exc)
        return False


def _preprocess(npy_array: np.ndarray) -> Optional[np.ndarray]:
    """(T, N) 배열 → (1, 1, FIXED_TIMESTEPS, FIXED_CELLS) float32."""
    try:
        arr = npy_array.astype(np.float32)
        T, N = arr.shape

        # 시간 축 패딩/잘라내기
        if T > _FIXED_TIMESTEPS:
            arr = arr[:_FIXED_TIMESTEPS]
        elif T < _FIXED_TIMESTEPS:
            pad = np.zeros((_FIXED_TIMESTEPS - T, N), dtype=np.float32)
            arr = np.vstack([arr, pad])

        # 셀 축 패딩/잘라내기
        if N < _FIXED_CELLS:
            pad = np.zeros((_FIXED_TIMESTEPS, _FIXED_CELLS - N), dtype=np.float32)
            arr = np.hstack([arr, pad])
        elif N > _FIXED_CELLS:
            arr = arr[:, :_FIXED_CELLS]

        # 전압 [V_MIN, V_MAX] → [0, 1]
        arr = (arr - _V_MIN) / (_V_MAX - _V_MIN)
        arr = np.clip(arr, 0.0, 1.0)

        return arr[np.newaxis, np.newaxis]   # (1, 1, 20, 98)

    except Exception as exc:
        logger.error("Battery preprocess failed: %s", exc)
        return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    exp_l = np.exp(logits - logits.max())
    return exp_l / exp_l.sum()


def predict(npy_array: np.ndarray) -> Dict:
    """CNN으로 배터리 상태와 결함 유형을 분류한다.

    Parameters
    ----------
    npy_array : shape (T, N) — 배터리 셀 전압 행렬

    Returns
    -------
    {
        "model_available"     : bool,
        "status"              : str,   # NORMAL | CAUTION | DEFECT
        "fault_type"          : str,   # none | cell_voltage | cell_deviation
        "status_probabilities": {cls: float},
        "fault_probabilities" : {cls: float},
    }
    """
    if not _try_load():
        return {
            "model_available":      False,
            "status":               "UNKNOWN",
            "fault_type":           "none",
            "status_probabilities": {},
            "fault_probabilities":  {},
        }

    tensor = _preprocess(npy_array)
    if tensor is None:
        return {
            "model_available":      True,
            "status":               "UNKNOWN",
            "fault_type":           "none",
            "status_probabilities": {},
            "fault_probabilities":  {},
        }

    s_logits, f_logits = _session.run(None, {"voltage_array": tensor})

    s_probs = _softmax(s_logits[0])
    f_probs = _softmax(f_logits[0])

    s_idx = int(np.argmax(s_probs))
    f_idx = int(np.argmax(f_probs))

    predicted_status     = _STATUS_CLASSES[s_idx] if s_idx < len(_STATUS_CLASSES) else "NORMAL"
    predicted_fault_type = _FAULT_CLASSES[f_idx]  if f_idx < len(_FAULT_CLASSES)  else "none"

    # NORMAL로 예측된 경우 결함 유형은 항상 none으로 강제
    if predicted_status == "NORMAL":
        predicted_fault_type = "none"

    # 프론트엔드가 소문자 키를 기대하므로 소문자로 변환
    s_prob_dict = {_STATUS_CLASSES[i].lower(): round(float(s_probs[i]), 4) for i in range(len(_STATUS_CLASSES))}
    f_prob_dict = {_FAULT_CLASSES[i]:          round(float(f_probs[i]), 4) for i in range(len(_FAULT_CLASSES))}

    return {
        "model_available":      True,
        "status":               predicted_status,
        "fault_type":           predicted_fault_type,
        "status_probabilities": s_prob_dict,
        "fault_probabilities":  f_prob_dict,
    }
