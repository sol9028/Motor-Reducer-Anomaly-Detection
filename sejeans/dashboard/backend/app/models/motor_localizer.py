"""Motor-reducer fault localizer — ONNX Runtime inference wrapper.

Loads motor_localizer.onnx (trained by scripts/train_motor_localizer.py)
and exposes a single `predict(image_path)` function that returns:
  - predicted_class : str   e.g. "ECC20"
  - probabilities   : dict  e.g. {"DEMAG": 0.02, "ECC10": 0.05, ...}
  - bbox            : list[float]  [x1, y1, x2, y2] normalised 0-1
                      (all zeros when class == NORMAL or model not loaded)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "weights"
_ONNX_PATH   = _WEIGHTS_DIR / "motor_localizer.onnx"
_META_PATH   = _WEIGHTS_DIR / "motor_localizer_meta.json"

# ── module-level state ─────────────────────────────────────────────────
_session         = None   # onnxruntime.InferenceSession
_class_names: List[str] = []
_img_size        = 224
_mean            = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_std             = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_loaded          = False
_has_score_head  = False  # True if ONNX model has 3 outputs (logits, bbox, anomaly_score)


def _try_load() -> bool:
    """Load the ONNX model once. Returns True if successful."""
    global _session, _class_names, _img_size, _loaded, _has_score_head

    if _loaded:
        return True
    if not _ONNX_PATH.exists():
        logger.info("motor_localizer.onnx not found – localizer disabled. "
                    "Run scripts/train_motor_localizer.py to enable it.")
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
            _class_names = meta["class_names"]
            _img_size    = meta.get("img_size", 224)
            _mean        = np.array(meta["mean"], dtype=np.float32)
            _std         = np.array(meta["std"],  dtype=np.float32)
        else:
            _class_names = ["DEMAG", "ECC10", "ECC20", "NORMAL", "REDUC"]

        # ONNX 출력 개수로 score_head 존재 여부 확인
        num_outputs     = len(_session.get_outputs())
        _has_score_head = num_outputs >= 3
        logger.info(
            "Motor localizer loaded: %s | classes: %s | score_head: %s",
            _ONNX_PATH.name, _class_names, _has_score_head,
        )

        _loaded = True
        return True

    except Exception as exc:
        logger.warning("Failed to load motor localizer: %s", exc)
        return False


def _preprocess(image_path: str) -> Optional[np.ndarray]:
    """Load a PNG and return a (1, 3, H, W) float32 tensor."""
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((_img_size, _img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)
        arr = (arr - _mean) / _std                       # normalise
        arr = arr.transpose(2, 0, 1)[np.newaxis]         # (1, 3, H, W)
        return arr.astype(np.float32)
    except Exception as exc:
        logger.error("Image preprocess failed (%s): %s", image_path, exc)
        return None


def predict(image_path: str) -> Dict:
    """Run the localizer on a spectrogram PNG.

    Returns a dict:
    {
      "model_available": bool,
      "predicted_class": str,          # e.g. "ECC20"
      "probabilities": {cls: prob},
      "bbox": [x1, y1, x2, y2],       # 0-1 normalised; [0,0,0,0] for NORMAL
      "has_bbox": bool,
      "anomaly_score": float | None,   # 0~100; None if model has no score_head
    }
    """
    if not _try_load():
        return {
            "model_available": False,
            "predicted_class": "UNKNOWN",
            "probabilities": {},
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "has_bbox": False,
            "anomaly_score": None,
        }

    tensor = _preprocess(image_path)
    if tensor is None:
        return {
            "model_available": True,
            "predicted_class": "UNKNOWN",
            "probabilities": {},
            "bbox": [0.0, 0.0, 0.0, 0.0],
            "has_bbox": False,
            "anomaly_score": None,
        }

    outputs   = _session.run(None, {"image": tensor})
    logits    = outputs[0]
    bbox_raw  = outputs[1]

    # anomaly_score: score_head 출력(sigmoid) × 100 → 0~100
    anomaly_score: Optional[float] = None
    if _has_score_head and len(outputs) >= 3:
        score_raw     = outputs[2]
        anomaly_score = round(float(score_raw[0][0]) * 100, 1)

    # softmax
    exp_l = np.exp(logits[0] - logits[0].max())
    probs = exp_l / exp_l.sum()

    pred_idx   = int(np.argmax(probs))
    pred_class = _class_names[pred_idx] if pred_idx < len(_class_names) else "UNKNOWN"
    prob_dict  = {_class_names[i]: round(float(probs[i]), 4)
                  for i in range(len(_class_names))}
    bbox       = [round(float(v), 4) for v in bbox_raw[0]]
    has_bbox   = pred_class != "NORMAL"

    if not has_bbox:
        bbox = [0.0, 0.0, 0.0, 0.0]

    return {
        "model_available": True,
        "predicted_class": pred_class,
        "probabilities":   prob_dict,
        "bbox":            bbox,
        "has_bbox":        has_bbox,
        "anomaly_score":   anomaly_score,
    }
