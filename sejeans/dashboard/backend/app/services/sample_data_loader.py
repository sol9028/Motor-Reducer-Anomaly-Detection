"""Sample data loader.

Scans the sample data directory at startup and builds an in-memory index of:
  * Battery NPY files + companion JSON labels
  * Motor PNG spectrograms + companion JSON labels

The index is keyed by car_model and fault_type so that demo endpoints can
quickly retrieve a random representative sample.

Directory layout (relative to this file → ../../샘플데이터/Sample/):

  01.원천데이터/
    1.모터_감속기/<CAR_MODEL>/<FAULT_TYPE>/<DATE>/<SENSOR>/  *.png
    2.배터리/<BAT_TYPE>/NPY/<CAR_MODEL>/<STATUS>/<DATE>/<SUBFOLDER>/  *.npy

  02.라벨링데이터/
    1.모터_감속기/<CAR_MODEL>/<FAULT_TYPE>/<DATE>/<SENSOR>/  *.json
    2.배터리/<BAT_TYPE>/JSON/<CAR_MODEL>/<STATUS>/<DATE>/<SUBFOLDER>/  *.json
"""
from __future__ import annotations

import json
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate sample root relative to *this* file
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent           # app/services/
_BACKEND_DIR = _THIS_DIR.parent.parent                # backend/
_SAMPLE_ROOT = _BACKEND_DIR / "../../샘플데이터/Sample"

# Motor sensor channel folders that contain spectrograms
_MOTOR_CHANNELS = ("Current_U", "Vib_Motor", "Vib_TM")

# ---------------------------------------------------------------------------
# Index structures
# ---------------------------------------------------------------------------

class BatterySample:
    __slots__ = ("npy_path", "json_path", "bat_type", "car_model", "status")

    def __init__(
        self,
        npy_path: Path,
        json_path: Path,
        bat_type: str,
        car_model: str,
        status: str,
    ) -> None:
        self.npy_path = npy_path
        self.json_path = json_path
        self.bat_type = bat_type
        self.car_model = car_model
        self.status = status   # NORMAL | CAUTION | DEFECT


class MotorSample:
    __slots__ = ("png_paths", "json_path", "car_model", "fault_type")

    def __init__(
        self,
        png_paths: Dict[str, Path],   # channel_name -> Path
        json_path: Path,
        car_model: str,
        fault_type: str,
    ) -> None:
        self.png_paths = png_paths
        self.json_path = json_path
        self.car_model = car_model
        self.fault_type = fault_type   # NORMAL | ECC10 | ECC20 | DEMAG | REDUC


# ---------------------------------------------------------------------------
# Module-level indexes (populated at startup)
# ---------------------------------------------------------------------------

_battery_index: Dict[str, List[BatterySample]] = {}   # key: "<car_model>/<status>"
_motor_index: Dict[str, List[MotorSample]] = {}        # key: "<car_model>/<fault_type>"
_all_battery: List[BatterySample] = []
_all_motor: List[MotorSample] = []

_initialized = False


# ---------------------------------------------------------------------------
# Index building helpers
# ---------------------------------------------------------------------------

def _add_battery(sample: BatterySample) -> None:
    key = f"{sample.car_model}/{sample.status}"
    _battery_index.setdefault(key, []).append(sample)
    _all_battery.append(sample)


def _add_motor(sample: MotorSample) -> None:
    key = f"{sample.car_model}/{sample.fault_type}"
    _motor_index.setdefault(key, []).append(sample)
    _all_motor.append(sample)


def _build_battery_index(sample_root: Path) -> None:
    """Walk 01.원천데이터/2.배터리/ and pair each NPY with its JSON label."""
    raw_bat_root = sample_root / "01.원천데이터" / "2.배터리"
    lbl_bat_root = sample_root / "02.라벨링데이터" / "2.배터리"

    if not raw_bat_root.exists():
        logger.warning("Battery raw data directory not found: %s", raw_bat_root)
        return

    for bat_type_dir in raw_bat_root.iterdir():
        if not bat_type_dir.is_dir():
            continue
        bat_type = bat_type_dir.name           # Vlt_96 | Vlt_98 | Dev_96 | Dev_98
        npy_root = bat_type_dir / "NPY"
        if not npy_root.exists():
            continue

        # Corresponding label root
        json_root = lbl_bat_root / bat_type / "JSON"

        for car_dir in npy_root.iterdir():
            if not car_dir.is_dir():
                continue
            car_model = car_dir.name           # IONIQ | KONA | NIRO

            for status_dir in car_dir.iterdir():
                if not status_dir.is_dir():
                    continue
                status = status_dir.name       # NORMAL | CAUTION | DEFECT

                # Collect all NPY files recursively under this status dir
                for npy_file in status_dir.rglob("*.npy"):
                    # Derive corresponding JSON path by replacing root + extension
                    relative = npy_file.relative_to(npy_root)
                    json_file = json_root / relative.with_suffix(".json")
                    if not json_file.exists():
                        # Try to find any json in the same folder as a fallback
                        json_candidates = list(json_file.parent.glob("*.json"))
                        if json_candidates:
                            json_file = json_candidates[0]
                        else:
                            continue

                    _add_battery(BatterySample(
                        npy_path=npy_file,
                        json_path=json_file,
                        bat_type=bat_type,
                        car_model=car_model,
                        status=status,
                    ))

    logger.info("Battery index: %d samples across %d keys", len(_all_battery), len(_battery_index))


def _build_motor_index(sample_root: Path) -> None:
    """Walk 01.원천데이터/1.모터_감속기/ and group PNGs by recording instance."""
    raw_motor_root = sample_root / "01.원천데이터" / "1.모터_감속기"
    lbl_motor_root = sample_root / "02.라벨링데이터" / "1.모터_감속기"

    if not raw_motor_root.exists():
        logger.warning("Motor raw data directory not found: %s", raw_motor_root)
        return

    for car_dir in raw_motor_root.iterdir():
        if not car_dir.is_dir():
            continue
        car_model = car_dir.name

        for fault_dir in car_dir.iterdir():
            if not fault_dir.is_dir():
                continue
            fault_type = fault_dir.name

            # Walk date sub-directories
            for date_dir in fault_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                # The Current_U channel is the primary spectrogram channel.
                # Collect all PNG files in Current_U and look for matching
                # Vib_Motor / Vib_TM files with the same stem.
                current_u_dir = date_dir / "Current_U"
                if not current_u_dir.exists():
                    # Fall back to any channel available
                    channels = [d for d in date_dir.iterdir() if d.is_dir()]
                    if not channels:
                        continue
                    current_u_dir = channels[0]

                for png_file in current_u_dir.glob("*.png"):
                    stem = png_file.stem
                    # Gather matching files from all three sensor channels
                    png_paths: Dict[str, Path] = {}
                    for channel in _MOTOR_CHANNELS:
                        channel_dir = date_dir / channel
                        candidate = channel_dir / f"{stem.rsplit('_', 1)[0]}_{channel}.png"
                        # The file name encodes the channel at the end; try exact match first
                        exact = channel_dir / png_file.name.replace(
                            current_u_dir.name, channel
                        )
                        if exact.exists():
                            png_paths[channel] = exact
                        elif channel_dir.exists():
                            # take the first PNG in the channel dir as a representative
                            fallbacks = list(channel_dir.glob("*.png"))
                            if fallbacks:
                                png_paths[channel] = fallbacks[0]

                    if not png_paths:
                        png_paths[current_u_dir.name] = png_file

                    # Corresponding JSON label
                    relative = png_file.relative_to(raw_motor_root)
                    json_file = (
                        lbl_motor_root
                        / relative.with_suffix(".json")
                    )
                    if not json_file.exists():
                        candidates = list(json_file.parent.glob("*.json"))
                        if candidates:
                            json_file = candidates[0]
                        else:
                            continue

                    _add_motor(MotorSample(
                        png_paths=png_paths,
                        json_path=json_file,
                        car_model=car_model,
                        fault_type=fault_type,
                    ))

    logger.info("Motor index: %d samples across %d keys", len(_all_motor), len(_motor_index))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(sample_root: Optional[Path] = None) -> None:
    """Scan the sample data directory and populate the in-memory index.

    Safe to call multiple times (subsequent calls are no-ops).
    """
    global _initialized
    if _initialized:
        return

    root = sample_root or _SAMPLE_ROOT.resolve()
    logger.info("Building sample data index from: %s", root)
    _build_battery_index(root)
    _build_motor_index(root)
    _initialized = True


def get_random_battery_sample(
    car_model: Optional[str] = None,
    status: Optional[str] = None,
) -> Optional[BatterySample]:
    """Return a random battery sample, optionally filtered."""
    if car_model and status:
        pool = _battery_index.get(f"{car_model}/{status}", [])
    elif car_model:
        pool = [s for s in _all_battery if s.car_model == car_model]
    elif status:
        pool = [s for s in _all_battery if s.status == status]
    else:
        pool = _all_battery

    return random.choice(pool) if pool else None


def get_random_motor_sample(
    car_model: Optional[str] = None,
    fault_type: Optional[str] = None,
) -> Optional[MotorSample]:
    """Return a random motor sample, optionally filtered."""
    if car_model and fault_type:
        pool = _motor_index.get(f"{car_model}/{fault_type}", [])
    elif car_model:
        pool = [s for s in _all_motor if s.car_model == car_model]
    elif fault_type:
        pool = [s for s in _all_motor if s.fault_type == fault_type]
    else:
        pool = _all_motor

    return random.choice(pool) if pool else None


def load_npy(path: Path) -> Optional[np.ndarray]:
    """Load a .npy file and return the array, or None on error."""
    try:
        arr = np.load(str(path))
        return arr
    except Exception as exc:
        logger.error("Failed to load NPY %s: %s", path, exc)
        return None


def load_json_label(path: Path) -> Dict[str, Any]:
    """Load a JSON label file, returning an empty dict on error."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.error("Failed to load JSON %s: %s", path, exc)
        return {}


def get_index_stats() -> Dict[str, Any]:
    """Return summary statistics about the loaded index."""
    return {
        "battery_total": len(_all_battery),
        "battery_keys": sorted(_battery_index.keys()),
        "motor_total": len(_all_motor),
        "motor_keys": sorted(_motor_index.keys()),
        "initialized": _initialized,
    }


def get_sample_root() -> Path:
    return _SAMPLE_ROOT.resolve()
