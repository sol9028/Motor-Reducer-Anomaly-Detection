"""Rule-based cell fault localizer.

Two fault modes are supported:
  * cell_voltage   – one or more cells have voltage significantly below the
                     pack mean (the most common DEFECT/CAUTION pattern).
  * cell_deviation – a cell's voltage varies excessively across timesteps
                     (indicates internal resistance growth or micro-short).

The function returns:
  * faulty_cell_indices : List[int]   (0-based)
  * cell_scores         : List[float] (per-cell anomaly score in [0, 1])

localize_faults() accepts an optional predicted_fault_type from the CNN model.
When provided it takes precedence over the bat_type heuristic.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# Voltage below this absolute threshold is considered suspect
_VOLTAGE_LOW_THRESHOLD = 3.5   # V
# Z-score magnitude above which a cell is flagged
_Z_SCORE_FLAG = 2.0


def localize_voltage_fault(
    npy_array: np.ndarray,
) -> Tuple[List[int], List[float]]:
    """Flag cells whose mean voltage deviates most below the pack mean.

    Parameters
    ----------
    npy_array : shape (T, N)

    Returns
    -------
    faulty_cell_indices, cell_scores (0-1, higher = more anomalous)
    """
    cell_means: np.ndarray = npy_array.mean(axis=0)   # (N,)
    pack_mean: float = float(cell_means.mean())
    pack_std: float = float(cell_means.std()) + 1e-9  # avoid /0

    # Negative z-score → cell is low relative to pack
    z_scores: np.ndarray = (cell_means - pack_mean) / pack_std

    # Normalise to [0, 1]: the more negative the z-score the higher the score
    raw_anomaly = np.clip(-z_scores, 0, None)  # only penalise low cells
    max_val = raw_anomaly.max() + 1e-9
    cell_scores: List[float] = (raw_anomaly / max_val).tolist()

    # Flag cells below the absolute threshold OR with large negative z-score
    below_thresh = cell_means < _VOLTAGE_LOW_THRESHOLD
    high_z = z_scores < -_Z_SCORE_FLAG
    flag_mask = below_thresh | high_z

    # If nothing is flagged, flag the single lowest cell
    if not flag_mask.any():
        flag_mask[int(np.argmin(cell_means))] = True

    faulty_cell_indices: List[int] = [int(i) for i in np.where(flag_mask)[0]]
    return faulty_cell_indices, [round(s, 4) for s in cell_scores]


def localize_deviation_fault(
    npy_array: np.ndarray,
) -> Tuple[List[int], List[float]]:
    """Flag cells whose intra-column std deviation is unusually high.

    High per-cell deviation suggests inconsistent charge/discharge behaviour
    (internal short, SEI layer variation, etc.).

    Parameters
    ----------
    npy_array : shape (T, N)

    Returns
    -------
    faulty_cell_indices, cell_scores
    """
    cell_std: np.ndarray = npy_array.std(axis=0)   # (N,)
    mean_std: float = float(cell_std.mean())
    std_of_std: float = float(cell_std.std()) + 1e-9

    z_scores: np.ndarray = (cell_std - mean_std) / std_of_std

    # Normalise to [0, 1]
    raw_anomaly = np.clip(z_scores, 0, None)
    max_val = raw_anomaly.max() + 1e-9
    cell_scores: List[float] = (raw_anomaly / max_val).tolist()

    flag_mask = z_scores > _Z_SCORE_FLAG
    if not flag_mask.any():
        flag_mask[int(np.argmax(cell_std))] = True

    faulty_cell_indices: List[int] = [int(i) for i in np.where(flag_mask)[0]]
    return faulty_cell_indices, [round(s, 4) for s in cell_scores]


def localize_faults(
    npy_array: np.ndarray,
    battery_type: str,
    classification: str,
    predicted_fault_type: Optional[str] = None,
) -> Tuple[str, List[int], List[float]]:
    """Dispatch to the correct localizer based on classification and fault type.

    Parameters
    ----------
    npy_array            : shape (T, N)
    battery_type         : "Vlt_96" | "Vlt_98" | "Dev_96" | "Dev_98"
    classification       : "NORMAL" | "CAUTION" | "DEFECT"
    predicted_fault_type : CNN 예측 결함 유형 (있으면 bat_type 휴리스틱보다 우선)
                           "none" | "cell_voltage" | "cell_deviation" | None

    Returns
    -------
    fault_type, faulty_cell_indices, cell_scores
    """
    if classification == "NORMAL":
        n_cells = npy_array.shape[1]
        return "none", [], [0.0] * n_cells

    # CNN 예측이 있으면 우선 사용, 없으면 bat_type으로 결정
    if predicted_fault_type and predicted_fault_type != "none":
        fault_type = predicted_fault_type
    elif battery_type.startswith("Dev"):
        fault_type = "cell_deviation"
    else:
        fault_type = "cell_voltage"

    if fault_type == "cell_deviation":
        indices, scores = localize_deviation_fault(npy_array)
    else:
        indices, scores = localize_voltage_fault(npy_array)

    return fault_type, indices, scores
