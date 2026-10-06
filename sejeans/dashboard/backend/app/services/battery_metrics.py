"""Compute real battery metrics from a (20, 96|98) numpy array."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def compute_battery_metrics(
    npy_array: np.ndarray,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Parameters
    ----------
    npy_array : shape (20, N) – rows are timesteps, columns are cells.
    metadata  : dict loaded from the companion JSON label file.

    Returns
    -------
    dict with keys: min_cell_voltage, voltage_deviation, temperature,
                    cell_count, cell_mean_voltages
    """
    # Per-cell mean across all 20 timesteps  -> shape (N,)
    cell_mean_voltage: np.ndarray = npy_array.mean(axis=0)

    # Absolute minimum voltage across all cells and timesteps
    min_cell_voltage: float = float(npy_array.min())

    # Std deviation per timestep across all cells -> shape (20,), then average
    cell_std_per_time: np.ndarray = npy_array.std(axis=1)
    voltage_deviation: float = float(cell_std_per_time.mean())

    temperature: Optional[float] = metadata.get("temperature", None)

    return {
        "min_cell_voltage": round(min_cell_voltage, 3),
        "voltage_deviation": round(voltage_deviation, 4),
        "temperature": round(temperature, 1) if temperature is not None else None,
        "cell_count": int(npy_array.shape[1]),
        "cell_mean_voltages": [round(v, 4) for v in cell_mean_voltage.tolist()],
    }
