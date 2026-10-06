"""State-of-Health proxy calculation.

Formula
-------
    soh = 100 - w1*voltage_deviation_norm
              - w2*min_voltage_deficit_norm
              - w3*anomaly_score_norm

All three penalty terms are normalised to [0, 1] before weighting, so the
maximum possible penalty equals the sum of the weights (= 40 points by
default), giving a floor of ~60 for completely degraded cells.

Weights (tunable defaults)
    w1 = 15   voltage deviation penalty
    w2 = 15   low-voltage deficit penalty
    w3 = 10   classification-derived anomaly penalty
"""
from __future__ import annotations

# Weights must sum <= 100
_W1_DEVIATION = 15.0
_W2_MIN_VOLTAGE = 15.0
_W3_ANOMALY = 10.0

# Reference bounds for normalisation
_DEVIATION_MAX = 0.10   # V – deviation above this is fully penalised
_VOLTAGE_HEALTHY = 4.2  # V – typical full-charge cell voltage
_VOLTAGE_CUTOFF = 3.0   # V – hard cutoff; deficit normalised against this range


def compute_soh_proxy(
    voltage_deviation: float,
    min_cell_voltage: float,
    classification: str,   # "NORMAL" | "CAUTION" | "DEFECT"
) -> float:
    """Return an SOH proxy in the range [0, 100].

    Parameters
    ----------
    voltage_deviation : mean of per-timestep cell-voltage std (from metrics)
    min_cell_voltage  : lowest single-cell voltage observed
    classification    : label-derived health class
    """
    # --- term 1: voltage spread penalty ---
    deviation_norm = min(voltage_deviation / _DEVIATION_MAX, 1.0)

    # --- term 2: minimum-voltage deficit penalty ---
    voltage_range = _VOLTAGE_HEALTHY - _VOLTAGE_CUTOFF  # 1.2 V
    deficit = max(_VOLTAGE_HEALTHY - min_cell_voltage, 0.0)
    min_voltage_norm = min(deficit / voltage_range, 1.0)

    # --- term 3: classification anomaly score ---
    class_score_map = {"NORMAL": 0.0, "CAUTION": 0.5, "DEFECT": 1.0}
    anomaly_norm = class_score_map.get(classification.upper(), 0.5)

    soh = (
        100.0
        - _W1_DEVIATION * deviation_norm
        - _W2_MIN_VOLTAGE * min_voltage_norm
        - _W3_ANOMALY * anomaly_norm
    )
    return round(max(min(soh, 100.0), 0.0), 1)
