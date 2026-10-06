"""Phase 1 자동 검증 스크립트 (V1-1, V1-2, V1-3).

실행: python scripts/verify_phase1.py
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logger import get_logger
from src.data import CSVLoader, build_sample_index

DATA_ROOT = PROJECT_ROOT / "샘플데이터" / "03.합성데이터" / "1.모터_감속기_시계열"


def v1_1(df: pd.DataFrame) -> bool:
    ngroups = df.groupby(["vehicle", "sensor", "fault_class"], observed=True).ngroups
    ok = ngroups == 45
    print(f"V1-1: ngroups(vehicle,sensor,fault_class) = {ngroups} -> "
          f"{'PASS' if ok else 'FAIL'} (expected 45)")
    return ok


_ALLOWED_ROWS = {2000, 4000, 6000, 8000, 10000}


def v1_2(df: pd.DataFrame) -> bool:
    """Rev. 2026-05-16-02: 가변 길이 허용 — shape[1]==4 and shape[0] in 허용 집합."""
    loader = CSVLoader()
    sample_path = df["csv_path"].iloc[0]
    meta_df, signal_df = loader.load(sample_path)

    shape_ok = (
        signal_df.shape[1] == 4 and signal_df.shape[0] in _ALLOWED_ROWS
    )
    cols_ok = {
        "vehicle", "sensor", "fault_class", "group_key", "zsplit",
    }.issubset(set(meta_df.columns))
    dtype_ok = all(str(signal_df[c].dtype) == "float32" for c in signal_df.columns)

    ok = shape_ok and cols_ok and dtype_ok
    print(f"V1-2: signal_df.shape = {signal_df.shape}, meta_cols include "
          f"vehicle/sensor/fault_class/group_key/zsplit -> "
          f"{'PASS' if ok else 'FAIL'} (allowed rows: {sorted(_ALLOWED_ROWS)})")
    print(f"      signal dtypes: {dict(signal_df.dtypes.astype(str))}")
    return ok


def v1_3() -> bool:
    """결측 폴더 시나리오: 가짜 fault_class 1개 추가 → WARN 로그가 잡혀야 한다."""
    # 임시로 path_indexer 로거에 메모리 핸들러를 부착
    logger = get_logger("data.path_indexer")
    buf = io.StringIO()
    mem_handler = logging.StreamHandler(buf)
    mem_handler.setLevel(logging.WARNING)
    logger.addHandler(mem_handler)
    try:
        # 존재하지 않는 fault_class 1개 + 정상 4개를 같이 넘겨서 WARN 1줄 이상 유도
        build_sample_index(
            DATA_ROOT,
            fault_classes=("NORMAL", "ECC10", "ECC20", "DEMAG", "REDUC", "__GHOST__"),
        )
    finally:
        logger.removeHandler(mem_handler)
    logs = buf.getvalue()
    warn_lines = [ln for ln in logs.splitlines() if "missing fault_dir" in ln]
    ok = len(warn_lines) >= 1
    print(f"V1-3: missing-folder WARN count = {len(warn_lines)} -> "
          f"{'PASS' if ok else 'FAIL'} (expected >= 1)")
    if warn_lines:
        print(f"      sample: {warn_lines[0]}")
    return ok


def main() -> int:
    df = build_sample_index(DATA_ROOT)
    print(f"[index] rows={len(df)}, unique group_key={df['group_key'].nunique()}")

    results = [v1_1(df), v1_2(df), v1_3()]
    summary = all(results)
    print(f"\n=== Phase 1 verification: "
          f"{'ALL PASS' if summary else 'FAIL'} ({sum(results)}/{len(results)}) ===")
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
