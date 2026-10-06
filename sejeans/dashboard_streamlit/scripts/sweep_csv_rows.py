"""Phase 2-0: 81개 CSV 행 수 sweep & chunk 수 집계 (Rev. 2026-05-16-02).

전체 CSV 를 순회해
  (a) 행 수 분포 {2000:?, 4000:?, 6000:?, 8000:?, 10000:?}
  (b) chunk_size=2000 기준 총 chunk 수
  (c) (vehicle, sensor, fault_class) 슬라이스별 chunk 수 표
를 산출하고 `artifacts/csv_row_sweep.csv` 에 저장한다.

실행:
    python scripts/sweep_csv_rows.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logger import get_logger  # noqa: E402
from src.data import CSVLoader, build_sample_index  # noqa: E402

_LOGGER = get_logger("scripts.sweep_csv_rows")

DATA_ROOT = PROJECT_ROOT / "샘플데이터" / "03.합성데이터" / "1.모터_감속기_시계열"
OUT_CSV = PROJECT_ROOT / "artifacts" / "csv_row_sweep.csv"

CHUNK_SIZE = 2_000
ALLOWED_ROWS = (2_000, 4_000, 6_000, 8_000, 10_000)


def main() -> int:
    index_df = build_sample_index(DATA_ROOT)
    if len(index_df) == 0:
        _LOGGER.error("sample index empty — abort")
        return 1

    loader = CSVLoader()
    rows: list[dict] = []
    for rec in index_df.itertuples(index=False):
        _, signal_df = loader.load(rec.csv_path)
        n_rows = len(signal_df)
        if n_rows % CHUNK_SIZE != 0:
            _LOGGER.warning(
                "non-divisible row count: %s -> %d rows",
                Path(rec.csv_path).name, n_rows,
            )
        n_chunks = n_rows // CHUNK_SIZE
        rows.append(
            {
                "csv_path": rec.csv_path,
                "vehicle": rec.vehicle,
                "fault_class": rec.fault_class,
                "sensor": rec.sensor,
                "group_key": rec.group_key,
                "n_rows": n_rows,
                "n_chunks": n_chunks,
            }
        )

    sweep_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sweep_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    _LOGGER.info("saved sweep csv: %s (rows=%d)", OUT_CSV, len(sweep_df))

    # (a) 행 수 분포
    print("\n=== (a) 행 수 분포 ===")
    row_dist = sweep_df["n_rows"].value_counts().sort_index()
    full_dist = {r: int(row_dist.get(r, 0)) for r in ALLOWED_ROWS}
    print(full_dist)
    extras = set(sweep_df["n_rows"].unique()) - set(ALLOWED_ROWS)
    if extras:
        print(f"WARN: 허용 집합 외 행 수 등장: {sorted(extras)}")

    # (b) 총 chunk 수
    total_chunks = int(sweep_df["n_chunks"].sum())
    print(f"\n=== (b) 총 chunk 수 (chunk_size={CHUNK_SIZE}) === {total_chunks}")

    # (c) (vehicle, sensor, fault_class) 슬라이스별 chunk 수
    print("\n=== (c) (vehicle, sensor, fault_class) 슬라이스별 chunk 수 ===")
    pivot = sweep_df.pivot_table(
        index=["vehicle", "sensor"],
        columns="fault_class",
        values="n_chunks",
        aggfunc="sum",
        fill_value=0,
    )
    pivot["TOTAL"] = pivot.sum(axis=1)
    print(pivot.to_string())

    # PLAN 갱신용 줄 (사람이 PLAN.md 의 <TBD> 자리에 복붙)
    print("\n--- PLAN <TBD> 갱신용 요약 ---")
    print(f"총 81 CSV → 행 수 분포 {full_dist} → chunk 분할 후 약 {total_chunks} 샘플.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
