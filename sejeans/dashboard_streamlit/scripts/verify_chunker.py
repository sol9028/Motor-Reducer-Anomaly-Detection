"""Phase 2-X 자동 검증 스크립트 (V2-1-chunker, V2-2-chunker).

검증 항목:
- V2-1-chunker: 10000행 CSV 1개 → 5개 chunk, 동일 group_key 상속, chunk_id 0~4
- V2-2-chunker (보강):
    * 4000행 → 2 chunk
    * 2000행 → 1 chunk
    * 3000행(배수 아님) → AssertionError
- 실측 CSV 1개로 end-to-end 검증 (csv_loader + chunker 조합)

실행: python scripts/verify_chunker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import CSVLoader, build_sample_index  # noqa: E402
from src.preprocess import split_into_chunks  # noqa: E402

DATA_ROOT = PROJECT_ROOT / "샘플데이터" / "03.합성데이터" / "1.모터_감속기_시계열"


def _make_signal(n_rows: int) -> pd.DataFrame:
    """`(n_rows, 4)` float32 더미 signal_df 를 만든다."""
    rng = np.random.default_rng(42)
    arr = rng.standard_normal(size=(n_rows, 4)).astype(np.float32)
    return pd.DataFrame(
        arr, columns=["peak_freq_bin", "band_start", "band_end", "rms"]
    )


def v2_1_chunker() -> bool:
    signal_df = _make_signal(10_000)
    chunks = split_into_chunks(signal_df, chunk_size=2_000, group_key="GK_TEST")

    cnt_ok = len(chunks) == 5
    len_ok = all(len(c) == 2_000 for c in chunks)
    gk_ok = all(c.attrs.get("group_key") == "GK_TEST" for c in chunks)
    id_ok = [c.attrs.get("chunk_id") for c in chunks] == [0, 1, 2, 3, 4]
    dtype_ok = all(
        all(str(c[col].dtype) == "float32" for col in c.columns) for c in chunks
    )

    ok = cnt_ok and len_ok and gk_ok and id_ok and dtype_ok
    print(
        "V2-1-chunker: 10000행 → 5 chunk / len=2000 / group_key 상속 / chunk_id 0~4 / "
        f"dtype float32 -> {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        print(f"      cnt_ok={cnt_ok}, len_ok={len_ok}, gk_ok={gk_ok}, "
              f"id_ok={id_ok}, dtype_ok={dtype_ok}")
    return ok


def v2_2_chunker() -> bool:
    # (a) 4000 → 2 chunk
    c4 = split_into_chunks(_make_signal(4_000), chunk_size=2_000)
    a = len(c4) == 2 and all(len(c) == 2_000 for c in c4)

    # (b) 2000 → 1 chunk
    c2 = split_into_chunks(_make_signal(2_000), chunk_size=2_000)
    b = len(c2) == 1 and len(c2[0]) == 2_000

    # (c) 3000 → AssertionError
    try:
        split_into_chunks(_make_signal(3_000), chunk_size=2_000)
        c = False
        msg = "AssertionError 미발생"
    except AssertionError as e:
        c = True
        msg = f"AssertionError 발생: {e}"

    ok = a and b and c
    print(
        f"V2-2-chunker: 4000→2 ({a}) / 2000→1 ({b}) / 3000→AssertErr ({c}) -> "
        f"{'PASS' if ok else 'FAIL'}"
    )
    print(f"      3000행 케이스: {msg}")
    return ok


def v2_end_to_end() -> bool:
    """실측 CSV 1개를 로드 → chunker 통과시켜 (행 수 / chunk_size) 가 정확히 맞는지 확인."""
    index_df = build_sample_index(DATA_ROOT)
    sample_row = index_df.iloc[0]
    loader = CSVLoader()
    _, signal_df = loader.load(sample_row["csv_path"])

    n_rows = len(signal_df)
    expected_chunks = n_rows // 2_000

    chunks = split_into_chunks(
        signal_df, chunk_size=2_000, group_key=sample_row["group_key"],
    )

    ok = (
        len(chunks) == expected_chunks
        and all(len(c) == 2_000 for c in chunks)
        and all(c.attrs["group_key"] == sample_row["group_key"] for c in chunks)
    )
    print(
        f"V2-E2E: 실측 CSV {Path(sample_row['csv_path']).name} "
        f"({n_rows}행) → {len(chunks)} chunk -> "
        f"{'PASS' if ok else 'FAIL'} (expected {expected_chunks})"
    )
    return ok


def main() -> int:
    results = [v2_1_chunker(), v2_2_chunker(), v2_end_to_end()]
    summary = all(results)
    print(
        f"\n=== Phase 2-X verification: "
        f"{'ALL PASS' if summary else 'FAIL'} ({sum(results)}/{len(results)}) ==="
    )
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
