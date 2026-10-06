"""Phase 2 전처리 모듈 pytest (TODO 2-5).

acceptance criteria (TODO V2-1 ~ V2-3 + Rev. 02 chunker):
- 짧음 케이스 → enforce_length(arr_1500, target=2000) shape (2000,) + pad 카운터 +1
- 긴 케이스 → enforce_length(arr_2500, target=2000) shape (2000,) + truncate 카운터 +1
- NaN 케이스 → 5% NaN 입력 → clean_signal 출력 NaN 0개 + ratio 0.05 부근
- chunker 케이스 → 6000행 → 3 chunk / 동일 group_key / chunk_id 0~2 / 비배수 → AssertionError

추가 validation (Rev. 02 panel_formatter):
- panel shape (N, 1, 2000), dtype float32
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocess import (
    clean_signal,
    enforce_length,
    get_stats,
    select_column,
    split_into_chunks,
    to_sktime_panel,
)


# ---------------------------------------------------------------------------
# 공통 fixture
# ---------------------------------------------------------------------------

_SIGNAL_COLS = ("peak_freq_bin", "band_start", "band_end", "rms")


@pytest.fixture(autouse=True)
def _reset_shape_stats():
    """각 테스트 시작 시 shape_guard 카운터 초기화 (테스트 격리)."""
    get_stats().reset()
    yield
    get_stats().reset()


def _make_signal_df(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """`(n_rows, 4)` float32 더미 signal_df 생성."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(size=(n_rows, len(_SIGNAL_COLS))).astype(np.float32)
    return pd.DataFrame(arr, columns=list(_SIGNAL_COLS))


# ---------------------------------------------------------------------------
# 케이스 1 — 짧음 (TODO V2-1, pad)
# ---------------------------------------------------------------------------


class TestEnforceLengthShort:
    def test_pad_short_input_to_target(self):
        arr = np.arange(1_500, dtype=np.float32)
        stats_before = get_stats()
        assert stats_before.pad_count == 0

        out = enforce_length(arr, target=2_000)

        assert out.shape == (2_000,), f"expected (2000,), got {out.shape}"
        assert out.dtype == np.float32
        # 앞 1500 은 원본, 뒤 500 은 0
        np.testing.assert_array_equal(out[:1_500], arr)
        np.testing.assert_array_equal(out[1_500:], np.zeros(500, dtype=np.float32))
        # pad 카운터 +1
        assert get_stats().pad_count == 1
        assert get_stats().truncate_count == 0
        assert get_stats().noop_count == 0

    def test_noop_when_already_target(self):
        arr = np.arange(2_000, dtype=np.float32)
        out = enforce_length(arr, target=2_000)
        assert out.shape == (2_000,)
        # noop 시에는 동일 객체 반환 (메모리 효율)
        assert out is arr
        assert get_stats().noop_count == 1


# ---------------------------------------------------------------------------
# 케이스 2 — 긴 입력 (truncate)
# ---------------------------------------------------------------------------


class TestEnforceLengthLong:
    def test_truncate_long_input_to_target(self):
        arr = np.arange(2_500, dtype=np.float32)
        out = enforce_length(arr, target=2_000)
        assert out.shape == (2_000,)
        np.testing.assert_array_equal(out, arr[:2_000])
        assert get_stats().truncate_count == 1
        assert get_stats().pad_count == 0

    def test_invalid_dims_raises(self):
        with pytest.raises(ValueError, match="1D"):
            enforce_length(np.zeros((10, 10), dtype=np.float32), target=2_000)


# ---------------------------------------------------------------------------
# 케이스 3 — NaN (TODO V2-2)
# ---------------------------------------------------------------------------


class TestCleanSignalNaN:
    def test_interpolate_5pct_nan(self):
        """V2-2: NaN 5% → 보간 후 NaN 0개, 결측 비율 0.05."""
        n = 2_000
        rng = np.random.default_rng(42)
        arr = rng.standard_normal(n).astype(np.float32)
        # 정확히 5% (=100개) NaN 주입, 단 양 끝은 valid 로 둬서 보간 안정성 확보
        nan_positions = rng.choice(np.arange(10, n - 10), size=100, replace=False)
        arr[nan_positions] = np.nan
        assert np.isnan(arr).sum() == 100

        cleaned, ratio = clean_signal(arr)

        assert cleaned.shape == (n,)
        assert cleaned.dtype == np.float32
        assert np.isnan(cleaned).sum() == 0, "NaN must be fully removed"
        assert ratio == pytest.approx(0.05, abs=1e-6), f"ratio={ratio}"

    def test_no_nan_returns_zero_ratio(self):
        arr = np.arange(2_000, dtype=np.float32)
        cleaned, ratio = clean_signal(arr)
        assert ratio == 0.0
        np.testing.assert_array_equal(cleaned, arr)

    def test_all_nan_falls_back_to_zeros(self):
        arr = np.full(2_000, np.nan, dtype=np.float32)
        cleaned, ratio = clean_signal(arr)
        assert ratio == 1.0
        np.testing.assert_array_equal(cleaned, np.zeros(2_000, dtype=np.float32))

    def test_edge_nan_extrapolates_nearest(self):
        """양 끝이 NaN 이어도 np.interp 가 nearest-neighbor 로 확장 → 잔여 NaN 0."""
        arr = np.linspace(0, 1, 100, dtype=np.float32)
        arr[:5] = np.nan
        arr[-5:] = np.nan
        cleaned, ratio = clean_signal(arr)
        assert np.isnan(cleaned).sum() == 0
        assert ratio == pytest.approx(10 / 100)


# ---------------------------------------------------------------------------
# 케이스 4 — chunker (TODO V2-1-chunker, V2-2-chunker, Rev. 02)
# ---------------------------------------------------------------------------


class TestChunker:
    def test_6000_rows_splits_into_3_chunks(self):
        signal_df = _make_signal_df(6_000)
        chunks = split_into_chunks(
            signal_df, chunk_size=2_000, group_key="GK_TEST_6000",
        )
        assert len(chunks) == 3
        for i, c in enumerate(chunks):
            assert len(c) == 2_000
            assert c.attrs["chunk_id"] == i
            assert c.attrs["group_key"] == "GK_TEST_6000"
            # dtype 보존
            for col in _SIGNAL_COLS:
                assert str(c[col].dtype) == "float32"

    def test_2000_rows_yields_single_chunk(self):
        chunks = split_into_chunks(_make_signal_df(2_000), chunk_size=2_000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 2_000

    def test_non_multiple_raises_assertion(self):
        with pytest.raises(AssertionError, match="multiple of chunk_size"):
            split_into_chunks(_make_signal_df(3_000), chunk_size=2_000)

    def test_invalid_chunk_size_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            split_into_chunks(_make_signal_df(2_000), chunk_size=0)


# ---------------------------------------------------------------------------
# 추가 — panel_formatter (TODO V2-3)
# ---------------------------------------------------------------------------


class TestPanelFormatter:
    def test_select_column_returns_1d_float32(self):
        signal_df = _make_signal_df(2_000)
        arr = select_column(signal_df, "rms")
        assert arr.ndim == 1
        assert arr.shape == (2_000,)
        assert arr.dtype == np.float32

    def test_select_column_rejects_invalid_name(self):
        signal_df = _make_signal_df(100)
        with pytest.raises(ValueError, match="col_name must be one of"):
            select_column(signal_df, "not_a_column")

    def test_to_sktime_panel_from_dataframes(self):
        """V2-3: panel shape (N, 1, 2000), dtype float32."""
        chunks = split_into_chunks(_make_signal_df(6_000), chunk_size=2_000)
        panel = to_sktime_panel(chunks, target=2_000, col_name="rms")
        assert panel.shape == (3, 1, 2_000)
        assert panel.dtype == np.float32

    def test_to_sktime_panel_from_1d_arrays(self):
        arrs = [np.arange(2_000, dtype=np.float32) for _ in range(4)]
        panel = to_sktime_panel(arrs, target=2_000)
        assert panel.shape == (4, 1, 2_000)
        assert panel.dtype == np.float32

    def test_to_sktime_panel_pads_short_input(self):
        """짧은 1D 입력 (1500) → shape_guard 통해 2000 으로 pad 후 panel 합류."""
        arrs = [np.arange(1_500, dtype=np.float32)]
        panel = to_sktime_panel(arrs, target=2_000)
        assert panel.shape == (1, 1, 2_000)
        assert get_stats().pad_count == 1

    def test_to_sktime_panel_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            to_sktime_panel([], target=2_000)

    def test_to_sktime_panel_dataframe_without_col_name_raises(self):
        chunks = split_into_chunks(_make_signal_df(2_000), chunk_size=2_000)
        with pytest.raises(ValueError, match="col_name=None"):
            to_sktime_panel(chunks, target=2_000)
