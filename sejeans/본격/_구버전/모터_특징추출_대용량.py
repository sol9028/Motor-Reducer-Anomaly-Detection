"""
모터-감속기 스펙트로그램 -> 시계열 특징 CSV  (대용량 배치 버전)

기존 노트북 대비 변경점
1) 병합 + 특징추출을 한 워커 안에서 융합 -> 중간 합성 PNG 디스크 I/O 제거 (SAVE_MERGED_PNG로 선택)
2) ProcessPoolExecutor로 그룹 단위 병렬 처리
3) 특징추출 컬럼 for문 -> numpy 완전 벡터화
4) 이미지 핸들 명시적 close (too many open files 방지)
5) 노트북이 아닌 .py 배치 실행 (Windows multiprocessing 안전)

실행:  python 모터_특징추출_대용량.py
"""

import os
import re
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image

# ===================== CONFIG =====================
BASE_DIR   = Path(r"샘플데이터\Sample\01.원천데이터\1.모터_감속기")
TS_OUTPUT_DIR = Path(r"샘플데이터\03.합성데이터\1.모터_감속기_시계열")

THRESHOLD_PERCENTILE = 90
MIN_BAND_HEIGHT      = 2
FREQ_FROM_BOTTOM     = True

# 합성 PNG도 따로 저장할지. 학습/디버깅에 굳이 필요 없으면 False가 훨씬 빠름.
SAVE_MERGED_PNG  = False
MERGED_PNG_DIR   = Path(r"샘플데이터\03.합성데이터\1.모터_감속기")
PNG_COMPRESS_LV  = 1          # 0~9. 낮을수록 빠름(용량은 큼)

# 워커 수. 외장하드(HDD)에서 읽으면 코어 다 쓰면 오히려 디스크 thrashing 나니
# HDD면 4~6 권장, SSD/Drive면 os.cpu_count()까지.
MAX_WORKERS = min(8, os.cpu_count() or 4)
# ==================================================


PATTERN = re.compile(
    r"^(\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"  # timestamp
    r"_(\d{2})"                                 # sequential num
    r"_(zsplit\d{3})"                           # zsplit num
    r"_(\d{3})"                                 # split num
    r"_(.+)\.png$"                              # sensor type
)


def parse_filename(fname):
    m = PATTERN.match(fname)
    if not m:
        return None
    ts, seq, zsplit, split_num, sensor = m.groups()
    return {
        "timestamp": ts, "seq": seq, "zsplit": zsplit,
        "split_num": int(split_num), "sensor": sensor,
        "group_key": f"{ts}_{seq}_{zsplit}_{sensor}",
    }


def collect_groups(base_dir):
    """전체 트리를 한 번만 훑어 그룹키별로 (split_num, path, info) 수집."""
    groups = defaultdict(list)
    for png_path in base_dir.rglob("*.png"):
        info = parse_filename(png_path.name)
        if info is None:
            continue
        groups[info["group_key"]].append((info["split_num"], png_path, info))
    return groups


def merge_images_horizontal(image_paths):
    """가로 병합. 핸들은 paste 직후 바로 닫는다."""
    imgs = []
    for p in image_paths:
        im = Image.open(p)
        im = im.convert("RGB")   # load 강제 + 모드 통일
        imgs.append(im)
    total_w = sum(im.width for im in imgs)
    max_h   = max(im.height for im in imgs)
    merged = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for im in imgs:
        merged.paste(im, (x, 0))
        x += im.width
        im.close()
    return merged


def image_to_energy_map(image):
    """RGB 스펙트로그램 -> 0~1 에너지 맵.
    주의: jet/viridis 컬러맵에서 채널 RMS는 에너지와 단조관계가 아님.
    정밀하게 가려면 컬러맵 역변환(LUT)이 맞지만, 여기선 기존 노트북 로직 그대로 둠."""
    arr = np.asarray(image, dtype=np.float32) / 255.0          # 이미 RGB
    energy = np.sqrt(np.mean(arr ** 2, axis=2))
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)
    return energy


def extract_timeseries_vectorized(energy,
                                  threshold_percentile=90,
                                  min_band_height=2,
                                  freq_from_bottom=True):
    """컬럼 for문 없이 numpy로 한 번에 특징 추출."""
    H, W = energy.shape
    threshold = np.percentile(energy, threshold_percentile)
    mask = energy >= threshold                       # (H, W)

    rms = np.sqrt(np.mean(energy ** 2, axis=0))      # (W,)
    count = mask.sum(axis=0)                          # (W,)
    valid = count >= min_band_height                  # (W,)

    peak_y = np.argmax(energy, axis=0)                # 가장 큰 row (top=0)
    # 각 컬럼의 최상단/최하단 True row
    y_top = np.argmax(mask, axis=0)                   # 첫 True (작은 y)
    y_bot = H - 1 - np.argmax(mask[::-1, :], axis=0)  # 마지막 True (큰 y)

    if freq_from_bottom:
        peak_freq_bin = (H - 1 - peak_y).astype(np.float32)
        band_start = (H - 1 - y_bot).astype(np.float32)  # 작은 bin
        band_end   = (H - 1 - y_top).astype(np.float32)  # 큰 bin
    else:
        peak_freq_bin = peak_y.astype(np.float32)
        band_start = y_top.astype(np.float32)
        band_end   = y_bot.astype(np.float32)

    # valid 아닌 컬럼은 NaN 처리 (기존 로직과 동일)
    nan = np.float32(np.nan)
    peak_freq_bin = np.where(valid, peak_freq_bin, nan)
    band_start    = np.where(valid, band_start, nan)
    band_end      = np.where(valid, band_end, nan)

    return pd.DataFrame({
        "time_idx": np.arange(W),
        "peak_freq_bin": peak_freq_bin,
        "band_start": band_start,
        "band_end": band_end,
        "rms": rms.astype(np.float32),
    }), threshold


def build_metadata_cols(df, sample_key, sample_paths, sample_info, base_dir):
    rel_parts = sample_paths[0].relative_to(base_dir).parts
    vehicle, fault_class, date, sensor = rel_parts[:4]
    df.insert(0, "group_key", sample_key)
    df.insert(1, "vehicle", vehicle)
    df.insert(2, "fault_class", fault_class)
    df.insert(3, "date", date)
    df.insert(4, "sensor", sensor)
    df.insert(5, "split_count", len(sample_paths))
    df.insert(6, "timestamp", sample_info["timestamp"])
    df.insert(7, "zsplit", sample_info["zsplit"])
    return df


def process_group(args):
    """워커 1개가 그룹 1개를 끝까지: 병합 -> (선택)PNG저장 -> 특징추출 -> CSV저장."""
    group_key, entries = args
    try:
        entries_sorted = sorted(entries, key=lambda x: x[0])
        image_paths = [e[1] for e in entries_sorted]
        sample_info = entries_sorted[0][2]

        merged = merge_images_horizontal(image_paths)

        if SAVE_MERGED_PNG:
            rel = image_paths[0].relative_to(BASE_DIR).parent
            out_name = (f"{sample_info['timestamp']}_{sample_info['seq']}_"
                        f"{sample_info['zsplit']}_merged_{sample_info['sensor']}.png")
            png_path = MERGED_PNG_DIR / rel / out_name
            png_path.parent.mkdir(parents=True, exist_ok=True)
            merged.save(png_path, compress_level=PNG_COMPRESS_LV)

        energy = image_to_energy_map(merged)
        merged.close()

        ts_df, _ = extract_timeseries_vectorized(
            energy,
            threshold_percentile=THRESHOLD_PERCENTILE,
            min_band_height=MIN_BAND_HEIGHT,
            freq_from_bottom=FREQ_FROM_BOTTOM,
        )
        ts_df = build_metadata_cols(ts_df, group_key, image_paths, sample_info, BASE_DIR)

        csv_path = TS_OUTPUT_DIR / f"{group_key}_timeseries.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        ts_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return (group_key, len(ts_df), None)
    except Exception as e:
        return (group_key, 0, repr(e))


def main():
    t0 = time.time()
    TS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("그룹 수집 중...")
    groups = collect_groups(BASE_DIR)
    total = len(groups)
    print(f"총 그룹 {total:,}개 / 파일 {sum(len(v) for v in groups.values()):,}개")
    print(f"워커 {MAX_WORKERS}개로 처리 시작\n")

    work = list(groups.items())
    done = fail = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_group, w) for w in work]
        for i, fut in enumerate(as_completed(futures), 1):
            gk, nrows, err = fut.result()
            if err:
                fail += 1
                print(f"  [실패] {gk}: {err}")
            else:
                done += 1
            if i % 500 == 0 or i == total:
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (total - i) / rate if rate else 0
                print(f"  {i:,}/{total:,}  ({rate:.0f} grp/s, ETA {eta/60:.1f}분)")

    print(f"\n완료: 성공 {done:,} / 실패 {fail:,} / {time.time()-t0:.1f}초")
    print(f"저장 위치: {TS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
