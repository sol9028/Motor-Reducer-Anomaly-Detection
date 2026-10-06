"""
모터-감속기 스펙트로그램 -> 시계열 특징 CSV  (v2)

기존 모터_특징추출_대용량.py 대비 변경점 (확정 3건)
1) LUT 역변환 : jet 컬러맵 역변환으로 픽셀 -> 실제 에너지 복원
                (기존 채널 RMS는 에너지와 단조관계가 아님 = 물리적 오류)
2) rownorm    : 행별 기저 제거 후 임계 -> band_start 포화 해소
                (기존 band_start 는 80~95% 가 상수 0 = 정보량 0)
3) rpm 컬럼   : 라벨링 JSON 의 metadata.speed 를 split 단위로 부착

추가(조건부, 기본 ON) : 확장 특징 12개
   band_ratio_0..7 / centroid / entropy / rolloff / flatness
   -> ENABLE_EXT = False 로 끄면 기존 4특징만 나온다.

추가(조건부, 기본 ON) : 차수축 확장특징 12개  ← 5단계 검증 반영
   ord_band_ratio_0..7 / ord_centroid / ord_entropy / ord_rolloff / ord_log_flatness
   주파수축을 rpm 비례로 리샘플링한 뒤 같은 확장특징을 뽑는다.
   -> ENABLE_ORD = False 로 끄면 A4/4-c 와 동일한 컬럼 구성이 된다.

   왜 절대축을 '대체'하지 않고 '추가'하는가 (5단계 결과):
     DEMAG recall  80.0% -> 84.4% (+4.4%p),  D->N 오분류 5건 -> 3건
     전체 정확도   68.0% -> 61.3% (-6.7%p),  ECC10 -11.1%p / REDUC -17.8%p
   차수축은 rpm 을 일부러 지우는 변환인데, rpm 은 ECC/REDUC 에는 실제
   판별정보다(NIRO 는 DEMAG 4591~5693 vs NORMAL 1951~3398 으로 rpm 만으로
   갈린다). 교란을 지우면 진짜 신호도 같이 지워진다. 그래서 둘은 상보적이고
   택일이 아니라 결합이 답이다. 최종 판정은 본데이터 학습으로 한다.

출력 포맷 : Parquet (기존 CSV 107GB -> 약 15GB)
   컬럼이 12->24개로 늘고 확장특징이 전부 float 이라
   CSV 로 두면 400GB 급이 된다. Parquet 은 숫자를 텍스트가 아닌
   이진수로 저장하고 컬럼 단위 압축이 들어간다.
   필요: pip install pyarrow

실행:  python 모터_특징추출_v2.py
      python 모터_특징추출_v2.py --limit 200     (앞 200그룹만, 리허설용)
      python 모터_특징추출_v2.py --workers 6
"""

import os
import re
import sys
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image

# ===================== CONFIG =====================
# --- 작업 데스크톱 경로 ---
BASE_DIR      = Path(r"D:\TS\1.모터-감속기_데이터")
LABEL_DIR     = Path(r"D:\라벨링데이터\1.모터-감속기_데이터")
WORK_DIR      = Path(r"C:\Users\jaesung\Desktop\매녀졸프")
TS_OUTPUT_DIR = WORK_DIR / "특징데이터"
RPM_CACHE     = WORK_DIR / "rpm_index.json"
INDEX_PATH    = WORK_DIR / "샘플인덱스.parquet"

# --- 출력 포맷 ---
OUT_FORMAT   = "parquet"     # "parquet" | "csv"
PARQUET_COMP = "zstd"        # zstd 가 snappy 보다 30~40% 더 작다
CSV_FLOAT_FMT = "%.5f"       # csv 로 뺄 때만 사용. 원본이 8bit 라 5자리면 충분
SLIM_META    = True          # 행마다 반복되는 문자열 메타를 빼고 인덱스로 분리
FLOAT_DTYPE  = np.float16    # 비율/스칼라 특징용. 원본이 8bit 라 손실 없음
                             # (위치 컬럼은 코드에서 float32 로 강제 유지)

# --- 특징 추출 ---
THRESHOLD_PERCENTILE = 90
MIN_BAND_HEIGHT      = 2
FREQ_FROM_BOTTOM     = True

USE_LUT   = True      # (1) LUT 역변환
USE_ROWNORM = True    # (2) 행별 기저 제거
ATTACH_RPM  = True    # (3) rpm 컬럼
ENABLE_EXT  = True    # 확장 특징 12개

N_BANDS    = 8        # 대역 비율 분할 수
ROLLOFF_P  = 0.85     # 누적 에너지 rolloff 지점

# --- 차수(order) 정규화: 5단계 검증 반영 ---
# rpm 이 다르면 같은 결함이 다른 주파수 bin 에 나타난다. 라벨링 폴리곤으로
# 실측한 결과 K = 결함bin/rpm 이 차종 무관하게 일정했다:
#   DEMAG 0.1333/0.1344/0.1342, ECC10·ECC20 0.1087~0.1091,
#   REDUC 0.1789(IONIQ)/0.1411/0.1420  <- IONIQ 만 감속비가 달라 K 가 다르다
# 즉 결함 주파수는 rpm 에 정비례하므로, 주파수축을 rpm 비례로 리샘플링하면
# rpm 이 달라도 같은 결함이 같은 자리에 온다.
ENABLE_ORD = True     # 차수축 확장특징 12개 추가 (ord_* 접두사)

# 기준 rpm. scale = rpm/REF_RPM 이므로 REF_RPM >= max(rpm) 이면 항상 압축만
# 일어나 잘림(NaN)이 원리적으로 0 이 된다. 3단계가 3300 을 써서 표본 225장 중
# 120장이 축 윗부분을 잃었고(커버리지 92.3%, 최악 58%) 그게 당시 실패 원인이었다.
# 5700 은 표본 최대 rpm 5693 을 덮는 값이다. 본데이터 rpm 이 더 높으면 올려야 한다.
REF_RPM    = 5700.0
ORD_BINS   = 0        # 0 = 원본 높이 유지. 절대축과 bin 수를 같게 해 비교 오염을 막는다.

LUT_QUANT  = 64       # 컬러맵 역변환 복셀 격자 해상도 (64 = 0.26MB, maxerr 4)
CMAP_NAME  = "jet"

# 병합 PNG 저장은 제거했다. 조각을 에너지로 바꿔 이어붙이는 구조라
# 거대 RGB 가 메모리에 존재하지 않는다(그게 MemoryError 의 원인이었다).

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


# ---------------------------------------------------------------- (3) rpm
def build_rpm_index(label_dir, cache_path):
    """라벨링 JSON 을 한 번 훑어 {png_stem: {"rpm":.., "temp":..}} 인덱스를 만든다.

    speed 는 split 마다 다르다 (같은 세션 안에서도 1935 -> 2341 -> 3272 처럼 변한다).
    따라서 세션 단위가 아니라 반드시 split 단위(파일명 stem)로 잡아야 한다.
    """
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
            print(f"rpm 인덱스 캐시 로드: {len(idx):,}건  ({cache_path})")
            return idx
        except Exception as e:
            print(f"  캐시 로드 실패, 재생성: {e!r}")

    if not label_dir.exists():
        print(f"!! 라벨링 경로 없음: {label_dir}  -> rpm 없이 진행")
        return {}

    print(f"rpm 인덱스 생성 중... ({label_dir})")
    t0 = time.time()
    idx, n = {}, 0
    for p in label_dir.rglob("*.json"):
        n += 1
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        md = d.get("metadata", {})
        rpm = md.get("speed")
        if rpm is None:
            continue
        fp = d.get("image", {}).get("file_path", "")
        stem = Path(fp).stem if fp else p.stem
        idx[stem] = {"rpm": float(rpm), "temp": md.get("temperature")}
        if n % 50000 == 0:
            print(f"  ... {n:,}개 스캔 ({time.time()-t0:.0f}s)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(idx, f)
    print(f"rpm 인덱스 {len(idx):,}건 / JSON {n:,}개 ({time.time()-t0:.0f}s)")
    print(f"  캐시 저장: {cache_path}")
    return idx


# ---------------------------------------------------------------- (1) LUT
_VOX = None   # 워커 프로세스별 1회 구축


def _build_voxel(q=LUT_QUANT, cmap_name=CMAP_NAME):
    """RGB 3차원 격자 -> 컬러맵 scalar 역참조 테이블.

    브루트포스 최근접 탐색은 픽셀당 256회 거리계산이라 대용량에서 감당이 안 된다.
    q^3 복셀에 미리 정답을 채워두면 조회가 O(1) 이 된다.
    q=64 에서 최대오차 4/255 로, 컬러맵 자체의 색 간격과 같은 수준이다.
    """
    import matplotlib
    cm = matplotlib.colormaps[cmap_name]
    lut = (np.asarray([cm(i / 255.0)[:3] for i in range(256)]) * 255.0)  # (256,3)

    g = (np.arange(q) + 0.5) * (255.0 / q)
    R, G, B = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([R.ravel(), G.ravel(), B.ravel()], axis=1)            # (q^3,3)

    out = np.empty(len(pts), dtype=np.float32)
    step = 200_000                                    # 청크: (step,256) 거리행렬
    for i in range(0, len(pts), step):
        chunk = pts[i:i + step]
        d = ((chunk[:, None, :] - lut[None, :, :]) ** 2).sum(axis=2)
        out[i:i + step] = d.argmin(axis=1) / 255.0
    return out.reshape(q, q, q)


def _get_voxel():
    global _VOX
    if _VOX is None:
        _VOX = _build_voxel()
    return _VOX


def rgb_to_energy(arr):
    """RGB 배열 -> 0~1 에너지 맵.

    기존 방식(채널 RMS)은 jet 에서 에너지와 단조관계가 아니다.
    scalar 0.063(짙은 남색)과 0.941(짙은 빨강)이 둘 다 밝기 0.45 로 나와,
    가장 약한 신호와 가장 강한 신호가 같은 값으로 뭉개진다.
    LUT 역변환은 색 -> 원래 scalar 를 직접 되돌린다.

    메모리 주의: uint8 -> int32 승격은 4배다. 가로 19만px 이미지에서
    2.9GB 를 한 번에 잡아 MemoryError 가 났었다. uint8 산술로 처리한다.
    """
    if not USE_LUT:
        a = arr.astype(np.float32) / 255.0
        e = np.sqrt(np.mean(a ** 2, axis=2))
        return (e - e.min()) / (e.max() - e.min() + 1e-8)

    vox = _get_voxel()
    q = vox.shape[0]
    shift = 8 - int(np.log2(q))                # q=64 -> 2비트 우시프트
    # uint8 유지 (>> 는 dtype 을 올리지 않는다). q 가 2의 거듭제곱이라 가능.
    r = arr[..., 0] >> shift
    g = arr[..., 1] >> shift
    b = arr[..., 2] >> shift
    return vox[r, g, b].astype(np.float32)


# ------------------------------------------------------------ (2) rownorm
def row_detrend(energy):
    """행(주파수)별 기저 제거.

    스펙트로그램은 저주파가 원래 밝고 고주파가 원래 어둡다.
    전체 90퍼센타일 하나로 자르면 저주파는 통째로 통과, 고주파는 통째로 탈락해
    band_start 가 화면 끝(0)에 붙어버린다. 실제로 Vib_Motor 는 94.6% 가 0 이었다.
    각 행에서 "그 주파수 기준 얼마나 특이한가" 로 바꾸면 이 경사가 사라진다.
    MAD 를 쓰는 건 스펙트로그램 값 분포가 두꺼운 꼬리를 갖기 때문이다.
    """
    med = np.median(energy, axis=1, keepdims=True)
    mad = np.median(np.abs(energy - med), axis=1, keepdims=True) + 1e-6
    return (energy - med) / (1.4826 * mad)


def extract_timeseries_vectorized(energy,
                                  threshold_percentile=THRESHOLD_PERCENTILE,
                                  min_band_height=MIN_BAND_HEIGHT,
                                  freq_from_bottom=FREQ_FROM_BOTTOM,
                                  use_rownorm=USE_ROWNORM):
    """컬럼 for문 없이 numpy로 한 번에 기본 4특징 추출.

    주의: rms 는 rownorm 이전의 실제 에너지에서 뽑는다.
          rownorm 값은 "특이도"라 물리적 크기가 아니다.
    """
    H, W = energy.shape

    rms = np.sqrt(np.mean(energy ** 2, axis=0))       # (W,)  실제 에너지 기준

    det = row_detrend(energy) if use_rownorm else energy
    threshold = np.percentile(det, threshold_percentile)
    mask = det >= threshold                           # (H, W)

    count = mask.sum(axis=0)
    valid = count >= min_band_height

    peak_y = np.argmax(det, axis=0)                   # 가장 특이한 row (top=0)
    y_top = np.argmax(mask, axis=0)                   # 첫 True (작은 y)
    y_bot = H - 1 - np.argmax(mask[::-1, :], axis=0)  # 마지막 True (큰 y)

    if freq_from_bottom:
        peak_freq_bin = (H - 1 - peak_y).astype(np.float32)
        band_start = (H - 1 - y_bot).astype(np.float32)
        band_end   = (H - 1 - y_top).astype(np.float32)
    else:
        peak_freq_bin = peak_y.astype(np.float32)
        band_start = y_top.astype(np.float32)
        band_end   = y_bot.astype(np.float32)

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
    }), float(threshold)


# ------------------------------------------------------------ 확장 특징
def extract_ext_features(energy, n_bands=N_BANDS, rolloff_p=ROLLOFF_P):
    """확장 특징 12개. rownorm 이전의 '실제 에너지'로 계산해야 비율이 의미를 갖는다.

    DEMAG 는 에너지 재분배 고장인데(기본파 약화 + 고조파 강화),
    기존 4특징은 위치 3 + 스칼라 1 이라 재분배를 담을 그릇이 없다.
    대역 비율은 총합이 1 로 고정되므로 재분배가 그대로 숫자로 남는다.

    NaN 안전 집계를 쓰는 이유: 차수축(to_order_axis)은 REF_RPM 보다 빠른
    구간에서 상단이 NaN 이 될 수 있다. 일반 sum/mean 은 NaN 을 전파하므로
    한 행만 NaN 이어도 그 열 전체가 NaN 이 된다. 절대축은 NaN 이 없어
    결과가 바뀌지 않는다(기존 A4 수치와 동일).
    """
    E = np.clip(energy, 0, None)
    flip = E[::-1, :]                                  # row0 = 저주파(하단)
    H, W = flip.shape

    tot = np.nansum(flip, axis=0) + 1e-9               # (W,)
    edges = np.linspace(0, H, n_bands + 1).astype(int)

    cols = {}
    for i in range(n_bands):
        cols[f"band_ratio_{i}"] = (np.nansum(flip[edges[i]:edges[i + 1]], axis=0) / tot)

    idx = np.arange(H, dtype=np.float64)[:, None]
    cols["centroid"] = np.nansum(flip * idx, axis=0) / tot

    pn = flip / tot
    cols["entropy"] = -np.nansum(pn * np.log(pn + 1e-12), axis=0)

    cum = np.nancumsum(flip, axis=0) / tot
    cols["rolloff"] = np.argmax(cum >= rolloff_p, axis=0).astype(np.float64)

    # flatness 는 10^-9 ~ 10^0 로 9자릿수를 걸친다.
    # 비율 그대로 두면 30% 가 float16 최소 정규값(6.1e-5) 아래로 언더플로우한다.
    # log 로 저장하면 -20~0 범위가 되어 float16 으로도 정밀도가 남고,
    # 분포도 한쪽에 뭉치지 않아 학습에 더 낫다.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_gm = np.nanmean(np.log(flip + 1e-12), axis=0)
        log_am = np.log(np.nanmean(flip, axis=0) + 1e-12)
    cols["log_flatness"] = log_gm - log_am

    return pd.DataFrame({k: v.astype(np.float32) for k, v in cols.items()})


# ------------------------------------------------------------ 차수 정규화
def to_order_axis(energy, rpm, ref_rpm=REF_RPM, n_out=0):
    """주파수축(세로)을 rpm 비례로 리샘플링해 '차수축'으로 바꾼다.

    energy : (H, W). 행 0 = 이미지 상단 = 고주파.
             내부에서 하단기준으로 뒤집어 계산한 뒤 같은 방향으로 되돌린다.

    차수 o(0~1) 가 가리키는 원본 bin = o * (H-1) * (rpm/ref_rpm)
      rpm < ref  -> scale<1 : 원본 아래쪽만 뽑아 늘려 담는다 (해상도 손실이 대가)
      rpm > ref  -> scale>1 : 원본 범위를 넘어 잘린다 (NaN)
    따라서 ref_rpm >= max(rpm) 이면 잘림이 0 이다. REF_RPM 주석 참고.

    선형보간을 쓰는 이유: 스펙트로그램은 이미 8bit 로 양자화된 데이터라
    고차보간을 써도 없던 정보가 생기지 않고 링잉만 늘어난다.
    """
    H, W = energy.shape
    if n_out <= 0:
        n_out = H
    flip = energy[::-1, :]                        # 행 0 = 저주파
    src = np.linspace(0.0, (H - 1) * (rpm / ref_rpm), n_out)

    valid = src <= (H - 1)
    sc = np.clip(src, 0, H - 1)
    i0 = np.floor(sc).astype(np.int32)
    i1 = np.minimum(i0 + 1, H - 1)
    w = (sc - i0).astype(np.float32)[:, None]

    out = flip[i0] * (1.0 - w) + flip[i1] * w
    if not valid.all():
        out[~valid] = np.nan
    return out[::-1, :]


def extract_ord_features(image_paths, widths, rpm_idx, n_bands=N_BANDS):
    """조각별로 차수축 변환 후 확장특징을 뽑아 가로로 이어붙인다.

    왜 조각 단위인가: rpm 은 split 마다 다르다. 표본에서 멀티스플릿 그룹의
    90% 가 rpm spread > 50 이고 중앙값이 677rpm, 최대 1879rpm 이었다.
    병합된 배열에 그룹 대표 rpm 하나로 리샘플링하면 대부분의 구간이 틀린
    배율로 늘어난다. 그래서 merge 이전 단계에서 조각마다 따로 변환한다.

    rpm 이 없는 조각은 변환 배율을 알 수 없으므로 NaN 으로 채운다.
    (학습 쪽 fill_nan 이 앞뒤로 메운다. 억지 대체값을 넣으면 틀린 차수가 된다.)
    """
    n_feat = n_bands + 4
    cols = [[] for _ in range(n_feat)]

    for p, w in zip(image_paths, widths):
        rec = rpm_idx.get(p.stem)
        rpm = rec.get("rpm") if rec else None
        if rpm is None or not np.isfinite(rpm) or rpm <= 0:
            for i in range(n_feat):
                cols[i].append(np.full(w, np.nan, dtype=np.float32))
            continue

        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        eo = to_order_axis(rgb_to_energy(arr), float(rpm), n_out=ORD_BINS)
        del arr
        sub = extract_ext_features(eo, n_bands=n_bands)
        del eo
        for i, c in enumerate(sub.columns):
            cols[i].append(sub[c].to_numpy())

    names = [f"ord_band_ratio_{i}" for i in range(n_bands)] + \
            ["ord_centroid", "ord_entropy", "ord_rolloff", "ord_log_flatness"]
    return pd.DataFrame(
        {n: np.concatenate(c).astype(np.float32) for n, c in zip(names, cols)})


# ------------------------------------------------------------ 병합/메타
def merge_energy_horizontal(image_paths):
    """조각을 하나씩 열어 에너지로 바꾼 뒤 가로 이어붙인다.

    거대 RGB 를 통째로 만들지 않는 게 핵심이다.
    zsplit000 은 조각이 90개가 넘어 병합하면 가로 19만px 이 되는데,
    RGB(uint8) 만 0.7GB, LUT 조회용 int32 승격까지 가면 2.9GB 라
    워커 8개가 동시에 하면 MemoryError 가 난다.
    에너지(float32 1채널)는 RGB 대비 1/3 이고, 조각 단위로 변환하면
    피크 메모리가 '가장 큰 조각 1장' 으로 고정된다.
    """
    heights, widths, energies = [], [], []
    for p in image_paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
        e = rgb_to_energy(arr)
        del arr
        heights.append(e.shape[0])
        widths.append(e.shape[1])
        energies.append(e)

    H = max(heights)
    W = sum(widths)
    out = np.zeros((H, W), dtype=np.float32)
    x = 0
    for e in energies:
        out[:e.shape[0], x:x + e.shape[1]] = e
        x += e.shape[1]
    energies.clear()
    return out, widths


def attach_rpm_column(df, image_paths, widths, rpm_idx):
    """split 마다 rpm 이 다르므로, 병합된 가로축 구간별로 서로 다른 값을 채운다.

    같은 세션 안에서도 speed 가 1935 -> 2341 -> 3272 처럼 변한다.
    세션 단위로 하나만 붙이면 틀린 값이 대부분 구간에 들어간다.
    """
    n = len(df)
    rpm_col = np.full(n, np.nan, dtype=np.float32)
    tmp_col = np.full(n, np.nan, dtype=np.float32)
    x = 0
    for p, w in zip(image_paths, widths):
        rec = rpm_idx.get(p.stem)
        if rec is not None:
            lo, hi = x, min(x + w, n)
            rpm_col[lo:hi] = rec.get("rpm", np.nan)
            t = rec.get("temp")
            if t is not None:
                tmp_col[lo:hi] = t
        x += w
    df["rpm"] = rpm_col
    df["temperature"] = tmp_col
    return df


def build_metadata_cols(df, sample_key, sample_paths, sample_info, base_dir):
    """메타 컬럼 부착.

    SLIM_META 면 행마다 반복되는 문자열을 넣지 않는다.
    'IONIQ','DEMAG','22_07_28_12_15_20' 같은 값이 2000행 전부에 복사되면
    전체 용량의 30~40% 가 메타가 된다. 파일 경로에 이미 있는 정보이므로
    샘플인덱스.parquet 한 벌로 따로 저장하고 group_key 로 join 한다.
    """
    rel_parts = sample_paths[0].relative_to(base_dir).parts
    vehicle, fault_class, date, sensor = rel_parts[:4]

    meta = {
        "group_key": sample_key,
        "vehicle": vehicle,
        "fault_class": fault_class,
        "date": date,
        "sensor": sensor,
        "split_count": len(sample_paths),
        "timestamp": sample_info["timestamp"],
        "zsplit": sample_info["zsplit"],
    }

    if SLIM_META:
        # 학습 시 그룹 분할에 반드시 필요한 최소 2개만 행에 남긴다.
        df.insert(0, "group_key", sample_key)
        df.insert(1, "fault_class", fault_class)
    else:
        for i, (k, v) in enumerate(meta.items()):
            df.insert(i, k, v)
    return df, meta


# ------------------------------------------------------------ 워커
_RPM_IDX = {}


def _init_worker(rpm_idx):
    global _RPM_IDX
    _RPM_IDX = rpm_idx
    if USE_LUT:
        _get_voxel()          # 워커당 1회 구축(약 3.6s), 이후 재사용


def process_group(args):
    """워커 1개가 그룹 1개를 끝까지: 병합 -> 에너지 -> 특징 -> CSV저장."""
    group_key, entries = args
    try:
        entries_sorted = sorted(entries, key=lambda x: x[0])
        image_paths = [e[1] for e in entries_sorted]
        sample_info = entries_sorted[0][2]

        energy, widths = merge_energy_horizontal(image_paths)

        ts_df, _ = extract_timeseries_vectorized(energy)

        if ENABLE_EXT:
            ext_df = extract_ext_features(energy)
            ts_df = pd.concat([ts_df, ext_df], axis=1)
        del energy

        # 차수축은 조각별 rpm 이 필요하므로 병합 배열이 아니라 원본 조각을 다시 읽는다.
        # (energy 를 먼저 해제해 피크 메모리가 겹치지 않게 한다)
        if ENABLE_ORD:
            ord_df = extract_ord_features(image_paths, widths, _RPM_IDX)
            if len(ord_df) != len(ts_df):        # 폭 합이 어긋나면 조용히 섞이면 안 된다
                raise ValueError(
                    f"ord 행수 불일치: ord={len(ord_df)} vs ts={len(ts_df)}")
            ts_df = pd.concat([ts_df, ord_df], axis=1)

        if ATTACH_RPM:
            ts_df = attach_rpm_column(ts_df, image_paths, widths, _RPM_IDX)

        ts_df, meta = build_metadata_cols(
            ts_df, group_key, image_paths, sample_info, BASE_DIR)

        # 원본이 8bit 이미지(256단계)라 float16(유효 3자리)이면 정보 손실이 없다.
        # 위치/시간 컬럼은 2048 을 넘을 수 있어 float16 정밀도가 부족하므로 제외.
        keep32 = {"time_idx", "peak_freq_bin", "band_start", "band_end",
                  "rolloff", "centroid", "rpm",
                  "ord_rolloff", "ord_centroid"}   # 위치성 컬럼은 float16 정밀도 부족
        for c in ts_df.columns:
            if ts_df[c].dtype.kind != "f":
                continue
            ts_df[c] = ts_df[c].astype(
                np.float32 if c in keep32 else FLOAT_DTYPE)
        if "time_idx" in ts_df.columns:
            ts_df["time_idx"] = ts_df["time_idx"].astype(np.int32)

        rel = image_paths[0].relative_to(BASE_DIR).parent
        out_dir = TS_OUTPUT_DIR / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        if OUT_FORMAT == "parquet":
            out_path = out_dir / f"{group_key}.parquet"
            ts_df.to_parquet(out_path, index=False,
                             compression=PARQUET_COMP, engine="pyarrow")
        else:
            out_path = out_dir / f"{group_key}_timeseries.csv"
            ts_df.to_csv(out_path, index=False, encoding="utf-8-sig",
                         float_format=CSV_FLOAT_FMT)

        meta["rows"] = len(ts_df)
        meta["bytes"] = out_path.stat().st_size
        return (group_key, len(ts_df), None, meta)
    except Exception as e:
        return (group_key, 0, repr(e), None)


# ------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="앞 N개 그룹만 처리(리허설)")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--rpm-only", action="store_true", help="rpm 인덱스만 만들고 종료")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("모터-감속기 특징추출 v2")
    print(f"  LUT역변환 {USE_LUT} / rownorm {USE_ROWNORM} / "
          f"rpm {ATTACH_RPM} / 확장특징 {ENABLE_EXT}")
    if ENABLE_ORD:
        print(f"  차수축 ON : 기준rpm {REF_RPM:.0f} / "
              f"bin {'원본유지' if ORD_BINS <= 0 else ORD_BINS}")
        print("     주의: 본데이터 rpm 최대값이 기준rpm 을 넘으면 축이 잘린다.")
        print("           먼저 --rpm-only 로 인덱스를 만들고 최대 rpm 을 확인할 것.")
    print(f"  출력 {OUT_FORMAT} / 슬림메타 {SLIM_META}")
    print("=" * 70)

    if OUT_FORMAT == "parquet":
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            print("!! pyarrow 없음.  pip install pyarrow  후 다시 실행")
            print("   (또는 OUT_FORMAT = \"csv\" 로 변경)")
            sys.exit(1)

    if not BASE_DIR.exists():
        print(f"!! 원천 경로 없음: {BASE_DIR}")
        sys.exit(1)

    TS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 저장 드라이브 여유 확인. 몇 시간 돌리고 용량부족으로 죽는 걸 막는다.
    import shutil
    du = shutil.disk_usage(TS_OUTPUT_DIR)
    print(f"\n저장 드라이브 여유 {du.free/2**30:,.1f} GB "
          f"/ 전체 {du.total/2**30:,.1f} GB")
    if du.free < 30 * 2**30:
        print("  !! 30GB 미만. 중단 위험이 크다.")

    rpm_idx = build_rpm_index(LABEL_DIR, RPM_CACHE) if ATTACH_RPM else {}

    # 차수축은 REF_RPM >= max(rpm) 이어야 잘림이 0 이다. 몇 시간 돌린 뒤
    # 축이 잘려 있는 걸 발견하는 사고를 막기 위해 시작 전에 실측으로 막는다.
    if ENABLE_ORD and rpm_idx:
        rv = np.array([r["rpm"] for r in rpm_idx.values()
                       if r.get("rpm") is not None], dtype=np.float64)
        if len(rv):
            print(f"\nrpm 실측 범위 {rv.min():.0f} ~ {rv.max():.0f} "
                  f"(중앙 {np.median(rv):.0f})")
            if rv.max() > REF_RPM:
                over = 100.0 * (rv > REF_RPM).mean()
                print(f"  !! REF_RPM({REF_RPM:.0f}) < 최대 rpm({rv.max():.0f}). "
                      f"조각 {over:.1f}% 에서 축이 잘린다(NaN).")
                print(f"     REF_RPM 을 {int(np.ceil(rv.max()/100)*100)} 이상으로 "
                      f"올리고 다시 실행할 것. 중단한다.")
                sys.exit(1)
            print(f"  차수축 잘림 없음 (REF_RPM {REF_RPM:.0f} >= 최대 {rv.max():.0f})")
            print(f"  최저rpm 조각 실효 해상도 {100*rv.min()/REF_RPM:.1f}% "
                  f"(저rpm 은 확대되므로 해상도를 잃는다 = 차수정규화의 대가)")

    if args.rpm_only:
        return

    print("\n그룹 수집 중...")
    groups = collect_groups(BASE_DIR)
    work = sorted(groups.items())
    if args.limit:
        work = work[:args.limit]
    total = len(work)
    print(f"총 그룹 {total:,}개 / 파일 {sum(len(v) for _, v in work):,}개")

    if rpm_idx:
        hit = sum(1 for _, es in work[:200] for _, p, _ in es if p.stem in rpm_idx)
        tot = sum(len(es) for _, es in work[:200])
        print(f"rpm 매칭율(앞 200그룹): {hit}/{tot} = {hit/max(tot,1)*100:.1f}%")
        if hit == 0:
            print("  !! 매칭 0건. 라벨링 경로/파일명 규칙 확인 필요. rpm 컬럼은 NaN 이 된다.")

    print(f"워커 {args.workers}개로 처리 시작\n")

    done = fail = 0
    errs, metas = [], []
    nbytes = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(rpm_idx,)) as ex:
        futures = [ex.submit(process_group, w) for w in work]
        for i, fut in enumerate(as_completed(futures), 1):
            gk, nrows, err, meta = fut.result()
            if err:
                fail += 1
                if len(errs) < 20:
                    errs.append(f"  [실패] {gk}: {err}")
            else:
                done += 1
                metas.append(meta)
                nbytes += meta["bytes"]
            if i % 500 == 0 or i == total:
                el = time.time() - t0
                rate = i / el if el else 0
                eta = (total - i) / rate if rate else 0
                proj = nbytes / max(done, 1) * total / 2**30
                print(f"  {i:,}/{total:,}  ({rate:.0f} grp/s, ETA {eta/60:.1f}분, "
                      f"현재 {nbytes/2**30:.1f}GB, 전체예상 {proj:.1f}GB)")

    for e in errs:
        print(e)
    if fail > len(errs):
        print(f"  ... 외 {fail - len(errs):,}건 더 실패")

    # 슬림메타로 뺀 정보를 인덱스 한 벌에 저장
    if metas:
        idx_df = pd.DataFrame(metas)
        try:
            idx_df.to_parquet(INDEX_PATH, index=False, engine="pyarrow")
        except Exception:
            idx_df.to_csv(INDEX_PATH.with_suffix(".csv"), index=False,
                          encoding="utf-8-sig")
        print(f"\n샘플 인덱스 {len(idx_df):,}행 저장: {INDEX_PATH}")

    print(f"\n완료: 성공 {done:,} / 실패 {fail:,} / {time.time()-t0:.1f}초")
    print(f"출력 용량 {nbytes/2**30:.2f} GB   (그룹당 평균 "
          f"{nbytes/max(done,1)/1024:.0f} KB)")
    if args.limit:
        print(f"  -> 전체 {len(groups):,}그룹 환산 시 약 "
              f"{nbytes/max(done,1)*len(groups)/2**30:,.1f} GB 예상")
    print(f"저장 위치: {TS_OUTPUT_DIR}")
    print("\n다음: python 추출결과_검증.py")


if __name__ == "__main__":
    main()
