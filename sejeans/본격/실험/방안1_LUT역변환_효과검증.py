"""
방안 1 (jet 컬러맵 LUT 역변환) 효과 검증 스크립트 - 샘플데이터 전용

목적
----
"밝기 RMS -> 에너지" 대신 "jet LUT 역변환 -> 에너지"를 썼을 때
DEMAG vs NORMAL 판별력이 실제로 올라가는지를 **본 파이프라인 수정 전에** 확인한다.

핵심 비교 대상은 '이미지'가 아니라 '모델이 먹는 4개 특징'이다.
따라서 두 방식으로 각각 CSV 특징을 뽑고, 동일 지표로 비교한다.

측정 지표
--------
1) DEMAG-NORMAL 분리도 (표준화 거리) : 클수록 좋음
2) 다른 클래스(ECC10/ECC20/REDUC) 분리도 : 방안1이 이걸 망가뜨리면 안 됨
3) band_start/band_end 포화율 : 90퍼센타일 임계가 전대역을 잡아버리는 비율
4) 소규모 LOO 분류 정확도 (샘플 수가 적으므로 참고용)

실행
----
python 본격\방안1_LUT역변환_효과검증.py

결과는 stdout + 방안1_검증결과.txt 로 저장된다.
"""

import io
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
from PIL import Image

# ===================== CONFIG =====================
BASE_DIR = Path(r"샘플데이터\Sample\01.원천데이터\1.모터_감속기")
OUT_TXT = Path(r"본격\방안1_검증결과.txt")

THRESHOLD_PERCENTILE = 90
MIN_BAND_HEIGHT = 2
FREQ_FROM_BOTTOM = True

GRID_Q = 64          # LUT 역변환 복셀 격자 해상도 (64 -> 0.26MB, 오차 <= LUT 자체 간격)
CHUNK_PX = 200_000   # 혹시 정밀모드 쓸 때 메모리 폭발 방지용
USE_EXACT = False    # True면 픽셀당 256색 완전탐색(느림). False면 복셀격자(빠름, 사실상 동일)
# ==================================================

_out_lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    _out_lines.append(s)
    print(s)


# --------------------------------------------------
# jet LUT 역변환
# --------------------------------------------------
def build_jet_lut():
    """matplotlib jet -> (256,3) 0~255 float32 LUT."""
    cmap = matplotlib.colormaps["jet"]          # cm.get_cmap은 최신 mpl에서 deprecated
    return (np.asarray([cmap(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.float32)


def build_voxel_grid(lut, q=GRID_Q):
    """RGB 공간을 q^3 격자로 나눠 각 셀의 최근접 LUT 인덱스를 미리 계산.
    한 번 만들어두면 이미지당 O(1) 룩업이라 매우 빠르다."""
    centers = (np.arange(q) + 0.5) * (256.0 / q)
    gr, gg, gb = np.meshgrid(centers, centers, centers, indexing="ij")
    pts = np.stack([gr.ravel(), gg.ravel(), gb.ravel()], 1).astype(np.float32)
    idx = np.empty(len(pts), dtype=np.uint8)
    for i in range(0, len(pts), 8192):
        ch = pts[i:i + 8192]
        d = ((ch[:, None, :] - lut[None, :, :]) ** 2).sum(2)
        idx[i:i + 8192] = d.argmin(1)
    return idx.reshape(q, q, q)


_LUT = build_jet_lut()
_GRID = build_voxel_grid(_LUT, GRID_Q)


def image_to_energy_lut(image):
    """[방안 1] jet 역변환으로 실제 스칼라 에너지(0~1) 복원."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    arr = arr[:, :, :3]

    if USE_EXACT:
        h, w, _ = arr.shape
        flat = arr.reshape(-1, 3).astype(np.float32)
        res = np.empty(len(flat), dtype=np.float32)
        # 반드시 청크로. 안 하면 (H*W, 256, 3) 배열이 수 GiB 잡힌다.
        for i in range(0, len(flat), CHUNK_PX):
            ch = flat[i:i + CHUNK_PX]
            d = ((ch[:, None, :] - _LUT[None, :, :]) ** 2).sum(2)
            res[i:i + CHUNK_PX] = d.argmin(1)
        return (res / 255.0).reshape(h, w).astype(np.float32)

    q = (arr.astype(np.int32) * GRID_Q) // 256
    np.clip(q, 0, GRID_Q - 1, out=q)
    return (_GRID[q[:, :, 0], q[:, :, 1], q[:, :, 2]] / 255.0).astype(np.float32)


def image_to_energy_bright(image):
    """[현행] 채널 밝기 RMS. jet에서 에너지와 비단조 -> 정보 손실."""
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    arr = arr[:, :, :3]
    energy = np.sqrt(np.mean(arr ** 2, axis=2))
    return (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)


# --------------------------------------------------
# 특징 추출 (본 파이프라인과 동일 로직)
# --------------------------------------------------
def extract_features(energy, threshold_percentile=THRESHOLD_PERCENTILE,
                     min_band_height=MIN_BAND_HEIGHT,
                     freq_from_bottom=FREQ_FROM_BOTTOM):
    H, W = energy.shape
    threshold = np.percentile(energy, threshold_percentile)
    mask = energy >= threshold

    rms = np.sqrt(np.mean(energy ** 2, axis=0))
    count = mask.sum(axis=0)
    valid = count >= min_band_height

    peak_y = np.argmax(energy, axis=0)
    y_top = np.argmax(mask, axis=0)
    y_bot = H - 1 - np.argmax(mask[::-1, :], axis=0)

    if freq_from_bottom:
        peak_freq_bin = (H - 1 - peak_y).astype(np.float32)
        band_start = (H - 1 - y_bot).astype(np.float32)
        band_end = (H - 1 - y_top).astype(np.float32)
    else:
        peak_freq_bin = peak_y.astype(np.float32)
        band_start = y_top.astype(np.float32)
        band_end = y_bot.astype(np.float32)

    nan = np.float32(np.nan)
    return pd.DataFrame({
        "peak_freq_bin": np.where(valid, peak_freq_bin, nan),
        "band_start": np.where(valid, band_start, nan),
        "band_end": np.where(valid, band_end, nan),
        "rms": rms.astype(np.float32),
    })


def merge_horizontal(paths):
    imgs = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        imgs.append(im)
    tw = sum(i.width for i in imgs)
    mh = max(i.height for i in imgs)
    merged = Image.new("RGB", (tw, mh), (255, 255, 255))
    x = 0
    for im in imgs:
        merged.paste(im, (x, 0))
        x += im.width
        im.close()
    return merged


# --------------------------------------------------
# 비교 지표
# --------------------------------------------------
def descriptor(df):
    """CSV 한 장 -> 고정 길이 서술자 (평균/표준편차/10%/90% x 4열)."""
    v = []
    for col in ["peak_freq_bin", "band_start", "band_end", "rms"]:
        x = df[col].values.astype(np.float64)
        if np.all(np.isnan(x)):
            v += [0.0, 0.0, 0.0, 0.0]
        else:
            v += [np.nanmean(x), np.nanstd(x),
                  np.nanpercentile(x, 10), np.nanpercentile(x, 90)]
    return np.array(v)


def separability(desc_by_class):
    """각 클래스 중심과 NORMAL 중심 사이의 표준화 거리."""
    cent = {c: np.mean(v, 0) for c, v in desc_by_class.items() if v}
    if "NORMAL" not in cent or len(cent) < 2:
        return {}
    allv = np.stack(list(cent.values()))
    sd = allv.std(0) + 1e-9
    n = cent["NORMAL"]
    return {c: float(np.linalg.norm((cent[c] - n) / sd))
            for c in cent if c != "NORMAL"}


def loo_accuracy(X, y):
    """샘플이 적으므로 1-NN Leave-One-Out. 참고 지표."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if len(X) < 3:
        return float("nan"), None
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    correct = 0
    preds = []
    for i in range(len(Z)):
        d = ((Z - Z[i]) ** 2).sum(1)
        d[i] = np.inf
        p = y[d.argmin()]
        preds.append(p)
        correct += (p == y[i])
    return correct / len(Z), np.array(preds)


def confusion(y, pred, labels):
    M = pd.DataFrame(0, index=labels, columns=labels)
    for a, b in zip(y, pred):
        M.loc[a, b] += 1
    return M


# --------------------------------------------------
def collect_groups():
    """(vehicle, cls, sensor, group_key) -> [png paths]  (split 병합 단위)"""
    import re
    PAT = re.compile(r"^(\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})_(\d{2})_(zsplit\d{3})_(\d{3})_(.+)\.png$")
    groups = defaultdict(list)
    for p in BASE_DIR.rglob("*.png"):
        m = PAT.match(p.name)
        if not m:
            continue
        ts, seq, zs, split, sensor = m.groups()
        rel = p.relative_to(BASE_DIR).parts
        vehicle, cls = rel[0], rel[1]
        key = (vehicle, cls, sensor, f"{ts}_{seq}_{zs}_{sensor}")
        groups[key].append((int(split), p))
    return {k: [p for _, p in sorted(v)] for k, v in groups.items()}


def main():
    t0 = time.time()
    P("=" * 78)
    P("방안 1 (jet LUT 역변환) 효과 검증 - 샘플데이터")
    P("=" * 78)
    P(f"LUT 격자 Q={GRID_Q}, 정밀모드={USE_EXACT}")

    groups = collect_groups()
    P(f"그룹 {len(groups)}개 발견\n")
    if not groups:
        P("!! PNG를 찾지 못했습니다. BASE_DIR 경로를 확인하세요.")
        return

    # jet 여부 사전 점검 -----------------------------------------
    P("--- [사전점검] 센서별 컬러맵이 정말 jet인가 ---")
    by_sensor = defaultdict(list)
    for (veh, cls, sensor, gk), paths in groups.items():
        by_sensor[sensor].append(paths[0])
    for sensor, paths in sorted(by_sensor.items()):
        arr = np.asarray(Image.open(paths[0]).convert("RGB"), dtype=np.float32)
        px = arr.reshape(-1, 3)
        sub = px[np.random.default_rng(0).choice(len(px), min(20000, len(px)), replace=False)]
        d = ((sub[:, None, :] - _LUT[None, :, :]) ** 2).sum(2).min(1)
        rmse = float(np.sqrt(d.mean()))
        P(f"  {sensor:10s} jet LUT 평균거리 {rmse:6.2f} (0에 가까울수록 jet 확실) "
          f"고유색 {len(np.unique(px, axis=0)):5d}")
    P("")

    # 두 방식으로 특징 추출 ---------------------------------------
    results = {"BRIGHT(현행)": defaultdict(lambda: defaultdict(list)),
               "LUT(방안1)": defaultdict(lambda: defaultdict(list))}
    satur = {"BRIGHT(현행)": [], "LUT(방안1)": []}
    keys = sorted(groups.keys())

    for i, key in enumerate(keys, 1):
        veh, cls, sensor, gk = key
        merged = merge_horizontal(groups[key])
        for name, fn in (("BRIGHT(현행)", image_to_energy_bright),
                         ("LUT(방안1)", image_to_energy_lut)):
            e = fn(merged)
            df = extract_features(e)
            results[name][(veh, sensor)][cls].append(descriptor(df))
            satur[name].append({
                "sensor": sensor,
                "bs0": 100 * float((df.band_start == 0).mean()),
                "be_max": 100 * float((df.band_end >= e.shape[0] - 1).mean()),
            })
        merged.close()
        if i % 10 == 0 or i == len(keys):
            P(f"  처리 {i}/{len(keys)}  ({time.time()-t0:.0f}s)")
    P("")

    # 지표 1: 포화율 ---------------------------------------------
    P("--- [지표1] band 특징 포화율 (%, 낮을수록 정보량 많음) ---")
    P(f"{'방식':14s} {'센서':11s} {'band_start=0':>13s} {'band_end=max':>13s}")
    for name in results:
        s = pd.DataFrame(satur[name])
        for sensor, g in s.groupby("sensor"):
            P(f"{name:14s} {sensor:11s} {g.bs0.mean():12.1f}% {g.be_max.mean():12.1f}%")
    P("")

    # 지표 2: 분리도 ---------------------------------------------
    P("--- [지표2] NORMAL 대비 클래스 분리도 (표준화 거리, 클수록 좋음) ---")
    P("    * DEMAG 항목이 핵심. 다른 클래스가 함께 떨어지면 안 됨.")
    for (veh, sensor) in sorted(set(k for name in results for k in results[name])):
        line_b = separability(results["BRIGHT(현행)"][(veh, sensor)])
        line_l = separability(results["LUT(방안1)"][(veh, sensor)])
        if not line_b or not line_l:
            continue
        P(f"\n  [{veh} / {sensor}]")
        P(f"    {'클래스':8s} {'현행':>8s} {'방안1':>8s} {'변화':>9s}")
        for c in ["DEMAG", "ECC10", "ECC20", "REDUC"]:
            if c in line_b and c in line_l:
                b, l = line_b[c], line_l[c]
                mark = "  <== 핵심" if c == "DEMAG" else ""
                P(f"    {c:8s} {b:8.3f} {l:8.3f} {(l-b):+9.3f}{mark}")

    # 지표 3: LOO 분류 -------------------------------------------
    P("\n--- [지표3] 1-NN Leave-One-Out 정확도 (샘플 적어 참고용) ---")
    labels = ["DEMAG", "ECC10", "ECC20", "NORMAL", "REDUC"]
    for name in results:
        X, y = [], []
        for (veh, sensor), byc in results[name].items():
            for c, vs in byc.items():
                for v in vs:
                    X.append(v)
                    y.append(c)
        acc, pred = loo_accuracy(X, y)
        P(f"\n  {name}: 전체 정확도 {acc:.3f}  (n={len(X)})")
        if pred is not None:
            M = confusion(np.array(y), pred, labels)
            dn = M.loc["DEMAG", "NORMAL"] if "DEMAG" in M.index else 0
            nd = M.loc["NORMAL", "DEMAG"] if "NORMAL" in M.index else 0
            P(f"    DEMAG->NORMAL 오분류 {dn}건 / NORMAL->DEMAG {nd}건")
            P("    혼동행렬(행=정답, 열=예측):")
            for line in M.to_string().split("\n"):
                P("      " + line)

    P("\n" + "=" * 78)
    P("판단 기준")
    P("  - [지표2]에서 DEMAG 분리도가 뚜렷이 증가하고 다른 클래스가 유지/증가하면 -> 방안1 채택")
    P("  - DEMAG 증가폭이 미미하면 -> 원인은 컬러맵이 아니라 특징 설계(방안2) 쪽")
    P("  - 샘플이 81개뿐이라 [지표3]은 방향 참고용. 최종 판단은 [지표2] 기준.")
    P("=" * 78)
    P(f"소요 {time.time()-t0:.1f}s")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_out_lines))
    print(f"\n저장: {OUT_TXT}")


if __name__ == "__main__":
    main()
