"""
2단계: 적응 임계 검증 — 방안1(LUT 역변환)의 순수 효과를 측정 가능하게 만든다

배경
----
1단계에서 컬러맵은 jet으로 확정됐다. 그런데 방안1을 적용하니 band 특징이
포화됐다(Vib_Motor band_start=0 이 3.4% -> 98.4%). 이 상태에서는 4개 특징 중
2개가 죽어 있어 방안1의 순수 효과를 알 수 없다.

예비 진단에서 밝혀진 중요한 사실
--------------------------------
band_start=0 폭증은 '임계가 잘못 걸려서 번진 것'만이 아니다.
LUT로 복원한 실제 에너지를 행별로 보면 이미지 최하단(저주파)이 0.79,
최상단이 0.19로 4배 이상 높다. 즉 저주파 대역이 실제로 강한 신호다.
밝기 방식은 이 사실 자체를 뭉개고 있었기 때문에 band_start가 0이 아니었던 것이고,
LUT 쪽이 물리적으로는 오히려 옳다.

따라서 문제는 두 겹이다.
  (1) 전역 단일 임계로는 '주파수축을 따라 기저 에너지가 크게 다른' 이미지를
      제대로 자를 수 없다.
  (2) band_start/band_end 라는 특징 정의 자체가 "임계를 넘는 최상/최하단 행"이라
      기저가 기울어져 있으면 곧바로 화면 끝에 붙어버린다.

그래서 이 스크립트는 임계 후보를 단순 비교하는 데 그치지 않고,
'행별 기저 제거(detrend)'까지 포함해 검증한다.

임계/전처리 후보
---------------
  pct90        : 현행 (전역 90퍼센타일)          - 기준선
  otsu         : 전역 Otsu 이진화
  bgmode3      : 배경 최빈값 + 3*배경산포
  colpct97     : 열별 97퍼센타일 (시간축 국소)
  rownorm_p90  : 행별 기저 제거 후 전역 p90      << 본命
  rownorm_otsu : 행별 기저 제거 후 Otsu

판정
----
포화율만 낮다고 좋은 게 아니다. 대역이 너무 좁아져도(width~0) 정보가 없다.
따라서 아래를 함께 본다.
  - 포화율(bs0/bemax)   : 낮아야 함
  - 유효폭(width)       : 극단(0 또는 화면전체)이 아니어야 함
  - empty%              : 마스크가 비는 열 비율. 높으면 NaN 폭증
  - DEMAG/NORMAL 분리도 : between/within. 이게 최종 판단 기준
    (클래스당 여러 파일을 써서 클래스 내 산포까지 반영)

실행
----
python 본격\2단계_적응임계_검증.py
결과: stdout + 2단계_적응임계결과.txt
"""

import io
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib

# ===================== CONFIG =====================
BASE_DIR = Path(r"샘플데이터\Sample\01.원천데이터\1.모터_감속기")
OUT_TXT = Path(r"본격\2단계_적응임계결과.txt")

VEHICLES = ["IONIQ", "KONA", "NIRO"]
SENSORS = ["Current_U", "Vib_Motor", "Vib_TM"]
CLASSES = ["NORMAL", "DEMAG", "ECC10", "ECC20", "REDUC"]

MAX_FILES_PER_CLASS = 6     # 클래스 내 산포 추정용
DOWNSCALE = 2               # 속도용 축소. 1이면 원본
GRID_Q = 64
MEDIAN_FILTER = True        # 1단계에서 보간노이즈 확정 -> 기본 ON
MIN_BAND_HEIGHT = 2
# ==================================================

_lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    _lines.append(s)
    print(s)


# ---------------- LUT 역변환 ----------------
def _build():
    cmap = matplotlib.colormaps["jet"]
    lut = (np.asarray([cmap(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.float32)
    q = GRID_Q
    c = (np.arange(q) + 0.5) * (256.0 / q)
    gr, gg, gb = np.meshgrid(c, c, c, indexing="ij")
    pts = np.stack([gr.ravel(), gg.ravel(), gb.ravel()], 1).astype(np.float32)
    idx = np.empty(len(pts), dtype=np.uint8)
    for i in range(0, len(pts), 8192):
        ch = pts[i:i + 8192]
        idx[i:i + 8192] = ((ch[:, None, :] - lut[None, :, :]) ** 2).sum(2).argmin(1)
    return lut, idx.reshape(q, q, q)


_LUT, _GRID = _build()


def median3(a):
    p = np.pad(a, 1, mode="edge")
    st = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]]
                   for i in range(3) for j in range(3)], 0)
    return np.median(st, axis=0)


def energy_lut(rgb):
    q = (rgb.astype(np.int32) * GRID_Q) // 256
    np.clip(q, 0, GRID_Q - 1, out=q)
    e = (_GRID[q[:, :, 0], q[:, :, 1], q[:, :, 2]] / 255.0).astype(np.float32)
    return median3(e) if MEDIAN_FILTER else e


def energy_bright(rgb):
    a = rgb.astype(np.float32) / 255.0
    e = np.sqrt(np.mean(a ** 2, axis=2))
    return (e - e.min()) / (e.max() - e.min() + 1e-8)


# ---------------- 임계 후보 ----------------
def otsu_threshold(e, nb=256):
    h, edges = np.histogram(e.ravel(), bins=nb, range=(0, 1))
    h = h.astype(np.float64)
    p = h / max(h.sum(), 1)
    om = np.cumsum(p)
    mu = np.cumsum(p * np.arange(nb))
    mt = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mt * om - mu) ** 2 / (om * (1 - om))
    return float(edges[np.nan_to_num(sb).argmax() + 1])


def bgmode_threshold(e, k=3.0):
    h, edges = np.histogram(e.ravel(), bins=256, range=(0, 1))
    mode = edges[h.argmax()] + (edges[1] - edges[0]) / 2
    left = e[e <= mode]
    sd = np.sqrt(np.mean((left - mode) ** 2)) if left.size > 10 else float(e.std())
    return float(mode + k * sd)


def row_detrend(e):
    """행(주파수)별 기저 제거.
    각 행의 중앙값을 빼고 행 산포로 나눠, '그 주파수 기준으로 얼마나 튀는가'로 바꾼다.
    저주파가 통째로 강한 구조 때문에 band가 화면 끝에 붙는 문제를 푼다."""
    med = np.median(e, axis=1, keepdims=True)
    mad = np.median(np.abs(e - med), axis=1, keepdims=True) + 1e-6
    z = (e - med) / (1.4826 * mad)
    return z


def thresholds_for(e):
    """이름 -> (임계, 대상배열). 임계는 스칼라 또는 (W,) 벡터."""
    z = row_detrend(e)
    zz = (z - z.min()) / (z.max() - z.min() + 1e-8)   # otsu용 0~1 정규화
    return [
        ("pct90(현행)", np.percentile(e, 90), e),
        ("otsu", otsu_threshold(e), e),
        ("bgmode3", bgmode_threshold(e, 3.0), e),
        ("colpct97", np.percentile(e, 97, axis=0), e),
        ("rownorm_p90", np.percentile(z, 90), z),
        ("rownorm_otsu", otsu_threshold(zz) * (z.max() - z.min()) + z.min(), z),
    ]


# ---------------- 특징 추출 ----------------
def features(arr, th):
    """현행과 동일한 정의로 band 특징 추출. th는 스칼라 또는 (W,)."""
    H, W = arr.shape
    mask = arr >= (th if np.isscalar(th) else th[None, :])
    valid = mask.sum(axis=0) >= MIN_BAND_HEIGHT

    y_top = np.argmax(mask, axis=0)
    y_bot = H - 1 - np.argmax(mask[::-1, :], axis=0)
    peak_y = np.argmax(arr, axis=0)

    peak = (H - 1 - peak_y).astype(np.float64)
    bs = (H - 1 - y_bot).astype(np.float64)
    be = (H - 1 - y_top).astype(np.float64)

    peak[~valid] = np.nan
    bs[~valid] = np.nan
    be[~valid] = np.nan
    return peak, bs, be, valid, H


def summarize(arr, th):
    peak, bs, be, valid, H = features(arr, th)
    with np.errstate(invalid="ignore"):
        width = np.nanmean(be - bs) if valid.any() else np.nan
    return dict(
        bs0=100 * float(np.nanmean(bs == 0)) if valid.any() else np.nan,
        bemax=100 * float(np.nanmean(be >= H - 1)) if valid.any() else np.nan,
        width=float(width) if valid.any() else np.nan,
        empty=100 * float((~valid).mean()),
        desc=descriptor(peak, bs, be),
    )


def descriptor(peak, bs, be):
    v = []
    for x in (peak, bs, be):
        if np.all(np.isnan(x)):
            v += [0.0, 0.0, 0.0, 0.0]
        else:
            v += [np.nanmean(x), np.nanstd(x),
                  np.nanpercentile(x, 10), np.nanpercentile(x, 90)]
    return np.array(v, dtype=np.float64)


def separation(dn, dd):
    """DEMAG vs NORMAL 분리도 = 클래스간거리 / 클래스내산포.
    1 미만이면 클래스 내 변동이 더 커서 판별 불가."""
    if len(dn) < 2 or len(dd) < 2:
        return np.nan
    N, D = np.array(dn), np.array(dd)
    allv = np.vstack([N, D])
    sd = allv.std(0) + 1e-9
    between = np.linalg.norm((N.mean(0) - D.mean(0)) / sd)
    within = (np.linalg.norm(N.std(0) / sd) + np.linalg.norm(D.std(0) / sd)) / 2
    return float(between / (within + 1e-9))


# ---------------- 수집 ----------------
def list_pngs(vehicle, cls, sensor):
    d = BASE_DIR / vehicle / cls
    if not d.exists():
        return []
    return sorted(d.glob(f"*/{sensor}/*.png"))[:MAX_FILES_PER_CLASS]


def load(p):
    a = np.asarray(Image.open(p).convert("RGB"))
    return a[::DOWNSCALE, ::DOWNSCALE]


def main():
    t0 = time.time()
    P("=" * 84)
    P("2단계: 적응 임계 검증 (LUT 역변환 기준)")
    P("=" * 84)
    P(f"차종 {VEHICLES} / median필터 {MEDIAN_FILTER} / 축소 1/{DOWNSCALE} "
      f"/ 클래스당 최대 {MAX_FILES_PER_CLASS}장")
    if not BASE_DIR.exists():
        P(f"!! 경로 없음: {BASE_DIR}")
        return
    P("")

    method_names = [n for n, _, _ in thresholds_for(np.zeros((8, 8), np.float32))]

    # (veh,sensor,method) -> {cls: [desc,...]}, 그리고 포화 통계 누적
    store = {}
    satur = {}
    n_img = 0

    for veh in VEHICLES:
        for sensor in SENSORS:
            for cls in CLASSES:
                for p in list_pngs(veh, cls, sensor):
                    e = energy_lut(load(p))
                    n_img += 1
                    for name, th, arr in thresholds_for(e):
                        st = summarize(arr, th)
                        store.setdefault((veh, sensor, name), {}).setdefault(cls, []).append(st["desc"])
                        satur.setdefault((sensor, name), []).append(
                            (st["bs0"], st["bemax"], st["width"], st["empty"]))
                    if n_img % 20 == 0:
                        P(f"  ... {n_img}장 처리 ({time.time()-t0:.0f}s)")

    # 참고: 현행 밝기 방식 기준선
    base_store = {}
    for veh in VEHICLES:
        for sensor in SENSORS:
            for cls in ("NORMAL", "DEMAG"):
                for p in list_pngs(veh, cls, sensor):
                    e = energy_bright(load(p))
                    st = summarize(e, np.percentile(e, 90))
                    base_store.setdefault((veh, sensor), {}).setdefault(cls, []).append(st["desc"])

    P(f"\n총 {n_img}장 처리 완료 ({time.time()-t0:.0f}s)\n")

    # ---------- 지표 A: 포화/폭 ----------
    P("=" * 84)
    P("[A] 마스크 건전성 (센서 x 임계방식)")
    P("=" * 84)
    P("  bs0/bemax = 대역이 화면 끝에 붙은 비율(%). 낮을수록 좋음")
    P("  width     = 평균 대역폭(px). 0에 가깝거나 화면전체면 정보 없음")
    P("  empty     = 마스크가 빈 열 비율(%). 높으면 NaN 폭증 -> 나쁨")
    P("")
    P(f"  {'센서':11s} {'방식':14s} {'bs0%':>7s} {'bemax%':>7s} {'width':>8s} {'empty%':>7s}")
    for sensor in SENSORS:
        for name in method_names:
            v = satur.get((sensor, name))
            if not v:
                continue
            a = np.array(v, dtype=np.float64)
            with np.errstate(invalid="ignore"):
                m = np.nanmean(a, axis=0)
            P(f"  {sensor:11s} {name:14s} {m[0]:7.1f} {m[1]:7.1f} {m[2]:8.0f} {m[3]:7.1f}")
        P("")

    # ---------- 지표 B: DEMAG/NORMAL 분리도 ----------
    P("=" * 84)
    P("[B] DEMAG vs NORMAL 분리도  << 최종 판단 기준")
    P("=" * 84)
    P("  = 클래스간 거리 / 클래스내 산포.  1 미만 = 판별 불가, 2 이상 = 뚜렷")
    P("")
    header = f"  {'차종/센서':22s} {'현행밝기':>9s}" + "".join(f"{n:>14s}" for n in method_names)
    P(header)
    best_overall = {}
    for veh in VEHICLES:
        for sensor in SENSORS:
            bs_ = base_store.get((veh, sensor), {})
            base_sep = separation(bs_.get("NORMAL", []), bs_.get("DEMAG", []))
            row = f"  {veh + '/' + sensor:22s} {base_sep:9.2f}"
            for name in method_names:
                d = store.get((veh, sensor, name), {})
                sep = separation(d.get("NORMAL", []), d.get("DEMAG", []))
                row += f"{sep:14.2f}"
                if not np.isnan(sep):
                    best_overall.setdefault(name, []).append(sep)
            P(row)
    P("")

    # ---------- 종합 ----------
    P("=" * 84)
    P("[C] 종합")
    P("=" * 84)
    base_all = []
    for veh in VEHICLES:
        for sensor in SENSORS:
            b = base_store.get((veh, sensor), {})
            s = separation(b.get("NORMAL", []), b.get("DEMAG", []))
            if not np.isnan(s):
                base_all.append(s)
    if base_all:
        P(f"  현행(밝기+pct90) 평균 분리도 : {np.mean(base_all):.3f}")
    ranked = sorted(((np.mean(v), k) for k, v in best_overall.items()), reverse=True)
    for m, k in ranked:
        P(f"  LUT + {k:16s} 평균 분리도 : {m:.3f}")
    P("")
    if ranked:
        top_m, top_k = ranked[0]
        ref = np.mean(base_all) if base_all else 0.0
        P(f"  최고: LUT + {top_k} ({top_m:.3f})")
        if top_m >= 2.0:
            P("  >> 판정: 분리 성공. 이 임계로 방안1을 본 파이프라인에 반영할 것.")
        elif top_m >= 1.0 and top_m > ref * 1.3:
            P("  >> 판정: 개선은 있으나 불충분. 방안2(대역 에너지 특징) 병행 필요.")
        else:
            P("  >> 판정: 임계를 바꿔도 DEMAG는 갈라지지 않는다.")
            P("     -> 원인은 임계가 아니라 '특징 정의' 쪽이다.")
            P("        현행 4특징은 위치3 + 스칼라1 뿐이라 에너지 재분배를 담을 그릇이 없다.")
            P("        방안2(주파수 대역별 에너지 비율)로 넘어갈 것.")
    P("")
    P("  주의: [A]에서 포화율이 낮아도 [B]가 안 오르면 그 임계는 의미 없다.")
    P("        반대로 [B]가 올라도 empty%가 높으면 NaN이 늘어 실제 학습에서 불리하다.")
    P("=" * 84)
    P(f"소요 {time.time()-t0:.1f}s")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print(f"\n저장: {OUT_TXT}")


if __name__ == "__main__":
    main()
