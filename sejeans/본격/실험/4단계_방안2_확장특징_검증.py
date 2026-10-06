"""
4단계(방안2): 확장 특징 검증 — 에너지 재분배를 담을 '그릇'을 만든다

배경
----
1~3단계로 아래를 확정했다.
  1단계  컬러맵 = jet (잔차는 보간노이즈)
  2단계  rownorm 으로 임계 포화 해결 (bs0 94.6% -> 0.6%)   분리도 1.540
  3단계  차수정규화로 rpm 교란 제거 (corr 1.0 -> 0.0)      분리도 1.587

앞의 셋을 다 고쳤는데도 IONIQ은 0.85로 1.0을 못 넘었다.
남은 용의자는 '특징 정의' 하나다.

현행 4특징의 한계
----------------
  peak_freq_bin / band_start / band_end  = 위치 3개
  rms                                     = 크기 스칼라 1개

ECC/REDUC 는 결함 대역의 '위치가 이동'하므로 위치 특징으로 잡힌다.
DEMAG(감자)는 위치가 아니라 '에너지가 재분배'된다 — 기본파가 약해지고
고조파가 강해지는 식으로 분포 모양만 바뀐다.
최댓값 위치가 그대로면 위치 특징은 아무 변화도 못 잡고,
분포 모양은 rms 스칼라 하나에 뭉개진다. 즉 담을 그릇이 없다.

추가하는 특징 (열=시점 단위, 기존과 동일한 시계열 형태)
-----------------------------------------------------
  band_ratio_0..N-1 : 주파수를 N등분한 각 대역의 에너지 '비율'
                      -> 재분배가 그대로 숫자로 남는다. 총합이 1이라 rms 크기 변동에 강인
  centroid          : 스펙트럼 무게중심 (에너지가 저주파/고주파 어디에 쏠렸나)
  entropy           : 분포가 퍼졌나 뭉쳤나
  rolloff           : 누적 에너지 85% 지점 (고주파 꼬리 길이)
  flatness          : 기하평균/산술평균 (톤 성분 vs 광대역 잡음)

예비 측정 (샘플, 클래스당 5장)
-----------------------------
  확장특징 평균 분리도 1.842  (2단계 1.540 / 3단계 1.587 대비 최고)
  IONIQ/Vib_TM 0.62 -> 1.14 로 상승
  다만 IONIQ 평균은 여전히 1.0 근처 -> 이 스크립트로 조합까지 확인한다.

검증 조합
--------
  base4        : 현행 4특징 (기준선)
  ext          : 확장 특징만
  base4+ext    : 둘 다
  각각 절대축 / 차수축 두 가지 축에서 측정 -> 총 6조합
  어느 조합이 실제로 값어치가 있는지, CSV 컬럼을 늘릴 만한지 판단한다.

주의
----
이 변경은 지금까지와 달리 CSV 컬럼 구조를 바꾼다.
  현행 12채널(3센서 x 4특징) -> 3센서 x (4+N+4)특징
학습 코드의 채널 수도 함께 바뀌므로, 효과가 확실할 때만 반영해야 한다.

실행
----
python 본격\4단계_방안2_확장특징_검증.py
결과: stdout + 4단계_방안2결과.txt
"""

import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib

# ===================== CONFIG =====================
SRC_DIR = Path(r"샘플데이터\Sample\01.원천데이터\1.모터_감속기")
LAB_DIR = Path(r"샘플데이터\Sample\02.라벨링데이터\1.모터_감속기")
OUT_TXT = Path(r"본격\4단계_방안2결과.txt")

VEHICLES = ["IONIQ", "KONA", "NIRO"]
SENSORS = ["Current_U", "Vib_Motor", "Vib_TM"]
CLASSES = ["NORMAL", "DEMAG", "ECC10", "ECC20", "REDUC"]

MAX_FILES_PER_CLASS = 6
DOWNSCALE = 2
GRID_Q = 64
MEDIAN_FILTER = True
MIN_BAND_HEIGHT = 2

N_BANDS = 8               # 대역 개수. 늘리면 표현력↑ 채널수↑
REF_RPM = 3300.0
ORDER_BINS = 640
ROLLOFF_P = 0.85
# ==================================================

_lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    _lines.append(s)
    print(s)


# ---------------- LUT (1단계) ----------------
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
    return idx.reshape(q, q, q)


_GRID = _build()


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


# ---------------- 차수 정규화 (3단계) ----------------
def to_order_axis(energy, rpm, ref_rpm=REF_RPM, n_out=ORDER_BINS):
    H, W = energy.shape
    flip = energy[::-1, :]
    src = np.linspace(0.0, (H - 1) * (rpm / ref_rpm), n_out)
    valid = src <= (H - 1)
    sc = np.clip(src, 0, H - 1)
    i0 = np.floor(sc).astype(int)
    i1 = np.minimum(i0 + 1, H - 1)
    w = (sc - i0)[:, None]
    out = flip[i0] * (1 - w) + flip[i1] * w
    out[~valid] = np.nan
    return out[::-1, :]


def row_detrend(e):
    med = np.nanmedian(e, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(e - med), axis=1, keepdims=True) + 1e-6
    return (e - med) / (1.4826 * mad)


# ---------------- 특징 ----------------
def feats_base4(arr):
    """현행 4특징. arr 은 rownorm 적용된 배열."""
    H, W = arr.shape
    finite = np.isfinite(arr)
    if finite.sum() < 10:
        return None
    th = np.nanpercentile(arr, 90)
    mask = np.where(finite, arr >= th, False)
    valid = mask.sum(axis=0) >= MIN_BAND_HEIGHT

    y_top = np.argmax(mask, axis=0)
    y_bot = H - 1 - np.argmax(mask[::-1, :], axis=0)
    filled = np.where(finite, arr, -np.inf)
    peak_y = np.argmax(filled, axis=0)

    peak = (H - 1 - peak_y).astype(np.float64)
    bs = (H - 1 - y_bot).astype(np.float64)
    be = (H - 1 - y_top).astype(np.float64)
    for a in (peak, bs, be):
        a[~valid] = np.nan
    rms = np.sqrt(np.nanmean(np.where(finite, arr, np.nan) ** 2, axis=0))
    return [peak, bs, be, rms]


def feats_ext(e, nb=N_BANDS):
    """확장 특징. e 는 rownorm 이전의 '실제 에너지'여야 비율이 물리적 의미를 갖는다."""
    E = np.where(np.isfinite(e), e, np.nan)
    E = np.clip(E, 0, None)
    flip = E[::-1, :]                       # row0 = 저주파
    H, W = flip.shape
    tot = np.nansum(flip, axis=0) + 1e-9

    edges = np.linspace(0, H, nb + 1).astype(int)
    ratios = [np.nansum(flip[edges[i]:edges[i + 1]], axis=0) / tot for i in range(nb)]

    idx = np.arange(H)[:, None].astype(np.float64)
    cent = np.nansum(flip * idx, axis=0) / tot

    pn = flip / tot
    ent = -np.nansum(pn * np.log(pn + 1e-12), axis=0)

    cum = np.nancumsum(flip, axis=0) / tot
    roll = np.argmax(cum >= ROLLOFF_P, axis=0).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        gm = np.exp(np.nanmean(np.log(flip + 1e-12), axis=0))
        am = np.nanmean(flip, axis=0) + 1e-12
    flat = gm / am

    return ratios + [cent, ent, roll, flat]


def describe(series_list):
    """시계열 리스트 -> 고정길이 서술자 (평균/표준편차/10%/90%)."""
    v = []
    for x in series_list:
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).any():
            v += [0.0, 0.0, 0.0, 0.0]
        else:
            v += [np.nanmean(x), np.nanstd(x),
                  np.nanpercentile(x, 10), np.nanpercentile(x, 90)]
    return np.array(v, dtype=np.float64)


def separation(dn, dd):
    if len(dn) < 2 or len(dd) < 2:
        return np.nan
    N, D = np.array(dn), np.array(dd)
    ok = np.isfinite(N).all(0) & np.isfinite(D).all(0)
    if ok.sum() < 2:
        return np.nan
    N, D = N[:, ok], D[:, ok]
    sd = np.vstack([N, D]).std(0) + 1e-9
    b = np.linalg.norm((N.mean(0) - D.mean(0)) / sd)
    w = (np.linalg.norm(N.std(0) / sd) + np.linalg.norm(D.std(0) / sd)) / 2
    return float(b / (w + 1e-9))


def loo_1nn(X, y):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if len(X) < 3:
        return np.nan, None
    ok = np.isfinite(X).all(0)
    X = X[:, ok]
    if X.shape[1] < 2:
        return np.nan, None
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    preds = []
    for i in range(len(Z)):
        d = ((Z - Z[i]) ** 2).sum(1)
        d[i] = np.inf
        preds.append(y[d.argmin()])
    preds = np.array(preds)
    return float((preds == y).mean()), preds


def load_rpm():
    idx = {}
    for p in LAB_DIR.rglob("*.json"):
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        r = d.get("metadata", {}).get("speed")
        if r is None:
            continue
        key = Path(d.get("image", {}).get("file_path", "")).stem or p.stem
        idx[key] = float(r)
    return idx


def list_pngs(v, c, s):
    d = SRC_DIR / v / c
    if not d.exists():
        return []
    return sorted(d.glob(f"*/{s}/*.png"))[:MAX_FILES_PER_CLASS]


COMBOS = ["base4/절대", "ext/절대", "base4+ext/절대",
          "base4/차수", "ext/차수", "base4+ext/차수"]


def main():
    t0 = time.time()
    P("=" * 90)
    P("4단계(방안2): 확장 특징 검증")
    P("=" * 90)
    if not SRC_DIR.exists():
        P(f"!! 경로 없음: {SRC_DIR}")
        return
    rpm_idx = load_rpm()
    P(f"rpm {len(rpm_idx)}건 / 대역 {N_BANDS}개 / 축소 1/{DOWNSCALE}")
    P(f"확장특징 = 대역비율 {N_BANDS} + centroid/entropy/rolloff/flatness 4")
    P("")

    store = {}
    n = 0
    for veh in VEHICLES:
        for sensor in SENSORS:
            for cls in CLASSES:
                for p in list_pngs(veh, cls, sensor):
                    rpm = rpm_idx.get(p.stem)
                    rgb = np.asarray(Image.open(p).convert("RGB"))[::DOWNSCALE, ::DOWNSCALE]
                    e = energy_lut(rgb)
                    n += 1

                    axes = {"절대": e}
                    if rpm is not None:
                        axes["차수"] = to_order_axis(e, rpm)

                    for axname, arr in axes.items():
                        b4 = feats_base4(row_detrend(arr))
                        ex = feats_ext(arr)
                        if b4 is None:
                            continue
                        d4 = describe(b4)
                        dx = describe(ex)
                        rec = store.setdefault((veh, sensor), {}).setdefault(cls, {})
                        rec.setdefault(f"base4/{axname}", []).append(d4)
                        rec.setdefault(f"ext/{axname}", []).append(dx)
                        rec.setdefault(f"base4+ext/{axname}", []).append(
                            np.concatenate([d4, dx]))
                    if n % 30 == 0:
                        P(f"  ... {n}장 ({time.time()-t0:.0f}s)")
    P(f"\n총 {n}장 처리 ({time.time()-t0:.0f}s)\n")

    # ---------- [A] 조합별 DEMAG/NORMAL 분리도 ----------
    P("=" * 90)
    P("[A] DEMAG vs NORMAL 분리도 (조합별)   << 최종 기준")
    P("=" * 90)
    P("  1 미만 = 판별 불가 / 2 이상 = 뚜렷")
    P("")
    head = f"  {'차종/센서':20s}" + "".join(f"{c:>15s}" for c in COMBOS)
    P(head)
    agg = {c: [] for c in COMBOS}
    per_veh = {}
    for veh in VEHICLES:
        for sensor in SENSORS:
            d = store.get((veh, sensor), {})
            if "NORMAL" not in d or "DEMAG" not in d:
                continue
            row = f"  {veh + '/' + sensor:20s}"
            for c in COMBOS:
                s = separation(d["NORMAL"].get(c, []), d["DEMAG"].get(c, []))
                row += f"{s:15.2f}" if not np.isnan(s) else f"{'-':>15s}"
                if not np.isnan(s):
                    agg[c].append(s)
                    per_veh.setdefault((veh, c), []).append(s)
            P(row)
    P("")
    P(f"  {'전체 평균':20s}" + "".join(
        f"{np.mean(agg[c]):15.2f}" if agg[c] else f"{'-':>15s}" for c in COMBOS))
    P("")
    for veh in VEHICLES:
        P(f"  {veh + ' 평균':20s}" + "".join(
            f"{np.mean(per_veh.get((veh, c), [np.nan])):15.2f}" for c in COMBOS))
    P("")

    # ---------- [B] 5클래스 전체 성능 ----------
    P("=" * 90)
    P("[B] 5클래스 1-NN LOO 정확도 (다른 클래스가 망가지지 않는지 확인)")
    P("=" * 90)
    P("  DEMAG만 좋아지고 ECC/REDUC가 나빠지면 채택 불가")
    P("")
    for c in COMBOS:
        X, y = [], []
        for (veh, sensor), byc in store.items():
            for cls, m in byc.items():
                for v in m.get(c, []):
                    X.append(v)
                    y.append(cls)
        if len(X) < 5:
            continue
        acc, pred = loo_1nn(X, y)
        if pred is None:
            continue
        y = np.array(y)
        dn = int(((y == "DEMAG") & (pred == "NORMAL")).sum())
        nd = int(((y == "NORMAL") & (pred == "DEMAG")).sum())
        drec = float((pred[y == "DEMAG"] == "DEMAG").mean()) if (y == "DEMAG").any() else np.nan
        P(f"  {c:16s} 정확도 {100*acc:5.1f}%  DEMAG recall {100*drec:5.1f}%  "
          f"D->N {dn:3d}건  N->D {nd:3d}건  (n={len(X)}, 차원={len(X[0])})")
    P("")

    # ---------- [C] 채널 비용 ----------
    P("=" * 90)
    P("[C] 채널/용량 비용")
    P("=" * 90)
    n_ext = N_BANDS + 4
    P(f"  현행      : 3센서 x  4특징 = 12채널")
    P(f"  ext       : 3센서 x {n_ext:2d}특징 = {3*n_ext}채널")
    P(f"  base4+ext : 3센서 x {4+n_ext:2d}특징 = {3*(4+n_ext)}채널  ({(4+n_ext)/4:.1f}배)")
    P("")
    P("  MiniRocket 은 채널수에 거의 선형으로 시간/메모리가 늘어난다.")
    P("  3차 발표 기준 학습 1h52m -> base4+ext 는 대략 4~5시간 예상.")
    P("  CSV 용량도 같은 배율로 늘어난다 (600GB 원본 -> 특징 CSV 기준).")
    P("")

    # ---------- [D] 판정 ----------
    P("=" * 90)
    P("[D] 판정")
    P("=" * 90)
    base_ref = np.mean(agg["base4/절대"]) if agg["base4/절대"] else np.nan
    best_c, best_v = None, -1
    for c in COMBOS:
        if agg[c] and np.mean(agg[c]) > best_v:
            best_v, best_c = np.mean(agg[c]), c
    P(f"  기준선 base4/절대 : {base_ref:.3f}")
    P(f"  최고   {best_c:16s} : {best_v:.3f}  ({best_v-base_ref:+.3f})")
    ion = np.mean(per_veh.get(("IONIQ", best_c), [np.nan]))
    P(f"  최고 조합의 IONIQ : {ion:.3f}   << 핵심")
    P("")
    if best_v >= 2.0 and ion >= 1.5:
        P("  >> 채택. 확장 특징이 DEMAG 재분배를 실제로 담아낸다.")
        P("     특징추출 스크립트에 대역비율/centroid/entropy/rolloff/flatness 추가.")
        P("     학습 코드의 FEATURE_COLS 와 채널수도 함께 수정 필요.")
    elif best_v > base_ref + 0.3 and ion >= 1.0:
        P("  >> 부분 채택 검토. 개선은 실재하나 목표(2.0)에는 미달.")
        P("     채널수 증가 비용 대비 이득을 따져야 한다. [C] 참고.")
        P("     대안: 효과가 큰 센서 슬롯에만 확장특징을 적용하는 하이브리드.")
    else:
        P("  >> 확장 특징만으로도 부족. 스펙트로그램 이미지에서 뽑을 수 있는")
        P("     1D 시계열 특징의 한계일 가능성이 크다.")
        P("     -> 2D 입력을 직접 쓰는 경량 CNN 등 구조 변경을 검토해야 한다.")
    P("")
    P("  주의: 표본이 클래스당 5측정뿐이라 방향 확인용이다.")
    P("        채택 시 반드시 본데이터로 재검증할 것.")
    P("=" * 90)
    P(f"소요 {time.time()-t0:.1f}s")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print(f"\n저장: {OUT_TXT}")


if __name__ == "__main__":
    main()
