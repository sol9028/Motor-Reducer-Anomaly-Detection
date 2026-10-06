"""
3단계: 차수(order) 정규화 검증 — rpm 교란 제거

배경
----
2단계에서 임계 포화는 해결됐으나(bs0 94.6% -> 0.6%) DEMAG 분리도는 평균 1.77에
그쳤다. 차종별로 뜯어보니 원인이 드러났다.

  차종    DEMAG rpm      NORMAL rpm     중첩    분리도
  IONIQ   2626~2986      2041~3574      100%     0.82
  KONA    2699~4107      2098~3201       40%     1.64
  NIRO    4591~5693      1951~3398        0%     3.17
  -> rpm 중첩률 vs 분리도 상관 -0.958

즉 NIRO가 잘 되는 건 결함을 잡아서가 아니라 rpm이 안 겹쳐서다.
실제로 rpm 값 하나만으로 NIRO는 DEMAG/NORMAL을 100% 분류한다.
현행 특징이 전부 '절대 주파수 bin 위치'이므로 모델은 결함이 아니라 회전수를 학습한다.

라벨링 JSON이 준 결정적 근거
---------------------------
annotations.polygons(전문가가 표시한 결함 대역)의 y중심과 rpm의 상관:
  IONIQ DEMAG 0.984 / KONA DEMAG 1.000 / NIRO DEMAG 0.999
  freq_bin / rpm 비율이 DEMAG는 0.1333~0.1344 (std 0.0004~0.0021) 로 차종 무관 일정

즉 결함 주파수는 rpm에 정비례한다. 이것이 차수 정규화의 물리적 근거다.
  freq_bin ≈ K × rpm    (DEMAG의 경우 K ≈ 0.134)
  order = freq_bin / (K_ref × rpm)  로 바꾸면 rpm이 달라도 같은 결함은 같은 위치

클래스별 K (결함 차수가 다르므로 값이 다른 게 정상):
  DEMAG ≈ 0.134,  REDUC ≈ 0.141~0.179,  ECC10/ECC20 ≈ 0.109
  ECC 계열은 corr이 0.58~0.84로 낮다 -> 편심은 rpm 비례가 덜 뚜렷

무엇을 하는가
------------
주파수축(세로)을 rpm에 비례해 리샘플링한다.
  변환 전: y축 = 절대 주파수 bin (0~1279)
  변환 후: y축 = 차수 (회전수의 몇 배인가)
rpm이 높으면 스펙트럼이 위로 늘어나 있으므로 압축, 낮으면 확장한다.

검증 항목
--------
  [A] 차수축에서 결함 대역 위치가 rpm과 무관해지는가 (정규화가 작동했나)
  [B] DEMAG vs NORMAL 분리도가 오르는가  << 최종 기준
  [C] rpm 단독 분류 정확도가 떨어지는가 (교란이 제거됐다는 반증)

실행
----
python 본격\3단계_차수정규화_검증.py
결과: stdout + 3단계_차수정규화결과.txt
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
OUT_TXT = Path(r"본격\3단계_차수정규화결과.txt")

VEHICLES = ["IONIQ", "KONA", "NIRO"]
SENSORS = ["Current_U", "Vib_Motor", "Vib_TM"]
CLASSES = ["NORMAL", "DEMAG", "ECC10", "ECC20", "REDUC"]

MAX_FILES_PER_CLASS = 6
DOWNSCALE = 2
GRID_Q = 64
MEDIAN_FILTER = True
MIN_BAND_HEIGHT = 2

# 차수 정규화 기준 rpm. 모든 이미지를 이 회전수 기준으로 맞춘다.
# 데이터 전체 rpm 중앙값 근처로 잡아야 리샘플 왜곡이 최소가 된다.
REF_RPM = 3300.0
ORDER_BINS = 640          # 차수축 해상도 (원본 1280의 절반)
MAX_ORDER_RATIO = 1.0     # 기준 대비 표현할 최대 배율
# ==================================================

_lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    _lines.append(s)
    print(s)


# ---------------- LUT 역변환 (1단계 확정) ----------------
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


# ---------------- 차수 정규화 ----------------
def to_order_axis(energy, rpm, ref_rpm=REF_RPM,
                  n_out=ORDER_BINS, max_ratio=MAX_ORDER_RATIO):
    """주파수축(세로)을 rpm 비례로 리샘플링.

    energy 는 (H, W), 행 0 = 이미지 상단(고주파), 행 H-1 = 하단(저주파).
    내부에서 '하단 기준 bin'으로 뒤집어 계산한 뒤 같은 방향으로 되돌린다.

    차수 o 에 해당하는 원본 bin = o * (rpm/ref_rpm) * (H-1) * max_ratio
    rpm 이 높으면 같은 차수가 더 위쪽 bin에 있으므로 압축된다.
    """
    H, W = energy.shape
    flip = energy[::-1, :]                       # 행 0 = 저주파(하단)
    scale = (rpm / ref_rpm)
    src = np.linspace(0.0, (H - 1) * max_ratio * scale, n_out)

    valid = src <= (H - 1)
    src_c = np.clip(src, 0, H - 1)
    i0 = np.floor(src_c).astype(int)
    i1 = np.minimum(i0 + 1, H - 1)
    w = (src_c - i0)[:, None]

    out = flip[i0] * (1 - w) + flip[i1] * w
    out[~valid] = np.nan                          # 기준 rpm보다 빠르면 상단이 비어버림
    return out[::-1, :], float(valid.mean())


# ---------------- 특징 ----------------
def row_detrend(e):
    """2단계에서 채택된 행별 기저 제거."""
    med = np.nanmedian(e, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(e - med), axis=1, keepdims=True) + 1e-6
    return (e - med) / (1.4826 * mad)


def band_features(arr):
    """rownorm + 전역 p90 (2단계 최선 조합 중 empty%가 0인 것)."""
    H, W = arr.shape
    finite = np.isfinite(arr)
    if finite.sum() < 10:
        return np.full(12, np.nan)
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

    v = []
    for x in (peak, bs, be):
        if np.all(np.isnan(x)):
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
    between = np.linalg.norm((N.mean(0) - D.mean(0)) / sd)
    within = (np.linalg.norm(N.std(0) / sd) + np.linalg.norm(D.std(0) / sd)) / 2
    return float(between / (within + 1e-9))


# ---------------- rpm 로딩 ----------------
def load_rpm_index():
    """JSON을 훑어 (차종, 클래스, 센서, 파일stem) -> rpm 매핑."""
    idx = {}
    poly = {}
    for p in LAB_DIR.rglob("*.json"):
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        meta = d.get("metadata", {})
        img = d.get("image", {})
        rpm = meta.get("speed")
        if rpm is None:
            continue
        key = Path(img.get("file_path", "")).stem or p.stem
        idx[key] = float(rpm)

        pol = d.get("annotations", {}).get("polygons") or []
        ys = []
        for q in pol:
            a = np.asarray(q, dtype=float)
            if a.ndim == 2 and a.shape[1] == 2 and len(a):
                ys.append(a[:, 1])
        if ys:
            poly[key] = (float(np.concatenate(ys).mean()),
                         float(img.get("height", 1280)))
    return idx, poly


def list_pngs(vehicle, cls, sensor):
    d = SRC_DIR / vehicle / cls
    if not d.exists():
        return []
    return sorted(d.glob(f"*/{sensor}/*.png"))[:MAX_FILES_PER_CLASS]


def main():
    t0 = time.time()
    P("=" * 84)
    P("3단계: 차수(order) 정규화 검증")
    P("=" * 84)
    if not SRC_DIR.exists():
        P(f"!! 경로 없음: {SRC_DIR}")
        return
    rpm_idx, poly_idx = load_rpm_index()
    P(f"rpm 인덱스 {len(rpm_idx)}건 / 폴리곤 {len(poly_idx)}건 로드")
    P(f"기준 rpm {REF_RPM:.0f} / 차수축 {ORDER_BINS}bin / 축소 1/{DOWNSCALE}")
    P("")

    # ---------- [A] 정규화 작동 확인 ----------
    P("=" * 84)
    P("[A] 결함 대역 위치가 rpm과 무관해지는가 (전문가 폴리곤 기준)")
    P("=" * 84)
    P("  절대축에서는 corr(rpm, 위치)가 1에 가깝다 = rpm에 끌려다님")
    P("  차수축에서 corr이 0에 가까워지면 정규화 성공")
    P("")
    P(f"  {'차종':7s} {'클래스':7s} {'n':>3s} {'절대축 corr':>11s} {'차수축 corr':>11s}")
    for veh in VEHICLES:
        for cls in CLASSES:
            rec = []
            for sensor in SENSORS:
                for p in list_pngs(veh, cls, sensor):
                    if p.stem not in poly_idx or p.stem not in rpm_idx:
                        continue
                    ycen, H = poly_idx[p.stem]
                    rpm = rpm_idx[p.stem]
                    fb = (H - 1) - ycen                       # 하단기준 절대 bin
                    order = fb / (rpm / REF_RPM)              # 차수축 위치
                    rec.append((rpm, fb, order))
            if len(rec) < 3:
                continue
            a = np.array(rec)
            c_abs = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
            c_ord = np.corrcoef(a[:, 0], a[:, 2])[0, 1]
            mark = "  <== 개선" if abs(c_ord) < abs(c_abs) - 0.15 else ""
            P(f"  {veh:7s} {cls:7s} {len(rec):3d} {c_abs:11.3f} {c_ord:11.3f}{mark}")
    P("")

    # ---------- 특징 추출 ----------
    P("=" * 84)
    P("[B] DEMAG vs NORMAL 분리도  << 최종 기준")
    P("=" * 84)
    store = {}
    n_img = 0
    miss = 0
    covers = []
    for veh in VEHICLES:
        for sensor in SENSORS:
            for cls in CLASSES:
                for p in list_pngs(veh, cls, sensor):
                    rpm = rpm_idx.get(p.stem)
                    if rpm is None:
                        miss += 1
                        continue
                    rgb = np.asarray(Image.open(p).convert("RGB"))[::DOWNSCALE, ::DOWNSCALE]
                    e = energy_lut(rgb)
                    n_img += 1

                    # (1) 절대축 (2단계 최선 = rownorm)
                    f_abs = band_features(row_detrend(e))
                    # (2) 차수축
                    eo, cov = to_order_axis(e, rpm)
                    covers.append(cov)
                    f_ord = band_features(row_detrend(eo))

                    store.setdefault((veh, sensor), {}).setdefault(cls, {"abs": [], "ord": []})
                    store[(veh, sensor)][cls]["abs"].append(f_abs)
                    store[(veh, sensor)][cls]["ord"].append(f_ord)
                    if n_img % 30 == 0:
                        P(f"  ... {n_img}장 ({time.time()-t0:.0f}s)")
    if miss:
        P(f"  ! rpm 없어 건너뛴 파일 {miss}개")
    P(f"  차수축 유효 커버리지 평균 {100*np.mean(covers):.1f}% "
      f"(낮으면 REF_RPM을 올릴 것)")
    P("")
    P(f"  {'차종/센서':22s} {'절대축(2단계)':>14s} {'차수축(3단계)':>14s} {'변화':>9s}")
    abs_all, ord_all = [], []
    per_veh = {}
    for veh in VEHICLES:
        for sensor in SENSORS:
            d = store.get((veh, sensor))
            if not d or "NORMAL" not in d or "DEMAG" not in d:
                continue
            sa = separation(d["NORMAL"]["abs"], d["DEMAG"]["abs"])
            so = separation(d["NORMAL"]["ord"], d["DEMAG"]["ord"])
            if np.isnan(sa) or np.isnan(so):
                continue
            abs_all.append(sa)
            ord_all.append(so)
            per_veh.setdefault(veh, []).append((sa, so))
            mark = "  <==" if so > sa + 0.3 else ("  !!" if so < sa - 0.3 else "")
            P(f"  {veh + '/' + sensor:22s} {sa:14.2f} {so:14.2f} {so-sa:+9.2f}{mark}")
    P("")
    P(f"  {'차종별 평균':22s}")
    for veh in VEHICLES:
        v = per_veh.get(veh)
        if not v:
            continue
        a = np.mean([x[0] for x in v])
        o = np.mean([x[1] for x in v])
        P(f"  {veh:22s} {a:14.2f} {o:14.2f} {o-a:+9.2f}")
    P("")

    # ---------- [C] rpm 단독 분류력 ----------
    P("=" * 84)
    P("[C] rpm 단독 분류 정확도 (교란 강도의 척도)")
    P("=" * 84)
    P("  높을수록 '모델이 결함 아닌 회전수를 학습할 여지'가 크다는 뜻")
    P("")
    for veh in VEHICLES:
        vals, labs = [], []
        for sensor in ["Current_U"]:
            for cls in ("DEMAG", "NORMAL"):
                for p in list_pngs(veh, cls, sensor):
                    r = rpm_idx.get(p.stem)
                    if r is None:
                        continue
                    vals.append(r)
                    labs.append(1 if cls == "DEMAG" else 0)
        if len(set(labs)) < 2:
            continue
        x, y = np.array(vals), np.array(labs)
        best = 0.0
        for t in np.unique(x):
            for s in (1, -1):
                pred = ((x >= t) if s > 0 else (x < t)).astype(int)
                best = max(best, (pred == y).mean())
        P(f"  {veh:7s} rpm 단독 최고정확도 {100*best:5.1f}%  (n={len(y)})")
    P("")

    # ---------- 종합 ----------
    P("=" * 84)
    P("[D] 종합")
    P("=" * 84)
    if abs_all and ord_all:
        ma, mo = np.mean(abs_all), np.mean(ord_all)
        P(f"  절대축(2단계) 평균 분리도 : {ma:.3f}")
        P(f"  차수축(3단계) 평균 분리도 : {mo:.3f}   ({mo-ma:+.3f})")
        P("")
        iv = per_veh.get("IONIQ")
        if iv:
            ia, io_ = np.mean([x[0] for x in iv]), np.mean([x[1] for x in iv])
            P(f"  IONIQ 단독 : {ia:.3f} -> {io_:.3f}  ({io_-ia:+.3f})   << 핵심 관전 포인트")
            P("     IONIQ은 rpm이 100% 중첩되는 차종이라, 여기가 오르면 차수정규화가 유효하다.")
        P("")
        if mo >= 2.0 and (not iv or io_ >= 1.5):
            P("  >> 판정: 차수 정규화 유효. 본 파이프라인에 반영할 것.")
            P("     (특징추출 단계에서 rpm으로 주파수축 리샘플링 추가)")
        elif mo > ma + 0.3:
            P("  >> 판정: 개선 있으나 목표 미달. 방안2(대역 에너지 특징) 병행 필요.")
            P("     차수축 위에서 대역별 에너지 비율을 뽑으면 효과가 더 클 수 있다.")
        else:
            P("  >> 판정: 차수 정규화만으로는 부족.")
            P("     rpm 교란은 실재하나, 현행 4특징(위치3+스칼라1)의 표현력 한계가 더 크다.")
            P("     -> 방안2로 넘어갈 것.")
    P("")
    P("  주의")
    P("   - [A]에서 corr이 안 떨어지면 리샘플링 자체가 잘못된 것이니 [B]는 무의미하다.")
    P("   - 커버리지가 낮으면 REF_RPM을 데이터 rpm 상위값 쪽으로 올려야 한다.")
    P("   - 표본이 클래스당 5측정뿐이라 방향 확인용이다. 본데이터 재검증 필요.")
    P("=" * 84)
    P(f"소요 {time.time()-t0:.1f}s")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print(f"\n저장: {OUT_TXT}")


if __name__ == "__main__":
    main()
