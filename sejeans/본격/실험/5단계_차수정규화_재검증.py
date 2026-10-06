"""
5단계: 차수(order) 정규화 재검증 — 확장특징 채택 이후 기준으로 다시 판정

왜 다시 하는가
--------------
3단계에서 차수정규화는 "부족"으로 판정났다(분리도 1.540 -> 1.587). 그러나
그 판정은 '기존 4특징' 위에서 내린 것이고, 그 뒤 4단계에서 확장특징이
채택됐다(A4/4-c, DEMAG recall 0.73~0.82 -> 0.88~0.97). 판정 기준선이
바뀌었으므로 차수축도 새 기준선 위에서 다시 재봐야 한다.

4단계 결과표에는 이미 힌트가 있었다. 차수축은 '전체 분리도'에서는 졌지만
(ext/절대 2.07 vs ext/차수 1.88) 정작 우리 목표인 DEMAG 쪽에서는 이겼다:

    ext/절대 : DEMAG recall 80.0%,  D->N 오분류 5건
    ext/차수 : DEMAG recall 88.9%,  D->N 오분류 0건   <<<

즉 차수정규화는 전체 분리도를 조금 내주고 DEMAG<->NORMAL 경계를 산다.
세 차종 혼동행렬의 남은 오차가 전부 그 한 경계에 몰려 있으니(A4 기준
IONIQ 219건 / KONA 662건 / NIRO 581건) 이 거래는 우리에게 유리할 수 있다.

3단계 구현의 결함 2개 (이 스크립트에서 고친다)
---------------------------------------------
(1) 축 잘림.  3단계는 REF_RPM=3300, MAX_ORDER_RATIO=1.0 이었다.
    표본 rpm 범위는 1781~5693 (3.2배). rpm 5693 인 NIRO 이미지는
    차수 0.58 위쪽이 전부 NaN 으로 잘린다. 실제 커버리지 92.3% 가 그것이고,
    NIRO/Vib_Motor 가 3.43 -> 2.49 로 떨어진 것도 이 때문으로 의심된다.
    고침: REF_RPM 을 rpm 최대값 쪽에 두어 모든 이미지가 '압축'만 되게 한다.
          (외삽 0, 잘림 0. 대신 저rpm 이미지는 해상도를 잃는다 = 대가)

(2) 절대축과 bin 수가 달라 비교가 오염.  3단계는 절대축 H(=640, 1/2축소)
    대 차수축 ORDER_BINS=640 이었는데, 커버리지 92%면 유효 bin 이 590 뿐이라
    같은 조건이 아니다. 고침: 유효 bin 수를 리포트에 함께 찍는다.

라벨링 JSON 이 준 물리적 근거 (이번에 전수 측정)
----------------------------------------------
K = 결함대역_bin / rpm 이 차종 무관하게 일정하다:

    DEMAG        0.1333 / 0.1344 / 0.1342   (std <= 0.002)
    ECC10/ECC20  0.1088 ~ 0.1091            (세 차종 + 두 클래스 전부 동일)
    REDUC        0.1789 / 0.1411 / 0.1420   (IONIQ 만 다름 = 감속비 차이)

ECC10 과 ECC20 의 K 가 같다는 건 두 클래스가 '주파수 위치'가 아니라
'진폭'으로 갈린다는 뜻이다. 차수정규화는 위치를 정렬하는 변환이므로
ECC10<->ECC20 은 원래부터 도와줄 수 없다(A4 에서 이미 혼동 미미하니 무해).
DEMAG 는 K 가 매우 안정적이라 정렬 이득을 가장 크게 받을 후보다.

검증 항목
--------
  [A] REF_RPM 스윕 — 커버리지/잘림을 실제로 없앤 설정을 찾는다
  [B] corr(rpm, 결함위치)  절대축 -> 차수축   (정규화가 작동했는지)
  [C] DEMAG vs NORMAL 분리도  (ext 특징 기준, 절대축 vs 차수축)
  [D] 5클래스 1-NN LOO + DEMAG recall + D->N 오분류   << 최종 기준
  [E] 판정

실행
----
python 본격\5단계_차수정규화_재검증.py
결과: stdout + 5단계_차수정규화재검증결과.txt
"""

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib

# Windows 콘솔이 cp949 라 em-dash 같은 문자에서 죽는다 (학습 v3 와 동일 처리)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ===================== CONFIG =====================
SRC_DIR = Path(r"샘플데이터\Sample\01.원천데이터\1.모터_감속기")
LAB_DIR = Path(r"샘플데이터\Sample\02.라벨링데이터\1.모터_감속기")
OUT_TXT = Path(r"본격\5단계_차수정규화재검증결과.txt")

VEHICLES = ["IONIQ", "KONA", "NIRO"]
SENSORS = ["Current_U", "Vib_Motor", "Vib_TM"]
CLASSES = ["NORMAL", "DEMAG", "ECC10", "ECC20", "REDUC"]

MAX_FILES_PER_CLASS = 6
DOWNSCALE = 2
GRID_Q = 64
MEDIAN_FILTER = True
MIN_BAND_HEIGHT = 2

N_BANDS = 8
ROLLOFF_P = 0.85

# 차수축 해상도. 절대축(1280/DOWNSCALE=640)과 같게 두어 bin 수 차이를 없앤다.
ORDER_BINS = 640

# REF_RPM 후보. 3300 은 3단계 설정(잘림 발생), 나머지는 잘림 제거용.
#   ref 를 데이터 최대 rpm 이상으로 두면 scale=rpm/ref <= 1 이라
#   src 축이 항상 원본 범위 안에 들어가 잘림이 원리적으로 0 이 된다.
REF_RPM_CANDIDATES = [3300.0, 4600.0, 5700.0]
REF_RPM_MAIN = 5700.0      # [C][D] 본 비교에 쓸 값. [A] 결과로 재확인한다.
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


# ---------------- 차수 정규화 (잘림 제거판) ----------------
def to_order_axis(energy, rpm, ref_rpm, n_out=ORDER_BINS):
    """주파수축(세로)을 rpm 비례로 리샘플링.

    energy: (H, W), 행 0 = 상단 = 고주파.  내부에서 하단기준으로 뒤집어 계산.

    차수 o(0~1) 가 가리키는 원본 bin = o * (H-1) * (rpm/ref_rpm)
      rpm 이 ref 보다 낮으면 scale<1 -> 원본의 아래쪽 일부만 쓰고 늘려 담는다
      rpm 이 ref 보다 높으면 scale>1 -> 원본 범위를 넘어가 잘린다(NaN)
    따라서 ref_rpm >= max(rpm) 이면 잘림이 0 이 된다.

    반환: (차수축 배열, 유효비율)
    """
    H, W = energy.shape
    flip = energy[::-1, :]                       # 행 0 = 저주파
    src = np.linspace(0.0, (H - 1) * (rpm / ref_rpm), n_out)

    valid = src <= (H - 1)
    sc = np.clip(src, 0, H - 1)
    i0 = np.floor(sc).astype(int)
    i1 = np.minimum(i0 + 1, H - 1)
    w = (sc - i0)[:, None]

    out = flip[i0] * (1 - w) + flip[i1] * w
    out[~valid] = np.nan
    return out[::-1, :], float(valid.mean())


def row_detrend(e):
    med = np.nanmedian(e, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(e - med), axis=1, keepdims=True) + 1e-6
    return (e - med) / (1.4826 * mad)


# ---------------- 특징 (4단계와 동일 정의) ----------------
def feats_ext(e, nb=N_BANDS):
    """확장특징 12개. rownorm 이전 실제 에너지로 계산해야 비율이 의미를 갖는다."""
    E = np.clip(np.where(np.isfinite(e), e, np.nan), 0, None)
    flip = E[::-1, :]
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
    return float((np.array(preds) == y).mean()), np.array(preds)


# ---------------- 라벨 로딩 ----------------
def load_labels():
    """(stem -> rpm), (stem -> (결함대역 y중심, H))"""
    rpm, poly = {}, {}
    for p in LAB_DIR.rglob("*.json"):
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception:
            continue
        md = d.get("metadata", {})
        r = md.get("speed")
        if r is None:
            continue
        img = d.get("image", {})
        key = Path(img.get("file_path", "")).stem or p.stem
        rpm[key] = float(r)

        ys = []
        for q in (d.get("annotations", {}).get("polygons") or []):
            a = np.asarray(q, dtype=float)
            if a.ndim == 2 and a.shape[1] == 2 and len(a):
                ys.append(a[:, 1])
        if ys:
            poly[key] = (float(np.concatenate(ys).mean()),
                         float(img.get("height", 1280)))
    return rpm, poly


def list_pngs(v, c, s):
    d = SRC_DIR / v / c
    if not d.exists():
        return []
    return sorted(d.glob(f"*/{s}/*.png"))[:MAX_FILES_PER_CLASS]


def main():
    t0 = time.time()
    P("=" * 90)
    P("5단계: 차수 정규화 재검증 (확장특징 채택 이후 기준)")
    P("=" * 90)
    if not SRC_DIR.exists():
        P(f"!! 경로 없음: {SRC_DIR}")
        return

    rpm_idx, poly_idx = load_labels()
    rv = np.array(list(rpm_idx.values()))
    P(f"rpm {len(rpm_idx)}건 / 폴리곤 {len(poly_idx)}건")
    P(f"rpm 범위 {rv.min():.0f} ~ {rv.max():.0f} (중앙 {np.median(rv):.0f}, "
      f"최대/최소 {rv.max()/rv.min():.2f}배)")
    P(f"차수축 {ORDER_BINS}bin / 절대축 {1280//DOWNSCALE}bin (동일하게 맞춤)")
    P("")

    # ---------- [A] REF_RPM 스윕 ----------
    P("=" * 90)
    P("[A] REF_RPM 스윕 — 잘림(NaN)을 없애는 설정 찾기")
    P("=" * 90)
    P("  차수축 유효비율 = 잘리지 않고 살아남은 bin 비율. 100% 여야 정상.")
    P("  3단계는 3300 을 썼다 -> 여기서 잘림이 확인되면 그게 3단계 실패 원인.")
    P("")
    P(f"  {'REF_RPM':>8s} {'유효비율평균':>12s} {'최소':>8s} {'잘린이미지수':>12s} "
      f"{'최저rpm해상도손실':>18s}")
    for ref in REF_RPM_CANDIDATES:
        covs = []
        for k, r in rpm_idx.items():
            n_valid = (np.linspace(0, (640 - 1) * (r / ref), ORDER_BINS) <= 639).mean()
            covs.append(n_valid)
        covs = np.array(covs)
        # 저rpm 이미지는 원본의 아래쪽 일부만 확대해 쓰므로 실효 해상도가 준다
        loss = rv.min() / ref
        P(f"  {ref:8.0f} {100*covs.mean():11.1f}% {100*covs.min():7.1f}% "
          f"{int((covs < 0.999).sum()):12d} {100*loss:17.1f}%")
    P("")
    P(f"  -> 본 비교에는 REF_RPM = {REF_RPM_MAIN:.0f} 사용")
    P("")

    # ---------- [B] corr(rpm, 결함위치) ----------
    P("=" * 90)
    P("[B] 결함대역 위치가 rpm 과 무관해지는가 (전문가 폴리곤 기준)")
    P("=" * 90)
    P("  절대축 corr ~ 1.0 = 위치가 rpm 에 끌려다님 (모델이 회전수를 학습할 여지)")
    P("  차수축 corr ~ 0.0 = 정규화 성공")
    P("")
    P(f"  {'차종':7s} {'클래스':7s} {'n':>3s} {'K평균':>8s} {'절대축':>8s} {'차수축':>8s}")
    for veh in VEHICLES:
        for cls in CLASSES:
            rec = []
            for s in SENSORS:
                for p in list_pngs(veh, cls, s):
                    if p.stem not in poly_idx or p.stem not in rpm_idx:
                        continue
                    yc, H = poly_idx[p.stem]
                    r = rpm_idx[p.stem]
                    fb = (H - 1) - yc                    # 하단기준 절대 bin
                    rec.append((r, fb, fb / (r / REF_RPM_MAIN), fb / r))
            if len(rec) < 3:
                continue
            a = np.array(rec)
            ca = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
            co = np.corrcoef(a[:, 0], a[:, 2])[0, 1]
            mk = "  <== 개선" if abs(co) < abs(ca) - 0.15 else ""
            P(f"  {veh:7s} {cls:7s} {len(rec):3d} {a[:,3].mean():8.4f} "
              f"{ca:8.3f} {co:8.3f}{mk}")
    P("")
    P("  K = 결함bin/rpm. 차종 간 K 가 같으면 '같은 차수' = 물리적으로 같은 결함.")
    P("  ECC10 과 ECC20 의 K 가 같다면 두 클래스는 위치가 아니라 진폭으로 갈린다")
    P("  -> 차수정규화는 ECC 계열엔 도움을 줄 수 없다(무해하면 통과).")
    P("")

    # ---------- 특징 추출 ----------
    P("=" * 90)
    P("[C] DEMAG vs NORMAL 분리도 — ext 특징, 절대축 vs 차수축")
    P("=" * 90)
    store = {}
    rows = []          # (veh, sensor, cls, f_abs, f_ord)
    n_img, miss = 0, 0
    covs = []
    for veh in VEHICLES:
        for s in SENSORS:
            for cls in CLASSES:
                for p in list_pngs(veh, cls, s):
                    r = rpm_idx.get(p.stem)
                    if r is None:
                        miss += 1
                        continue
                    rgb = np.asarray(Image.open(p).convert("RGB"))[::DOWNSCALE, ::DOWNSCALE]
                    e = energy_lut(rgb)
                    n_img += 1

                    f_abs = describe(feats_ext(e))
                    eo, cov = to_order_axis(e, r, REF_RPM_MAIN)
                    covs.append(cov)
                    f_ord = describe(feats_ext(eo))

                    store.setdefault((veh, s), {}).setdefault(
                        cls, {"abs": [], "ord": []})
                    store[(veh, s)][cls]["abs"].append(f_abs)
                    store[(veh, s)][cls]["ord"].append(f_ord)
                    rows.append((veh, s, cls, f_abs, f_ord))
                    if n_img % 40 == 0:
                        P(f"  ... {n_img}장 ({time.time()-t0:.0f}s)")
    if miss:
        P(f"  ! rpm 없어 건너뛴 파일 {miss}개")
    P(f"  총 {n_img}장 / 차수축 유효비율 평균 {100*np.mean(covs):.1f}% "
      f"(100% 여야 잘림 없음)")
    P("")

    P(f"  {'차종/센서':22s} {'ext/절대':>10s} {'ext/차수':>10s} {'변화':>9s}")
    aa, oo = [], []
    per_veh = {}
    for veh in VEHICLES:
        for s in SENSORS:
            d = store.get((veh, s))
            if not d or "NORMAL" not in d or "DEMAG" not in d:
                continue
            sa = separation(d["NORMAL"]["abs"], d["DEMAG"]["abs"])
            so = separation(d["NORMAL"]["ord"], d["DEMAG"]["ord"])
            if np.isnan(sa) or np.isnan(so):
                continue
            aa.append(sa)
            oo.append(so)
            per_veh.setdefault(veh, []).append((sa, so))
            mk = "  <==" if so > sa + 0.3 else ("  !!" if so < sa - 0.3 else "")
            P(f"  {veh + '/' + s:22s} {sa:10.2f} {so:10.2f} {so-sa:+9.2f}{mk}")
    P("")
    for veh in VEHICLES:
        v = per_veh.get(veh)
        if not v:
            continue
        a = np.mean([x[0] for x in v])
        o = np.mean([x[1] for x in v])
        P(f"  {veh + ' 평균':22s} {a:10.2f} {o:10.2f} {o-a:+9.2f}")
    if aa:
        P(f"  {'전체 평균':22s} {np.mean(aa):10.2f} {np.mean(oo):10.2f} "
          f"{np.mean(oo)-np.mean(aa):+9.2f}")
    P("")

    # ---------- [D] 5클래스 1-NN ----------
    P("=" * 90)
    P("[D] 5클래스 1-NN LOO — DEMAG recall 과 D->N 오분류   << 최종 기준")
    P("=" * 90)
    P("  4단계 참고치(REF_RPM=3300, 잘림 있던 상태):")
    P("    ext/절대  정확도 68.0%  DEMAG recall 80.0%  D->N 5건  N->D 12건")
    P("    ext/차수  정확도 54.7%  DEMAG recall 88.9%  D->N 0건  N->D  7건")
    P("")
    y = np.array([r[2] for r in rows])
    for name, col in (("ext/절대", 3), ("ext/차수", 4)):
        X = np.array([r[col] for r in rows])
        acc, pred = loo_1nn(X, y)
        if pred is None:
            continue
        dm = y == "DEMAG"
        rec = float((pred[dm] == "DEMAG").mean())
        d2n = int(((y == "DEMAG") & (pred == "NORMAL")).sum())
        n2d = int(((y == "NORMAL") & (pred == "DEMAG")).sum())
        P(f"  {name:12s} 정확도 {100*acc:5.1f}%  DEMAG recall {100*rec:5.1f}%  "
          f"D->N {d2n:3d}건  N->D {n2d:3d}건  (n={len(y)}, 차원={X.shape[1]})")
    P("")
    P("  클래스별 recall")
    P(f"  {'클래스':8s} {'ext/절대':>10s} {'ext/차수':>10s}")
    res = {}
    for name, col in (("abs", 3), ("ord", 4)):
        X = np.array([r[col] for r in rows])
        _, pred = loo_1nn(X, y)
        res[name] = pred
    for cls in CLASSES:
        m = y == cls
        if not m.any():
            continue
        ra = float((res["abs"][m] == cls).mean())
        ro = float((res["ord"][m] == cls).mean())
        mk = "  <==" if ro > ra + 0.05 else ("  !!" if ro < ra - 0.05 else "")
        P(f"  {cls:8s} {100*ra:9.1f}% {100*ro:9.1f}%{mk}")
    P("")

    # ---------- [E] 판정 ----------
    P("=" * 90)
    P("[E] 판정")
    P("=" * 90)
    dm = y == "DEMAG"
    ra = float((res["abs"][dm] == "DEMAG").mean())
    ro = float((res["ord"][dm] == "DEMAG").mean())
    d2n_a = int(((y == "DEMAG") & (res["abs"] == "NORMAL")).sum())
    d2n_o = int(((y == "DEMAG") & (res["ord"] == "NORMAL")).sum())
    acc_a = float((res["abs"] == y).mean())
    acc_o = float((res["ord"] == y).mean())

    P(f"  DEMAG recall : {100*ra:.1f}% -> {100*ro:.1f}%  ({100*(ro-ra):+.1f}%p)")
    P(f"  D->N 오분류  : {d2n_a}건 -> {d2n_o}건")
    P(f"  전체 정확도  : {100*acc_a:.1f}% -> {100*acc_o:.1f}%  "
      f"({100*(acc_o-acc_a):+.1f}%p)")
    P("")
    if ro >= ra + 0.03 and acc_o >= acc_a - 0.03:
        P("  >> 채택 권고. DEMAG 가 오르고 전체도 지켜졌다.")
        P("     특징추출 v3 에 차수축 채널을 추가하고 본데이터로 확인할 것.")
    elif ro >= ra + 0.03:
        P("  >> 조건부 채택(병행) 권고.")
        P("     DEMAG 는 올랐으나 전체 정확도가 내려갔다. 절대축을 버리지 말고")
        P("     '절대축 + 차수축' 두 벌을 채널로 함께 넣는 게 맞다.")
        P("     근거: 차수축은 위치를 정렬해 DEMAG 를 살리지만, 그 과정에서")
        P("           rpm 자체가 갖고 있던 판별정보(ECC/REDUC)를 지운다.")
        P("           둘은 상보적이므로 택일이 아니라 결합이 답이다.")
    else:
        P("  >> 기각. 잘림을 고쳐도 DEMAG 이득이 없다.")
        P("     차수정규화는 여기서 종료하고 다른 축(임계/대역수/센서가중)을 볼 것.")
    P("")
    P("  주의")
    P("   - 표본이 클래스당 5측정(총 225장)뿐이다. 방향 확인용이며 확정은")
    P("     본데이터 학습(모터_학습_v3.py)으로 해야 한다.")
    P("   - [A] 유효비율이 100% 가 아니면 [C][D] 는 잘림이 섞인 값이다.")
    P("   - 1-NN LOO 는 세션 누수를 막지 않는다(같은 주행의 다른 조각이")
    P("     이웃이 될 수 있다). 절대/차수 비교에는 공정하나 절대값은 낙관적이다.")
    P("=" * 90)
    P(f"소요 {time.time()-t0:.1f}s")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    io.open(OUT_TXT, "w", encoding="utf-8").write("\n".join(_lines))
    print(f"\n저장: {OUT_TXT}")


if __name__ == "__main__":
    main()
