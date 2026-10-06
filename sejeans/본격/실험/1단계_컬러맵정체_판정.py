"""
1단계: 스펙트로그램 PNG의 실제 컬러맵 정체 판정

배경
----
방안1 검증에서 사전점검이 예상 밖 수치를 냈다.
  jet LUT 평균거리: Current_U 28.72 / Vib_Motor 33.25 / Vib_TM 23.29
순수 jet 렌더라면 0에 가까워야 하는데 그렇지 않았고, 고유색도 1257개로
256색 jet의 약 5배였다. 즉 "jet이다"라는 전제가 확인되지 않은 상태다.

이 스크립트는 방안1을 더 손대기 전에 그 전제를 판정한다.
LUT 역변환은 컬러맵을 정확히 알아야만 성립하므로, 여기서 결론이 나기 전에는
방안1/방안2 어느 쪽도 확정할 수 없다.

판정 항목
--------
A. 파일 실체    : 포맷/모드/알파 (RGBA인데 RGB로 읽으면 왜곡될 수 있음)
B. 후보 비교    : jet 외 11개 컬러맵과 거리 비교 -> 애초에 jet이 맞는가
C. 변형 모델    : 알파블렌딩 / 감마 적용된 jet인가
D. 잔차 구조    : 잔차가 '엣지'에 몰리는가(보간 노이즈) '고르게' 퍼지는가(다른 컬러맵)
                  -> 이게 핵심 판별. 보간 노이즈면 jet 확정이고 역변환 가능.
E. 복원 가능성  : 역변환된 스칼라가 실제로 쓸 만한 분포인가

실행
----
python 본격\1단계_컬러맵정체_판정.py

결과는 stdout + 1단계_컬러맵판정결과.txt 로 저장된다.
"""

import io
import time
from pathlib import Path

import numpy as np
import matplotlib
from PIL import Image

# ===================== CONFIG =====================
BASE_DIR = Path(r"샘플데이터\Sample\01.원천데이터\1.모터_감속기")
OUT_TXT = Path(r"본격\1단계_컬러맵판정결과.txt")

SENSORS = ["Current_U", "Vib_Motor", "Vib_TM"]
N_FILES_PER_SENSOR = 3      # 센서당 검사할 파일 수 (여러 차종/클래스 섞이도록)
N_SAMPLE_PX = 15000         # 파일당 샘플 픽셀 수
SEED = 0

CANDIDATES = ["jet", "turbo", "nipy_spectral", "gist_rainbow", "rainbow",
              "hsv", "gist_ncar", "viridis", "gist_stern", "CMRmap",
              "gnuplot2", "brg"]
# ==================================================

_lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    _lines.append(s)
    print(s)


def lut_of(name):
    """컬러맵 -> (256,3) 0~255 LUT."""
    cm = matplotlib.colormaps[name]      # cm.get_cmap은 deprecated
    return (np.asarray([cm(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.float32)


def nn_dist(sub, lut):
    """샘플 픽셀 -> 최근접 LUT 색까지의 유클리드 거리와 인덱스."""
    D = ((sub[:, None, :] - lut[None, :, :]) ** 2).sum(2)
    return np.sqrt(D.min(1)), D.argmin(1)


def pick_files():
    """센서별로 서로 다른 차종/클래스에서 파일을 고른다."""
    out = {}
    for s in SENSORS:
        allp = sorted(BASE_DIR.rglob(f"*{s}*.png"))
        if not allp:
            out[s] = []
            continue
        # 경로 앞부분(차종/클래스)이 다양하도록 고르게 샘플
        step = max(1, len(allp) // N_FILES_PER_SENSOR)
        out[s] = allp[::step][:N_FILES_PER_SENSOR]
    return out


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    P("=" * 78)
    P("1단계: 스펙트로그램 컬러맵 정체 판정")
    P("=" * 78)

    files = pick_files()
    total = sum(len(v) for v in files.values())
    if total == 0:
        P("!! PNG를 찾지 못했습니다. BASE_DIR 경로 확인 필요.")
        return
    P(f"검사 대상 {total}개 파일 (센서당 최대 {N_FILES_PER_SENSOR}개)\n")

    luts = {c: lut_of(c) for c in CANDIDATES}

    # ---------- A. 파일 실체 ----------
    P("-" * 78)
    P("[A] 파일 실체 점검 (RGBA/알파/비트depth)")
    P("-" * 78)
    P("  * 알파가 255 고정이 아니면 RGB로만 읽을 때 색이 왜곡된다.")
    for s, ps in files.items():
        for p in ps[:1]:
            im = Image.open(p)
            arr = np.asarray(im)
            note = ""
            if im.mode == "RGBA":
                a = arr[:, :, 3]
                note = f"alpha min{a.min()} max{a.max()}"
                note += " (균일 -> 무해)" if a.min() == a.max() == 255 else " (!! 비균일 - 왜곡 가능)"
            P(f"  {s:10s} {im.format} {im.mode} {im.size} {note}")
    P("")

    # ---------- B/C/D 파일별 분석 ----------
    verdicts = []
    for s, ps in files.items():
        for p in ps:
            img = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
            H, W, _ = img.shape
            px = img.reshape(-1, 3)
            n = min(N_SAMPLE_PX, len(px))
            sel = rng.choice(len(px), n, replace=False)
            sub = px[sel]

            P("=" * 78)
            P(f"[{s}] {p.name}")
            P(f"  크기 {W}x{H}, 고유색 {len(np.unique(px, axis=0)):,}개")
            P("-" * 78)

            # --- B. 후보 컬러맵 비교 ---
            P("  [B] 후보 컬러맵 거리 (mean/median/within5, 작을수록 일치)")
            res = []
            for c, L in luts.items():
                d, _ = nn_dist(sub, L)
                res.append((d.mean(), np.median(d), (d < 5).mean(), c))
            res.sort()
            for m, md, f5, c in res[:5]:
                star = "  <== 최적" if c == res[0][3] else ""
                P(f"      {c:14s} mean {m:6.2f}  median {md:6.2f}  within5 {100*f5:5.1f}%{star}")
            best_name = res[0][3]
            jet_rank = [c for _, _, _, c in res].index("jet") + 1
            P(f"      -> 최적 {best_name}, jet 순위 {jet_rank}위")

            # --- C. jet 변형 모델 ---
            P("  [C] jet 변형 모델 검정 (순수 jet보다 나은 게 있는가)")
            base = luts["jet"]
            d0, _ = nn_dist(sub, base)
            cands = [("jet 순수", base)]
            for bg, bgn in ((255, "white"), (0, "black")):
                for al in (0.85, 0.9, 0.95):
                    cands.append((f"alpha={al} on {bgn}", base * al + bg * (1 - al)))
            for g in (0.5, 0.7, 1.4, 2.0):
                gi = (np.linspace(0, 1, 256) ** g * 255).astype(int).clip(0, 255)
                cands.append((f"gamma={g}", base[gi]))
            best_var, best_var_d = "jet 순수", d0.mean()
            for name, L in cands:
                d, _ = nn_dist(sub, L)
                flag = ""
                if d.mean() < best_var_d - 1e-9:
                    best_var, best_var_d = name, d.mean()
                if name == "jet 순수":
                    flag = "  (기준)"
                P(f"      {name:22s} mean {d.mean():6.2f}{flag}")
            P(f"      -> 최적 변형: {best_var} (mean {best_var_d:.2f})")

            # --- D. 잔차 구조 (핵심 판별) ---
            P("  [D] 잔차 구조 분석  << 핵심")
            P("      잔차가 엣지(고gradient)에 몰리면 = 리샘플링/보간 노이즈 -> jet 확정")
            P("      잔차가 균일하면              = 컬러맵 자체가 다름   -> jet 아님")
            d, idx = nn_dist(sub, base)
            ys, xs = np.unravel_index(sel, (H, W))
            grad = (np.abs(np.diff(img, axis=0, prepend=img[:1])).sum(2)
                    + np.abs(np.diff(img, axis=1, prepend=img[:, :1])).sum(2))
            gv = grad[ys, xs]
            bins = [(0, 5), (5, 20), (20, 60), (60, np.inf)]
            means = []
            for lo, hi in bins:
                m = (gv >= lo) & (gv < hi)
                if m.sum() > 10:
                    mv = d[m].mean()
                    means.append(mv)
                    P(f"      gradient[{lo:>3},{str(hi):>4}) n={m.sum():6d}  평균잔차 {mv:6.2f}")
            flat_resid = means[0] if means else float("nan")
            ratio = (means[-1] / means[0]) if len(means) >= 2 and means[0] > 0 else float("nan")
            P(f"      평탄부 잔차 {flat_resid:.2f} / 엣지 잔차 비율 {ratio:.2f}배")

            # --- E. 복원 스칼라 분포 ---
            hist = np.histogram(idx, bins=16, range=(0, 256))[0]
            hp = 100 * hist / max(hist.sum(), 1)
            P("  [E] 복원 스칼라 분포 (16구간 %, 한쪽에 쏠리면 임계설정 주의)")
            P("      " + " ".join(f"{x:4.1f}" for x in hp))

            # --- 파일 판정 ---
            if best_name != "jet" and jet_rank > 2:
                v = f"jet 아님 (최적 {best_name})"
            elif flat_resid < 6 and ratio > 1.3:
                v = "jet 확정 (잔차=보간노이즈)"
            elif flat_resid < 12 and ratio > 1.2:
                v = "jet 유력 (보간노이즈 우세)"
            elif flat_resid >= 12:
                v = "jet 의심 (평탄부 잔차 큼)"
            else:
                v = "판정보류"
            P(f"  >> 판정: {v}")
            verdicts.append((s, p.name, best_name, jet_rank, best_var,
                             flat_resid, ratio, v))
            P("")

    # ---------- 종합 ----------
    P("=" * 78)
    P("종합 판정")
    P("=" * 78)
    P(f"{'센서':11s} {'최적맵':13s} {'jet순위':>7s} {'최적변형':16s} {'평탄잔차':>8s} {'엣지비':>7s}  판정")
    for s, fn, bn, jr, bv, fr, rt, v in verdicts:
        P(f"{s:11s} {bn:13s} {jr:7d} {bv:16s} {fr:8.2f} {rt:7.2f}  {v}")

    jet_ok = sum(1 for x in verdicts if x[2] == "jet")
    P("")
    P(f"jet이 최적인 파일: {jet_ok}/{len(verdicts)}")
    P("")
    P("해석 가이드")
    P("  1) 'jet 확정/유력'이 대부분 -> 컬러맵은 jet이 맞다.")
    P("     방안1 실패 원인은 역변환이 아니라 '역변환 + 90퍼센타일 임계' 충돌이다.")
    P("     -> 2단계(적응 임계)로 진행. 방안1 폐기하지 말 것.")
    P("  2) 'jet 아님'이 다수 -> 최적맵 컬럼의 컬러맵으로 LUT를 교체해야 한다.")
    P("  3) [C]에서 순수 jet보다 나은 변형이 있으면 그 변형으로 LUT를 만들어야 한다.")
    P("  4) [E] 분포가 저구간에 심하게 쏠려 있으면, 90퍼센타일 임계가 전대역을")
    P("     잡아버리는 지표1 포화의 직접 증거다.")
    P("")
    P("주의: 잔차가 보간노이즈라면 역변환 전 median/bilateral 필터로")
    P("      엣지 노이즈를 줄이면 복원 정확도가 올라간다.")
    P("=" * 78)
    P(f"소요 {time.time()-t0:.1f}s")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with io.open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines))
    print(f"\n저장: {OUT_TXT}")


if __name__ == "__main__":
    main()
