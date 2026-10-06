"""
LUT 역변환으로 복원한 에너지 시각화

목적
----
지금까지는 숫자로만 봤다. 실제로
  - 밝기(현행) 방식이 에너지를 어떻게 망가뜨리는지
  - LUT 역변환이 무엇을 복원하는지
  - 90퍼센타일 임계가 왜 포화되는지
를 눈으로 확인한다.

생성 이미지
----------
1) LUT복원_비교_{센서}.png       : 클래스별 원본/밝기/LUT 3단 비교
2) LUT복원_임계문제_{센서}.png   : 히스토그램 + 임계선 + 마스크 (포화 원인)
3) LUT복원_컬러맵곡선.png        : jet 밝기 비단조성 자체를 보여주는 그림
4) LUT복원_주파수프로파일.png    : DEMAG vs NORMAL 주파수 분포 (핵심 근거)

실행
----
python 본격\LUT복원_시각화.py

한글 폰트가 없으면 자동으로 영문 라벨로 대체된다.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")            # 창 안 띄우고 파일로만 저장
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

# ===================== CONFIG =====================
BASE_DIR = Path(r"샘플데이터\Sample\01.원천데이터\1.모터_감속기")
OUT_DIR = Path(r"본격\시각화")

VEHICLE = "IONIQ"                       # 오분류가 가장 심한 차종
SENSORS = ["Vib_TM", "Vib_Motor", "Current_U"]
CLASSES = ["NORMAL", "DEMAG", "ECC10", "ECC20", "REDUC"]

GRID_Q = 64
THRESHOLD_PERCENTILE = 90
MEDIAN_FILTER = True                    # 보간노이즈 억제 (1단계에서 확정된 사항)
DOWNSCALE = 2                           # 표시용 축소 배율 (1=원본)
DPI = 110
# ==================================================


# ---------- 한글 폰트 ----------
def setup_font():
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"]:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return False


HAS_KR = setup_font()


def L(kr, en):
    """한글 폰트 있으면 한글, 없으면 영문."""
    return kr if HAS_KR else en


# ---------- LUT ----------
def build_jet_lut():
    cm = matplotlib.colormaps["jet"]
    return (np.asarray([cm(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.float32)


def build_voxel_grid(lut, q=GRID_Q):
    centers = (np.arange(q) + 0.5) * (256.0 / q)
    gr, gg, gb = np.meshgrid(centers, centers, centers, indexing="ij")
    pts = np.stack([gr.ravel(), gg.ravel(), gb.ravel()], 1).astype(np.float32)
    idx = np.empty(len(pts), dtype=np.uint8)
    for i in range(0, len(pts), 8192):
        ch = pts[i:i + 8192]
        idx[i:i + 8192] = ((ch[:, None, :] - lut[None, :, :]) ** 2).sum(2).argmin(1)
    return idx.reshape(q, q, q)


_LUT = build_jet_lut()
_GRID = build_voxel_grid(_LUT)


def median3(a):
    """3x3 median. scipy 없이 numpy만으로."""
    p = np.pad(a, 1, mode="edge")
    st = np.stack([p[i:i + a.shape[0], j:j + a.shape[1]]
                   for i in range(3) for j in range(3)], 0)
    return np.median(st, axis=0)


def energy_lut(rgb):
    """[방안1] jet 역변환 -> 실제 스칼라 에너지."""
    q = (rgb.astype(np.int32) * GRID_Q) // 256
    np.clip(q, 0, GRID_Q - 1, out=q)
    e = (_GRID[q[:, :, 0], q[:, :, 1], q[:, :, 2]] / 255.0).astype(np.float32)
    return median3(e) if MEDIAN_FILTER else e


def energy_bright(rgb):
    """[현행] 밝기 RMS + min-max 정규화."""
    a = rgb.astype(np.float32) / 255.0
    e = np.sqrt(np.mean(a ** 2, axis=2))
    return (e - e.min()) / (e.max() - e.min() + 1e-8)


def find_png(vehicle, cls, sensor):
    d = BASE_DIR / vehicle / cls
    if not d.exists():
        return None
    ps = sorted(d.glob(f"*/{sensor}/*.png"))
    return ps[0] if ps else None


def load_rgb(p):
    a = np.asarray(Image.open(p).convert("RGB"))
    return a[::DOWNSCALE, ::DOWNSCALE]


def band_mask(e, pct=THRESHOLD_PERCENTILE):
    """현행 로직과 동일한 마스크 + band_start/end."""
    H = e.shape[0]
    th = np.percentile(e, pct)
    m = e >= th
    y_top = np.argmax(m, axis=0)
    y_bot = H - 1 - np.argmax(m[::-1, :], axis=0)
    bs = H - 1 - y_bot
    be = H - 1 - y_top
    return m, th, bs, be


# =========================================================
# 그림 1: 원본 / 밝기 / LUT 3단 비교
# =========================================================
def fig_compare(sensor):
    cols = [c for c in CLASSES if find_png(VEHICLE, c, sensor)]
    if not cols:
        return None
    fig, axes = plt.subplots(3, len(cols), figsize=(3.1 * len(cols), 9.6))
    if len(cols) == 1:
        axes = axes.reshape(3, 1)

    for j, cls in enumerate(cols):
        rgb = load_rgb(find_png(VEHICLE, cls, sensor))
        eb, el = energy_bright(rgb), energy_lut(rgb)

        axes[0, j].imshow(rgb)
        axes[0, j].set_title(cls, fontsize=13, fontweight="bold",
                             color="crimson" if cls in ("DEMAG", "NORMAL") else "black")
        axes[1, j].imshow(eb, cmap="gray", vmin=0, vmax=1)
        axes[2, j].imshow(el, cmap="gray", vmin=0, vmax=1)

        axes[1, j].set_xlabel(f"mean {eb.mean():.3f}  std {eb.std():.3f}", fontsize=8)
        axes[2, j].set_xlabel(f"mean {el.mean():.3f}  std {el.std():.3f}", fontsize=8)
        for i in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    axes[0, 0].set_ylabel(L("원본 PNG\n(jet)", "Original\n(jet)"), fontsize=11)
    axes[1, 0].set_ylabel(L("현행: 밝기 RMS\n(정보 손실)", "Current: brightness\n(lossy)"),
                          fontsize=11, color="darkred")
    axes[2, 0].set_ylabel(L("방안1: LUT 역변환\n(에너지 복원)", "Fix: LUT inverse\n(recovered)"),
                          fontsize=11, color="darkgreen")
    fig.suptitle(L(f"[{VEHICLE} / {sensor}] 에너지 복원 비교  —  밝을수록 에너지 높음",
                   f"[{VEHICLE} / {sensor}] Energy recovery  —  brighter = higher"),
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =========================================================
# 그림 2: 임계 포화 원인
# =========================================================
def fig_threshold(sensor):
    cols = [c for c in ["NORMAL", "DEMAG"] if find_png(VEHICLE, c, sensor)]
    if not cols:
        return None
    fig, axes = plt.subplots(len(cols), 3, figsize=(15, 4.6 * len(cols)))
    if len(cols) == 1:
        axes = axes.reshape(1, 3)

    for i, cls in enumerate(cols):
        rgb = load_rgb(find_png(VEHICLE, cls, sensor))
        eb, el = energy_bright(rgb), energy_lut(rgb)
        mb, thb, _, _ = band_mask(eb)
        ml, thl, _, _ = band_mask(el)

        ax = axes[i, 0]
        ax.hist(eb.ravel(), bins=100, alpha=0.55, label=L("현행(밝기)", "brightness"),
                color="tab:red", density=True)
        ax.hist(el.ravel(), bins=100, alpha=0.55, label=L("방안1(LUT)", "LUT"),
                color="tab:green", density=True)
        ax.axvline(thb, color="darkred", ls="--", lw=2,
                   label=L(f"임계 현행 {thb:.2f}", f"th cur {thb:.2f}"))
        ax.axvline(thl, color="darkgreen", ls="--", lw=2,
                   label=L(f"임계 LUT {thl:.2f}", f"th LUT {thl:.2f}"))
        ax.set_title(L(f"{cls}: 에너지 분포와 90% 임계", f"{cls}: energy dist & 90% th"),
                     fontsize=11)
        ax.set_yscale("log")
        ax.legend(fontsize=8)

        sat_b = 100 * (mb[-1, :]).mean()
        sat_l = 100 * (ml[-1, :]).mean()
        axes[i, 1].imshow(mb, cmap="gray")
        axes[i, 1].set_title(L(f"{cls} 현행 마스크 (최하행 점유 {sat_b:.0f}%)",
                               f"{cls} current mask (bottom {sat_b:.0f}%)"), fontsize=10)
        axes[i, 2].imshow(ml, cmap="gray")
        axes[i, 2].set_title(L(f"{cls} LUT 마스크 (최하행 점유 {sat_l:.0f}%)",
                               f"{cls} LUT mask (bottom {sat_l:.0f}%)"),
                             fontsize=10,
                             color="crimson" if sat_l > sat_b + 10 else "black")
        for k in (1, 2):
            axes[i, k].set_xticks([])
            axes[i, k].set_yticks([])

    fig.suptitle(L(f"[{VEHICLE} / {sensor}] 90퍼센타일 임계가 포화되는 원인"
                   "  —  마스크가 하얗게 번지면 band 특징이 죽는다",
                   f"[{VEHICLE} / {sensor}] Why the 90th-percentile threshold saturates"),
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# =========================================================
# 그림 3: jet 밝기 비단조성
# =========================================================
def fig_curve():
    lut01 = _LUT / 255.0
    x = np.arange(256) / 255.0
    br = np.sqrt(np.mean(lut01 ** 2, axis=1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].imshow(lut01[None, :, :], aspect="auto",
                   extent=[0, 1, 0, 1])
    axes[0].set_yticks([])
    axes[0].set_xlabel(L("실제 에너지 (0~1)", "true energy (0-1)"))
    axes[0].set_title(L("jet 컬러맵", "jet colormap"), fontsize=12)

    axes[1].plot(x, br, lw=2.2, color="tab:red")
    peak = x[br.argmax()]
    axes[1].axvline(peak, color="gray", ls=":", lw=1.5)
    axes[1].annotate(L(f"여기서 꺾임 ({peak:.2f})", f"turns here ({peak:.2f})"),
                     xy=(peak, br.max()), xytext=(peak + 0.16, br.max() - 0.02),
                     arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
    lo, hi = 0.06, 0.94
    for v in (lo, hi):
        axes[1].plot([v], [br[int(v * 255)]], "o", ms=9, color="darkblue")
    axes[1].annotate("", xy=(lo, br[int(lo * 255)]), xytext=(hi, br[int(hi * 255)]),
                     arrowprops=dict(arrowstyle="<->", color="darkblue", lw=1.6))
    axes[1].text(0.5, br[int(lo * 255)] - 0.055,
                 L("최저↔최고 에너지가\n같은 밝기로 접힘",
                   "lowest & highest energy\nfold to same brightness"),
                 ha="center", fontsize=9.5, color="darkblue")
    axes[1].set_xlabel(L("실제 에너지", "true energy"))
    axes[1].set_ylabel(L("밝기 RMS (현행이 쓰는 값)", "brightness RMS (current)"))
    axes[1].set_title(L("현행 방식: 비단조 → 정보 손실", "Current: non-monotonic"),
                      fontsize=12, color="darkred")
    axes[1].grid(alpha=0.3)

    axes[2].plot(x, x, lw=2.2, color="tab:green")
    axes[2].set_xlabel(L("실제 에너지", "true energy"))
    axes[2].set_ylabel(L("복원 값", "recovered"))
    axes[2].set_title(L("방안1: LUT 역변환 → 1:1 복원", "Fix: LUT inverse (1:1)"),
                      fontsize=12, color="darkgreen")
    axes[2].grid(alpha=0.3)

    fig.suptitle(L("왜 밝기를 에너지로 쓰면 안 되는가",
                   "Why brightness cannot be used as energy"),
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# =========================================================
# 그림 4: DEMAG vs NORMAL 주파수 프로파일 (핵심 근거)
# =========================================================
def fig_profile():
    fig, axes = plt.subplots(1, len(SENSORS), figsize=(5.4 * len(SENSORS), 4.6))
    if len(SENSORS) == 1:
        axes = [axes]
    for k, sensor in enumerate(SENSORS):
        ax = axes[k]
        for cls, color in [("NORMAL", "tab:blue"), ("DEMAG", "tab:red")]:
            p = find_png(VEHICLE, cls, sensor)
            if p is None:
                continue
            rgb = load_rgb(p)
            for e, ls, lab in [(energy_bright(rgb), ":", L("현행", "cur")),
                               (energy_lut(rgb), "-", L("LUT", "LUT"))]:
                prof = e.mean(axis=1)[::-1]
                prof = prof / (prof.sum() + 1e-9)
                ax.plot(np.linspace(0, 1, len(prof)), prof, ls=ls, color=color,
                        lw=2.0 if ls == "-" else 1.2, alpha=0.9 if ls == "-" else 0.5,
                        label=f"{cls} {lab}")
        ax.set_title(sensor, fontsize=12)
        ax.set_xlabel(L("주파수 (낮음 → 높음)", "frequency (low -> high)"))
        if k == 0:
            ax.set_ylabel(L("정규화 에너지", "normalized energy"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(L(f"[{VEHICLE}] DEMAG vs NORMAL 주파수 분포"
                   "  —  실선(LUT)이 벌어져야 판별 가능",
                   f"[{VEHICLE}] DEMAG vs NORMAL frequency profile"),
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("LUT 역변환 복원 시각화")
    print("=" * 70)
    print(f"차종 {VEHICLE} / median필터 {MEDIAN_FILTER} / 축소 1/{DOWNSCALE}")
    if not HAS_KR:
        print("! 한글 폰트를 찾지 못해 영문 라벨로 출력합니다.")
    if not BASE_DIR.exists():
        print(f"!! 경로 없음: {BASE_DIR}")
        return
    print()

    saved = []

    f = fig_curve()
    p = OUT_DIR / "LUT복원_컬러맵곡선.png"
    f.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(f)
    saved.append(p)
    print(f"  저장 {p.name}")

    for s in SENSORS:
        for fn, tag in [(fig_compare, "비교"), (fig_threshold, "임계문제")]:
            f = fn(s)
            if f is None:
                print(f"  건너뜀 {s} ({tag}) - 파일 없음")
                continue
            p = OUT_DIR / f"LUT복원_{tag}_{s}.png"
            f.savefig(p, dpi=DPI, bbox_inches="tight")
            plt.close(f)
            saved.append(p)
            print(f"  저장 {p.name}")

    f = fig_profile()
    p = OUT_DIR / "LUT복원_주파수프로파일.png"
    f.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(f)
    saved.append(p)
    print(f"  저장 {p.name}")

    print(f"\n완료 {len(saved)}개 / {time.time()-t0:.1f}s")
    print(f"위치: {OUT_DIR.resolve()}")
    print()
    print("보는 법")
    print("  컬러맵곡선  : 가운데 그래프가 산 모양이면 밝기=에너지가 성립 안 함")
    print("  비교        : 2행(현행)이 뿌옇고 3행(LUT)에 구조가 살아나면 복원 성공")
    print("  임계문제    : LUT 마스크가 하얗게 번지면 그게 band 특징이 죽는 이유")
    print("  주파수프로파일: 실선 파랑/빨강이 갈라져야 DEMAG 판별 가능")


if __name__ == "__main__":
    main()
