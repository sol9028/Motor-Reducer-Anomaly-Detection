"""최종 발표(slides_final.md) 차트. 수치 출처는 각 블록 주석 참고.

실행: python slides/charts/final_charts.py  (시스템분석 폴더에서)
"""
import sys

sys.path.insert(0, r"C:\Users\kingm\.claude\skills\slides")
import slidechart as sc

OUT = "slides/assets/"

# 1. 특징 구성별 Macro-F1 (차종 평균) — 보고서/논문_Overleaf/tables/ablation.tex
fig, ax = sc.figure("col")
sc.bar(ax, ["15ch\n+rpm", "18ch\n+온도", "51ch\n절대축", "51ch\n차수축", "87ch\n두 축 결합"],
       [0.8814, 0.8730, 0.9635, 0.9562, 0.9729], emphasize=[4], fmt="{:.4f}")
ax.set_ylim(0.85, 0.99)
sc.save(fig, OUT + "final_ablation.png")

# 2. 라즈베리파이 판정 지연 vs 입력 주기 — tables/latency.tex (9,996개, t1·b1 p50)
fig, ax = sc.figure("col")
sc.bar(ax, ["판정 지연\n(p50)", "입력 주기\n(10초 청크)"], [455.04, 10000], emphasize=[0],
       fmt="{:,.0f}ms", horizontal=True)
ax.set_xlim(0, 12000)
sc.save(fig, OUT + "final_pi_budget.png")

# 3. 정확도 비교 — 발표자료/7조 2학기 1차.pdf 19쪽 (99.65%는 무작위 분할, 테스트 표본 다름)
fig, ax = sc.figure("col")
sc.bar(ax, ["AI Hub\n공개 평가\n(무작위 분할)", "AI Hub\n세션 분할", "87ch 경량\n세션 분할"],
       [99.65, 96.33, 96.90], emphasize=[2], fmt="{:.2f}%")
ax.set_ylim(94, 100.6)
sc.save(fig, OUT + "final_acc_compare.png")

# 4. 효과크기와 신뢰구간 — 효과크기 보충 분석 해설 (Δ, 세션 bootstrap CI, 비열등 δ=1%p)
fig, ax = sc.figure("col")
ax.grid(False)
ax.set_ylim(-1, 1.6)
ax.set_xlim(-1.6, 2.0)
ax.tick_params(axis="y", left=False, labelleft=False)
ax.axvspan(-1.6, -1.0, color=sc.HIGHLIGHT, alpha=0.30, linewidth=0)
ax.text(-1.3, 1.35, "1%p 넘는\n손실 구간", ha="center", va="center", color=sc.INK, fontsize=15)
ax.axvline(0, color=sc.SUB, linewidth=1.5, linestyle="--")
ax.text(0.05, 1.45, "차이 0", ha="left", color=sc.SUB, fontsize=15)
# 양측 95% CI
ax.plot([-0.43, 1.62], [0.6, 0.6], color=sc.POINT, linewidth=4)
ax.plot(0.567, 0.6, "o", color=sc.POINT, markersize=14)
ax.text(0.567, 0.82, "Δ = +0.57%p", ha="center", fontweight="bold", color=sc.INK)
ax.text(-0.43, 0.32, "-0.43", ha="center", color=sc.INK)
ax.text(1.62, 0.32, "+1.62", ha="center", color=sc.INK)
ax.text(2.0, 0.6, " 양측 95% CI", va="center", color=sc.SUB, fontsize=15)
# 단측 95% 하한
ax.annotate("", xy=(-0.28, -0.4), xytext=(1.9, -0.4),
            arrowprops=dict(arrowstyle="-[", color=sc.BASE, linewidth=3))
ax.text(-0.36, -0.4, "단측 하한 -0.28 ", ha="right", va="center", color=sc.INK)
ax.set_xlabel("87ch - AI Hub 정확도 차이 (%p)")
sc.save(fig, OUT + "final_effect_ci.png")
