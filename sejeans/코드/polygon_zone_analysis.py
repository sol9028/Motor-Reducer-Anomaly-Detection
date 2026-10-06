# -*- coding: utf-8 -*-
"""
센서별 Polygon Y 대역 분석
- 모든 JSON 라벨 파일을 읽어 센서(Current_U / Vib_Motor / Vib_TM)별로
  결함 신호가 나타나는 주파수 대역(y_min, y_max)을 통계 분석한다.
- 결과로 도출된 FAULT_ZONE을 특징 추출 파이프라인에 고정값으로 사용한다.
"""

import json
import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 윈도우 콘솔 출력 인코딩 강제 설정
sys.stdout.reconfigure(encoding="utf-8")

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
DATA_DIR = (
    r"c:\Users\kingm\OneDrive\문서\새 폴더\OneDrive"
    r"\바탕 화면\학교\4학년 1학기\시스템분석"
    r"\샘플데이터\Sample\02.라벨링데이터\1.모터_감속기"
)
IMG_HEIGHT = 1280  # PNG 높이 (주파수 축 총 bin 수)

# ── 1. 전체 JSON 파일 읽기 ────────────────────────────────────────────────────
records = []

json_paths = glob.glob(os.path.join(DATA_DIR, "**", "*.json"), recursive=True)
print(f"총 JSON 파일 수: {len(json_paths)}")

for json_path in json_paths:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    fault_class = data["category"]["name"]
    sensor_name = data["image"]["sensor_name"]
    car_model   = data["metadata"]["car_model"]
    polygons    = data["annotations"]["polygons"]

    # polygon 유효성 확인
    is_empty = (
        not polygons
        or polygons == [[]]
        or len(polygons[0]) == 0
    )

    if is_empty:
        y_min, y_max, bandwidth = None, None, None
    else:
        coords    = polygons[0]              # 첫 번째(유일한) polygon
        y_coords  = [pt[1] for pt in coords]
        y_min     = min(y_coords)
        y_max     = max(y_coords)
        bandwidth = y_max - y_min

    records.append({
        "fault_class": fault_class,
        "sensor_name": sensor_name,
        "car_model":   car_model,
        "has_polygon": not is_empty,
        "y_min":       y_min,
        "y_max":       y_max,
        "bandwidth":   bandwidth,
    })

df = pd.DataFrame(records)
print("\n[클래스 x 센서 x polygon 유무 분포]")
print(df.groupby(["fault_class", "sensor_name"])["has_polygon"]
        .value_counts()
        .unstack(fill_value=0)
        .to_string())

# ── 2. polygon 있는 샘플만 추출 ───────────────────────────────────────────────
df_valid = df[df["has_polygon"]].copy()

# ── 3. 센서별 통계 ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("[센서별 Polygon Y 대역 통계]")
print("="*60)

sensor_zones = {}

for sensor, grp in df_valid.groupby("sensor_name"):
    y_min_vals = grp["y_min"].values
    y_max_vals = grp["y_max"].values
    bw_vals    = grp["bandwidth"].values

    stats = {
        "count":        len(grp),
        "y_min_mean":   np.mean(y_min_vals),
        "y_min_median": np.median(y_min_vals),
        "y_min_p5":     np.percentile(y_min_vals, 5),
        "y_max_mean":   np.mean(y_max_vals),
        "y_max_median": np.median(y_max_vals),
        "y_max_p95":    np.percentile(y_max_vals, 95),
        "bw_mean":      np.mean(bw_vals),
        "bw_std":       np.std(bw_vals),
    }

    # 권장 fault_zone: 하한은 5th percentile, 상한은 95th percentile
    zone = (int(stats["y_min_p5"]), int(stats["y_max_p95"]))
    sensor_zones[sensor] = zone

    print(f"\n  센서: {sensor}  (polygon 보유 샘플 수: {stats['count']})")
    print(f"    y_min - mean: {stats['y_min_mean']:.1f} | median: {stats['y_min_median']:.1f} | p5:  {stats['y_min_p5']:.1f}")
    print(f"    y_max - mean: {stats['y_max_mean']:.1f} | median: {stats['y_max_median']:.1f} | p95: {stats['y_max_p95']:.1f}")
    print(f"    bandwidth - mean: {stats['bw_mean']:.1f} +/- {stats['bw_std']:.1f} px")
    print(f"    >> FAULT_ZONE: y = [{zone[0]}, {zone[1]}]")

# ── 4. 결함 클래스 x 센서별 상세 통계 ────────────────────────────────────────
print("\n" + "="*60)
print("[결함 클래스 x 센서별 y_min / y_max 평균]")
print("="*60)

pivot_min = df_valid.pivot_table(
    values="y_min", index="sensor_name", columns="fault_class",
    aggfunc="mean"
).round(1)

pivot_max = df_valid.pivot_table(
    values="y_max", index="sensor_name", columns="fault_class",
    aggfunc="mean"
).round(1)

print("\n  y_min (평균):")
print(pivot_min.to_string())
print("\n  y_max (평균):")
print(pivot_max.to_string())

# ── 5. 최종 FAULT_ZONE 출력 ───────────────────────────────────────────────────
print("\n" + "="*60)
print("[최종 FAULT_ZONE - 파이프라인에 하드코딩할 값]")
print("="*60)
print("\nFAULT_ZONE = {")
for sensor, (y0, y1) in sensor_zones.items():
    coverage_pct = (y1 - y0) / IMG_HEIGHT * 100
    print(f'    "{sensor}": ({y0}, {y1}),   # 전체 높이의 {coverage_pct:.1f}% 커버')
print("}")

# ── 6. 시각화 ─────────────────────────────────────────────────────────────────
sensors       = sorted(df_valid["sensor_name"].unique())
fault_classes = sorted(df_valid["fault_class"].unique())
colors        = {"DEMAG": "#e74c3c", "ECC10": "#3498db", "ECC20": "#2ecc71", "REDUC": "#9b59b6"}

fig, axes = plt.subplots(1, len(sensors), figsize=(5 * len(sensors), 6), sharey=True)
if len(sensors) == 1:
    axes = [axes]

for ax, sensor in zip(axes, sensors):
    grp_sensor = df_valid[df_valid["sensor_name"] == sensor]

    for i, fault in enumerate(fault_classes):
        grp_fault = grp_sensor[grp_sensor["fault_class"] == fault]
        if grp_fault.empty:
            continue
        rng = np.random.default_rng(seed=42)
        x = i + rng.uniform(-0.2, 0.2, size=len(grp_fault))
        ax.scatter(x, grp_fault["y_min"], color=colors.get(fault, "gray"),
                   marker="v", s=30, alpha=0.6)
        ax.scatter(x, grp_fault["y_max"], color=colors.get(fault, "gray"),
                   marker="^", s=30, alpha=0.6)

    # 권장 zone 표시
    zone = sensor_zones[sensor]
    ax.axhspan(zone[0], zone[1], alpha=0.12, color="orange")
    ax.axhline(zone[0], color="orange", linewidth=1.5, linestyle="--")
    ax.axhline(zone[1], color="orange", linewidth=1.5, linestyle="--")
    ax.text(len(fault_classes) - 0.5, zone[0] - 15,
            f"y={zone[0]}", color="darkorange", fontsize=8, ha="right")
    ax.text(len(fault_classes) - 0.5, zone[1] + 10,
            f"y={zone[1]}", color="darkorange", fontsize=8, ha="right")

    ax.set_title(sensor, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(fault_classes)))
    ax.set_xticklabels(fault_classes, rotation=15)
    ax.set_ylim(IMG_HEIGHT, 0)   # y=0이 이미지 상단이므로 역방향
    ax.set_ylabel("pixel row (frequency axis)")
    ax.grid(axis="y", alpha=0.3)

# 공통 범례
legend_elements = [
    mpatches.Patch(color=c, label=f, alpha=0.8)
    for f, c in colors.items()
    if f in fault_classes
] + [mpatches.Patch(color="orange", alpha=0.3, label="FAULT_ZONE (p5~p95)")]

fig.legend(handles=legend_elements, loc="upper center",
           ncol=len(legend_elements), bbox_to_anchor=(0.5, 1.02))

plt.suptitle("Polygon Y Range per Sensor", fontsize=14, y=1.05)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "polygon_zone_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n시각화 저장 완료: {out_path}")
