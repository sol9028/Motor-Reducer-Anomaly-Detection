"""
배터리 라벨링데이터 온도(temperature) 시각화 스크립트
- 02.라벨링데이터/2.배터리 하위 모든 JSON 파일에서 metadata.temperature 추출
- 카테고리(NORMAL/CAUTION/DEFECT), 센서타입(Dev/Vlt), 차종별로 시각화
"""

import os
import json
import glob
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 데이터 수집 ──────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "Sample", "02.라벨링데이터", "2.배터리")

records = []  # 각 JSON에서 추출한 정보 저장

for root, dirs, files in os.walk(BASE_DIR):
    for fname in files:
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(root, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            temp = data.get('metadata', {}).get('temperature')
            category = data.get('category', {}).get('name', 'UNKNOWN')
            car_model = data.get('metadata', {}).get('car_model', 'UNKNOWN')
            speed = data.get('metadata', {}).get('speed', 0)
            measure_date = data.get('metadata', {}).get('measure_date', '')

            # 상위 폴더에서 센서 타입(Dev_96, Vlt_98 등) 추출
            rel = os.path.relpath(fpath, BASE_DIR)
            sensor_type = rel.split(os.sep)[0]  # Dev_96, Dev_98, Vlt_96, Vlt_98

            if temp is not None:
                records.append({
                    'temperature': temp,
                    'category': category,
                    'car_model': car_model,
                    'sensor_type': sensor_type,
                    'speed': speed,
                    'measure_date': measure_date,
                    'file': fname
                })
        except Exception as e:
            print(f"[WARN] {fpath}: {e}")

print(f"총 {len(records)}개의 JSON 파일에서 온도 데이터 수집 완료\n")

if len(records) == 0:
    print("수집된 데이터가 없습니다. 경로를 확인해주세요.")
    exit()

# ── 데이터 정리 ──────────────────────────────────────────────────────
categories = sorted(set(r['category'] for r in records))
car_models = sorted(set(r['car_model'] for r in records))
sensor_types = sorted(set(r['sensor_type'] for r in records))

cat_colors = {'NORMAL': '#2ecc71', 'CAUTION': '#f39c12', 'DEFECT': '#e74c3c'}

# ── 시각화 ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('배터리 라벨링데이터 — 온도(Temperature) 분석', fontsize=18, fontweight='bold', y=0.98)

# ① 카테고리별 온도 분포 (Box Plot)
ax1 = axes[0, 0]
box_data = []
box_labels = []
box_colors = []
for cat in categories:
    temps = [r['temperature'] for r in records if r['category'] == cat]
    if temps:
        box_data.append(temps)
        box_labels.append(f"{cat}\n(n={len(temps)})")
        box_colors.append(cat_colors.get(cat, '#95a5a6'))

bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.5,
                 boxprops=dict(linewidth=1.5),
                 medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_title('① 카테고리별 온도 분포', fontsize=14, fontweight='bold')
ax1.set_ylabel('온도 (°C)', fontsize=12)
ax1.grid(axis='y', alpha=0.3)

# ② 센서 타입별 카테고리별 온도 평균 (Grouped Bar)
ax2 = axes[0, 1]
x = np.arange(len(sensor_types))
width = 0.25
for i, cat in enumerate(categories):
    means = []
    for st in sensor_types:
        temps = [r['temperature'] for r in records if r['sensor_type'] == st and r['category'] == cat]
        means.append(np.mean(temps) if temps else 0)
    bars = ax2.bar(x + i * width, means, width, label=cat,
                   color=cat_colors.get(cat, '#95a5a6'), alpha=0.8, edgecolor='white', linewidth=0.5)
    # 값 표시
    for bar, val in zip(bars, means):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_xticks(x + width)
ax2.set_xticklabels(sensor_types, fontsize=10)
ax2.set_title('② 센서타입 × 카테고리별 평균 온도', fontsize=14, fontweight='bold')
ax2.set_ylabel('평균 온도 (°C)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# ③ 차종별 온도 분포 (Violin + Strip)
ax3 = axes[1, 0]
model_data = []
model_labels = []
for model in car_models:
    temps = [r['temperature'] for r in records if r['car_model'] == model]
    if temps:
        model_data.append(temps)
        model_labels.append(f"{model}\n(n={len(temps)})")

if model_data:
    parts = ax3.violinplot(model_data, showmeans=True, showmedians=True)
    # 색상 지정
    model_palette = ['#3498db', '#9b59b6', '#1abc9c', '#e67e22']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(model_palette[i % len(model_palette)])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('#e74c3c')
    parts['cmedians'].set_color('#2c3e50')
    # 개별 데이터 포인트 (strip)
    for i, temps in enumerate(model_data):
        jitter = np.random.normal(0, 0.04, len(temps))
        ax3.scatter([i + 1] * len(temps) + jitter, temps,
                    alpha=0.4, s=15, color=model_palette[i % len(model_palette)], zorder=3)
    ax3.set_xticks(range(1, len(model_labels) + 1))
    ax3.set_xticklabels(model_labels, fontsize=10)

ax3.set_title('③ 차종별 온도 분포 (Violin Plot)', fontsize=14, fontweight='bold')
ax3.set_ylabel('온도 (°C)', fontsize=12)
ax3.grid(axis='y', alpha=0.3)

# ④ 속도 vs 온도 산점도 (카테고리별 색상)
ax4 = axes[1, 1]
for cat in categories:
    speeds = [r['speed'] for r in records if r['category'] == cat]
    temps = [r['temperature'] for r in records if r['category'] == cat]
    ax4.scatter(speeds, temps, label=cat, alpha=0.6, s=40,
                color=cat_colors.get(cat, '#95a5a6'), edgecolors='white', linewidth=0.5)
ax4.set_title('④ 속도 vs 온도 (카테고리별)', fontsize=14, fontweight='bold')
ax4.set_xlabel('속도 (km/h)', fontsize=12)
ax4.set_ylabel('온도 (°C)', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# 저장
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'temperature_visualization.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"시각화 저장 완료: {save_path}")


# ── 요약 통계 출력 ───────────────────────────────────────────────────
print("\n" + "="*60)
print(" 온도 요약 통계")
print("="*60)
for cat in categories:
    temps = [r['temperature'] for r in records if r['category'] == cat]
    if temps:
        print(f"\n  [{cat}] (n={len(temps)})")
        print(f"   평균: {np.mean(temps):.2f} C")
        print(f"   중앙값: {np.median(temps):.2f} C")
        print(f"   최소: {np.min(temps):.2f} C | 최대: {np.max(temps):.2f} C")
        print(f"   표준편차: {np.std(temps):.2f} C")
