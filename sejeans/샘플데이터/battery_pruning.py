# -*- coding: utf-8 -*-
"""
배터리 고장 예측 — Pruning (가지치기) 경량화
=============================================
학습된 Baseline 1D CNN 모델에 Pruning을 적용하여
모델 크기와 추론 속도를 비교합니다.

사용법:
    1. 먼저 battery_cnn.py 를 실행하여 results/best_model.pt 생성
    2. python battery_pruning.py
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# battery_cnn.py 에서 모델과 데이터 관련 클래스/함수 재사용
from battery_cnn import (
    BatteryCNN, BatteryDataset, collect_data, preprocess_data,
    CLASS_NAMES, RESULTS_DIR, DEVICE
)

BATCH_SIZE = 16
FINETUNE_EPOCHS = 20
FINETUNE_LR = 5e-4

# =====================================================================
# 1. 모델 크기 & 추론 속도 측정 유틸
# =====================================================================
def get_model_size_mb(model):
    """모델의 실제 크기 (MB) 계산 — 0인 파라미터 제외"""
    total_bytes = 0
    for name, param in model.named_parameters():
        # 0이 아닌 파라미터만 카운트 (Pruning 효과 반영)
        nonzero = torch.count_nonzero(param).item()
        total_bytes += nonzero * param.element_size()
    for name, buf in model.named_buffers():
        if 'mask' not in name:  # pruning mask는 제외
            total_bytes += buf.nelement() * buf.element_size()
    return total_bytes / (1024 * 1024)


def get_sparsity(model):
    """전체 파라미터 중 0의 비율 (희소성)"""
    total, zeros = 0, 0
    for name, param in model.named_parameters():
        total += param.numel()
        zeros += (param == 0).sum().item()
    return zeros / total * 100 if total > 0 else 0


def measure_inference_time(model, sample_input, n_runs=100):
    """평균 추론 시간 측정 (ms)"""
    model.eval()
    sample_input = sample_input.to(DEVICE)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(sample_input)

    # 측정
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            model(sample_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return np.mean(times), np.std(times)


def evaluate_model(model, val_loader):
    """Accuracy 및 예측 결과 반환"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    return acc, all_preds, all_labels


# =====================================================================
# 2. Pruning 적용 함수
# =====================================================================
def apply_unstructured_pruning(model, amount=0.5):
    """
    비구조적 Pruning: 각 Conv1d & Linear 레이어의 가중치 중
    L1-norm이 작은 것부터 amount 비율만큼 0으로 만듦
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model


def apply_structured_pruning(model, amount=0.3):
    """
    구조적 Pruning: Conv1d 레이어의 필터(출력 채널) 단위로 제거
    실제 추론 속도 향상에 효과적
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
    return model


def make_pruning_permanent(model):
    """Pruning mask를 실제 가중치에 반영하고 mask 제거"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass  # pruning이 적용되지 않은 레이어
    return model


# =====================================================================
# 3. Fine-tuning (재학습)
# =====================================================================
def finetune(model, train_loader, val_loader, epochs, lr):
    """Pruning 후 성능 회복을 위한 재학습"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        acc, _, _ = evaluate_model(model, val_loader)
        if acc > best_acc:
            best_acc = acc

        if epoch % 5 == 0:
            print(f"    Fine-tune Epoch {epoch}/{epochs} | Val Acc: {acc:.4f}")

    return model, best_acc


# =====================================================================
# 4. 비교 시각화
# =====================================================================
def plot_comparison(results):
    """Baseline vs Pruning 결과 비교 차트"""
    names = [r["name"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    sizes = [r["size_mb"] for r in results]
    times = [r["infer_ms"] for r in results]
    sparsities = [r["sparsity"] for r in results]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63'][:len(names)]

    # 1) Accuracy
    bars = axes[0].bar(names, accs, color=colors)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("정확도 비교")
    axes[0].set_ylim(0, 105)
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', fontsize=10)

    # 2) Model Size
    bars = axes[1].bar(names, sizes, color=colors)
    axes[1].set_ylabel("Size (MB)")
    axes[1].set_title("모델 크기 비교")
    for bar, val in zip(bars, sizes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                     f'{val:.3f}', ha='center', fontsize=10)

    # 3) Inference Time
    bars = axes[2].bar(names, times, color=colors)
    axes[2].set_ylabel("Time (ms)")
    axes[2].set_title("추론 시간 비교")
    for bar, val in zip(bars, times):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.2f}', ha='center', fontsize=10)

    # 4) Sparsity
    bars = axes[3].bar(names, sparsities, color=colors)
    axes[3].set_ylabel("Sparsity (%)")
    axes[3].set_title("희소성 (0인 파라미터 비율)")
    axes[3].set_ylim(0, 105)
    for bar, val in zip(bars, sparsities):
        axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "pruning_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[결과 저장] {save_path}")


# =====================================================================
# 5. 메인 실행
# =====================================================================
def main():
    print("=" * 60)
    print("배터리 고장 예측 — Pruning 경량화")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # --- 데이터 준비 ---
    samples = collect_data()
    X, y = preprocess_data(samples)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_loader = DataLoader(BatteryDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(BatteryDataset(X_val, y_val),
                            batch_size=BATCH_SIZE, shuffle=False)

    # --- Baseline 모델 로드 ---
    model_path = os.path.join(RESULTS_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print("[오류] best_model.pt가 없습니다. 먼저 battery_cnn.py를 실행해주세요.")
        return

    baseline = BatteryCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    baseline.load_state_dict(torch.load(model_path, weights_only=True, map_location=DEVICE))
    print("[로드 완료] Baseline 모델 로드")

    # 샘플 입력 (추론 시간 측정용)
    sample_input = torch.randn(1, 1, X.shape[1]).to(DEVICE)

    # --- 결과 저장 리스트 ---
    results = []

    # ===== (A) Baseline 측정 =====
    print("\n[1/3] Baseline 모델 평가 중...")
    acc_base, _, _ = evaluate_model(baseline, val_loader)
    size_base = get_model_size_mb(baseline)
    time_base, _ = measure_inference_time(baseline, sample_input)
    sparsity_base = get_sparsity(baseline)

    results.append({
        "name": "Baseline",
        "accuracy": acc_base,
        "size_mb": size_base,
        "infer_ms": time_base,
        "sparsity": sparsity_base,
    })
    print(f"  Accuracy: {acc_base:.4f} | Size: {size_base:.4f} MB | "
          f"Infer: {time_base:.3f} ms | Sparsity: {sparsity_base:.1f}%")

    # ===== (B) 비구조적 Pruning 50% =====
    print("\n[2/3] 비구조적 Pruning (50%) 적용 중...")
    model_unstr = copy.deepcopy(baseline)
    model_unstr = apply_unstructured_pruning(model_unstr, amount=0.5)

    print("  → Fine-tuning...")
    model_unstr, _ = finetune(model_unstr, train_loader, val_loader,
                              FINETUNE_EPOCHS, FINETUNE_LR)
    make_pruning_permanent(model_unstr)

    acc_unstr, preds_unstr, labels_unstr = evaluate_model(model_unstr, val_loader)
    size_unstr = get_model_size_mb(model_unstr)
    time_unstr, _ = measure_inference_time(model_unstr, sample_input)
    sparsity_unstr = get_sparsity(model_unstr)

    results.append({
        "name": "Unstructured\n50%",
        "accuracy": acc_unstr,
        "size_mb": size_unstr,
        "infer_ms": time_unstr,
        "sparsity": sparsity_unstr,
    })
    print(f"  Accuracy: {acc_unstr:.4f} | Size: {size_unstr:.4f} MB | "
          f"Infer: {time_unstr:.3f} ms | Sparsity: {sparsity_unstr:.1f}%")

    # ===== (C) 구조적 Pruning 30% =====
    print("\n[3/3] 구조적 Pruning (30%) 적용 중...")
    model_str = copy.deepcopy(baseline)
    model_str = apply_structured_pruning(model_str, amount=0.3)

    print("  → Fine-tuning...")
    model_str, _ = finetune(model_str, train_loader, val_loader,
                            FINETUNE_EPOCHS, FINETUNE_LR)
    make_pruning_permanent(model_str)

    acc_str, preds_str, labels_str = evaluate_model(model_str, val_loader)
    size_str = get_model_size_mb(model_str)
    time_str, _ = measure_inference_time(model_str, sample_input)
    sparsity_str = get_sparsity(model_str)

    results.append({
        "name": "Structured\n30%",
        "accuracy": acc_str,
        "size_mb": size_str,
        "infer_ms": time_str,
        "sparsity": sparsity_str,
    })
    print(f"  Accuracy: {acc_str:.4f} | Size: {size_str:.4f} MB | "
          f"Infer: {time_str:.3f} ms | Sparsity: {sparsity_str:.1f}%")

    # --- Pruning 모델 저장 ---
    torch.save(model_unstr.state_dict(),
               os.path.join(RESULTS_DIR, "pruned_unstructured_50.pt"))
    torch.save(model_str.state_dict(),
               os.path.join(RESULTS_DIR, "pruned_structured_30.pt"))

    # --- 최종 비교표 출력 ---
    print("\n" + "=" * 70)
    print(f"{'모델':<22} {'정확도':>8} {'크기(MB)':>10} {'추론(ms)':>10} {'희소성':>8}")
    print("-" * 70)
    for r in results:
        name = r['name'].replace('\n', ' ')
        print(f"{name:<22} {r['accuracy']:>7.4f} {r['size_mb']:>10.4f} "
              f"{r['infer_ms']:>10.3f} {r['sparsity']:>7.1f}%")
    print("=" * 70)

    # --- Classification Report (최고 Pruning 모델) ---
    best_pruned = max(results[1:], key=lambda r: r['accuracy'])
    if best_pruned['name'].startswith("Unstructured"):
        print("\n[Best Pruned] 비구조적 Pruning Classification Report:")
        print(classification_report(labels_unstr, preds_unstr,
                                    target_names=CLASS_NAMES, zero_division=0))
    else:
        print("\n[Best Pruned] 구조적 Pruning Classification Report:")
        print(classification_report(labels_str, preds_str,
                                    target_names=CLASS_NAMES, zero_division=0))

    # --- 시각화 ---
    plot_comparison(results)

    print("\n[완료] results/ 폴더에서 pruning_comparison.png 를 확인하세요.")


if __name__ == "__main__":
    main()
