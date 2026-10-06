# -*- coding: utf-8 -*-
"""
배터리 고장 예측 — 1D CNN 분류 모델
====================================
AI Hub '자율주행 고장진단 데이터' 중 2.배터리 데이터를 사용하여
NORMAL / CAUTION / DEFECT 3클래스를 분류합니다.

사용법:
    python battery_cnn.py
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # GUI 없이 저장만
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 0. 설정
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "Sample", "01.원천데이터", "2.배터리")
LABEL_DIR = os.path.join(BASE_DIR, "Sample", "02.라벨링데이터", "2.배터리")

CLASS_MAP = {"NORMAL": 0, "CAUTION": 1, "DEFECT": 2}
CLASS_NAMES = ["NORMAL", "CAUTION", "DEFECT"]

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================================
# 1. 데이터 수집 — 폴더 구조에서 npy + 클래스 자동 추출
# =====================================================================
def collect_data():
    """
    폴더 구조:
      {Dev_96,Dev_98,Vlt_96,Vlt_98}/NPY/{차종}/{NORMAL|CAUTION|DEFECT}/{날짜}/{센서명}/xxx.npy
    """
    samples = []   # (npy_path, class_label)
    
    npy_files = glob.glob(os.path.join(RAW_DIR, "**", "*.npy"), recursive=True)
    
    for fpath in npy_files:
        # 경로에서 클래스 추출 (NORMAL / CAUTION / DEFECT)
        parts = fpath.replace("\\", "/").split("/")
        label = None
        for part in parts:
            if part.upper() in CLASS_MAP:
                label = CLASS_MAP[part.upper()]
                break
        
        if label is not None:
            samples.append((fpath, label))
    
    print(f"[데이터 수집] 총 {len(samples)}개 샘플 발견")
    for cls_name, cls_id in CLASS_MAP.items():
        count = sum(1 for _, l in samples if l == cls_id)
        print(f"  - {cls_name}: {count}개")
    
    return samples


# =====================================================================
# 2. 데이터 전처리 — 길이 통일 + 정규화
# =====================================================================
def preprocess_data(samples):
    """npy 파일들을 로드하고 2D → 1D 평탄화, 길이 통일, 정규화"""
    arrays = []
    labels = []
    
    for fpath, label in samples:
        try:
            arr = np.load(fpath).astype(np.float32)
        except Exception as e:
            print(f"  [경고] 파일 로드 실패: {fpath} — {e}")
            continue
        
        # 2D 배열이면 평탄화(flatten)
        if arr.ndim > 1:
            arr = arr.flatten()
        
        arrays.append(arr)
        labels.append(label)
    
    # 길이 통일: 가장 긴 배열 기준으로 zero-padding (또는 자르기)
    max_len = max(len(a) for a in arrays)
    print(f"[전처리] 배열 길이 범위: {min(len(a) for a in arrays)} ~ {max_len}")
    print(f"[전처리] 통일 길이: {max_len}")
    
    X = np.zeros((len(arrays), max_len), dtype=np.float32)
    for i, arr in enumerate(arrays):
        length = min(len(arr), max_len)
        X[i, :length] = arr[:length]
    
    # 정규화 (Z-score)
    mean = X.mean()
    std = X.std() + 1e-8
    X = (X - mean) / std
    
    y = np.array(labels, dtype=np.int64)
    
    print(f"[전처리] 최종 데이터 shape: X={X.shape}, y={y.shape}")
    return X, y


# =====================================================================
# 3. PyTorch Dataset
# =====================================================================
class BatteryDataset(Dataset):
    def __init__(self, X, y):
        # (N, seq_len) → (N, 1, seq_len) : Conv1D 입력 형태
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =====================================================================
# 4. 1D CNN 모델 정의
# =====================================================================
class BatteryCNN(nn.Module):
    """
    경량 1D CNN 분류 모델
    Conv1D 블록 3개 + Global Average Pooling + FC
    """
    def __init__(self, num_classes=3):
        super(BatteryCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2),
            
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =====================================================================
# 5. 학습 함수
# =====================================================================
def train_model(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # --- Validate ---
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_running_loss += loss.item() * X_batch.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # 최고 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pt"))
    
    print(f"\n[학습 완료] Best Val Accuracy: {best_val_acc:.4f}")
    return history


# =====================================================================
# 6. 평가 및 시각화
# =====================================================================
def evaluate_and_plot(model, val_loader, history, device):
    # --- 예측 ---
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # --- Classification Report ---
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))
    
    # --- Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1) Loss 그래프
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2) Accuracy 그래프
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3) Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[2])
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")
    axes[2].set_title("Confusion Matrix")
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, "training_results.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[결과 저장] {save_path}")


# =====================================================================
# 7. 메인 실행
# =====================================================================
def main():
    print("=" * 60)
    print("배터리 고장 예측 — 1D CNN 분류 모델")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # 1) 데이터 수집
    samples = collect_data()
    if len(samples) == 0:
        print("[오류] 데이터를 찾을 수 없습니다. 경로를 확인해주세요.")
        print(f"  원천데이터 경로: {RAW_DIR}")
        return
    
    # 2) 전처리
    X, y = preprocess_data(samples)
    
    # 3) Train/Val 분할 (80:20, 층화 추출)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[분할] Train: {len(y_train)}개, Val: {len(y_val)}개")
    
    # 4) DataLoader 생성
    train_ds = BatteryDataset(X_train, y_train)
    val_ds = BatteryDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5) 모델 생성
    model = BatteryCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[모델] BatteryCNN — 총 파라미터: {total_params:,}개")
    print(model)
    
    # 6) 학습
    print(f"\n[학습 시작] Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")
    history = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, DEVICE)
    
    # 7) 최고 모델 로드 후 평가
    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "best_model.pt"), weights_only=True))
    evaluate_and_plot(model, val_loader, history, DEVICE)
    
    print("\n[완료] results/ 폴더에서 결과를 확인하세요.")


if __name__ == "__main__":
    main()
