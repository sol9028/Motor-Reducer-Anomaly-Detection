# -*- coding: utf-8 -*-
"""
배터리 이상 분류 모델 학습
============================
입력  : (20, 96|98) 배터리 셀 전압 NPY 배열
출력 1: 상태 분류   (NORMAL / CAUTION / DEFECT)
출력 2: 결함 유형   (none / cell_voltage / cell_deviation)

전략
-----
- (20, N) 배열을 (1, 20, 98) 2D 그레이스케일 텐서로 변환
- 경량 2D CNN + 멀티태스크 헤드
- 학습 완료 후 ONNX 내보내기 → backend/weights/battery_classifier.onnx

실행
-----
    C:/Users/kingm/miniforge3/python.exe train_battery_classifier.py
"""

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
SAMPLE_ROOT = SCRIPT_DIR.parent.parent / "샘플데이터" / "Sample"
RAW_BAT     = SAMPLE_ROOT / "01.원천데이터"  / "2.배터리"
LBL_BAT     = SAMPLE_ROOT / "02.라벨링데이터" / "2.배터리"
OUT_DIR     = SCRIPT_DIR.parent / "backend" / "weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 고정 입력 크기 (96셀은 패딩으로 98셀로 맞춤)
FIXED_TIMESTEPS = 20
FIXED_CELLS     = 98
BATCH_SIZE      = 32
EPOCHS          = 30
LR              = 1e-3
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전압 정규화 범위
V_MIN = 3.0
V_MAX = 4.2

STATUS_CLASSES = ["CAUTION", "DEFECT", "NORMAL"]       # 알파벳순
FAULT_CLASSES  = ["cell_deviation", "cell_voltage", "none"]  # 알파벳순
STATUS2IDX     = {c: i for i, c in enumerate(STATUS_CLASSES)}
FAULT2IDX      = {c: i for i, c in enumerate(FAULT_CLASSES)}

print(f"Device : {DEVICE}")
print(f"샘플 루트: {SAMPLE_ROOT}")


# ─────────────────────────────────────────────
# 1. 데이터 수집
# ─────────────────────────────────────────────
def collect_samples():
    """NPY 파일과 라벨을 수집한다."""
    records = []

    for bat_type_dir in RAW_BAT.iterdir():
        if not bat_type_dir.is_dir():
            continue
        bat_type = bat_type_dir.name           # Vlt_96 | Vlt_98 | Dev_96 | Dev_98
        npy_root = bat_type_dir / "NPY"
        if not npy_root.exists():
            continue

        json_root = LBL_BAT / bat_type / "JSON"

        for car_dir in npy_root.iterdir():
            if not car_dir.is_dir():
                continue

            for status_dir in car_dir.iterdir():
                if not status_dir.is_dir():
                    continue
                status = status_dir.name.upper()    # NORMAL | CAUTION | DEFECT
                if status not in STATUS2IDX:
                    continue

                # 결함 유형: NORMAL이면 none, Dev면 cell_deviation, Vlt면 cell_voltage
                if status == "NORMAL":
                    fault_type = "none"
                elif bat_type.startswith("Dev"):
                    fault_type = "cell_deviation"
                else:
                    fault_type = "cell_voltage"

                for npy_file in status_dir.rglob("*.npy"):
                    records.append({
                        "npy_path":  str(npy_file),
                        "status":    status,
                        "fault":     fault_type,
                        "s_idx":     STATUS2IDX[status],
                        "f_idx":     FAULT2IDX[fault_type],
                    })

    print(f"\n수집된 샘플: {len(records)}개")
    s_cnt = Counter(r["status"] for r in records)
    f_cnt = Counter(r["fault"]  for r in records)
    print("  상태별:", dict(sorted(s_cnt.items())))
    print("  결함별:", dict(sorted(f_cnt.items())))
    return records


# ─────────────────────────────────────────────
# 2. 전처리
# ─────────────────────────────────────────────
def preprocess(npy_path: str) -> np.ndarray | None:
    """(T, N) 배열 → (1, FIXED_TIMESTEPS, FIXED_CELLS) float32 텐서.

    - 전압 [V_MIN, V_MAX] → [0, 1] 정규화
    - 시간 축이 20이 아니면 잘라내거나 패딩
    - 셀 수가 FIXED_CELLS보다 작으면 오른쪽에 zero-padding
    """
    try:
        arr = np.load(npy_path).astype(np.float32)
    except Exception:
        return None

    if arr.ndim != 2:
        return None

    # 시간 축 정규화
    T, N = arr.shape
    if T > FIXED_TIMESTEPS:
        arr = arr[:FIXED_TIMESTEPS]
    elif T < FIXED_TIMESTEPS:
        pad_t = np.zeros((FIXED_TIMESTEPS - T, N), dtype=np.float32)
        arr = np.vstack([arr, pad_t])

    # 셀 축 정규화
    if N < FIXED_CELLS:
        pad_n = np.zeros((FIXED_TIMESTEPS, FIXED_CELLS - N), dtype=np.float32)
        arr = np.hstack([arr, pad_n])
    elif N > FIXED_CELLS:
        arr = arr[:, :FIXED_CELLS]

    # 전압 → [0, 1]
    arr = (arr - V_MIN) / (V_MAX - V_MIN)
    arr = np.clip(arr, 0.0, 1.0)

    return arr[np.newaxis]   # (1, 20, 98)


# ─────────────────────────────────────────────
# 3. Dataset
# ─────────────────────────────────────────────
class BatteryDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        arr = preprocess(rec["npy_path"])

        if arr is None:
            arr = np.zeros((1, FIXED_TIMESTEPS, FIXED_CELLS), dtype=np.float32)

        if self.augment:
            # 미세 노이즈 + 전압 오프셋
            arr = arr + np.random.normal(0, 0.005, arr.shape).astype(np.float32)
            arr = arr + np.random.uniform(-0.008, 0.008)
            arr = np.clip(arr, 0.0, 1.0)

        tensor = torch.from_numpy(arr)
        s_label = torch.tensor(rec["s_idx"], dtype=torch.long)
        f_label = torch.tensor(rec["f_idx"], dtype=torch.long)
        return tensor, s_label, f_label


# ─────────────────────────────────────────────
# 4. 모델: 경량 2D CNN + 멀티태스크 헤드
# ─────────────────────────────────────────────
class BatteryClassifier(nn.Module):
    """배터리 상태 분류 + 결함 유형 분류를 동시에 수행하는 경량 CNN."""

    def __init__(self, num_status: int, num_fault: int):
        super().__init__()
        # 특징 추출기: (1, 20, 98) → (128, 1, 1)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # → (32, 10, 49)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # → (64, 5, 24)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),            # → (128, 1, 1)
        )

        # 헤드 1: 상태 분류 (NORMAL / CAUTION / DEFECT)
        self.status_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_status),
        )

        # 헤드 2: 결함 유형 분류 (none / cell_voltage / cell_deviation)
        self.fault_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_fault),
        )

    def forward(self, x):
        feat         = self.features(x).flatten(1)    # (B, 128)
        status_logit = self.status_head(feat)          # (B, 3)
        fault_logit  = self.fault_head(feat)           # (B, 3)
        return status_logit, fault_logit


# ─────────────────────────────────────────────
# 5. 학습
# ─────────────────────────────────────────────
def train():
    records = collect_samples()
    if not records:
        raise RuntimeError("샘플을 찾을 수 없습니다. SAMPLE_ROOT를 확인하세요.")

    np.random.seed(42)
    idx   = np.random.permutation(len(records))
    split = int(len(idx) * 0.8)
    tr_recs = [records[i] for i in idx[:split]]
    va_recs = [records[i] for i in idx[split:]]

    tr_ds = BatteryDataset(tr_recs, augment=True)
    va_ds = BatteryDataset(va_recs, augment=False)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = BatteryClassifier(len(STATUS_CLASSES), len(FAULT_CLASSES)).to(DEVICE)
    ce_loss   = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_pt_path = OUT_DIR / "battery_classifier_best.pt"

    for epoch in range(1, EPOCHS + 1):
        # ── 학습
        model.train()
        tr_loss = 0.0
        tr_s_correct = tr_f_correct = 0

        for tensors, s_labels, f_labels in tr_dl:
            tensors  = tensors.to(DEVICE)
            s_labels = s_labels.to(DEVICE)
            f_labels = f_labels.to(DEVICE)

            optimizer.zero_grad()
            s_logits, f_logits = model(tensors)

            loss_s = ce_loss(s_logits, s_labels)
            loss_f = ce_loss(f_logits, f_labels)
            loss   = loss_s + loss_f          # 동일 가중치
            loss.backward()
            optimizer.step()

            tr_loss      += loss.item() * len(tensors)
            tr_s_correct += (s_logits.argmax(1) == s_labels).sum().item()
            tr_f_correct += (f_logits.argmax(1) == f_labels).sum().item()

        # ── 검증
        model.eval()
        va_s_correct = va_f_correct = va_total = 0
        with torch.no_grad():
            for tensors, s_labels, f_labels in va_dl:
                tensors  = tensors.to(DEVICE)
                s_labels = s_labels.to(DEVICE)
                f_labels = f_labels.to(DEVICE)
                s_logits, f_logits = model(tensors)
                va_s_correct += (s_logits.argmax(1) == s_labels).sum().item()
                va_f_correct += (f_logits.argmax(1) == f_labels).sum().item()
                va_total     += len(tensors)

        tr_s_acc = tr_s_correct / len(tr_ds)
        tr_f_acc = tr_f_correct / len(tr_ds)
        va_s_acc = va_s_correct / va_total
        va_f_acc = va_f_correct / va_total
        scheduler.step()

        # 두 태스크 평균 정확도로 best 모델 선택
        va_avg = (va_s_acc + va_f_acc) / 2

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"loss={tr_loss/len(tr_ds):.4f} | "
              f"tr_status={tr_s_acc:.3f} tr_fault={tr_f_acc:.3f} | "
              f"va_status={va_s_acc:.3f} va_fault={va_f_acc:.3f}")

        if va_avg > best_val_acc:
            best_val_acc = va_avg
            torch.save(model.state_dict(), best_pt_path)
            print(f"  >> Best saved (va_avg={va_avg:.3f})")

    print(f"\n최고 검증 평균 정확도: {best_val_acc:.3f}")
    return model, best_pt_path


# ─────────────────────────────────────────────
# 6. ONNX 내보내기
# ─────────────────────────────────────────────
def export_onnx(model, pt_path: Path):
    onnx_path = OUT_DIR / "battery_classifier.onnx"
    model.eval()
    dummy = torch.zeros(1, 1, FIXED_TIMESTEPS, FIXED_CELLS)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["voltage_array"],
        output_names=["status_logits", "fault_logits"],
        dynamic_axes={
            "voltage_array": {0: "batch"},
            "status_logits": {0: "batch"},
            "fault_logits":  {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX 저장: {onnx_path}")

    meta = {
        "status_classes":  STATUS_CLASSES,
        "fault_classes":   FAULT_CLASSES,
        "fixed_timesteps": FIXED_TIMESTEPS,
        "fixed_cells":     FIXED_CELLS,
        "v_min":           V_MIN,
        "v_max":           V_MAX,
    }
    meta_path = OUT_DIR / "battery_classifier_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"메타데이터 저장: {meta_path}")
    return onnx_path


# ─────────────────────────────────────────────
# 7. 빠른 검증
# ─────────────────────────────────────────────
def quick_validate_onnx(onnx_path: Path):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    dummy = np.random.rand(1, 1, FIXED_TIMESTEPS, FIXED_CELLS).astype(np.float32)
    s_logits, f_logits = sess.run(None, {"voltage_array": dummy})
    pred_status = STATUS_CLASSES[int(np.argmax(s_logits[0]))]
    pred_fault  = FAULT_CLASSES [int(np.argmax(f_logits[0]))]
    print(f"ONNX 검증 OK | 상태: {pred_status} | 결함유형: {pred_fault}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("배터리 이상 분류 모델 학습")
    print("=" * 55)
    model, pt_path = train()

    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    onnx_path = export_onnx(model, pt_path)
    quick_validate_onnx(onnx_path)

    print("\n완료! backend/weights/battery_classifier.onnx 를 백엔드가 자동으로 로드합니다.")
