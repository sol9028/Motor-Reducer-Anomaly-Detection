# -*- coding: utf-8 -*-
"""
모터-감속기 이상 위치 탐지 모델 학습
======================================
입력  : Current_U STFT 스펙트로그램 PNG (2000×1280)
출력  : ① 결함 유형 분류 (NORMAL/ECC10/ECC20/DEMAG/REDUC)
        ② 이상 영역 바운딩박스 (x1,y1,x2,y2 — 0~1 정규화)

전략
-----
- MobileNetV2 (ImageNet pretrained) + 멀티태스크 헤드
- bbox 손실: NORMAL 클래스는 마스킹하여 제외
- 학습 완료 후 ONNX 내보내기 → backend/weights/motor_localizer.onnx

실행
-----
    # miniforge (torch 있는 환경)
    C:/Users/kingm/miniforge3/python.exe train_motor_localizer.py
"""

import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
SAMPLE_ROOT = SCRIPT_DIR.parent.parent / "샘플데이터" / "Sample"
RAW_MOTOR   = SAMPLE_ROOT / "01.원천데이터"  / "1.모터_감속기"
LBL_MOTOR   = SAMPLE_ROOT / "02.라벨링데이터" / "1.모터_감속기"
OUT_DIR     = SCRIPT_DIR.parent / "backend" / "weights"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 40
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["DEMAG", "ECC10", "ECC20", "NORMAL", "REDUC"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS2IDX   = {c: i for i, c in enumerate(CLASS_NAMES)}

# 결함 유형별 이상 점수 목표값 (0~1, 학습 시 MSELoss 사용)
_SCORE_TARGETS = {
    "NORMAL": 0.05,
    "ECC10":  0.22,
    "ECC20":  0.38,
    "DEMAG":  0.50,
    "REDUC":  0.45,
}

print(f"Device: {DEVICE}")
print(f"Sample root: {SAMPLE_ROOT}")


# ─────────────────────────────────────────────
# 1. 데이터 수집 (Current_U 채널만 사용)
# ─────────────────────────────────────────────
def collect_samples():
    """PNG + JSON 페어를 수집하고 bbox를 추출한다."""
    records = []  # {"img_path", "label", "bbox": [x1,y1,x2,y2] or None}

    for png in RAW_MOTOR.rglob("*.png"):
        rel   = png.relative_to(RAW_MOTOR)
        parts = rel.parts  # car / fault / date / channel / file

        if len(parts) < 5:
            continue
        channel = parts[3]
        if channel != "Current_U":     # Current_U만 사용
            continue

        fault = parts[1].upper()
        if fault not in CLASS2IDX:
            continue

        # 대응 JSON
        json_path = LBL_MOTOR / rel.with_suffix(".json")
        if not json_path.exists():
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            lbl = json.load(f)

        img_w = lbl["image"]["width"]   # 2000
        img_h = lbl["image"]["height"]  # 1280
        polys  = lbl["annotations"]["polygons"]

        # bbox 추출 (정규화)
        bbox = None
        if polys and polys[0]:
            pts = polys[0]
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            bbox = [
                min(xs) / img_w,
                min(ys) / img_h,
                max(xs) / img_w,
                max(ys) / img_h,
            ]

        records.append({
            "img_path": str(png),
            "label":    CLASS2IDX[fault],
            "fault":    fault,
            "bbox":     bbox,           # NORMAL이면 None
        })

    print(f"수집된 샘플: {len(records)}개")
    from collections import Counter
    cnt = Counter(r["fault"] for r in records)
    for k, v in sorted(cnt.items()):
        print(f"  {k}: {v}")
    return records


# ─────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────
class MotorDataset(Dataset):
    def __init__(self, records, transform=None):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["img_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(rec["label"], dtype=torch.long)

        # bbox: NORMAL → [0,0,0,0], has_bbox=0
        if rec["bbox"] is not None:
            bbox    = torch.tensor(rec["bbox"], dtype=torch.float32)
            has_box = torch.tensor(1.0)
        else:
            bbox    = torch.zeros(4, dtype=torch.float32)
            has_box = torch.tensor(0.0)

        score_target = torch.tensor(
            [_SCORE_TARGETS.get(rec["fault"], 0.05)], dtype=torch.float32
        )
        return img, label, bbox, has_box, score_target


def make_transforms(train: bool):
    if train:
        return T.Compose([
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ─────────────────────────────────────────────
# 3. 모델: MobileNetV2 + 멀티태스크 헤드
# ─────────────────────────────────────────────
class MotorLocalizer(nn.Module):
    """분류 + 바운딩박스 회귀를 동시에 수행하는 경량 모델."""

    def __init__(self, num_classes: int):
        super().__init__()
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # 특징 추출기 (1280차원 feature)
        self.features = base.features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        feat_dim      = base.last_channel  # 1280

        # 헤드 1: 분류
        self.cls_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, num_classes),
        )
        # 헤드 2: bbox 회귀 (sigmoid → 0~1 정규화)
        self.bbox_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )
        # 헤드 3: 이상 점수 회귀 (sigmoid → 0~1, 추론 시 ×100 → 0~100점)
        self.score_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat   = self.features(x)
        feat   = self.pool(feat).flatten(1)   # (B, 1280)
        logits = self.cls_head(feat)          # (B, num_classes)
        bbox   = self.bbox_head(feat)         # (B, 4)
        score  = self.score_head(feat)        # (B, 1)
        return logits, bbox, score


# ─────────────────────────────────────────────
# 4. 학습
# ─────────────────────────────────────────────
def train():
    records = collect_samples()
    if len(records) == 0:
        raise RuntimeError("샘플을 찾을 수 없습니다. SAMPLE_ROOT를 확인하세요.")

    # 80/20 분할
    np.random.seed(42)
    idx   = np.random.permutation(len(records))
    split = int(len(idx) * 0.8)
    tr_recs = [records[i] for i in idx[:split]]
    va_recs = [records[i] for i in idx[split:]]

    tr_ds = MotorDataset(tr_recs, make_transforms(True))
    va_ds = MotorDataset(va_recs, make_transforms(False))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MotorLocalizer(NUM_CLASSES).to(DEVICE)
    ce_loss   = nn.CrossEntropyLoss()
    bbox_loss = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_pt_path = OUT_DIR / "motor_localizer_best.pt"

    for epoch in range(1, EPOCHS + 1):
        # ── 학습
        model.train()
        tr_loss, tr_correct = 0.0, 0
        for imgs, labels, bboxes, has_box, score_targets in tr_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            bboxes, has_box = bboxes.to(DEVICE), has_box.to(DEVICE)
            score_targets = score_targets.to(DEVICE)

            optimizer.zero_grad()
            logits, pred_bbox, pred_score = model(imgs)

            loss_cls  = ce_loss(logits, labels)
            # bbox 손실은 has_box==1인 샘플에만 적용
            loss_bbox = bbox_loss(pred_bbox, bboxes).mean(dim=1)  # (B,)
            loss_bbox = (loss_bbox * has_box).sum() / (has_box.sum() + 1e-6)
            loss_score = nn.functional.mse_loss(pred_score, score_targets)

            loss = loss_cls + 2.0 * loss_bbox + loss_score
            loss.backward()
            optimizer.step()

            tr_loss    += loss.item() * len(imgs)
            tr_correct += (logits.argmax(1) == labels).sum().item()

        # ── 검증
        model.eval()
        va_correct, va_total = 0, 0
        with torch.no_grad():
            for imgs, labels, bboxes, has_box, score_targets in va_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits, _, _ = model(imgs)
                va_correct += (logits.argmax(1) == labels).sum().item()
                va_total   += len(imgs)

        tr_acc = tr_correct / len(tr_ds)
        va_acc = va_correct / va_total
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"loss={tr_loss/len(tr_ds):.4f} | "
              f"tr_acc={tr_acc:.3f} | va_acc={va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_pt_path)
            print(f"  >> Best saved (val_acc={va_acc:.3f})")

    print(f"\n최고 검증 정확도: {best_val_acc:.3f}")
    return model, best_pt_path


# ─────────────────────────────────────────────
# 5. ONNX 내보내기
# ─────────────────────────────────────────────
def export_onnx(model, pt_path: Path):
    onnx_path = OUT_DIR / "motor_localizer.onnx"
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits", "bbox", "anomaly_score"],
        dynamic_axes={
            "image":        {0: "batch"},
            "logits":       {0: "batch"},
            "bbox":         {0: "batch"},
            "anomaly_score": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"ONNX 저장: {onnx_path}")

    # 메타데이터 저장 (클래스 이름, 이미지 크기)
    import json
    meta = {
        "class_names": CLASS_NAMES,
        "img_size": IMG_SIZE,
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
        "score_targets": _SCORE_TARGETS,
    }
    meta_path = OUT_DIR / "motor_localizer_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"메타데이터 저장: {meta_path}")
    return onnx_path


# ─────────────────────────────────────────────
# 6. 빠른 검증
# ─────────────────────────────────────────────
def quick_validate_onnx(onnx_path: Path):
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(onnx_path))
    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    logits, bbox, anomaly_score = sess.run(None, {"image": dummy})
    pred_class = CLASS_NAMES[int(np.argmax(logits[0]))]
    score_val  = float(anomaly_score[0][0]) * 100
    print(f"ONNX 검증 OK | 예측 클래스: {pred_class} | bbox: {bbox[0].round(3)} | anomaly_score: {score_val:.1f}")


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("모터-감속기 이상 위치 탐지 모델 학습")
    print("=" * 55)
    model, pt_path = train()

    # 최적 가중치 로드 후 내보내기
    model.load_state_dict(torch.load(pt_path, map_location=DEVICE))
    onnx_path = export_onnx(model, pt_path)
    quick_validate_onnx(onnx_path)

    print("\n완료! backend/weights/motor_localizer.onnx 를 백엔드가 자동으로 로드합니다.")
