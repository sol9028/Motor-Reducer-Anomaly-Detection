> **[DEPRECATION NOTE — 2026-05-17]**
> 본 PRD는 **본 데이터(600GB) 단계 복귀 시 재활용 예정**이며, 현 파일럿(샘플 81 CSV) 단계에서는 다음 PRD가 우선한다:
> `prd_260517_모터감속기_파일럿_feasibility.md`
> 본 문서의 정량 성공 지표(mIoU ≥ 0.95, FPS ≥ 30, per-class Acc ≥ 90% 등)와 YOLOv11-Seg/Quantization/KD 아키텍처는 **본 데이터 단계에서 검증**한다. 파일럿 단계에서는 파이프라인 동작 검증·엣지 배포 형태 산출만 목표로 한다.

---

# PRD: 전기차 모터-감속기 및 배터리 이상 탐지 기술 경량화

---

## 1. 문제 정의

자율주행 전기차에서 모터-감속기와 배터리 팩은 탑승자 안전과 직결된다.
기존 클라우드 기반 진단 방식은 네트워크 지연, 통신 비용, 오프라인 불가 문제가 있고,
기존 딥러닝 진단 모델(DeepLabV3+, LSTM)은 정확도는 높지만 차량 내 실시간 추론에 필요한 연산량을 초과한다.

**핵심 문제**: 차량 내 엣지 디바이스에서 실시간으로 구동 가능한 경량 이상 탐지 모델이 없다.

---

## 2. 목표 및 성공 지표

| 목표 | 측정 지표 | 목표값 |
|------|-----------|--------|
| 이상 탐지 정확도 유지 | mIoU / F1-score | ≥ 0.95 (경량화 후) |
| 추론 속도 | FPS (Colab 환경) | ≥ 30 FPS |
| 모델 크기 축소 | 파라미터 수 / 용량 | 베이스 모델 대비 ≥ 50% 감소 |
| 결함 유형 분류 | 다중 클래스 분류 정확도 | ≥ 90% per class |

---

## 3. 범위 (Scope)

### In Scope
- 모터-감속기 이상 탐지: ECC10, ECC20, DEMAG, REDUC, NORMAL 5-class 분류
- 배터리 이상 탐지: 셀전압 결함, 셀편차 결함, 정상 3-class 분류
- YOLOv11-Seg 기반 베이스 모델 학습
- Quantization + Knowledge Distillation 경량화
- 실시간 모니터링 대시보드 (웹)

### Out of Scope
- 실제 차량 탑재 및 하드웨어 검증 (라즈베리파이 등)
- 실시간 스트리밍 데이터 수집 파이프라인
- 클라우드 연동 서비스

---

## 4. 데이터 요구사항

### 모터-감속기 데이터
- 형식: PNG (STFT 스펙트로그램)
- 차종: IONIQ, KONA, NIRO
- 채널: Current_U, Vib_Motor, Vib_TM
- 규모: 550,800건
- 클래스: NORMAL / ECC10 / ECC20 / DEMAG / REDUC

### 배터리 데이터
- 형식: NPY (수치 행렬 - 시간 × 셀번호)
- 차종: IONIQ, KONA, NIRO
- 채널: 96/98개 셀 전압, 전압 편차
- 규모: 220,320건
- 클래스: NORMAL / 셀전압 결함 / 셀편차 결함

---

## 5. 시스템 아키텍처

```
[원천 데이터]
  ├── STFT PNG (모터-감속기)  →  [YOLOv11-Seg 베이스 모델]
  └── NPY 행렬 (배터리)       →  [분류 모델 (CNN or MLP)]
                                        ↓
                              [경량화: Quantization + KD]
                                        ↓
                              [경량 추론 엔진 (ONNX/TensorRT)]
                                        ↓
                              [모니터링 대시보드]
                                ├── SOH 게이지
                                ├── 이상 점수 트렌드
                                └── Grad-CAM 리포트
```

---

## 6. 기술 요구사항

### 모델
- 베이스: YOLOv11-Seg (Ultralytics 2024.09)
- 경량화 기법 1: Quantization (FP32 → INT8)
- 경량화 기법 2: Knowledge Distillation (Teacher: YOLOv11-Seg large / Student: YOLOv11-Seg nano)
- 출력 형식: ONNX 또는 TensorRT 호환

### 학습 환경
- Google Colab (GPU: T4 또는 A100)
- Python 3.10+, PyTorch, Ultralytics, ONNX Runtime

### 대시보드
- Python 기반 웹 (Streamlit 또는 Gradio)
- Grad-CAM 시각화 포함

---

## 7. 업무 분장

| 담당자 | 담당 영역 |
|--------|-----------|
| 박세진 | 배터리 모델 구축, 배터리 경량화, 성능 평가 |
| 박솔 | 모터-감속기 모델 구축, 모터 경량화, 대시보드 구현 |

---

## 8. 제한사항 및 리스크

| 리스크 | 대응 |
|--------|------|
| 경량화 시 정확도 하락 | 목표 mIoU를 0.95로 현실화, 다양한 기법 비교 |
| False Positive 과다 | 임계값 튜닝, 알림 우선순위 시스템 |
| 하드웨어 검증 불가 | Colab에서 추론 시간 시뮬레이션으로 대체 |
| 데이터 불균형 | 클래스 가중치 조정, 오버샘플링 |

---

## 9. 참고 베이스라인

- DeepLabV3+: mIoU 99.6%, 연산량 과다
- YOLOv8: YOLOv11 대비 파라미터 많음
- LSTM 기반 모델: 시계열에 강하나 실시간 추론 느림
