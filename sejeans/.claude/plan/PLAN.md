# 구현 계획서

> PRD 기반 단계별 개발 계획 | 박세진 담당: 배터리 파트

---

## Phase 0. 환경 구축 (1주)

- [ ] Python 환경 세팅 (Colab Pro 또는 로컬 venv)
- [ ] 필수 패키지 설치: ultralytics, torch, onnx, onnxruntime, shap, gradcam
- [ ] 데이터 경로 정리 및 폴더 구조 확정
- [ ] 샘플 PNG 1장, NPY 1개 로드 확인

---

## Phase 1. 데이터 전처리 (1~2주)

### 1-1. 모터-감속기 (박솔)
- PNG 이미지 크기 통일 (224×224 or 640×640)
- train/val/test 분할 (7:1.5:1.5)
- YOLO 포맷 라벨 변환 (클래스: 0~4)

### 1-2. 배터리 (박세진)
- NPY 행렬 로드 및 정규화
- 결함 라벨 매핑: 0=NORMAL, 1=셀전압 결함, 2=셀편차 결함
- train/val/test 분할 (7:1.5:1.5)
- 클래스 불균형 분석 → 가중치 설정

**산출물**: 전처리된 데이터셋 + 클래스 분포 시각화

---

## Phase 2. 베이스 모델 학습 (2~3주)

### 2-1. 모터-감속기 베이스 (박솔)
- YOLOv11-Seg (large 또는 medium) 파인튜닝
- 학습: epochs=50, img=640, batch=16
- 평가: mIoU, F1, Confusion Matrix

### 2-2. 배터리 베이스 (박세진)
- 입력: NPY 행렬 → CNN 또는 MLP 분류기
- 또는 행렬을 이미지화 → YOLOv11-Seg 적용 검토
- 평가: Accuracy, F1-score per class, ROC-AUC

**산출물**: 베이스 모델 가중치 (.pt) + 성능 리포트

---

## Phase 3. 모델 경량화 (2~3주)

### 3-1. Quantization (양자화)
- FP32 → INT8 변환 (Post-Training Quantization)
- ONNX 변환: `yolo export format=onnx`
- TensorRT 변환 검토 (Colab 환경 가능 시)
- 경량화 전후 성능 비교: mIoU, FPS, 모델 크기

### 3-2. Knowledge Distillation (지식 증류)
- Teacher: YOLOv11-Seg large (베이스 모델)
- Student: YOLOv11-Seg nano
- Soft label로 Student 학습
- 경량화 전후 성능 비교

### 3-3. 최종 모델 선정
- Quantization vs KD vs 둘 다 적용: 성능/속도/크기 3-way 비교표 작성
- 최적 조합 선정

**산출물**: 경량화 모델 (.onnx) + 비교 실험 결과 표

---

## Phase 4. 대시보드 구현 (1~2주, 박솔 주도 / 박세진 배터리 파트 연동)

- Streamlit 또는 Gradio 기반 웹 UI
- 기능:
  - 이미지/NPY 업로드 → 실시간 추론 결과 표시
  - SOH 게이지 차트 (정상/주의/경고)
  - 이상 점수 시계열 트렌드
  - Grad-CAM 히트맵 시각화
  - 자동 진단 리포트 생성

**산출물**: 실행 가능한 웹 대시보드

---

## Phase 5. 평가 및 발표 준비 (1주)

- 전체 파이프라인 통합 테스트
- 최종 성능 지표 정리 (베이스 vs 경량화 비교)
- 발표 PPT 작성 (채점 기준 반영: 문제정의, 완성도, 실현가능성)
- 최종 보고서 작성

---

## 일정 요약

| Phase | 내용 | 담당 | 기간 |
|-------|------|------|------|
| 0 | 환경 구축 | 전체 | 1주 |
| 1 | 데이터 전처리 | 분담 | 1~2주 |
| 2 | 베이스 모델 학습 | 분담 | 2~3주 |
| 3 | 경량화 | 분담 | 2~3주 |
| 4 | 대시보드 | 박솔 주도 | 1~2주 |
| 5 | 평가 및 발표 | 전체 | 1주 |

---

## 기술 스택 확정

```
언어: Python 3.10+
프레임워크: PyTorch, Ultralytics (YOLOv11)
경량화: ONNX, onnxruntime, TensorRT (선택)
시각화: Grad-CAM (pytorch-grad-cam), Plotly
대시보드: Streamlit
실험 관리: WandB 또는 TensorBoard
환경: Google Colab Pro
```
