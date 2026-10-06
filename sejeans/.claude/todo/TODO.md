# TODO

> 박세진 담당 태스크 위주 (배터리 파트) | 상태: [ ] 미착수 / [x] 완료 / [-] 진행중

---

## Phase 0. 환경 구축

- [ ] Colab 환경에서 ultralytics 설치 및 YOLOv11-Seg 로드 확인
- [ ] 배터리 NPY 파일 1개 로드 테스트 (shape, dtype 확인)
- [ ] 샘플 PNG 1장 로드 + matplotlib으로 시각화 확인
- [ ] 폴더 구조 정의 및 README 작성

```
project/
  data/
    motor/   # PNG
    battery/ # NPY
  models/
    base/
    lightweight/
  notebooks/
    01_eda.ipynb
    02_motor_train.ipynb
    03_battery_train.ipynb
    04_quantization.ipynb
    05_kd.ipynb
  dashboard/
    app.py
```

---

## Phase 1. 배터리 데이터 전처리

- [ ] 전체 NPY 파일 목록 파악 및 클래스별 샘플 수 확인
- [ ] NPY 행렬 shape 통일 (시간축 길이 확인)
- [ ] 정규화 방식 결정 (Min-Max or Standard)
- [ ] 클래스 라벨 매핑: 0=NORMAL, 1=셀전압 결함, 2=셀편차 결함
- [ ] 클래스 불균형 비율 계산 → 클래스 가중치 계산
- [ ] train/val/test 분할 (70/15/15) 및 저장
- [ ] 클래스 분포 bar chart 시각화 저장

---

## Phase 2. 배터리 베이스 모델

- [ ] 모델 구조 결정 (CNN1D / CNN2D / MLP / Transformer 중 선택)
  - NPY 행렬 → 이미지화 후 CNN2D 우선 시도
- [ ] 학습 코드 작성 (train loop + val loop)
- [ ] 학습 실행 (epoch=50 이상, early stopping)
- [ ] 성능 평가: Accuracy, F1(macro), Confusion Matrix, ROC-AUC
- [ ] 베이스 모델 가중치 저장 (`models/base/battery_base.pt`)
- [ ] 오분류 샘플 분석 (어떤 결함을 놓치는지)

---

## Phase 3. 배터리 모델 경량화

### Quantization
- [ ] ONNX 변환: `torch.onnx.export()`
- [ ] Post-Training Quantization 적용 (onnxruntime)
- [ ] 경량화 전후 비교: Accuracy, 모델 크기(MB), 추론 시간(ms)

### Knowledge Distillation
- [ ] Teacher 모델: 베이스 모델 (큰 모델)
- [ ] Student 모델: 더 작은 네트워크 정의
- [ ] KD 학습 코드 작성 (soft label loss + hard label loss)
- [ ] 학습 후 성능 평가

### 비교 실험
- [ ] 베이스 vs Quantization vs KD vs KD+Quantization 비교표 작성

---

## Phase 4. 대시보드 연동 (박솔과 협업)

- [ ] 배터리 추론 함수 모듈화 (`predict_battery(npy_path) → result`)
- [ ] Streamlit에서 배터리 결과 표시 연동 확인

---

## Phase 5. 발표 준비

- [ ] 실험 결과 정리 (표 + 그래프)
- [ ] 발표 PPT 배터리 파트 슬라이드 작성
- [ ] 최종 보고서 배터리 파트 작성

---

## 당장 할 일 (오늘/이번 주)

1. [ ] Colab 환경 세팅 + 패키지 설치
2. [ ] 배터리 NPY 파일 탐색적 분석 (EDA) — shape, 분포, 결측치 확인
3. [ ] 모터 PNG 샘플 확인 (박솔과 데이터 구조 공유)
