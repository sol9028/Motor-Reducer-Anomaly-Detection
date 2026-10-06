# 1D CNN을 활용한 배터리 고장 예측

## 배경
AI Hub "자율주행 고장진단 데이터"의 `2.배터리` npy 시계열 데이터를 **1D CNN 분류 모델**로 학습하여 배터리 상태(정상/주의/결함)를 예측합니다. CNN은 LSTM 대비 **학습 속도가 빠르고 모델이 가볍습니다.**

### 데이터 요약
| 항목 | 내용 |
|------|------|
| 원천데이터 | [.npy](file:///c:/Users/kingm/OneDrive/%EB%AC%B8%EC%84%9C/%EC%83%88%20%ED%8F%B4%EB%8D%94/OneDrive/%EB%B0%94%ED%83%95%20%ED%99%94%EB%A9%B4/%ED%95%99%EA%B5%90/4%ED%95%99%EB%85%84%201%ED%95%99%EA%B8%B0/%EC%8B%9C%EC%8A%A4%ED%85%9C%EB%B6%84%EC%84%9D/%EC%83%98%ED%94%8C%EB%8D%B0%EC%9D%B4%ED%84%B0/Sample/01.%EC%9B%90%EC%B2%9C%EB%8D%B0%EC%9D%B4%ED%84%B0/2.%EB%B0%B0%ED%84%B0%EB%A6%AC/Vlt_98/NPY/NIRO/NORMAL/0708/volt_BC/22-07-08_062035_01_004_volt_BC.npy) 배열 (셀 전압/편차 시계열) |
| 라벨링데이터 | [.json](file:///c:/Users/kingm/OneDrive/%EB%AC%B8%EC%84%9C/%EC%83%88%20%ED%8F%B4%EB%8D%94/OneDrive/%EB%B0%94%ED%83%95%20%ED%99%94%EB%A9%B4/%ED%95%99%EA%B5%90/4%ED%95%99%EB%85%84%201%ED%95%99%EA%B8%B0/%EC%8B%9C%EC%8A%A4%ED%85%9C%EB%B6%84%EC%84%9D/%EC%83%98%ED%94%8C%EB%8D%B0%EC%9D%B4%ED%84%B0/Sample/02.%EB%9D%BC%EB%B2%A8%EB%A7%81%EB%8D%B0%EC%9D%B4%ED%84%B0/2.%EB%B0%B0%ED%84%B0%EB%A6%AC/Vlt_96/JSON/IONIQ/CAUTION/0830/volt_BC/22-08-30_090821_01_005_volt_BC.json) (카테고리, 메타데이터, polygon) |
| 클래스 | `NORMAL`(0), `CAUTION`(1), `DEFECT`(2) |
| 데이터 유형 | Dev_96, Dev_98(셀편차), Vlt_96, Vlt_98(셀전압) |
| 차종 | KONA, NIRO, IONIQ |

---

## Proposed Changes — 전체 파이프라인

### Step 1: 환경 구축
```bash
pip install torch torchvision numpy scikit-learn matplotlib seaborn
```

### Step 2: 데이터 로드 + 전처리 + 1D CNN 학습 + 평가

#### [NEW] [battery_cnn.py](file:///c:/Users/kingm/OneDrive/문서/새%20폴더/OneDrive/바탕%20화면/학교/4학년%201학기/시스템분석/샘플데이터/battery_cnn.py)

하나의 파이썬 스크립트에 전체 파이프라인을 구현:

1. **데이터 로드**: 폴더 구조에서 npy 파일 + 클래스(NORMAL/CAUTION/DEFECT) 자동 수집
2. **전처리**: 정규화, 패딩/트렁케이션으로 길이 통일, Train/Val 분할
3. **1D CNN 모델**: Conv1D → BatchNorm → ReLU → MaxPool 블록 × 3 → FC → 3클래스 분류
4. **학습**: CrossEntropyLoss + Adam optimizer
5. **평가**: Accuracy, Confusion Matrix, Classification Report 출력

---

## Verification Plan
- 학습 Loss/Accuracy 곡선 확인
- Confusion Matrix로 클래스별 정확도 확인
- 전체 Accuracy 및 F1-score 확인
