> **[DEPRECATION NOTE — 2026-05-17]**
> 본 PRD는 `StratifiedGroupKFold(n_splits=3)` + (차종×센서) 9개 모델 × 4컬럼 grid search = **36회 학습**을 전제하였으나, 샘플 인덱스 실측 결과 **KONA-DEMAG/KONA-ECC20 슬롯에서 group 수 = 1** 로 확인되어 stratified group K-Fold가 ValueError로 실행 불가하다. 따라서 본 PRD의 핵심 가정(9개 모델 분리, 36-grid, splits.json 공유, NFR 정량 목표)은 **본 데이터(600GB) 단계 복귀 시 재활용**한다.
>
> 현 파일럿(샘플 81 CSV) 단계에서는 다음 PRD가 우선한다:
> `prd_260517_모터감속기_파일럿_feasibility.md`
> 파일럿에서는 단일 hold-out split · 통합 모델 1개 · 동작 검증 중심으로 단순화한다. 신호 채널·chunk=2000·MiniROCKET+Ridge 분류기 등 도메인/알고리즘 설계는 그대로 승계한다.

---

# PRD: 모터 감속기 시계열 데이터 기반 MiniROCKET 분류 파이프라인

> 7조 | 박세진(C321027) · 박솔 | 홍익대 시스템분석/설계

---

## 1. 이전 단계의 문제점

이전 PRD(`prd_260501_모터감속기_신호_특징벡터_변환_파이프라인.md`, v2.0)는 PNG 스펙트로그램 → 76차원 수작업 특징 → SVM/RF 분류 파이프라인을 정의했다. 해당 접근은 도메인 지식을 강하게 반영하는 장점이 있으나, 다음 한계가 있어 **대안 모델**을 별도로 설계할 필요가 있다.

- **PNG 변환 손실**: 원본 시계열을 STFT → PNG로 변환하는 과정에서 양자화·컬러맵 매핑·해상도 제한으로 미세 진폭 정보가 손실된다.
- **수작업 특징의 표현력 한계**: 76차원 벡터는 사전에 정의된 통계량(band energy, peak, ROI 등)만 포착하며, 학습 단계에서 새로운 시계열 패턴을 자동 발견할 수 없다.
- **튜닝 비용**: `prominence`, `distance`, fault_zone 등 파라미터가 데이터 분포 변동에 민감하다.
- **다른 모델 계열과의 비교 부재**: 시계열 자체를 직접 학습하는 최신 경량 분류기와의 성능 비교 베이스라인이 없다.
- **PNG 미생성 환경 대응 불가**: 원본 CSV만 보유한 차량/시점에서는 76차원 파이프라인이 동작하지 않는다.

> 본 PRD는 v2.0 파이프라인을 대체하지 않으며, **동일 split·동일 metric으로 비교 평가되는 대안 모델**로 병행 운용된다.

---

## 2. 해결 방법

### 2-1. 기본 전략

원본 시계열 CSV를 입력으로 받아 **MiniROCKET 변환 + 선형 분류기**로 5-class(NORMAL/ECC10/ECC20/DEMAG/REDUC) 분류를 수행한다. 모델은 **(차종 × 센서)별로 독립 학습**하여 9개를 운용하며, 각 모델의 입력 채널은 4개 후보 컬럼(`peak_freq_bin`, `band_start`, `band_end`, `rms`) 중 **검증 metric이 가장 우수한 단일 컬럼(univariate)을 grid search로 선택**한다.

핵심 설계 결정 요약:

| 결정 항목 | 선택 | 근거 |
|---|---|---|
| 모델 분리 단위 | (차종 × 센서)별 9개 모델 | 차종별 회전 특성·센서별 신호 특성 차이가 커서 단일 모델로는 일반화 한계 |
| 입력 채널 방식 | univariate (4개 컬럼 중 best 1개) | 컬럼별 노이즈·정보량이 다르며 grid search로 (차종 × 센서)별 최적 선택 |
| 시퀀스 길이 | 원본 10,000 step 그대로 사용 | 다운샘플링·윈도잉으로 인한 정보 손실 회피, MiniROCKET은 긴 시퀀스에 강건 |
| 분류기 | RidgeClassifierCV (대안 LogisticRegression) | MiniROCKET 공식 권장 조합, 정규화 강도 자동 선택 |
| 결정론성 | random_state 전 모듈 고정 | 동일 시드 → 동일 결과 재현성 100% 보장 |

---

### 2-2. 데이터 명세

| 항목 | 값 |
|---|---|
| 경로 | `샘플데이터\03.합성데이터\1.모터_감속기_시계열\{차종}\{고장클래스}\{날짜}\{센서}\*.csv` |
| 차종 | IONIQ, KONA, NIRO (3종) |
| 고장 클래스 | NORMAL, ECC10, ECC20, DEMAG, REDUC (5종) |
| 센서 | Current_U, Vib_Motor, Vib_TM (3종) |
| CSV 1개 단위 | 1 샘플 = 10,000 time step × 12 컬럼 |
| 메타 컬럼 (8) | group_key, vehicle, fault_class, date, sensor, timestamp, zsplit, time_idx |
| 시계열 수치 컬럼 (4) | peak_freq_bin, band_start, band_end, rms |

> CSV 내 시계열 수치 컬럼 4개는 각각 STFT 후 frame별로 산출된 파생 신호이며, MiniROCKET 입력 후보로 동등하게 취급한다.

---

### 2-3. 모델 분리 구조 — 9개 (차종 × 센서) 모델

```
                       ┌────────────────────────────┐
[CSV 입력]   ──────►   │  메타 라우터               │   ──────►  [해당 모델]
(vehicle, sensor)      │  (vehicle, sensor) → key   │
                       └────────────────────────────┘
                                  │
   ┌──────────────────────────────┼──────────────────────────────┐
   ▼                              ▼                              ▼
IONIQ-Current_U            KONA-Current_U                 NIRO-Current_U
IONIQ-Vib_Motor            KONA-Vib_Motor                 NIRO-Vib_Motor
IONIQ-Vib_TM               KONA-Vib_TM                    NIRO-Vib_TM
   (5-class 분류)             (5-class 분류)                 (5-class 분류)
```

- 총 9개 분류기. 각 분류기는 NORMAL/ECC10/ECC20/DEMAG/REDUC 5-class.
- 추론 시 입력 메타데이터(vehicle, sensor)를 키로 `model_registry.json`을 조회해 해당 모델 1개만 호출한다.
- 단일 (차종 × 센서) 모델은 다른 (차종 × 센서) 데이터로 학습·평가되지 않는다.

---

### 2-4. 입력 채널 선택 — 4컬럼 Grid Search 전략

각 (차종 × 센서) 조합에 대해 4개 시계열 컬럼을 독립 univariate 입력으로 학습·평가하고, 검증 metric(매크로 F1) 기준으로 최적 컬럼 1개를 선택한다.

```
(차종 × 센서) 9개 × (peak_freq_bin, band_start, band_end, rms) 4개
= 36개 (MiniROCKET + RidgeClassifierCV) 학습

→ 각 (차종 × 센서)별로 val macro-F1 최댓값 컬럼 선택

→ 최종 9개 모델 + config.json (vehicle, sensor) → best_column 매핑 저장
```

#### Grid Search 결과 매핑 예시 (config.json)

```json
{
  "IONIQ-Current_U":  { "best_column": "rms",            "val_macro_f1": 0.91 },
  "IONIQ-Vib_Motor":  { "best_column": "peak_freq_bin",  "val_macro_f1": 0.88 },
  "IONIQ-Vib_TM":     { "best_column": "band_end",       "val_macro_f1": 0.90 },
  "KONA-Current_U":   { "best_column": "rms",            "val_macro_f1": 0.93 },
  "...": "..."
}
```

추론 시 라우터는 `(vehicle, sensor)` 키로 best_column을 조회한 뒤 해당 컬럼만 슬라이싱하여 MiniROCKET에 전달한다.

---

### 2-5. 전체 파이프라인

```
CSV 로드 (10,000 행 × 12 컬럼)
        ↓
[1단계] 메타 컬럼 분리
        vehicle, sensor, fault_class, group_key, zsplit 추출
        ↓
[2단계] 시계열 컬럼 선택
        훈련 시: 4개 컬럼 모두 후보 (grid search)
        추론 시: config.json의 best_column 1개
        ↓
[3단계] Shape 검증 및 정규화
        - 길이 == 10,000 검증, 부족 시 zero-pad, 초과 시 절단
        - NaN/Inf 검출 → 선형 보간 또는 0 대체 후 로그 기록
        ↓
[4단계] sktime panel format 변환
        shape: (n_samples, 1, 10000)
        ↓
[5단계] MiniRocket.fit_transform
        → 약 10,000개의 random convolutional features
        ↓
[6단계] RidgeClassifierCV.fit (class_weight='balanced')
        → 5-class 예측
        ↓
[7단계] (차종, 센서) 모델 라우터에 등록
        model_registry[(vehicle, sensor)] = (transformer, classifier, best_column)
        ↓
[추론] (vehicle, sensor) → 라우터 → 모델 1개 호출 → 예측 라벨 + softmax 점수
```

---

### 2-6. 데이터 분할 전략 — 누수 방지

동일 `zsplit` 또는 동일 `group_key` 내 샘플이 train/val/test에 동시 포함되면 데이터 누수가 발생한다. 다음 규칙을 강제한다.

| 단계 | 규칙 |
|---|---|
| 분할 단위 | `group_key`(또는 `(vehicle, sensor, date, zsplit)` 조합) 단위 |
| 분할 비율 | train 70% / val 15% / test 15% |
| 클래스 균형 | `StratifiedGroupKFold` 사용 — fault_class별 비율 유지 + group 누수 차단 |
| 시드 | random_state=42 고정, split 결과를 `splits.json`으로 저장 |
| 비교 기준 | v2.0 PNG 파이프라인과 **동일 split 파일**을 공유하여 동일 조건 벤치마크 |

---

## 3. 기대 효과와 리스크

### 기대 효과

- **PNG 변환 손실 회피**: 원본 시계열에서 직접 특징을 추출하여 신호 정보 보존.
- **자동 특징 학습**: MiniROCKET이 ~10,000개 random convolutional kernel로 다양한 시계열 패턴을 자동 추출 → 수작업 튜닝 부담 감소.
- **추론 속도**: MiniROCKET은 GPU 없이 CPU에서 단일 샘플 100ms 이내 추론 가능 → 엣지 디바이스 적합.
- **비교 베이스라인 확보**: v2.0(76차원 hand-crafted) vs 본 PRD(MiniROCKET) 동일 split 평가표 작성으로 모델 선택 근거 마련.
- **모델 분리에 따른 정밀도**: (차종 × 센서)별 9개 모델 분리로 차종 간 회전 특성·센서 응답 차이를 모델 구조 자체로 흡수.

### 리스크 및 대응 방안

| 리스크 | 영향 | 대응 방안 |
|---|---|---|
| CSV 길이가 10,000이 아닌 케이스 | shape 불일치로 학습/추론 실패 | 로드 직후 길이 검증, 부족 시 0-pad, 초과 시 절단 후 경고 로그 |
| NaN/Inf 결측값 | MiniROCKET fit 실패 | 선형 보간 1차 시도 → 실패 시 0 대체, 결측 비율 로그 기록 |
| 4컬럼 best 선택이 (차종 × 센서)별로 달라 운영 복잡도 증가 | 추론 라우팅 오류 가능성 | `config.json`에 매핑 명시, 라우터에서 키 미존재 시 명시적 에러 |
| 클래스 불균형 (NORMAL >> 결함) | 결함 클래스 Recall 저하 | `RidgeClassifierCV(class_weight='balanced')`, 필요 시 stratified sampling 추가 |
| 동일 `zsplit`/`group_key` 누수 | val/test 성능 과대 평가 | `StratifiedGroupKFold` 강제, splits.json 버전 관리 |
| MiniROCKET 변환 메모리 폭증 | OOM | 배치 단위(`batch_size=512`) 변환, 변환 결과는 float32로 저장 |
| 36회 학습으로 인한 시간 비용 | 학습 파이프라인 지연 | sktime 멀티스레딩(`n_jobs=-1`), 컬럼별 학습 병렬화 |
| 9개 모델 디스크 크기 초과 | 엣지 탑재 불가 | 각 모델 직렬화 시 joblib + zlib 압축, 총 ≤ 200MB 검증 |
| v2.0 파이프라인과의 비교 평가 누락 | 모델 선택 근거 약화 | 동일 split·동일 metric으로 벤치마크 표 필수 작성 |
| 라이브러리 버전 차이 (sktime API 변경) | 재현 불가 | `requirements.txt`에 sktime 버전 고정, MiniRocket import 경로 명시 |
| 시계열 컬럼이 사실상 동일 정보 | grid search가 무의미 | 컬럼 간 상관계수 사전 분석, 0.99 이상이면 단일 컬럼 고정 |

---

## 4. 목표 및 성공 지표

| 지표 | 목표값 | 측정 방법 |
|---|---|---|
| 모델 수 | 정확히 9개 (3차종 × 3센서) | `len(model_registry) == 9` |
| 5-class Accuracy (각 모델) | ≥ 0.90 | 테스트 세트 분류 정확도, 9개 모델 모두 충족 |
| Macro F1 (각 모델) | ≥ 0.85 | sklearn `f1_score(average='macro')` |
| 결함 클래스 Recall (안전 직결) | ≥ 0.85 | ECC10/ECC20/DEMAG/REDUC 각 클래스의 Recall |
| 단일 샘플 추론 시간 (CPU) | ≤ 100ms | 변환 + 분류기 예측 합산, 100회 평균 |
| 9개 모델 총 디스크 크기 | ≤ 200MB | joblib 압축 직렬화 후 합산 |
| 학습 재현성 | 100% | 동일 시드 → 동일 weights·동일 metric |
| Grid Search 커버리지 | 100% | 9 × 4 = 36회 학습 모두 완료 후 best_column 선정 |
| Shape 검증 커버리지 | 100% | 길이 ≠ 10,000인 모든 케이스 패딩/절단 처리, 오류 0건 |
| NaN 처리 커버리지 | 100% | NaN 포함 샘플 자동 처리, 실패 0건 |
| v2.0 대비 벤치마크 | 동일 split·동일 metric 표 | `benchmark.md`에 두 모델 결과 병기 |

---

## 5. 아키텍처

```
[입력]
원본 시계열 CSV (10,000 행 × 12 컬럼)
메타: vehicle, sensor, fault_class, group_key, zsplit, timestamp, time_idx
신호: peak_freq_bin, band_start, band_end, rms

        ↓ CSVLoader
          - pandas.read_csv() → DataFrame
          - 메타 컬럼 / 신호 컬럼 분리
          - 길이 == 10,000 검증, pad/truncate

        ↓ NanHandler
          - NaN/Inf 검출 → 선형 보간 → 실패 시 0 대체
          - 결측 비율 로그

        ↓ ColumnRouter                            ← 핵심 컴포넌트
          - 훈련 모드: 4개 컬럼 모두 후보
          - 추론 모드: config.json의 best_column 1개 선택
          - 출력 shape: (1, 10000) per sample

        ↓ PanelFormatter
          - sktime panel format 변환
          - shape: (n_samples, 1, 10000)

        ↓ MiniRocketTransformer                   ← sktime
          - num_kernels=10,000 (기본값)
          - random_state=42 고정
          - fit_transform → (n_samples, ~10000) feature matrix

        ↓ RidgeClassifierCV                       ← sklearn
          - alphas=np.logspace(-3, 3, 10)
          - class_weight='balanced'
          - 5-class 예측 (NORMAL, ECC10, ECC20, DEMAG, REDUC)

        ↓ ModelRegistry
          - key: (vehicle, sensor) → (transformer, classifier, best_column, metrics)
          - 9개 모델 joblib 직렬화

        ↓ InferenceRouter
          - (vehicle, sensor) 메타 → ModelRegistry 조회
          - 단일 모델 호출 → 라벨 + decision_function 점수

[출력]
예측 라벨 (5-class) + 신뢰도 점수 + 모델 메타 (best_column, val_f1)
```

---

## 6. 기술 요구사항

### 기능 요구사항

| ID | 요구사항 | 우선순위 |
|---|---|---|
| FR-01 | CSV를 pandas로 로드하고 메타/신호 컬럼을 분리한다 | 필수 |
| FR-02 | 시계열 길이가 10,000이 아닐 시 zero-pad 또는 절단하여 (10000,)을 보장한다 | 필수 |
| FR-03 | NaN/Inf 값 검출 시 선형 보간 → 실패 시 0 대체하고 결측 비율을 로그에 남긴다 | 필수 |
| FR-04 | 훈련 단계에서 (차종 × 센서) 9 × 4 = 36회 학습을 수행한다 | 필수 |
| FR-05 | 각 (차종 × 센서)별 val macro-F1이 최대인 컬럼을 best_column으로 선택한다 | 필수 |
| FR-06 | 최종 9개 모델(transformer + classifier)을 joblib로 직렬화한다 | 필수 |
| FR-07 | (vehicle, sensor) → (best_column, val_f1) 매핑을 config.json으로 저장한다 | 필수 |
| FR-08 | sktime panel format (n, 1, 10000)으로 변환하여 MiniRocket에 입력한다 | 필수 |
| FR-09 | RidgeClassifierCV에 class_weight='balanced'를 적용한다 | 필수 |
| FR-10 | 모든 모듈에서 random_state=42를 고정하여 재현성을 보장한다 | 필수 |
| FR-11 | 추론 시 (vehicle, sensor) 메타로 단일 모델만 호출한다 | 필수 |
| FR-12 | `StratifiedGroupKFold`로 group_key 누수를 차단하며 split 결과를 splits.json으로 저장한다 | 필수 |
| FR-13 | model_registry에 키 미존재 시 명시적 KeyError를 발생시킨다 | 필수 |
| FR-14 | v2.0 PNG 파이프라인과 동일 split·동일 metric으로 benchmark.md 표를 생성한다 | 권장 |
| FR-15 | 배치 변환 시 `batch_size=512` 단위로 메모리 사용량을 제한한다 | 권장 |

### 비기능 요구사항

| ID | 요구사항 | 기준값 |
|---|---|---|
| NFR-01 | 단일 샘플 추론 시간 (변환 + 예측, CPU) | ≤ 100ms |
| NFR-02 | 9개 모델 총 디스크 크기 (joblib 압축 후) | ≤ 200MB |
| NFR-03 | 동일 시드 → 동일 결과 재현율 | 100% |
| NFR-04 | 결함 클래스 Recall (안전 직결) | ≥ 0.85 |
| NFR-05 | 5-class Accuracy / Macro F1 | ≥ 0.90 / ≥ 0.85 |
| NFR-06 | 36회 학습 총 소요 시간 (Colab T4 기준) | ≤ 6시간 |
| NFR-07 | 모듈 단위테스트 가능성 | CSVLoader / NanHandler / ColumnRouter / MiniRocketWrapper / Registry 각 독립 |

### 기술 스택

| 항목 | 라이브러리 |
|---|---|
| 데이터 로드 | pandas, numpy |
| 시계열 변환 | sktime (`sktime.transformations.panel.rocket.MiniRocket`) |
| 분류기 | scikit-learn (`RidgeClassifierCV`, 대안 `LogisticRegression`) |
| 검증 분할 | scikit-learn (`StratifiedGroupKFold`) |
| 직렬화 | joblib (zlib 압축) |
| 로깅 | logging (표준 라이브러리) |
| 개발 환경 | Jupyter Notebook, Python 3.10+ |
| 학습 환경 | Google Colab (CPU 또는 T4 GPU, MiniROCKET은 CPU만으로 충분) |

---

## 변경 이력

| 버전 | 날짜 | 내용 |
|---|---|---|
| v1.0 | 2026-05-16 | 최초 작성 — 모터감속기 시계열 CSV 기반 MiniROCKET 분류 파이프라인 신규 추가. (차종 × 센서) 9개 모델 분리, 4개 시계열 컬럼 grid search 전략, sktime MiniRocket + RidgeClassifierCV 조합, group_key 누수 방지 split, v2.0 PNG 파이프라인과의 벤치마크 비교 명시 |
