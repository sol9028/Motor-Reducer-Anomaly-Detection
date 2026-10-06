# PRD: 모터 감속기 시계열 분류 파이프라인 — 파일럿(Feasibility) 단계

> 7조 | 박세진(C321027) · 박솔 | 홍익대 시스템분석/설계
> 작성일: 2026-05-17
> 단계: **Pilot / Feasibility Study** (샘플 81 CSV)
> 후속 단계: 본 데이터(600GB, 550,800 샘플) 검증 단계

---

## 0. 본 PRD의 위치 (Scope Statement)

본 PRD는 본 프로젝트의 **파일럿(feasibility study) 단계** 요구사항을 정의한다. 본 단계의 목적은 다음과 같다.

| 항목 | 파일럿 단계 (본 PRD) | 본 데이터 단계 (후속) |
|---|---|---|
| 데이터 규모 | 샘플 81 CSV (chunk 분할 후 ≈225개) | 550,800 샘플 (600GB) |
| 1차 목표 | **파이프라인이 end-to-end 동작하는가** | 모델 성능·엣지 NFR 정량 검증 |
| 모델 성능 | 측정·기록은 하되 정량 목표값 없음 | Accuracy ≥ 0.90, Macro-F1 ≥ 0.85 등 (이전 PRD 승계) |
| 모델 분리 | **센서별 모델 3개**(Current_U / Vib_Motor / Vib_TM, 차종 비분리) | (차종 × 센서) 9개 모델 분리 + 36-grid |
| Split 전략 | 단순 hold-out (GroupShuffleSplit 1회) | StratifiedGroupKFold(n_splits=3+) |
| v2.0 PNG 벤치마크 | 미수행 | 동일 splits.json 공유로 정량 비교 |
| 엣지 NFR | 측정 코드만 존재, 목표값 부재 | ≤ 100ms 추론, ≤ 200MB 디스크 정량 검증 |

> 본 PRD는 이전 3건의 PRD(`prd_0`, `prd_260501`, `prd_260516`)를 **대체하지 않고 일시적으로 우선**한다. 본 데이터 단계 진입 시 이전 PRD의 가정들이 복귀한다.

---

## 1. 이전 단계의 문제점 (Blocker 분석)

직전 PRD인 `prd_260516_모터감속기_시계열_miniROCKET_분류기.md` (v1.0)와 그 PLAN(`plan_260516_*.md`, Rev. 2026-05-16-02)은 다음 가정 위에서 작성되었다.

1. 9개 (차종 × 센서) 슬롯 각각이 `StratifiedGroupKFold(n_splits=3)`로 분할 가능
2. 각 슬롯에서 5 클래스 × 최소 3개 부모 CSV 확보로 fault_class stratification + group_key 누수 차단 동시 충족
3. 4개 컬럼 grid search × 9개 슬롯 = 36회 학습 후 slot별 best_column 선정

### 1-1. 실측 데이터로 본 가정 깨짐

`artifacts/sample_index.parquet`와 `artifacts/csv_row_sweep.csv`(2026-05-17 실측, 81 CSV) 기준 (vehicle × sensor) 슬롯별 부모 CSV(=group) 수는 다음과 같다.

| model_key | DEMAG | ECC10 | ECC20 | NORMAL | REDUC |
|---|---|---|---|---|---|
| IONIQ × 3센서 (각각) | 1 | 1 | 2 | 1 | 4 |
| KONA × 3센서 (각각) | 1 | 2 | 1 | 2 | 2 |
| NIRO × 3센서 (각각) | 1 | 3 | 3 | 1 | 2 |

- **즉시 실패**: `KONA-{DEMAG, ECC20}`처럼 group 수 = 1인 클래스는 `StratifiedGroupKFold(n_splits=3)`에서 `ValueError`로 학습 자체가 불가하다.
- **잠재 실패**: 다른 슬롯도 group 2~4개 수준으로, 3-fold stratification이 fold별 양성 0개를 생성할 위험이 크다.
- **근본 원인**: 샘플 데이터(81 CSV)는 group 수준 stratified CV에 **구조적으로 부적합**하다. 데이터 자체의 분포 한계이며 알고리즘 튜닝으로 해소되지 않는다.

### 1-2. 부수 한계

- **9개 모델 분리 운영의 신뢰도 부족**: 슬롯당 5 클래스 × 평균 2개 group으로는 (차종 × 센서)별 일반화 성능 비교의 통계적 유의성 확보가 불가능하다.
- **NFR 정량 검증 어려움**: 9개 모델 × 36-grid 학습 시 latency/디스크 측정값이 본 데이터 단계 수치를 대표하지 못한다.
- **v2.0 PNG 벤치마크 동시 진행 부담**: 샘플 단계에서 양쪽 파이프라인 모두 split 누수 위험에 노출되어 비교 결과 신뢰도가 낮다.

---

## 2. 해결 방법 (파일럿 단계 단순화 전략)

### 2-1. 기본 전략

> **"파일럿은 동작 검증, 성능은 본 데이터에서"** 원칙을 채택한다.

1. **센서별 모델 3개** 학습 — `Current_U` / `Vib_Motor` / `Vib_TM` 각각을 별도 5 클래스 분류기로 학습한다. **차종은 합쳐서 학습**(latent 변수 처리)하여 차종 정보는 메타 컬럼으로만 보존한다.
2. **단순 hold-out split** — `GroupShuffleSplit(n_splits=1, test_size=0.2)`로 train/val 1회 분할. stratification 포기. **센서별 모델 각각에 동일 방식 적용**.
3. **MiniROCKET + LogisticRegression(multinomial)** — 특징 추출기는 MiniROCKET 유지, 분류기는 `LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, class_weight='balanced', max_iter=1000, random_state=42)` 채택. 네이티브 `predict_proba` 지원으로 5클래스 확률을 직접 출력하여 대시보드의 AS·확률 차트 요구에 직결한다.
4. **chunk_size=2000 전처리 파이프 유지** — 가변 길이 CSV를 2,000행 단위 chunk로 분할, 같은 부모 CSV의 chunk는 동일 group_key 상속.
5. **엣지 배포 형태로 모델 저장** — joblib 압축. NFR 측정 코드는 갖추되 목표값은 본 데이터 단계로 이전.
6. **Streamlit 데모 대시보드 — 청크 순차 스트리밍 시뮬레이션, 센서별 Anomaly Score(AS) 시계열, 5클래스 확률, 이상 탐지 로그** — 단일 페이지 Streamlit 앱으로 (1) "데모 시작" 버튼 클릭 시 3개 센서(Current_U/Vib_Motor/Vib_TM)의 val 셋 청크가 sensor 기준 정렬 후 0.5~1초 간격으로 순차 스트리밍, (2) 청크 도착 시마다 센서별 모델 추론 → AS·확률·로그 갱신, (3) 위젯 4종(AS 게이지·AS 시계열 라인 차트·5클래스 확률 차트·이상 탐지 로그 테이블) 표시. 모델 3개는 `@st.cache_resource`로 1회 로드, 청크 스트리밍은 `st.empty()` placeholder + `time.sleep` 패턴 채택.

**센서 분리 / 차종 비분리 결정 근거**

- **센서 분리 필수**: 전류 신호(Current_U)와 진동 신호(Vib_Motor, Vib_TM)는 물리량·단위·스펙트럼 분포가 모두 달라 단일 모델로 합치면 학습이 깨진다. MiniROCKET 커널은 신호 도메인 가정을 하지 않지만, Ridge 단계에서 채널 도메인 혼재가 결정 경계를 왜곡한다.
- **차종 비분리 (파일럿 한정)**: 차종 × 클래스당 부모 CSV가 1~4개(§1-1)에 불과해 차종 분리 시 group_key 부족으로 `GroupShuffleSplit` hold-out도 불안정해진다. 센서별로 차종 3개를 합치면 group_key 분포가 (Current_U 예) DEMAG=3 / ECC10=6 / ECC20=6 / NORMAL=4 / REDUC=8 (min ≥ 3)이 되어 hold-out이 안정 동작한다.
- **본 데이터 단계 확장 경로**: model_key 차원만 `(sensor)` → `(vehicle, sensor)`로 확장하고 학습 루프에 outer loop를 추가하면 9개 모델 분리로 자연 전환된다(§7).

### 2-2. 데이터 명세 (파일럿)

| 항목 | 값 |
|---|---|
| 경로 | `샘플데이터\03.합성데이터\1.모터_감속기_시계열\{차종}\{고장클래스}\{날짜}\{센서}\*.csv` |
| 차종 | IONIQ, KONA, NIRO |
| 고장 클래스 | NORMAL, ECC10, ECC20, DEMAG, REDUC |
| 센서 | Current_U, Vib_Motor, Vib_TM |
| 부모 CSV 수 | 81 |
| chunk 분할 후 샘플 수 | 약 225 (chunk_size=2000 기준, 실측값은 `artifacts/csv_row_sweep.csv`) |
| 신호 컬럼 | peak_freq_bin, band_start, band_end, rms (4종, 후보) |
| 메타 컬럼 | vehicle, sensor, fault_class, group_key, zsplit, chunk_id 등 |

### 2-3. Split 전략 (파일럿 한정)

```
[단계 1] sample_index.parquet 로드
[단계 2] chunker로 2,000행 단위 분할 → chunk 단위 인덱스 생성
[단계 3] GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
         - 분할 단위: group_key (부모 CSV)
         - 같은 부모 CSV의 chunk는 train/val에 동시 등장하지 않음
[단계 4] splits.json 저장 (스키마는 본 데이터 단계와 호환)
[단계 5] 클래스 분포는 로그로만 기록, stratification은 포기
```

| 결정 항목 | 파일럿 선택 | 근거 |
|---|---|---|
| Split 방식 | `GroupShuffleSplit(n_splits=1)` | group 누수만 차단, stratification은 본 데이터 단계로 이전 |
| Fold 수 | 1 (hold-out) | 3-fold가 group 수 부족으로 실행 불가 |
| test_size | 0.2 | 단순한 80/20 |
| 시드 | random_state=42 | 재현성 확보 |
| **group_key 누수 차단 메커니즘** | **코드에 존재**(GroupShuffleSplit 사용 + 사후 assertion) | 본 데이터 단계에서 그대로 사용 가능 |

> **본 데이터 단계 전환 시**: `GroupShuffleSplit` 대신 `StratifiedGroupKFold(n_splits=3+)`로 교체. splits.json 스키마는 동일 유지하여 코드 변경 최소화.

### 2-4. 센서별 모델 구조 (3개)

```
[입력] CSV (vehicle, sensor 메타 포함)
        ↓
[sensor 필터] sensor ∈ {Current_U, Vib_Motor, Vib_TM} 각각 분리
        ↓ (센서별 동일 파이프, 차종은 합쳐서 학습)
[전처리] chunker(2000) → NaN 정리 → 1개 컬럼 선택(기본: rms)
        ↓
[Panel 변환] (N, 1, 2000)  float32
        ↓
[MiniRocket] num_kernels=10000, random_state=42  (센서별 독립 인스턴스)
        ↓
[LogisticRegression(multinomial, L2, class_weight='balanced')] 5-class  (센서별 독립 학습)
        ↓
[출력] 라벨 + predict_proba 5클래스 확률 + AS (= 1 - P(NORMAL))
```

- **model_key 차원**: 파일럿은 `(sensor)` 1차원 — 3개 모델. 본 데이터 단계는 `(vehicle, sensor)` 2차원 — 9개 모델로 확장(§7).
- **컬럼 선택**: 파일럿 단계는 grid search 미수행. **`rms` 단일 컬럼 고정**. (근거: 직전 PRD에서도 rms가 동률 시 1순위였고, 컬럼 선택 자체가 본 데이터 단계의 NFR-06 검증 대상)
- **class_weight**: chunking으로도 클래스 불균형 잔존(센서별 REDUC=8 vs DEMAG=3 등) → LogisticRegression의 `class_weight='balanced'` 적용.
- **모델 저장**: `models/pilot_{sensor}.joblib` 형식으로 3개(`pilot_current_u.joblib` / `pilot_vib_motor.joblib` / `pilot_vib_tm.joblib`). joblib zlib 압축. 본 데이터 단계의 ModelRegistry 인터페이스와 동일 형식.
- **학습 데이터 규모(센서별)**: 부모 CSV 27개(3차종 합), chunk ≈ 75개. group_key 분포는 (Current_U 예) DEMAG=3 / ECC10=6 / ECC20=6 / NORMAL=4 / REDUC=8 → min ≥ 3으로 `GroupShuffleSplit` hold-out 안정 동작.

### 2-5. 분류기 결정: MiniROCKET (sktime) + LogisticRegression(multinomial) 명시

본 절은 §2-4의 모델 구조에서 사용되는 분류기 선택을 명시적으로 고정한다.

| 결정 항목 | 파일럿 선택 | 근거 |
|---|---|---|
| 특징 추출기 | **`sktime.transformations.panel.rocket.MiniRocket`** (`num_kernels=10000`, `random_state=42`) | 커널 가중치가 학습 없이 고정되어 모델 크기/지연이 데이터 규모에 거의 무관. 엣지 배포 친화적. |
| 분류기 | **`sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, class_weight='balanced', max_iter=1000, random_state=42)`** | 선형 모델·직렬화/추론 비용 작음 + 네이티브 `predict_proba`로 진짜 확률 출력 → AS 계산과 5클래스 확률 시각화 직결. softmax 근사 우회 불필요. |
| 입력 텐서 형식 | **`(N, 1, 2000)` `float32` sktime panel** (panel_formatter 출력 그대로, **센서별 모델 공통**) | univariate 1채널 panel로 고정. MiniRocket 입력 규약 충족. 센서 간 채널 혼합 없음. |
| 채널(센서) 선택 | **3개 센서 모두 사용 — 센서별 독립 모델 3개** (`Current_U` / `Vib_Motor` / `Vib_TM`) | 전류/진동 도메인 차이로 단일 모델 통합 시 학습 붕괴 위험. multivariate(3 센서 동시 입력) 적용은 본 데이터 단계로 이전. |
| 신호 컬럼 | `rms` 단일 컬럼 (§2-4 승계) | 컬럼 grid search(4종 × 9슬롯 = 36-grid)는 본 데이터 단계 NFR-06 검증 대상. |

**결정 근거 요약**

- **엣지 배포 적합성**: MiniROCKET의 커널은 고정 난수로 생성되므로 학습 산출물은 사실상 LogisticRegression 가중치 행렬만 직렬화하면 된다. 선형 모델 계수 행렬 모양은 RidgeClassifierCV와 동일하여 직렬화 크기·추론 지연이 사실상 같다. 결과적으로 PSC-04 "모델 저장/로드 후 동일 출력"과 본 데이터 단계 NFR-02 "≤ 200MB 디스크" 양쪽 모두에 유리하다.
- **확률 출력의 직접적 필요성**: 대시보드 요구(센서별 AS 게이지·AS 시계열·5클래스 확률 차트)가 네이티브 확률 출력을 전제로 한다. RidgeClassifierCV는 `decision_function` 점수만 제공하여 softmax 근사를 별도로 구현해야 하나, LogisticRegression은 `predict_proba`로 진짜 확률(합=1)을 직접 출력한다.
- **MiniROCKET-Ridge 페어링 정통성 vs 본 PRD 맥락**: MiniROCKET 원논문은 Ridge 페어링이나, 대용량 데이터 속도용 선택이며 파일럿 규모(≈225 샘플)에서 차이 무의미. 확률 출력이 대시보드 요구(5클래스 확률·AS)에 직접 필요하여 LogisticRegression으로 교체. 본 데이터 단계 전환 시 코드 변경 1줄(분류기 인스턴스 교체) 수준이며 ModelRegistry 인터페이스 영향 없음.
- **소표본 강건성**: chunk 분할 후 ≈225 샘플이라는 파일럿 데이터 규모에서도 `solver='lbfgs'`로 학습이 수렴한다. 데이터 크기 민감도가 낮아 단일 hold-out split(§2-3)과 호환된다. 미수렴 시 `solver='saga'` + L2 폴백 경로 보유(§3 리스크 표).
- **multinomial 명시 사유**: `multi_class='multinomial'`은 5클래스 softmax 기반 확률을 생성하여 AS 계산(=1-P(NORMAL))과 시각화에 직결된다. One-vs-Rest 대비 클래스 간 확률의 합=1 보장.
- **9슬롯 × 4 컬럼 = 36-grid 비교**는 본 PRD 범위에서 제외하며 §4-3 Out of Scope에 이미 명시된 대로 본 데이터 단계로 이전한다.
- **차종 분리(9개 모델)**는 본 PRD 범위에서 제외한다 — 파일럿은 센서별 3개 모델까지만, 차종 분리는 §4-3 Out of Scope 및 §7 Exit Criteria에 명시.
- **정량 성능 목표는 본 절에서도 부여하지 않는다** — 파일럿 톤(§4-1·4-2) 유지.

### 2-6. Anomaly Score (AS) 정의

본 절은 대시보드 위젯과 시계열 기록의 핵심 스칼라 지표인 **Anomaly Score (AS)** 의 정의·계산 절차·시각화 임계·확장 경로를 고정한다.

| 항목 | 정의 |
|---|---|
| 정의식 | **`AS = 1 - P(NORMAL)`** |
| 값 범위 | `AS ∈ [0, 1]` (0 = 정상, 1 = 결함 확정) |
| 계산 단위 | 청크 단위(센서별 독립). 각 청크의 `predict_proba` 출력에서 NORMAL 클래스 확률을 추출하여 1에서 뺀 값 |
| 입력 | 센서별 모델의 `predict_proba(X_chunk)` 결과 (shape: `(1, 5)`) |
| 출력 | 스칼라 `AS` + 5클래스 확률 벡터 `(P(NORMAL), P(ECC10), P(ECC20), P(DEMAG), P(REDUC))` |
| 누적 형식 | (chunk_idx, sensor, AS, top_class_prob, predicted_label) 시계열 — `st.session_state.as_history`에 센서별 독립 리스트로 저장 |

**시각화 색상 임계**

| AS 구간 | 색상 | 의미 |
|---|---|---|
| `AS < 0.3` | 녹색 | 정상 |
| `0.3 ≤ AS ≤ 0.7` | 황색 | 경계/주의 |
| `AS > 0.7` | 적색 | 이상 의심/결함 확정 |

**SOH 용어 폐기 사유**

- SOH(State of Health)는 배터리 도메인의 표준 용어(전류 적산·내부저항 기반)로, 모터-감속기 회전체 결함 진단에는 도메인 정합성이 떨어진다.
- 본 프로젝트명 "Anomaly Detection"과 직접 매칭되는 PHM(Prognostics and Health Management) 분야 표준 톤인 **Anomaly Score (AS)** 를 채택한다.
- AS는 단일 스칼라(`1 - softmax 1개 값`)로 계산되어 정의가 명확하고 위젯 통합이 용이하다.

**본 데이터 단계 확장 경로 — FSI(Fault Severity Index)**

- 파일럿 단계: `AS = 1 - P(NORMAL)` 단일 정의(결함 종류별 중증도 미반영).
- 본 데이터 단계: 결함 종류별 가중치를 반영한 **FSI(Fault Severity Index)** 검토.
  ```
  FSI = Σ w_c · P(c)
       c ∈ {NORMAL, ECC10, ECC20, DEMAG, REDUC}

  초기 가중치(도메인 전문가 협의 필요):
    w_NORMAL = 0.0
    w_ECC10  = 0.3
    w_ECC20  = 0.5
    w_DEMAG  = 0.8
    w_REDUC  = 1.0
  ```
- **인터페이스 호환성**: FSI도 [0, 1] 스칼라 출력을 유지하여 AS 게이지/AS 시계열 위젯 코드 변경 없이 교체 가능. 가중치만 `configs/anomaly_score.yaml`로 외부화하여 본 데이터 단계 진입 시 정의식만 교체.

---

## 3. 기대 효과와 리스크

### 기대 효과

- **블로커 해소**: StratifiedGroupKFold ValueError 우회로 파이프라인이 끝까지 실행됨.
- **본 데이터 단계 코드 자산화**: chunker / NaN handler / Panel formatter / MiniROCKET wrapper / Registry / Router / Dashboard 스캐폴딩이 그대로 본 데이터 단계로 이전 가능.
- **누수 차단 메커니즘 사전 검증**: GroupShuffleSplit + group_key assertion 코드 경로가 본 데이터 단계의 StratifiedGroupKFold 교체 후에도 그대로 활용 가능.
- **엣지 배포 형태 확인**: 모델 직렬화/로드/추론 latency 측정 인프라가 본 데이터 단계에서 곧바로 NFR 검증으로 전환됨.
- **학기말 발표 자료 확보**: 정량 성능을 약속하지 않더라도 "feasibility 검증 완료 + 본 데이터 단계 계획"이라는 명확한 학술적 서사 제공.

### 리스크 및 대응 방안

| 리스크 | 영향 | 대응 방안 |
|---|---|---|
| 센서별 모델이 차종 간 신호 차이를 흡수 못함 | 파일럿 분류 성능 매우 낮을 수 있음 | **성능은 평가 대상이 아님을 PRD에 명시**, 본 데이터 단계 (차종 × 센서) 9-모델 분리 복귀로 해소 |
| 센서별 group_key 분포 불균형(예: REDUC=8 vs DEMAG=3) | hold-out 시 특정 클래스 train/val 한쪽 쏠림 | split 사후 검증으로 train/val 각각 5 클래스 ≥ 1 샘플 충족 시까지 시드 회전(최대 10회), 미충족 시 해당 센서 모델은 사유 로깅 후 PSC-01 부분 통과 처리 |
| chunk_size=2000 가정이 본 데이터에서 변경 | 코드 광범위 수정 | chunk_size를 `configs/preprocess.yaml`로 외부화, 본 데이터 단계 첫 sweep에서 재선정 |
| GroupShuffleSplit이 fault_class를 train에 0개로 만들 가능성 | 학습 자체 실패 | split 결과 사후 검증 — train/val 각각 5 클래스 모두 ≥ 1 샘플 충족 시까지 시드 회전(최대 10회) |
| 단일 hold-out으로 인한 평가 편향 | 평가 신뢰도 낮음 | **성능 평가가 목표가 아님**을 명시. 평가 코드는 코드만 동작하면 됨 |
| 본 데이터 단계 전환 시 코드 ABI 변경 | 재작업 비용 | 모든 핵심 인터페이스(splits.json, config.json, registry_manifest.json) 스키마를 본 데이터 단계와 동일 형식으로 미리 고정 |
| MiniROCKET 추론 latency 측정값이 본 데이터 단계에서 무의미 | NFR 검증 실효성 없음 | 파일럿 단계 측정값은 "코드 동작 증빙"용으로만 사용, 정량 NFR은 본 데이터 단계 Phase 8로 이전 |
| 대시보드 스캐폴딩이 본 데이터 단계 요구와 어긋남 | UI 재작업 | 파일럿 대시보드는 Streamlit 단일 앱으로 4종 위젯(AS 게이지·AS 시계열·5클래스 확률·이상 탐지 로그)만 구현, 시각화 확장은 본 데이터 단계 결과에 맞춰 추가 |
| 클래스 불균형(REDUC 4 vs DEMAG 1 등) | balanced 가중치만으로 부족 | 본 데이터 단계 진입 전제로 묵인. 단 분포 통계는 매 학습 시 로그 |
| 본 데이터 600GB 적재 환경 미정 | 본 데이터 단계 시작 시점 지연 | 본 데이터 단계 진입 조건을 "스토리지 확보 + I/O 처리량 ≥ X MB/s" 로 별도 정의(본 PRD 범위 밖) |
| LogisticRegression 수렴 실패 가능 (≈225 샘플 + 10000 차원 MiniROCKET 특징) | 학습 단계 실패 | `max_iter=1000`, `solver='lbfgs'` 기본. 미수렴(`ConvergenceWarning`) 시 `solver='saga'` + L2 폴백, 미수렴 사실은 학습 로그에 기록하고 모델 카드에 동반 기록 |
| AS 정의가 단순(=1-P(NORMAL))이라 결함 종류별 중증도 미반영 | 시계열 차트가 결함 심각도 차이를 표현 못함 | 파일럿은 단일 스칼라(AS)로 충분(0~1 범위 정상/이상 이분 시각화 목적). 본 데이터 단계에서 FSI(가중) 정의로 확장 검토. 인터페이스(0~1 스칼라) 호환되므로 대시보드 코드 영향 없음 |
| Streamlit 청크 스트리밍 시 `st.rerun` 루프 부하 | 브라우저 지연·CPU 과부하 | 청크 간격 0.5초 이상 유지, `st.empty()` placeholder 갱신 패턴 권장. 위젯 4종은 placeholder 컨테이너로 in-place 업데이트. 모델 3개는 `@st.cache_resource`로 1회 로드하여 재로드 비용 제거 |

---

## 4. 목표 및 성공 지표 (파일럿)

### 4-1. Pilot Success Criteria (필수)

본 단계는 **성능 정량 목표가 없다**. 대신 다음 6개 항목을 모두 충족하면 파일럿 단계 완료로 간주한다.

| ID | 항목 | 측정 방법 | 통과 기준 |
|---|---|---|---|
| PSC-01 | end-to-end 파이프라인 동작 | 데이터 로드 → 전처리 → 학습 → 추론 → 대시보드 표시까지 단일 명령으로 실행 | 예외 없이 완료, 1건 이상 추론 결과가 대시보드에 표시됨. **데모 시작 버튼 클릭 시 3개 센서(Current_U/Vib_Motor/Vib_TM) 청크가 순차 스트리밍되며 AS 게이지·AS 시계열·5클래스 확률·이상 탐지 로그가 실시간 갱신됨** |
| PSC-02 | 모델 저장/로드 가능 | joblib 저장 후 신규 Python 세션에서 로드 → 동일 입력 → 동일 출력 | 출력 라벨/점수 100% 일치 |
| PSC-03 | 엣지 배포 가능 형태 | joblib 파일 단일 산출물 1개, 의존성 `requirements.txt` 잠금 | `pip install -r requirements.txt` clean 후 추론 성공 |
| PSC-04 | 추론 latency 측정 가능 | `measure_inference_latency()` 함수가 mean/p50/p95/p99 반환 | 측정값이 numeric으로 출력, 환경 메타(CPU·OS·Python·sktime 버전) 동반 기록 |
| PSC-05 | group_key 누수 차단 메커니즘 | train/val의 group_key 교집합 검사 함수 존재 + 호출됨 | 교집합 ∅ assertion 통과 |
| PSC-06 | 본 데이터 단계 인터페이스 호환 | splits.json, config.json, registry_manifest.json 스키마가 직전 PRD 정의와 일치 | JSON Schema 검증 통과 |

### 4-2. 측정만 하되 목표값 없음 (Non-Binding Metrics)

다음은 파일럿 단계에서 기록만 하고, **수치 통과 기준을 두지 않는다.** 본 데이터 단계로 이전.

| 항목 | 측정 | 본 데이터 단계 목표값(참고) |
|---|---|---|
| Validation Accuracy | 기록 | ≥ 0.90 |
| Validation Macro-F1 | 기록 | ≥ 0.85 |
| 결함 클래스별 Recall | 기록 | ≥ 0.85 |
| 단일 샘플 추론 latency | 기록 | ≤ 100ms |
| 모델 디스크 크기 | 기록 | ≤ 200MB |
| 학습 총 소요 시간 | 기록 | ≤ 6시간 (T4 기준) |

### 4-3. Out of Scope (본 데이터 단계로 이전)

다음 항목은 **파일럿 단계에서 수행하지 않는다.**

- 9개 (차종 × 센서) 슬롯 분리 모델 운영 → 본 데이터 단계 복귀 (파일럿은 센서별 3개 모델까지만, 차종은 합쳐서 학습)
- 4개 컬럼 × 9개 슬롯 = 36-grid search → 본 데이터 단계 복귀
- v2.0 PNG 76차원 파이프라인과의 정량 벤치마크(`benchmark.md`) → 본 데이터 단계 복귀
- 엣지 NFR 정량 통과/실패 판정(NFR-01 ≤100ms, NFR-02 ≤200MB 등) → 본 데이터 단계 복귀
- splits.json의 v2.0 PNG 팀과의 해시 공유 협의 → 본 데이터 단계 복귀
- 모델 카드 9개 작성 → 본 데이터 단계 복귀 (파일럿은 센서별 모델 카드 3개만 작성)
- StratifiedGroupKFold 기반 다중 fold 평가 → 본 데이터 단계 복귀

---

## 5. 시스템 아키텍처 (파일럿)

```
[원천]
샘플데이터/03.합성데이터/1.모터_감속기_시계열/
  {차종}/{고장}/{날짜}/{센서}/*.csv  (81 CSV, 가변 길이 2000~10000)

        ↓ path_indexer + csv_loader
[Sample Index]
artifacts/sample_index.parquet
  (csv_path, vehicle, sensor, fault_class, group_key, n_rows, n_chunks)

        ↓ chunker(chunk_size=2000)
[Chunk Index]
  N ≈ 225 chunk, 같은 부모 CSV는 동일 group_key 상속

        ↓ sensor 필터 (Current_U / Vib_Motor / Vib_TM 각각, 차종 합산)
[Per-sensor Chunk Index]  ×3
  센서별 부모 CSV 27개, chunk ≈ 75개

        ↓ GroupShuffleSplit(n_splits=1, test_size=0.2, seed=42)  ×3 (센서별 독립)
[Splits]
artifacts/splits_{sensor}.json  ×3  (train_groups / val_groups)
  + assert_no_group_leakage() 센서별 통과

        ↓ NaN handler + select_column('rms') + panel_formatter
[Panel]  (N, 1, 2000), float32  (센서별)

        ↓ MiniRocket(num_kernels=10000, random_state=42)  ×3 (센서별 독립 인스턴스)
[Features]  (N, ~10000)  (센서별)

        ↓ LogisticRegression(multinomial, L2, class_weight='balanced')  ×3 (센서별 독립 학습)
[Models]
  models/pilot_current_u.joblib
  models/pilot_vib_motor.joblib
  models/pilot_vib_tm.joblib

        ↓ ModelRegistry(센서별 3개 모델 등록)
[Dispatcher]
  predict(signal, sensor_name)
    → registry[sensor_name].predict_proba(signal)
  (파일럿: sensor 1차원 dispatch
   본 데이터: (vehicle, sensor) 2차원 dispatch로 확장)

        ↓ InferenceRouter
[Inference]  (csv_path) → sensor 메타 추출 → dispatcher 라우팅
            → label + predict_proba(5클래스) + AS + latency + model_key

        ↓
[Streamlit App] streamlit_app.py
  @st.cache_resource: ModelRegistry 3개 모델 로드
  st.session_state: chunk_queue, as_history, prob_history, log_buffer

  [데모 시작 버튼] st.button("데모 시작")
      ↓
  [Chunk Streaming Loop]
    val 셋 청크 (sensor 정렬) → 0.5~1초 간격 공급
    st.empty() placeholder + time.sleep 패턴
      ↓
    각 청크: dispatcher → predict_proba → AS = 1-P(NORMAL)
      ↓
  [위젯 4종 갱신]
    - AS 게이지 (센서 3개) — 현재 시점 AS, 색상 임계(0.3/0.7)
    - AS 시계열 라인 차트 (센서 3개 또는 통합) — 청크 누적 추이
    - 5클래스 확률 차트 (막대/도넛) — P(NORMAL/ECC10/ECC20/DEMAG/REDUC)
    - 이상 탐지 로그 테이블 (누적) — (timestamp, sensor, predicted_label, top_class_prob, AS, latency_ms)
```

---

## 6. 기술 요구사항 (파일럿)

### 6-1. 기능 요구사항 (FR)

| ID | 요구사항 | 우선순위 |
|---|---|---|
| FR-01 | path_indexer가 81 CSV를 전수 스캔하여 sample_index.parquet 생성 | 필수 |
| FR-02 | csv_loader가 가변 길이 CSV(2000~10000행)를 메타/신호 분리하여 반환 | 필수 |
| FR-03 | chunker가 신호 DataFrame을 chunk_size=2000 단위로 분할, 같은 부모 CSV의 chunk는 동일 group_key 상속 | 필수 |
| FR-04 | NaN handler가 선형 보간 → 실패 시 0 대체, 결측 비율 로그 | 필수 |
| FR-05 | select_column 헬퍼가 기본 컬럼 'rms'를 슬라이싱하여 (N, 1, 2000) panel 생성 | 필수 |
| FR-06 | 센서별(3개)로 GroupShuffleSplit 단일 train/val split 생성, `splits_{sensor}.json` 3개 저장 | 필수 |
| FR-07 | assert_no_group_leakage 함수가 train/val의 group_key 교집합 = ∅임을 센서별 검증 | 필수 |
| FR-08 | 센서별 3개 모델 각각 MiniRocket(num_kernels=10000, random_state=42) + **LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, class_weight='balanced', max_iter=1000, random_state=42)** 학습 | 필수 |
| FR-09 | 센서별 모델 joblib 직렬화 3개(`pilot_current_u.joblib` / `pilot_vib_motor.joblib` / `pilot_vib_tm.joblib`, zlib 압축) | 필수 |
| FR-10 | ModelRegistry가 센서별 3개 모델을 로드하고 sensor key 기반 dispatcher를 통해 predict 호출 가능. `config.json`의 `model_assignments`는 3개 키(센서명) 보유 | 필수 |
| FR-11 | InferenceRouter가 CSV 경로 입력 시 sensor 메타를 추출해 dispatcher로 라우팅, 라벨/**predict_proba 5클래스 확률**/latency/model_key를 dict로 반환 | 필수 |
| FR-12 | 모든 모듈에서 random_state=42 고정 (센서별 모델 3개 모두 동일 시드) | 필수 |
| FR-13 | **Streamlit 앱(streamlit_app.py)** 이 추론 결과를 받아 위젯에 표시 (사용된 model_key 필드 포함) | 필수 |
| FR-14 | **Streamlit 대시보드**가 4종 위젯(AS 게이지/AS 시계열/5클래스 확률 차트/이상 탐지 로그)을 표시 | 필수 |
| FR-15 | splits.json / config.json / `registry_manifest.json` 스키마가 본 데이터 단계 명세와 일치. 파일럿 단계 `registry_manifest.json`의 `models` 배열 길이는 **3**, `config.json`의 `model_assignments`는 **3개 키** | 권장 |
| FR-16 | latency 측정 함수가 mean/p50/p95/p99 + 환경 메타(OS, CPU 모델명, Python 버전) 반환 | 권장 |
| FR-17 | **데모 시작 버튼**(`st.button("데모 시작")`)이 청크 큐(`st.session_state.chunk_queue`)를 순차 소비, 청크 도착 시마다 센서별 모델 추론 호출 → AS/확률/로그 갱신. val 셋 청크는 sensor 기준 정렬 후 0.5~1초 간격 공급 | 필수 |
| FR-18 | **Anomaly Score AS = 1 - P(NORMAL) 계산**, 청크별 시계열 누적(센서별 독립 시계열). `st.session_state.as_history[sensor]` 리스트에 (chunk_idx, AS) 튜플 추가 | 필수 |
| FR-19 | **predict_proba 출력이 5클래스 확률(합=1) 보장**, 시각화에 직접 사용 (softmax 근사 우회 없음) | 필수 |
| FR-20 | **이상 탐지 로그 테이블**에 (timestamp, sensor, predicted_label, top_class_prob, AS, latency_ms) 컬럼 표시, 청크 누적(`st.session_state.log_buffer`) | 필수 |

### 6-2. 비기능 요구사항 (NFR) — 파일럿 한정

| ID | 요구사항 | 기준값 |
|---|---|---|
| NFR-01 | 동일 시드 재실행 시 splits.json sha256 동일 | 100% 일치 |
| NFR-02 | 동일 시드 재실행 시 학습 모델의 predict 결과 100% 동일 | 100% 일치 |
| NFR-03 | end-to-end 명령 1회 실행 시간 (로컬 CPU) | ≤ 30분 (참고치) |
| NFR-04 | 모든 모듈 단위테스트 가능 (chunker, nan_handler, split, router) | pytest 통과 |
| NFR-05 | 추론 latency 측정값 numeric 출력 | 100% (목표값 미설정) |

> **NFR-01~02**는 통과 기준이 있지만, 이는 "재현성"이라는 학술적 필수 조건이며 모델 성능 정량 목표가 아니다.

### 6-3. 기술 스택

| 항목 | 라이브러리 |
|---|---|
| 데이터 로드 | pandas, numpy |
| 시계열 변환 | sktime (`MiniRocket`) |
| 분류기 | scikit-learn (`LogisticRegression`, `GroupShuffleSplit`) |
| 직렬화 | joblib (zlib 압축) |
| 대시보드 | **Streamlit (streamlit ≥ 1.30)** |
| 시각화 | **plotly 또는 altair (Streamlit 내장 차트 우선)** |
| 로깅 | logging (표준 라이브러리) |
| 테스트 | pytest |
| 개발 환경 | Jupyter Notebook, Python 3.10+ |

---

## 7. 본 데이터 단계 전환 조건 (Exit Criteria)

본 PRD가 종료되고 직전 3건의 PRD가 복귀하는 조건은 다음과 같다.

1. **본 데이터(600GB) 적재 완료** — 스토리지·I/O 확보
2. **파일럿 PSC-01 ~ PSC-06 6개 항목 전부 통과**
3. **모든 인터페이스(splits.json, config.json, registry_manifest.json) 스키마 본 데이터 단계 호환 확인**
4. **본 데이터 단계 PRD(직전 3건) 전수 재검토 — Deprecation 노트 제거**

전환 시 변경되는 핵심 항목:
- `GroupShuffleSplit(n_splits=1)` → `StratifiedGroupKFold(n_splits=3+)`
- **센서별 3개 모델** → **(차종 × 센서) 9개 모델 분리** (model_key 차원 `(sensor)` 1차원 → `(vehicle, sensor)` 2차원 확장)
- dispatcher 키 확장: `registry[sensor_name]` → `registry[(vehicle_name, sensor_name)]` (registry_manifest.json / config.json 스키마는 동일 유지, 키 tuple만 확장)
- 학습 루프에 vehicle outer loop 1개 추가 — 코드 변경 최소
- 단일 컬럼 'rms' 고정 → 4개 컬럼 grid search (36회 학습)
- **분류기 자체는 LogisticRegression(multinomial, L2, class_weight='balanced') 그대로 본 데이터 단계로 이전 — 변경 없음** (네이티브 `predict_proba` 인터페이스 유지, ModelRegistry 영향 없음)
- AS → FSI(Fault Severity Index, 가중) 전환 검토 — `configs/anomaly_score.yaml` 외부화 후 가중치 정의식만 교체, 위젯 인터페이스(0~1 스칼라) 유지
- 성능 정량 목표 활성화 (Acc ≥ 0.90, Macro-F1 ≥ 0.85, 결함 Recall ≥ 0.85)
- NFR 정량 검증 활성화 (≤ 100ms, ≤ 200MB)
- v2.0 PNG 파이프라인과 splits.json 공유 협의 개시

---

## 변경 이력

| 버전 | 날짜 | 내용 |
|---|---|---|
| v1.0 | 2026-05-17 | 최초 작성 — 샘플 81 CSV의 group 수 부족(KONA-DEMAG/ECC20=1) 으로 `StratifiedGroupKFold(n_splits=3)` 실행 불가 블로커 발생. 파일럿 단계 단순화 전략으로 (1) 단일 통합 모델 1개 채택, (2) `GroupShuffleSplit(n_splits=1, test_size=0.2)`로 split 단순화, (3) `rms` 단일 컬럼 고정으로 36-grid 보류, (4) Pilot Success Criteria 6개(PSC-01~06)로 성공 기준 재정의, (5) 정량 성능 목표/엣지 NFR 정량 통과/v2.0 벤치마크/9-모델 분리/36-grid를 모두 "본 데이터 단계 이전"으로 명시. 직전 3건 PRD(`prd_0`, `prd_260501`, `prd_260516`)는 삭제 않고 Deprecation 노트 추가하여 본 데이터 단계 복귀 자산으로 보존. 신호 채널/chunk_size=2000/MiniROCKET+Ridge 도메인 설계와 group_key 누수 차단 메커니즘은 그대로 승계. |
| v1.1 | 2026-05-17 | miniROCKET 분류기 명시 — §2-5 "분류기 결정: MiniROCKET (sktime) + RidgeClassifierCV 명시" 절 신설. (1) 특징 추출기 `sktime`의 `MiniRocket(num_kernels=10000, random_state=42)`, (2) 분류기 `sklearn.linear_model.RidgeClassifierCV(class_weight='balanced')`, (3) 입력 텐서 `(N, 1, 2000)` `float32` sktime panel(panel_formatter 출력 그대로), (4) 파일럿 채널은 단일 — `Current_U` 우선(multivariate는 본 데이터 단계 이전)로 결정 사항을 표로 고정. 결정 근거(엣지 배포 적합성·소표본 강건성·직전 PRD 도메인 결정 계승·정량 성능 목표 미부여)를 명시. PSC-01~06 및 §4 성공 지표/Out of Scope/§5 아키텍처는 영향 없음. |
| v1.2 | 2026-05-17 | 센서별 모델 3개 분리, 차종 비분리 결정 명시 — 파일럿 모델 구성을 **단일 통합 모델 1개 → 센서별 모델 3개(Current_U / Vib_Motor / Vib_TM)** 로 변경. **차종은 합쳐서 학습**(latent 변수 처리)하여 model_key는 `(sensor)` 1차원 유지. (1) §0 Scope Statement 표 모델 분리 행 갱신, (2) §2-1 기본 전략 1번 항목 재서술 및 센서 분리/차종 비분리 결정 근거(전류·진동 도메인 차이로 통합 학습 붕괴 위험, 차종별 group 수 부족(1~4)으로 hold-out 불안정) 추가, (3) §2-4 모델 구조를 센서별 3개 파이프로 재작성 — 입력 `(N, 1, 2000)` float32 univariate 유지, 저장 파일 `models/pilot_{sensor}.joblib` 3개, 센서별 group_key 분포(Current_U 예: DEMAG=3 / ECC10=6 / ECC20=6 / NORMAL=4 / REDUC=8, min ≥ 3) 명시, (4) §2-5 분류기 결정 표에서 채널 선택 행을 "Current_U 우선" → "3센서 모두 사용, 센서별 독립 모델 3개"로 변경, (5) §3 리스크 표에서 단일 모델 차종 흡수 리스크를 센서별 모델 톤으로 수정 + 센서별 group_key 불균형 신규 리스크/대응 추가, (6) §4-3 Out of Scope에서 모델 카드 9개 → 센서별 모델 카드 3개 작성으로 정정, (7) §5 아키텍처에 sensor 필터·센서별 splits·센서별 MiniRocket/Ridge·dispatcher(`predict(signal, sensor_name) → registry[sensor_name].predict(signal)`) 노드 추가, (8) §6-1 FR-06/07/08/09/10/11/12/13/14/15를 센서별 3개 모델 운영 기준으로 갱신 — `registry_manifest.json`의 `models` 배열 길이 3, `config.json`의 `model_assignments` 키 3개, (9) §7 Exit Criteria에 model_key 차원 `(sensor)` → `(vehicle, sensor)` 확장과 dispatcher 키 tuple 확장, vehicle outer loop 추가가 코드 변경 최소화 경로임을 명시. PSC-01~06 통과 기준·정량 성능 목표 미부여·NFR 정량 본 데이터 단계 이전 원칙은 v1.0~v1.1 그대로 유지. |
| v1.3 | 2026-05-17 | **분류기 교체·Anomaly Score 도입·대시보드 스택 교체** — 3가지 핵심 변경: (1) **분류기 교체**: RidgeClassifierCV → `LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, class_weight='balanced', max_iter=1000, random_state=42)`. 네이티브 `predict_proba` 확보 → softmax 근사 불필요. MiniROCKET 페어링 정통성은 약간 양보하나 파일럿 규모(≈225 샘플)에서 무관, 대시보드 확률 요구(5클래스 확률·AS)에 직결. 본 데이터 단계 전환 시 코드 변경 1줄. (2) **Anomaly Score (AS) 도입**: `AS = 1 - P(NORMAL)`, [0, 1] 스칼라. PHM(Prognostics and Health Management) 표준 톤. SOH 용어 폐기(배터리 도메인 용어로 모터-감속기에 부적합). 청크 단위·센서별 독립 시계열 누적. 색상 임계 0.3(녹색)/0.7(적색). 본 데이터 단계에서 FSI(가중) 정의로 확장 경로 명시(`w_NORMAL=0, w_ECC10=0.3, w_ECC20=0.5, w_DEMAG=0.8, w_REDUC=1.0` 초안, 도메인 전문가 협의 필요), 인터페이스(0~1 스칼라) 호환 유지. (3) **대시보드 스택 교체**: FastAPI+Vite/TS/Tailwind → **Streamlit 단일 앱**(`streamlit_app.py`). "데모 시작" 버튼 클릭 시 val 셋 청크가 sensor 기준 정렬 후 0.5~1초 간격 순차 스트리밍, `@st.cache_resource`로 모델 3개 1회 로드, `st.empty()` placeholder + `time.sleep` 패턴. 위젯 4종: AS 게이지(센서 3개)·AS 시계열 라인 차트·5클래스 확률 차트·이상 탐지 로그 테이블(timestamp/sensor/predicted_label/top_class_prob/AS/latency_ms 컬럼). 갱신 범위: §2-1(3·6번 항목)·§2-4(모델 구조 다이어그램·class_weight 문장)·§2-5(분류기 결정 표·근거 단락) 갱신, §2-6(AS 정의·색상 임계·SOH 폐기 사유·FSI 확장 경로) 신설, §3 리스크 표 3행 추가(LogisticRegression 수렴 실패·AS 단순 정의·Streamlit st.rerun 부하), §4-1 PSC-01 통과 기준에 데모 버튼 기준 추가, §5 아키텍처에서 RidgeClassifierCV/FastAPI+Vite/TS/Tailwind 블록을 LogisticRegression + Streamlit 단일 앱 구조로 재작성, §6-1 FR-08/11/13/14 갱신 + FR-17~20 신설, §6-3 기술 스택 행 교체(FastAPI+Vite/TS/Tailwind 2개 행 삭제, Streamlit + plotly/altair 2개 행 추가, 분류기 행 LogisticRegression으로 갱신), §7 Exit Criteria에 분류기 자체는 LogisticRegression 그대로 본 데이터 단계로 이전(변경 없음) 및 AS→FSI 전환 검토 추가. 추론 자체(MiniROCKET 특징·센서별 3개 모델·GroupShuffleSplit·chunk_size=2000·`rms` 단일 컬럼·`(sensor)` 1차원 model_key)와 PSC-01~06 통과 기준·정량 성능 목표 미부여 원칙·본 데이터 단계 호환성(splits.json/config.json/registry_manifest.json)은 그대로 유지. |
