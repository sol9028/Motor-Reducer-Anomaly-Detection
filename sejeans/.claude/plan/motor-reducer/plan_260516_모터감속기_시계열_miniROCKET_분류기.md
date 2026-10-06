# PLAN: 모터 감속기 시계열 MiniROCKET 분류 파이프라인 구현 계획

> **[DEPRECATED — 2026-05-17] 본 PLAN은 본 데이터(600GB) 단계 복귀용 자산으로 보존된다.**
> 파일럿 단계에서는 본 PLAN을 사용하지 않는다. 활성 PLAN은 다음 파일이다:
> `.claude/plan/motor-reducer/plan_260517_모터감속기_파일럿_feasibility.md`
>
> **Deprecation 사유**: 샘플 81 CSV의 group(부모 CSV) 수 부족(KONA-DEMAG/ECC20=1)으로 `StratifiedGroupKFold(n_splits=3)` 실행이 `ValueError`로 즉시 실패함. 9슬롯 × 36-grid 가정도 슬롯당 평균 group 2~4개로 통계적 유의성 확보 불가.
>
> **본 데이터 단계 복귀 조건** (PRD `prd_260517_모터감속기_파일럿_feasibility.md` §7):
> 1. 본 데이터(600GB) 적재 완료
> 2. 파일럿 PSC-01~06 6개 항목 전부 통과
> 3. 인터페이스 스키마(splits.json / config.json / registry_manifest.json) 본 데이터 단계 호환 확인
> 4. 직전 3건 PRD Deprecation 노트 제거 후 본 PLAN 재활성화
>
> 본 PLAN의 Phase 0/1/2 산출물은 파일럿 단계에서 그대로 인계되어 사용 중이며, Phase 3 이후(StratifiedGroupKFold, 9슬롯 분리, 36-grid, v2.0 벤치마크, 엣지 NFR 정량 검증)는 본 데이터 단계에서 재활성화 예정.

---

> 7조 | 박세진(C321027) · 박솔 | 홍익대 시스템분석/설계
> 근거 PRD: `.claude/prd/motor-reducer/prd_260516_모터감속기_시계열_miniROCKET_분류기.md`
> 작성일: 2026-05-16 | 데드라인: 2026-06-13 (학기말 발표)
> 작업 가정: 박세진 1인, 주 10~15시간(약 4주, 총 40~60시간)
> Rev. 2026-05-16-02 (chunking & n_splits=3 반영)

## 데이터 양상 (Rev. 2026-05-16-02)

- 각 PNG STFT 스펙트로그램 = 2,000행(시간축 픽셀 2,000) 단위.
- CSV 1개 = 1~5개의 시간 인접 PNG를 이어 붙인 산출물 → 행 수는 `{2000, 4000, 6000, 8000, 10000}` 중 하나(즉 PNG 개수 × 2000).
- 전처리 단계에서 모든 CSV를 2,000행 chunk로 분할하면 원본 PNG 단위 샘플로 환원되며, 분할 후 (vehicle, sensor, fault_class) 슬라이스의 표본 수가 균형 복원됨(예: DEMAG 3 CSV × 5 chunk = 15, REDUC 9×1 + 3×2 = 15).
- 누수 방지를 위해 같은 부모 CSV에서 잘린 chunk들은 **동일 group_key**를 공유(시간 인접·동일 모터 지문의 train/test 누수 차단).
- 총 81 CSV → 행 수 분포 `{2000: 21, 4000: 24, 6000: 9, 8000: 6, 10000: 21}` → chunk 분할 후 **총 225 샘플** (모든 9개 (vehicle × sensor) 슬롯이 5 클래스 × 5 chunk = 25 샘플로 완전 균형). Phase 2 작업 2-0에서 `artifacts/csv_row_sweep.csv` 산출 후 본 섹션 갱신 (2026-05-17 실측, `scripts/sweep_csv_rows.py`).

---

## 0. 작업 개요 및 일정 요약

| Phase | 제목 | 예상 소요 | 목표 완료일 |
|---|---|---|---|
| Phase 0 | 환경 셋업 및 의존성 고정 | 3h | 05-17 |
| Phase 1 | 데이터 인덱싱·로더 (CSVLoader) | 5h | 05-19 |
| Phase 2 | 전처리 모듈 (NaN·Shape·Panel 변환) | 4h | 05-20 |
| Phase 3 | Split 설계 (StratifiedGroupKFold, splits.json) | 4h | 05-22 |
| Phase 4 | 단일 (차종×센서×컬럼) 학습 프로토타입 | 5h | 05-24 |
| Phase 5 | 36회 grid search 자동화 + best_column 선택 | 8h | 05-28 |
| Phase 6 | ModelRegistry · InferenceRouter | 5h | 05-30 |
| Phase 7 | 평가 (9모델 metric, classification_report, CM) | 4h | 06-01 |
| Phase 8 | 엣지 NFR 검증 (≤100ms, ≤200MB) | 4h | 06-03 |
| Phase 9 | v2.0 PNG 파이프라인 벤치마크 비교 | 5h | 06-07 |
| Phase 10 | 문서화 (README, 모델 카드, config 스키마) | 3h | 06-10 |
| 예비 | 발표 자료·버퍼 | 5h | 06-13 |

총 예상 55h, 4주 분산.

---

## Phase 0 — 환경 셋업 및 의존성 고정

### 목표
sktime/scikit-learn 버전 차이로 인한 재현성 깨짐을 사전에 차단하고, 모든 후속 Phase에서 동일한 시드/버전 정책을 사용할 수 있도록 기반 환경을 고정한다.

### 작업
1. `pyproject.toml` 또는 `requirements.txt` 작성 — `sktime==0.26.*`, `scikit-learn==1.4.*`, `pandas==2.2.*`, `numpy==1.26.*`, `joblib==1.4.*` 등 마이너 버전까지 고정.
2. MiniRocket import 경로 명시 — `from sktime.transformations.panel.rocket import MiniRocket` (버전별 경로 차이 검증).
3. 공통 시드 모듈 `src/common/seed.py` 작성 — `set_global_seed(42)` 함수에서 `random.seed`, `np.random.seed`, `os.environ['PYTHONHASHSEED']` 일괄 설정.
4. 로깅 설정 모듈 `src/common/logger.py` — INFO/WARN 분기 + 파일 로테이션.
5. Colab/로컬 양쪽에서 동작 검증 (Colab T4 노트북 1개 셀에서 `MiniRocket().fit_transform(np.random.randn(2,1,2000))` smoke test). **윈도우 길이 10000 → 2000 변경(Rev. 2026-05-16-02)에 따라 V0-2 smoke test 재실행 필요 (TODO)**.

### Deliverables
- `requirements.txt` (절대경로: `src/requirements.txt`)
- `src/common/seed.py`, `src/common/logger.py`
- `notebooks/00_smoke_test.ipynb` — 환경 검증용 노트북
- `docs/env_setup.md` — 설치 절차 1페이지

### 검증
- `pip install -r requirements.txt` clean install 성공
- V0-2 smoke test: 입력 `(2, 1, 2000)` (변경 전 `(2, 1, 10000)`)에서 MiniRocket 변환 결과 피처 차원이 9000~11000 범위 (MiniRocket 출력은 chunk 길이 아닌 `num_kernels=10000`에 의존하므로 윈도우 축소 후에도 동일 범위 유지)
- `set_global_seed(42)` 두 번 호출 후 `np.random.rand()` 값이 동일 (NFR-03 재현성 100%)
- **TODO: 윈도우 길이 변경(10000 → 2000) 반영 후 V0-2 재실행해 위 검증 갱신**

---

## Phase 1 — 데이터 인덱싱 및 CSVLoader 구현

### 목표
`샘플데이터/03.합성데이터/1.모터_감속기_시계열/{차종}/{고장}/{날짜}/{센서}/*.csv` 디렉터리 트리를 스캔해 전체 샘플 인덱스 DataFrame을 만들고, 단일 CSV를 로드해 메타·신호로 분리하는 CSVLoader 클래스를 제공한다.

### 작업
1. `src/data/path_indexer.py` — `build_sample_index(root) -> pd.DataFrame` 함수 작성. 컬럼: `csv_path, vehicle, fault_class, date, sensor, group_key`.
2. 폴더 트리 스캔 시 9개(차종×센서) × 5개(고장) = 45개 leaf 디렉터리 모두 발견 확인.
3. `src/data/csv_loader.py` — `class CSVLoader: load(path) -> (meta_df, signal_df)`. signal_df는 4개 컬럼(`peak_freq_bin, band_start, band_end, rms`)만 포함.
4. 라벨 인코딩 매핑 `LABEL_MAP = {NORMAL:0, ECC10:1, ECC20:2, DEMAG:3, REDUC:4}` 고정.
5. 인덱스 통계 출력 — (차종 × 센서 × 고장)별 샘플 수 분포 표.

### Deliverables
- `src/data/path_indexer.py`
- `src/data/csv_loader.py`
- `src/data/label_map.py`
- `artifacts/sample_index.parquet` — 전체 샘플 인덱스 캐시
- `notebooks/01_data_index_eda.ipynb` — 분포 통계 + 결측 폴더 점검

### 검증
- `build_sample_index()`로 생성된 인덱스의 `(vehicle, sensor, fault_class)` unique 조합 수 = 45
- **V1-2 (완화, Rev. 2026-05-16-02)**: 임의 CSV 1개 로드 시 `signal_df.shape[1] == 4 and signal_df.shape[0] in {2000, 4000, 6000, 8000, 10000}` 가변 길이 허용. csv_loader는 더 이상 길이를 강제하지 않으며, 2000행 chunk 분할은 Phase 2 `chunker.py` 책임으로 이관. `meta_df`에는 `vehicle/sensor/fault_class/group_key/zsplit` 모두 포함.
- 결측 폴더(샘플 0건) 발견 시 WARN 로그 + 인덱스에서 제외

---

## Phase 2 — 전처리 모듈 (Chunking·Shape·NaN·Panel 변환)

### 목표
가변 길이(2000~10000행) CSV를 2,000행 chunk로 분할해 원본 PNG 단위 샘플로 환원하고, 결측·shape 보정을 거쳐 MiniRocket panel `(N, 1, 2000)`로 안전하게 변환하는 전처리 파이프라인을 만든다.

### 작업
0. **작업 2-0: 81개 CSV 행 수 sweep & chunk 수 집계 (신규, 예상 20분)** — 전체 81 CSV를 순회하며 행 수 분포 `{2000:?, 4000:?, 6000:?, 8000:?, 10000:?}` 산출, chunk 분할 시 총 샘플 수 및 (vehicle, sensor, fault_class) 슬라이스별 chunk 수 표 생성. 결과는 `artifacts/csv_row_sweep.csv`로 저장하고 문서 상단 "데이터 양상" 섹션의 `<TBD>` 자리 표시자 갱신.
1. **`src/preprocess/chunker.py` (신규, Rev. 2026-05-16-02)** — `split_into_chunks(signal_df, chunk_size=2000) -> List[pd.DataFrame]`:
   - 가드: `assert len(signal_df) % chunk_size == 0` (입력 길이가 chunk_size의 배수여야 함, 위반 시 명확한 에러 메시지)
   - 부모 CSV의 `group_key`를 모든 chunk가 동일하게 상속 (메타 컬럼에 보존)
   - 각 chunk에 0-indexed `chunk_id` 부여하여 추적 가능
   - 반환: chunk DataFrame 리스트 (각 길이 정확히 2000)
2. `src/preprocess/shape_guard.py` — `enforce_length(arr, target=2000)`: chunker 뒤단에서 정확히 2000행 보장하는 sanity check 역할로 격하. 부족 시 0-pad, 초과 시 절단, 처리 건수 카운터 로깅 (chunker가 정상 동작하면 호출 시 항상 no-op이어야 함).
3. `src/preprocess/nan_handler.py` — `clean_signal(arr)`: 선형 보간 1차 → 실패 시 0 대체, 결측 비율 반환.
4. `src/preprocess/panel_formatter.py` — `to_sktime_panel(signal_2d_list) -> np.ndarray` shape `(N, 1, 2000)`.
5. 4개 컬럼 중 1개를 선택해 univariate panel을 만드는 헬퍼 `select_column(signal_df, col_name)`.
6. 단위 테스트 — `tests/test_preprocess.py` (pytest): 짧은 시퀀스/긴 시퀀스/NaN 포함 + chunker 분할 케이스 4종.

### Deliverables
- `src/preprocess/chunker.py` (신규)
- `src/preprocess/shape_guard.py`
- `src/preprocess/nan_handler.py`
- `src/preprocess/panel_formatter.py`
- `tests/test_preprocess.py`
- `artifacts/csv_row_sweep.csv` (작업 2-0 산출물)

### 검증
- **V2-0 (신규)**: 81 CSV sweep 결과의 chunk 총합이 (vehicle, sensor, fault_class) 슬라이스별로 표로 정리되고, 문서 상단 `<TBD>` 자리 표시자가 실제 값으로 갱신됨
- **V2-1 (신규, chunker)**: 10000행 CSV 1개 입력 → 5개 chunk 반환, 모든 chunk가 동일 `group_key`를 상속, `chunk_id`는 0~4로 0-indexed 부여
- **V2-2 (신규, chunker)**: 3000행처럼 chunk_size의 배수가 아닌 입력 → `AssertionError` 발생
- 길이 1,500 입력(예외 케이스) → `enforce_length(arr, target=2000)` 출력 (2000,), pad 카운터 +1 (FR-02)
- NaN 5% 포함 입력 → 보간 후 NaN 0개, 결측 비율 0.05 로그 (FR-03)
- panel shape이 `(N, 1, 2000)`이며 dtype float32 (NFR-02 메모리 절감)

---

## Phase 3 — Split 설계 (StratifiedGroupKFold + splits.json 공유)

### 목표
`group_key` 단위 누수를 차단하면서 fault_class 비율을 유지하는 train/val/test 분할을 만들고, v2.0 PNG 파이프라인과 **동일 split 파일**을 공유해 벤치마크의 공정성을 보장한다.

### 작업
1. `src/split/group_split.py` — `make_splits(sample_index, n_splits=3, val_ratio=0.15, test_ratio=0.15, seed=42)` 작성.
2. 1차로 `StratifiedGroupKFold(n_splits=3)`로 test fold 1개를 떼고, 잔여에서 다시 fold 1개를 val로 분리하는 2단계 전략.
   - **n_splits=3 선정 근거 (Rev. 2026-05-16-02)**: chunking 이후에도 그룹(부모 CSV 단위)은 chunk 분할로 늘어나지 않으며, 각 (vehicle, sensor, fault_class) 슬라이스의 최소 `group_key` 개수가 3(예: IONIQ-DEMAG 3 CSV, NIRO-NORMAL 3 CSV)이다. 7-fold 시 일부 fold에 양성 클래스가 0개가 되어 stratified 보장이 불가능하므로, **안전한 최댓값인 3-fold**를 채택. 누수 차단(같은 부모 CSV의 chunk를 같은 fold에 묶기)을 우선시(Option A).
3. (차종 × 센서)별로 독립 split — 9개 모델은 각각 자기 슬라이스의 split만 사용.
4. 결과를 `artifacts/splits.json`으로 저장. 스키마: `{model_key: {train: [csv_path,...], val: [...], test: [...]}}`.
5. v2.0 팀과 공유 — `docs/split_contract.md`에 splits.json 포맷·해시 명시. v2.0 측에서 동일 파일을 로드하도록 협의.
6. 누수 검사 함수 `assert_no_group_leakage(splits)` — train/val/test의 group_key 교집합이 비어야 함.

### Deliverables
- `src/split/group_split.py`
- `artifacts/splits.json` (model_key 9개 × {train, val, test})
- `artifacts/splits.sha256` — 무결성 해시
- `docs/split_contract.md` — v2.0 파이프라인과의 공유 계약서
- `tests/test_split.py` — 누수·비율·시드 재현성 테스트

### 검증
- 9개 모델 각각의 splits에서 `group_key` 교집합 = ∅ (FR-12)
- 동일 seed 2회 실행 시 splits.json sha256 동일 (NFR-03)
- fault_class 비율이 train/val/test에서 ±2%p 이내로 유지
- v2.0 측에서 동일 splits.json 로드 시 sample 수가 정확히 일치

---

## Phase 4 — 단일 (차종 × 센서 × 컬럼) 학습 프로토타입

### 목표
36회 자동화에 진입하기 전, **IONIQ × Current_U × rms** 1건을 end-to-end로 돌려 모든 모듈 인터페이스를 검증하고 학습 시간·메모리 베이스라인을 측정한다.

### 작업
1. `src/train/single_run.py` — 함수 `train_single(vehicle, sensor, column, splits, seed=42) -> dict` 작성.
2. 절차: split 로드 → CSV 일괄 로드 → **chunker로 2,000행 분할** → preprocess → panel 변환 `(N, 1, 2000)` → `MiniRocket(num_kernels=10000, random_state=42).fit_transform` → `RidgeClassifierCV(alphas=np.logspace(-3,3,10))` → val metric 계산.
   - **`class_weight='balanced'` 권고 제거 (Rev. 2026-05-16-02)**: chunking으로 (vehicle, sensor, fault_class) 슬라이스간 sample 수 균형이 복원되므로 기본값(`class_weight=None`)으로 학습. 단 sweep 결과 미세 불균형이 남는 슬롯이 발견되면 fallback 옵션으로 `class_weight='balanced'`를 활성화 가능.
3. 학습 시간/메모리 프로파일링 (`time.perf_counter`, `tracemalloc`).
4. 결과 dict 스키마: `{val_acc, val_macro_f1, val_recall_per_class, train_time_sec, peak_mem_mb}`.
5. 노트북 `notebooks/04_prototype_single.ipynb`에서 결과 시각화.

### Deliverables
- `src/train/single_run.py`
- `notebooks/04_prototype_single.ipynb`
- `artifacts/prototype_metrics.json` — 단일 학습 결과 baseline
- `docs/prototype_findings.md` — 학습 시간·메모리 1페이지 요약

### 검증
- val macro-F1 > 0.5 (랜덤 0.2보다 유의미하게 높음 — 파이프라인 정상)
- 단일 학습 소요 ≤ 10분 (T4 또는 로컬 CPU 기준) — 36회 자동화의 6h NFR-06 역산 충족
- 동일 seed 2회 실행 시 val metric 소수점 6자리까지 동일 (FR-10)

---

## Phase 5 — 36회 Grid Search 자동화 + best_column 선택

### 목표
9개 (차종 × 센서) × 4개 컬럼 = **36회 학습**을 단일 진입점에서 자동 실행하고, 각 (차종 × 센서)별 val macro-F1 최댓값 컬럼을 best_column으로 선정해 `config.json`을 생성한다. 본 Phase가 본 프로젝트의 핵심.

### 작업
1. `src/train/grid_search.py` — `class GridSearchRunner`:
   - 입력: sample_index, splits.json, columns=['peak_freq_bin','band_start','band_end','rms']
   - 외부 루프: 9개 (차종 × 센서), 내부 루프: 4개 컬럼
   - 진행률 표시 (`tqdm` total=36), 중간 실패 시 재시도 1회 → 그래도 실패면 결과에 `status=failed` 기록 후 계속 진행 (전체 중단 금지)
2. **체크포인트 정책** — 각 학습 종료 시 `artifacts/grid/{vehicle}_{sensor}_{column}/` 폴더에 `model.joblib`, `metrics.json`, `predictions.parquet` 저장. 중간 중단 후 재실행 시 완료된 셀은 스킵.
3. **병렬화** — sktime MiniRocket 내부 `n_jobs=-1`, 외부 루프는 직렬(메모리 안정성 우선). Colab T4에서 단일 실행 가정.
4. **메모리 가드** — 각 (차종 × 센서) 슬라이스 학습 후 `del transformer, classifier; gc.collect()`로 누수 차단.
5. `select_best_column(grid_results) -> dict` — 각 (차종 × 센서)별 val_macro_f1 argmax 선택. 동률 시 우선순위 `rms > peak_freq_bin > band_end > band_start` (rms가 가장 안정적 가정).
6. `config.json` 작성 — 스키마:
   ```json
   {
     "version": "1.0",
     "seed": 42,
     "models": {
       "IONIQ-Current_U": {"best_column": "rms", "val_macro_f1": 0.91, "val_acc": 0.93, "model_path": "models/IONIQ_Current_U.joblib"}
     }
   }
   ```
7. 36회 결과 요약 `grid_summary.csv` (행 36개, 열: vehicle, sensor, column, val_acc, val_macro_f1, val_recall_per_class, train_time, model_size_mb).

### Deliverables
- `src/train/grid_search.py`
- `scripts/run_grid_search.py` — CLI 진입점 (`python scripts/run_grid_search.py --config configs/grid.yaml`)
- `configs/grid.yaml` — 컬럼 리스트·시드·체크포인트 경로
- `artifacts/grid/` — 36개 폴더 (각 model.joblib, metrics.json)
- `artifacts/grid_summary.csv` — 36행 통합 결과표
- `artifacts/config.json` — 9개 best_column 매핑 (FR-07)
- `logs/grid_search_YYYYMMDD.log`

### 검증
- 완료된 학습 = 36건, `grid_summary.csv` 행 수 36 (FR-04)
- 9개 (차종 × 센서) 각각 best_column 정확히 1개 선정 (FR-05)
- 총 학습 시간 Colab T4 ≤ 6시간 (NFR-06)
- 학습 중 OOM 0건 (배치/메모리 가드 정상)
- 체크포인트 재실행 테스트: 절반 완료 상태에서 재실행 시 완료분 스킵 확인

---

## Phase 6 — ModelRegistry · InferenceRouter 구현

### 목표
9개 best 모델을 단일 레지스트리에 등록하고, 추론 시 `(vehicle, sensor)` 메타로 해당 모델 1개만 호출하는 InferenceRouter를 제공한다.

### 작업
1. `src/serve/model_registry.py` — `class ModelRegistry`:
   - `load(config_path)` → config.json 읽고 9개 모델 joblib 로드
   - `get(vehicle, sensor)` → `(transformer, classifier, best_column)` 반환, 키 미존재 시 `KeyError` 명시 (FR-13)
   - `size_mb()` → 메모리 상주 모델 총 크기
2. **직렬화 정책** — joblib `compress=('zlib', 6)`, 각 모델 별도 파일. 총 9개 파일.
3. `src/serve/inference_router.py` — `class InferenceRouter`:
   - `predict(csv_path) -> dict` — CSV 로드 → preprocess → best_column 슬라이스 → MiniRocket transform → classifier predict → 라벨 + `decision_function` 점수 + 메타 반환
   - 결과 dict: `{label, label_name, scores, model_key, best_column, inference_time_ms}`
4. **워밍업** — Registry 로드 직후 더미 입력 1회 predict (sklearn lazy init 제거).
5. 단위 테스트 — 9개 키 모두 조회 성공, 미존재 키(`FOO-BAR`) KeyError 발생.

### Deliverables
- `src/serve/model_registry.py`
- `src/serve/inference_router.py`
- `models/` — 9개 `.joblib` 파일 (압축)
- `tests/test_registry.py`, `tests/test_router.py`
- `artifacts/registry_manifest.json` — 모델 파일 sha256 + 크기

### 검증
- `len(registry) == 9` (PRD §4 모델 수 지표)
- 미존재 키 조회 시 `KeyError` 발생 (FR-13)
- 단일 CSV → 라벨 반환 end-to-end 정상
- 모델 9개 합산 디스크 크기 ≤ 200MB (NFR-02 1차 확인)

---

## Phase 7 — 평가 (9모델 metric · classification_report · confusion matrix)

### 목표
**test split** 기준으로 9개 모델 각각의 분류 성능을 측정하고, PRD §4 성공 지표(Acc ≥ 0.90, Macro-F1 ≥ 0.85, 결함 Recall ≥ 0.85) 충족 여부를 모델별로 판정한다.

### 작업
1. `src/eval/evaluate.py` — `evaluate_all_models(registry, splits) -> pd.DataFrame` 작성.
2. 모델별 metric: accuracy, macro-F1, weighted-F1, per-class precision/recall/f1, confusion matrix (5×5).
3. **결함 클래스 Recall 집계** — ECC10/ECC20/DEMAG/REDUC 4개 클래스 각각 ≥ 0.85 여부 PASS/FAIL 컬럼 추가.
4. 시각화 `notebooks/07_evaluation.ipynb` — 9개 confusion matrix heatmap (3×3 grid) + per-model bar chart.
5. `eval_report.md` 자동 생성기 — 모델별 분류 리포트 + PASS/FAIL 요약.

### Deliverables
- `src/eval/evaluate.py`
- `artifacts/eval/eval_results.csv` (행 9, 열: 각 metric + PASS/FAIL)
- `artifacts/eval/confusion_matrices.png` (3×3 그리드)
- `artifacts/eval/classification_reports/` — 모델별 9개 txt
- `notebooks/07_evaluation.ipynb`
- `docs/eval_report.md` — 자동 생성 요약 (PRD 성공 지표 연결표 포함)

### 검증
- 9개 모델 모두 test acc ≥ 0.90 (NFR-05) — 미달 시 어느 모델·어느 클래스가 미달인지 명시
- 결함 4클래스 Recall ≥ 0.85 (NFR-04)
- macro-F1 ≥ 0.85 (NFR-05)
- 미달 케이스 발견 시 `docs/eval_report.md`에 원인 가설(샘플 부족, 컬럼 선택 재검토 등) 기록

---

## Phase 8 — 엣지 NFR 검증 (≤100ms 추론 · ≤200MB 디스크)

### 목표
PRD NFR-01(≤100ms), NFR-02(≤200MB)을 측정 기반으로 명시적 검증해 엣지 디바이스 탑재 적합성을 증명한다.

### 작업
1. `src/bench/latency.py` — `measure_inference_latency(router, samples=100) -> dict`.
   - 워밍업 5회 → 본 측정 100회 → 평균/중앙값/p95/p99 산출
   - 9개 모델 각각 독립 측정 + 통합 평균
   - 측정 환경: CPU only (엣지 대표 환경), Colab CPU 런타임에서 실행
2. `src/bench/disk_size.py` — 9개 joblib 파일 합산 크기 + 메모리 상주 크기(`pympler.asizeof`) 측정.
3. **압축 옵션 비교** — `compress=0/3/6/9` 4가지 모드에서 디스크 크기 vs 로드 시간 trade-off 표 작성, 최적값 선정.
4. 결과 미달 시 fallback 계획:
   - 100ms 초과 → `num_kernels=5000`으로 축소 후 재학습 시 metric 영향 분석
   - 200MB 초과 → MiniRocket transformer만 sparse 직렬화, classifier만 dense
5. `bench_report.md` 작성 — 측정 환경·결과·통과 여부 명시.

### Deliverables
- `src/bench/latency.py`, `src/bench/disk_size.py`
- `scripts/run_bench.py`
- `artifacts/bench/latency_results.csv` (모델별 mean/p50/p95/p99)
- `artifacts/bench/disk_results.csv`
- `docs/bench_report.md` — NFR PASS/FAIL + 환경 표

### 검증
- 9개 모델 단일 샘플 추론 평균 ≤ 100ms (NFR-01)
- 9개 모델 joblib 총 크기 ≤ 200MB (NFR-02)
- p95 ≤ 150ms (실시간 안정성 보조 지표)
- 미달 시 fallback 적용 결과까지 문서화

---

## Phase 9 — v2.0 PNG 파이프라인 벤치마크 비교

### 목표
**동일 splits.json·동일 metric** 기준으로 v2.0(76차원 hand-crafted + SVM/RF) vs 본 PRD(MiniROCKET + Ridge) 결과를 병기한 `benchmark.md`를 작성한다. 모델 선택 근거의 핵심 산출물.

### 작업
1. v2.0 팀과 splits.json 공유 (Phase 3에서 합의된 계약) — sha256 확인.
2. v2.0 측 평가 결과 CSV 수령 (`artifacts/v2_eval_results.csv` 동일 스키마).
3. `src/bench/compare.py` — 두 파이프라인 결과 join 후 모델 단위 비교표 생성.
   - 컬럼: model_key, v2_acc, mini_acc, Δacc, v2_macro_f1, mini_macro_f1, Δf1, v2_size_mb, mini_size_mb, v2_latency_ms, mini_latency_ms
4. (차종 × 센서)별 우세 모델 표시 (Δ > 0이면 MiniROCKET 우세).
5. 통합 결정 가이드 — 9개 슬롯 각각에 어느 파이프라인을 운영에 채택할지 권고.
6. 시각화: 9개 모델 × 2 파이프라인의 metric 막대 그래프, latency vs accuracy 산점도.

### Deliverables
- `src/bench/compare.py`
- `artifacts/bench/comparison_table.csv` (9행)
- `artifacts/bench/comparison_chart.png`
- `docs/benchmark.md` — 비교표 + 모델 선택 권고 (FR-14)
- `docs/split_sha_log.md` — splits.json 해시 일치 증빙

### 검증
- splits.json sha256이 v2.0 측과 100% 일치
- 9개 model_key 모두 양 파이프라인 결과 join (누락 0)
- 권고 결정이 9개 슬롯 모두에 대해 명시
- benchmark.md가 PRD §4 "v2.0 대비 벤치마크" 지표 충족

---

## Phase 10 — 문서화 (README · 모델 카드 · config 스키마)

### 목표
타 팀원·교수 채점·후속 작업자가 코드 없이도 시스템을 이해하고 재현·운영할 수 있도록 최종 문서를 정비한다.

### 작업
1. `README.md` 작성 — 설치·데이터 경로·학습 명령·추론 예제·디렉터리 트리·각 산출물 위치.
2. **모델 카드 9개** — `docs/model_cards/{vehicle}_{sensor}.md`. 각 카드: 학습 데이터 분포, best_column, 시드, 성능, 알려진 한계, 결함 Recall 표.
3. `docs/config_schema.md` — config.json·splits.json·registry_manifest.json 스키마 명세 (JSON Schema 형식).
4. `docs/api_reference.md` — ModelRegistry·InferenceRouter 공개 API 시그니처.
5. 발표용 슬라이드 초안 `docs/presentation_outline.md` — 5분 발표 흐름.

### Deliverables
- `README.md` (프로젝트 루트)
- `docs/model_cards/` 9개 .md
- `docs/config_schema.md`
- `docs/api_reference.md`
- `docs/presentation_outline.md`

### 검증
- README만 보고 신규 작업자가 30분 내 학습/추론 명령을 재현 가능 (자체 dry-run)
- 9개 모델 카드 누락 0
- 모든 산출물 경로가 README에 링크됨
- PRD §4 성공 지표 11개 항목이 docs 어딘가에서 모두 검증 가능 (추적표 포함)

---

## 진행 상황 보드 (체크박스)

- [ ] Phase 0 — 환경 셋업
- [ ] Phase 1 — CSVLoader
- [ ] Phase 2 — 전처리 모듈
- [ ] Phase 3 — Split 설계 + v2.0 공유
- [ ] Phase 4 — 단일 학습 프로토타입
- [ ] Phase 5 — **36회 grid search 자동화 (핵심)**
- [ ] Phase 6 — ModelRegistry · InferenceRouter
- [ ] Phase 7 — 평가 (9모델 metric)
- [ ] Phase 8 — **엣지 NFR 검증 (100ms/200MB)**
- [ ] Phase 9 — **v2.0 벤치마크 비교**
- [ ] Phase 10 — 문서화

---

## 변경 이력

| 버전 | 날짜 | 내용 |
|---|---|---|
| v1.0 | 2026-05-16 | 최초 작성 — PRD v1.0(2026-05-16) 기반 11개 Phase 도출. 핵심 강조점: (1) Phase 5의 36회 grid search 체크포인트·메모리 가드·동률 선정 규칙 구체화, (2) Phase 3·9에서 v2.0과의 splits.json 해시 공유 계약 명시, (3) Phase 8을 엣지 NFR(100ms·200MB) 전용 검증으로 분리하고 미달 시 fallback 계획 포함 |
| Rev. 2026-05-16-02 | 2026-05-16 | 데이터 확인 결과 반영. (1) 윈도우 길이 10000 → 2000 전역 축소(MiniRocket `num_kernels=10000`은 유지), (2) Phase 2에 신규 모듈 `chunker.py`와 작업 2-0(81 CSV 행 수 sweep) 추가하여 가변 길이 CSV를 2,000행 chunk로 분할·원본 PNG 단위로 환원, (3) 같은 부모 CSV chunk가 동일 `group_key` 상속하도록 누수 차단(Option A) 유지, (4) Phase 3 `StratifiedGroupKFold` n_splits 7 → 3 (최소 group 수 3 제약), (5) Phase 1 V1-2 가변 길이 허용으로 완화, (6) Phase 4 `class_weight='balanced'` 권고 제거(chunking으로 균형 복원, fallback 옵션화), (7) Phase 0 V0-2 smoke test 재실행 TODO 추가, (8) "데이터 양상" 섹션 신설 및 총 샘플 수 `<TBD>` 자리 표시자 추가. Phase 0/1 완료 상태 표시는 유지. |
| DEPRECATED | 2026-05-17 | 본 PLAN을 deprecated 처리하고 본 데이터 단계 복귀 자산으로 보존. 사유: 샘플 81 CSV의 group 수 부족(KONA-DEMAG/ECC20=1)으로 `StratifiedGroupKFold(n_splits=3)` 즉시 실패 + 9슬롯 × 36-grid 통계 유의성 확보 불가. 파일럿 단계 활성 PLAN은 `plan_260517_모터감속기_파일럿_feasibility.md`로 교체. Phase 0/1/2 산출물은 파일럿 단계에서 인계 사용 중. 본 데이터 단계 진입 시 PRD `prd_260517_*.md` §7 Exit Criteria 충족 후 본 PLAN 재활성화 예정. |
