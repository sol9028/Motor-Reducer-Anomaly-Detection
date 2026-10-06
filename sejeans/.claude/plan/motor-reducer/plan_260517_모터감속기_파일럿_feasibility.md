# PLAN: 모터 감속기 파일럿(Feasibility) 단계 구현 계획

> 7조 | 박세진(C321027) · 박솔 | 홍익대 시스템분석/설계
> 근거 PRD: `.claude/prd/motor-reducer/prd_260517_모터감속기_파일럿_feasibility.md` (v1.2)
> 작성일: 2026-05-17 | 데드라인: 2026-06-13 (학기말 발표, D-27)
> 작업 가정: 박세진 1인, 주 10~15시간 (잔여 약 4주, 총 30~40시간)
> 단계: **Pilot / Feasibility Study** (샘플 81 CSV → chunk 분할 후 ≈225)
> 후속 단계: 본 데이터(600GB, 550,800 샘플) 검증 단계 PRD/PLAN 복귀 예정

---

## 0. 본 PLAN의 위치 (Scope Statement)

본 PLAN은 PRD `prd_260517_모터감속기_파일럿_feasibility.md` (v1.2)의 파일럿 단계 요구를 구현 단위 Phase로 분해한다. 직전 PLAN(`plan_260516_모터감속기_시계열_miniROCKET_분류기.md`, Rev. 2026-05-16-02)은 deprecated 처리되며, 9슬롯 × 36-grid, `StratifiedGroupKFold(n_splits=3)` 등 본 데이터 단계 가정은 본 PLAN에서 일괄 제거되어 본 데이터 단계로 이전된다.

### 파일럿 단계 핵심 원칙
- 성능 정량 목표 없음 (Pilot Success Criteria 6개 만족이 곧 완료)
- **센서별 모델 3개**(Current_U / Vib_Motor / Vib_TM, 차종 합산), 단일 컬럼(`rms`) 고정
- `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` — 누수 차단만 보장 (센서별 독립 적용)
- 인터페이스(`splits_{sensor}.json` / `config.json` / `registry_manifest.json`) 스키마는 **본 데이터 단계와 호환 형식으로 사전 고정**

### 일정 요약

| Phase | 제목 | 상태 | 예상 소요 | 목표 완료일 |
|---|---|---|---|---|
| Phase 0 | 환경 셋업 및 의존성 고정 | DONE | — | (완료) |
| Phase 1 | 데이터 인덱싱 + CSVLoader | DONE | — | (완료) |
| Phase 2 | 전처리 모듈 (chunker·shape_guard·nan_handler·panel_formatter) | DONE | — | (완료) |
| Phase 3 | GroupShuffleSplit 센서별(3) 단일 hold-out 분할 | TODO | 3.5h | 2026-05-20 |
| Phase 4 | MiniRocket + RidgeClassifierCV 센서별 모델 3개 학습 | TODO | 6h | 2026-05-24 |
| Phase 5 | 추론 인터페이스 + FastAPI/Vite 대시보드 연동 | TODO | 8h | 2026-06-03 |
| Phase 6 | 파일럿 결과 정리 + 본 데이터 단계 전환 체크리스트 | OPTIONAL | 4h | 2026-06-10 |
| 예비 | 발표 자료·버퍼 | — | 5h | 2026-06-13 |

> 누적: DONE 3개 + 신규 4개 Phase. 신규 합계 약 21.5h(예비 제외).

---

## Phase 0 — 환경 셋업 및 의존성 고정 (DONE)

### 인계 산출물
직전 PLAN Phase 0의 산출물을 그대로 인계한다. 본 PLAN에서 재구현하지 않는다.

- `src/common/seed.py` — `set_global_seed(42)`
- `src/common/logger.py` — INFO/WARN 분기 + 파일 로테이션
- (기존) requirements / smoke test 노트

### Deliverables
- (재사용) `src/common/seed.py`, `src/common/logger.py`
- 본 PLAN에서 신규 추가 없음

### Acceptance Criteria
- `set_global_seed(42)` 2회 호출 후 동일 난수 생성 (재현성 보장) — 직전 PLAN에서 검증 완료

### 본 데이터 단계 이전 항목
- 없음 (Phase 0 자산은 본 데이터 단계에서도 동일하게 사용)

---

## Phase 1 — 데이터 인덱싱 + CSVLoader (DONE)

### 인계 산출물
직전 PLAN Phase 1의 산출물을 그대로 인계한다.

- `src/data/path_indexer.py` — `build_sample_index(root) -> pd.DataFrame`
- `src/data/csv_loader.py` — `CSVLoader.load(path) -> (meta_df, signal_df)`
- `src/data/label_map.py` — `LABEL_MAP = {NORMAL:0, ECC10:1, ECC20:2, DEMAG:3, REDUC:4}`
- `artifacts/sample_index.parquet` — 전체 샘플 인덱스 캐시
- `scripts/verify_phase1.py`

### Deliverables
- (재사용) 위 5개 산출물
- 본 PLAN에서 신규 추가 없음

### Acceptance Criteria
- `sample_index.parquet`의 `(vehicle, sensor, fault_class)` 조합 수 = 45 — 직전 PLAN에서 검증 완료
- 가변 길이 CSV(2000~10000행) 로드 시 `signal_df.shape[1] == 4`

### 본 데이터 단계 이전 항목
- 없음 (path_indexer / csv_loader / label_map은 본 데이터 단계에서도 그대로 사용)

---

## Phase 2 — 전처리 모듈 (DONE)

### 인계 산출물
직전 PLAN Phase 2의 산출물을 그대로 인계한다.

- `src/preprocess/chunker.py` — `split_into_chunks(signal_df, chunk_size=2000)` + group_key 상속 + chunk_id 부여
- `src/preprocess/shape_guard.py` — `enforce_length(arr, target=2000)` sanity check
- `src/preprocess/nan_handler.py` — `clean_signal(arr)` 선형 보간 → 0 대체
- `src/preprocess/panel_formatter.py` — `to_sktime_panel(...)` `(N, 1, 2000) float32` + `select_column(signal_df, col_name)` 헬퍼
- `tests/test_preprocess.py` — 19 케이스 PASS
- `tests/conftest.py`
- `artifacts/csv_row_sweep.csv` — 81 CSV sweep 결과
- `scripts/sweep_csv_rows.py`, `scripts/verify_chunker.py`

### Deliverables
- (재사용) 위 8개 산출물
- 본 PLAN에서 신규 추가 없음

### Acceptance Criteria
- pytest `tests/test_preprocess.py` 19/19 PASS — 직전 PLAN에서 검증 완료
- panel shape `(N, 1, 2000)`, dtype `float32`

### 본 데이터 단계 이전 항목
- `chunk_size=2000` 가정의 본 데이터 재검토 (본 데이터 단계 첫 sweep에서 재선정 가능)
- 다중 채널/다중 컬럼 panel 생성 (multivariate panel) — 파일럿은 단일 채널, 단일 컬럼만 사용

---

## Phase 3 — GroupShuffleSplit 센서별(3) 단일 hold-out 분할 (신규)

### 목표
샘플 81 CSV의 group(부모 CSV) 수 부족으로 `StratifiedGroupKFold(n_splits=3)`가 실행 불가한 블로커를 회피하기 위해, **센서별로 독립**으로 `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`를 적용해 train/val 1회 분할한다. **stratification은 포기하되 group_key 누수 차단 메커니즘은 센서별로 유지**하며, `splits_{sensor}.json` 3개 스키마를 본 데이터 단계와 호환 형식으로 사전 고정한다.

### 작업
1. `src/split/group_split.py` 신규 — 함수 시그니처:
   ```python
   def make_pilot_splits(
       chunk_index: pd.DataFrame,   # chunker 출력 = sample_index의 chunk 확장본
       sensor: str,                 # "Current_U" | "Vib_Motor" | "Vib_TM"
       test_size: float = 0.2,
       random_state: int = 42,
       max_seed_retries: int = 10,
   ) -> dict
   ```
   - 호출 측에서 `for sensor in ("Current_U", "Vib_Motor", "Vib_TM"):` 루프 진입 → 센서별 독립 split 1회씩 (총 3회)
   - 입력 `chunk_index`를 `sensor == <인자>`로 필터링 후 `GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)`로 train/val 단일 분할
   - 분할 단위: `group_key` (부모 CSV 단위, chunk는 group_key 상속)
   - 사후 검증 — 해당 센서의 train/val 각각 5 클래스 모두 ≥ 1 샘플 확인. 미달 시 `random_state`를 +1씩 회전(최대 10회). 모두 실패 시 `RuntimeError`로 명시적 실패.
   - 센서별 group_key 분포 참고 (PRD §2-4) — Current_U 예: DEMAG=3 / ECC10=6 / ECC20=6 / NORMAL=4 / REDUC=8 (min ≥ 3)으로 hold-out 안정 동작 기대
2. 누수 차단 함수 `assert_no_group_leakage(splits: dict) -> None`:
   - `set(splits['train']['group_keys']) & set(splits['val']['group_keys']) == set()` assertion
   - 실패 시 `AssertionError` 발생 (PSC-05 충족)
   - 센서별 splits 결과 3개 각각에 대해 호출
3. `splits_{sensor}.json` 3개 저장 — **본 데이터 단계 호환 스키마로 사전 고정** (`splits_current_u.json` / `splits_vib_motor.json` / `splits_vib_tm.json`):
   ```json
   {
     "schema_version": "1.0",
     "stage": "pilot",
     "sensor": "Current_U",
     "split_strategy": "GroupShuffleSplit",
     "n_splits": 1,
     "test_size": 0.2,
     "random_state": 42,
     "seed_retries_used": 0,
     "folds": [
       {
         "fold_id": 0,
         "train": {
           "group_keys": ["..."],
           "chunk_ids": ["..."],
           "class_counts": {"NORMAL": 0, "ECC10": 0, "ECC20": 0, "DEMAG": 0, "REDUC": 0}
         },
         "val": {
           "group_keys": ["..."],
           "chunk_ids": ["..."],
           "class_counts": {"NORMAL": 0, "ECC10": 0, "ECC20": 0, "DEMAG": 0, "REDUC": 0}
         }
       }
     ],
     "leakage_check": {"passed": true, "intersection_size": 0}
   }
   ```
   - `folds`를 배열로 둠으로써 본 데이터 단계의 `StratifiedGroupKFold(n_splits=3+)` 출력과 동일 컨테이너로 직렬화 가능 (파일럿은 길이 1, 본 데이터는 길이 3+).
   - 본 데이터 단계는 동일 형식 파일이 9개((vehicle × sensor))로 확장.
4. SHA256 해시 산출 — `artifacts/splits_{sensor}.sha256` 3개 (NFR-01 재현성 검증용).
5. 실행 스크립트 `scripts/make_splits.py` — CLI 진입점, 3센서 루프 실행.
6. 단위 테스트 `tests/test_split.py` (3센서 모두 커버):
   - 센서 3종 각각 동일 seed 2회 실행 → 해당 `splits_{sensor}.json` 바이트 단위 일치
   - 누수 차단 함수가 의도된 leak 입력에 대해 `AssertionError` 발생 (센서별)
   - 센서 3종 각각 5 클래스 모두 train/val에 ≥ 1 샘플 존재

### Deliverables
- `src/split/group_split.py`
- `scripts/make_splits.py`
- `tests/test_split.py`
- `artifacts/splits_current_u.json`, `artifacts/splits_vib_motor.json`, `artifacts/splits_vib_tm.json` (스키마 v1.0, 각 fold 1개)
- `artifacts/splits_current_u.sha256`, `artifacts/splits_vib_motor.sha256`, `artifacts/splits_vib_tm.sha256`

### Acceptance Criteria
- pytest `tests/test_split.py` 전체 PASS (3센서 케이스)
- 센서 3종 각각의 `splits_{sensor}.json`의 `leakage_check.passed == true`, `intersection_size == 0`
- 센서별 train/val의 `class_counts` 5개 키 모두 ≥ 1 (또는 seed_retries 로그 후 최종 통과)
- 센서 3종 각각 동일 seed 2회 재실행 시 `splits_{sensor}.json` sha256 100% 일치 (NFR-01)

### 검증 항목 (V접두사 ↔ PSC 매핑)
- **V3-1** ↔ **PSC-05**: 센서별 group_key 교집합 ∅ assertion 통과 (3회)
- **V3-2** ↔ **PSC-06**: `splits_{sensor}.json` 스키마가 위 정의된 v1.0과 일치 (JSON Schema 검증, 3파일)
- **V3-3** ↔ **NFR-01**: 센서별 재실행 시 sha256 일치 (3쌍)

### 의존성
- Phase 0, 1, 2 산출물 (sample_index.parquet + chunker)

### 예상 소요
3.5h (구현 1.5h + 센서 루프/추가 테스트 1.5h + 검증 0.5h)

### 본 데이터 단계 이전 항목
- `GroupShuffleSplit` → `StratifiedGroupKFold(n_splits=3+)` 교체 (스키마 `folds` 배열 길이만 1 → 3+로 확장, 코드 변경은 split 함수 내부 알고리즘만)
- 센서별 splits 3개 → (vehicle × sensor) splits 9개로 확장 (동일 스키마, 파일명 컨벤션만 확장)
- 5 클래스 stratification 보장 (파일럿은 fault_class 분포 로그만)
- v2.0 PNG 파이프라인과의 `splits.json` 해시 공유 협의 (PRD §4-3 Out of Scope)
- group 수 부족 사례의 본 데이터 검증 — 본 데이터에서는 group 수가 충분하므로 누수 차단 메커니즘이 실효성 있게 작동함을 본 단계에서 재검증

### 리스크/블로커
- 센서별 group_key 분포 불균형(예: REDUC=8 vs DEMAG=3 등) → 특정 클래스가 train/val 한쪽 쏠림 가능. seed 회전(최대 10회)으로 회피, 그래도 실패 시 PRD §3 리스크 표대로 해당 센서 모델은 사유 로깅 후 PSC-01 부분 통과 처리.
- 최악의 경우 1회 fallback으로 `test_size=0.15` 또는 `0.25` 시도 후, 그래도 실패하면 PRD §2-3 단서대로 "코드만 동작하면 됨" 정책 적용 (val 일부 클래스 0개 허용, 로그로 기록).

---

## Phase 4 — MiniRocket + RidgeClassifierCV 센서별 모델 3개 학습 (신규)

### 목표
PRD §2-4·§2-5에서 결정한 **센서별 모델 3개**(Current_U / Vib_Motor / Vib_TM, 차종 합산)를 학습한다. 차종 분리(9슬롯)·36-grid·multivariate는 모두 본 데이터 단계로 이전하고, 파일럿은 **3개 센서 × 단일 컬럼(`rms`) × 센서당 1개 모델 = 총 3개 모델**을 학습한다. 각 센서마다 MiniRocket 인스턴스와 Ridge 분류기를 독립으로 학습한다.

### 작업
1. `src/train/pilot_train.py` 신규 — 함수 시그니처:
   ```python
   def train_pilot_per_sensor(
       sample_index: pd.DataFrame,
       splits_by_sensor: dict,    # {"Current_U": splits_dict, "Vib_Motor": ..., "Vib_TM": ...}
       sensors: tuple = ("Current_U", "Vib_Motor", "Vib_TM"),
       column: str = "rms",
       num_kernels: int = 10000,
       random_state: int = 42,
   ) -> dict  # {sensor: {"model": joblib_dict, "metrics": {...}, "path": "models/pilot_{sensor}.joblib"}}
   ```
2. 학습 절차 (outer 센서 루프):
   ```
   for sensor in sensors:
       splits = splits_by_sensor[sensor]            # Phase 3의 splits_{sensor}.json 로드 결과
       train/val group_keys 획득
       sample_index에서 sensor == <센서> 필터 → group_key로 train/val 분리
       각 CSV: CSVLoader.load → split_into_chunks(chunk_size=2000) → clean_signal → select_column('rms') → to_sktime_panel
       train panel (N_train, 1, 2000) float32, val panel (N_val, 1, 2000) float32
       센서별 독립 MiniRocket(num_kernels=10000, random_state=42).fit_transform(train_panel) → feature (N_train, ~10000)
       센서별 독립 RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight='balanced').fit(features, labels)
       val transform → predict → 센서별 metric (acc, macro-F1, per-class recall) 기록
   ```
3. **non-binding metric 기록** — `artifacts/pilot_metrics.json` (센서별 dict):
   ```json
   {
     "current_u": {
       "val_acc": 0.0,
       "val_macro_f1": 0.0,
       "val_recall_per_class": {"NORMAL": 0.0, "ECC10": 0.0, "ECC20": 0.0, "DEMAG": 0.0, "REDUC": 0.0},
       "train_time_sec": 0.0,
       "n_train_samples": 0,
       "n_val_samples": 0,
       "model_disk_mb": 0.0
     },
     "vib_motor": { "...": "..." },
     "vib_tm":    { "...": "..." },
     "env": {"python": "...", "sktime": "...", "sklearn": "...", "os": "...", "cpu": "..."}
   }
   ```
   - 통과 기준 없음 (PRD §4-2 Non-Binding Metrics)
4. 모델 직렬화 — 센서별 3개 (`models/pilot_current_u.joblib`, `models/pilot_vib_motor.joblib`, `models/pilot_vib_tm.joblib`, joblib `compress=('zlib', 6)`):
   - 각 파일: `{"transformer": MiniRocket, "classifier": RidgeClassifierCV, "meta": {...}}` dict 구조
5. **`registry_manifest.json` 스키마 사전 고정** (본 데이터 단계 호환, `models` 배열 길이 **3**):
   ```json
   {
     "schema_version": "1.0",
     "stage": "pilot",
     "models": [
       {
         "model_id": "pilot_current_u",
         "vehicle": "ALL",
         "sensor": "Current_U",
         "column": "rms",
         "model_path": "models/pilot_current_u.joblib",
         "sha256": "...",
         "disk_mb": 0.0,
         "num_kernels": 10000,
         "random_state": 42,
         "trained_at": "2026-05-2x",
         "metrics": {"val_acc": 0.0, "val_macro_f1": 0.0}
       },
       {
         "model_id": "pilot_vib_motor",
         "vehicle": "ALL",
         "sensor": "Vib_Motor",
         "column": "rms",
         "model_path": "models/pilot_vib_motor.joblib",
         "sha256": "...",
         "disk_mb": 0.0,
         "num_kernels": 10000,
         "random_state": 42,
         "trained_at": "2026-05-2x",
         "metrics": {"val_acc": 0.0, "val_macro_f1": 0.0}
       },
       {
         "model_id": "pilot_vib_tm",
         "vehicle": "ALL",
         "sensor": "Vib_TM",
         "column": "rms",
         "model_path": "models/pilot_vib_tm.joblib",
         "sha256": "...",
         "disk_mb": 0.0,
         "num_kernels": 10000,
         "random_state": 42,
         "trained_at": "2026-05-2x",
         "metrics": {"val_acc": 0.0, "val_macro_f1": 0.0}
       }
     ]
   }
   ```
   - 본 데이터 단계는 `models` 배열 길이만 3 → 9로 확장 (vehicle outer loop 1개 추가, vehicle별·sensor별 모델 9개).
6. **`config.json` 스키마 사전 고정** (본 데이터 단계 호환, `model_assignments` 키 **3개**):
   ```json
   {
     "schema_version": "1.0",
     "stage": "pilot",
     "seed": 42,
     "chunk_size": 2000,
     "default_column": "rms",
     "model_assignments": {
       "ALL-Current_U": "pilot_current_u",
       "ALL-Vib_Motor": "pilot_vib_motor",
       "ALL-Vib_TM":    "pilot_vib_tm"
     }
   }
   ```
   - 본 데이터 단계는 `model_assignments` 키 3개 → 9개 `{vehicle}-{sensor}` 매핑 확장.
7. 모델 재로드 검증 — 신규 Python 세션에서 3개 joblib 모두 `joblib.load(...)` → 동일 입력 → 동일 라벨/decision_function 100% 일치 (PSC-02 충족, 센서별).
8. 단위 테스트 `tests/test_pilot_train.py`:
   - 학습 함수 호출 → 산출물(센서별 joblib 3개 + `pilot_metrics.json` + `registry_manifest.json`) 생성 확인
   - 동일 seed 2회 학습 시 센서별 val 라벨 100% 일치 (NFR-02, 3회)
   - 저장/로드 후 predict 동일성 (PSC-02, 3회)

### Deliverables
- `src/train/pilot_train.py`
- `scripts/run_pilot_train.py` — CLI 진입점 (센서 루프 실행)
- `tests/test_pilot_train.py`
- `models/pilot_current_u.joblib`, `models/pilot_vib_motor.joblib`, `models/pilot_vib_tm.joblib`
- `artifacts/pilot_metrics.json` (센서별 dict)
- `artifacts/registry_manifest.json` (스키마 v1.0, `models` 배열 길이 3)
- `artifacts/config.json` (스키마 v1.0, `model_assignments` 키 3개)

### Acceptance Criteria
- pytest `tests/test_pilot_train.py` 전체 PASS (3센서 케이스)
- `pilot_metrics.json`의 센서별 dict 모든 필드 numeric 출력 (값 자체는 평가하지 않음)
- 모델 저장/로드 후 동일 입력 → 동일 출력 100% 일치 (PSC-02, 센서별 3회)
- `registry_manifest.json`(`models` 길이=3)·`config.json`(`model_assignments` 키=3)이 위 정의된 스키마 v1.0과 일치 (PSC-06)
- 동일 seed 2회 학습 시 센서별 val 라벨 100% 일치 (NFR-02)

### 검증 항목 (V접두사 ↔ PSC 매핑)
- **V4-1** ↔ **PSC-02**: joblib 저장/로드 round-trip 동일 출력 (3센서)
- **V4-2** ↔ **PSC-03**: `requirements.txt` clean install 후 추론 성공 (Phase 5와 공유 검증)
- **V4-3** ↔ **PSC-06**: `registry_manifest.json`(3-entry) / `config.json`(3-key) 스키마 v1.0 준수
- **V4-4** ↔ **NFR-02**: 동일 seed 재학습 시 센서별 predict 100% 일치

### 의존성
- Phase 3 (`splits_current_u.json` / `splits_vib_motor.json` / `splits_vib_tm.json`)

### 예상 소요
6h (학습 코드+센서 루프 2.5h + 직렬화·스키마 1h + 테스트 1.5h + 환경 메타 수집 0.5h + 검증 0.5h)

### 본 데이터 단계 이전 항목
- (차종 × 센서) 9개 모델 분리 학습 → 본 PLAN은 센서별 3개 모델까지만 (파일럿 3개 → 본 데이터 9개로 vehicle outer loop 1개만 추가)
- 4개 컬럼 × 9슬롯 = 36-grid search → 본 PLAN은 컬럼 1개(`rms`) 고정
- multivariate (3 센서 동시 입력) 학습 → 본 PLAN은 센서별 독립 univariate 모델 3개
- 모델 카드 9개 작성 → 본 PLAN은 센서별 모델 카드 3개만 (Phase 6에서)
- 정량 성능 목표 (Acc ≥ 0.90, Macro-F1 ≥ 0.85, 결함 Recall ≥ 0.85) 활성화

### 리스크/블로커
- 센서별 모델이 차종 간 신호 차이를 흡수 못해 특정 센서 모델의 val 분류 성능이 매우 낮을 수 있음 → **PRD 명시대로 성능은 평가 대상이 아님** (PSC-01만 동작 충족하면 통과).
- 센서별 group_key 분포 불균형(REDUC=8 vs DEMAG=3 등) → 특정 센서 모델에서 학습/평가 분포 편향 가능. `class_weight='balanced'`로 1차 완화, 그래도 부족하면 본 데이터 단계로 이전 명시.

---

## Phase 5 — 추론 인터페이스 (sensor key dispatcher) + FastAPI/Vite 대시보드 연동 (신규)

### 목표
파일럿 모델 3개(센서별)를 ModelRegistry 3-entry로 등록하고 **sensor key 기반 dispatcher** 패턴으로 라우팅한다. FastAPI 백엔드를 통해 단일 CSV 추론 결과를 Vite + TypeScript + Tailwind 대시보드에 1건 이상 표시한다(사용된 model_key/sensor 식별 가능). 추론 latency를 numeric으로 측정 가능한 인터페이스를 제공하되 수치 목표값은 부여하지 않는다. **PSC-01·03·04를 본 Phase에서 충족**한다.

### 작업
1. `src/serve/model_registry.py` 신규 (파일럿 버전, sensor key dispatcher):
   ```python
   class ModelRegistry:
       def load(self, config_path: str) -> None: ...
       def get(self, sensor: str) -> object: ...
           # 파일럿: registry["Current_U"|"Vib_Motor"|"Vib_TM"] → 해당 센서 모델 dict
           # 미존재 키 → KeyError
       def list_models(self) -> list[dict]: ...
   ```
   - PRD §5 표기 그대로의 dispatcher: `predict(signal, sensor_name) → registry[sensor_name].predict(signal)`
   - 본 데이터 단계는 `(vehicle, sensor)` tuple key 매핑으로 확장 (인터페이스 동일, key shape만 확장).
2. `src/serve/inference_router.py` 신규:
   ```python
   class InferenceRouter:
       def predict(self, csv_path: str) -> dict:
           # 1) CSV 경로 → 메타에서 sensor 추출 (path_indexer 규칙 재사용)
           # 2) registry.get(sensor) → dispatcher가 센서별 모델 반환
           # 3) CSVLoader → chunker → clean_signal → select_column → panel
           # 4) MiniRocket.transform → RidgeClassifierCV.predict + decision_function
           # 반환: {label, label_name, scores, model_id, sensor, latency_ms, n_chunks}
           #       model_id는 사용된 센서 모델 식별자 (pilot_current_u / pilot_vib_motor / pilot_vib_tm)
   ```
   - 워밍업 1회 (sklearn lazy init 제거)
3. `src/bench/latency.py` 신규:
   ```python
   def measure_inference_latency(router, samples: int = 50, warmup: int = 5) -> dict:
       # 반환: {mean_ms, p50_ms, p95_ms, p99_ms, n_samples, env: {...}}
   ```
   - **수치 목표값 없음** (PRD §4-2). numeric 출력만 확인하면 PSC-04 통과.
4. FastAPI 백엔드 `backend/app.py` (기존 스캐폴딩 활용 가정, 없으면 신규):
   - `GET /api/health` → `{"status": "ok"}`
   - `POST /api/predict` body `{"csv_path": "..."}` → `InferenceRouter.predict(...)` 결과 JSON 반환(`model_id`/`sensor` 포함)
   - `GET /api/model_meta` → `registry_manifest.json` 내용 반환 (3개 모델 entry 모두 포함) + latency 측정값
   - CORS 활성화 (frontend dev server 연결용)
   - 실행: `uvicorn backend.app:app --reload --port 8000`
5. Vite + TypeScript + Tailwind 프론트엔드 `frontend/` (기존 스캐폴딩 활용 가정):
   - 최소 화면 1개: CSV 경로 입력란 + "추론" 버튼 + 결과 카드
   - 결과 카드: 예측 라벨 / decision 점수 5개 / **사용된 model_key(sensor)** / model_id / latency_ms / n_chunks
   - 모델 메타 패널: seed, chunk_size, num_kernels + 센서별(3개) val_acc/val_macro_f1 요약 테이블
6. **end-to-end 단일 명령 검증** `scripts/run_pilot_e2e.py`:
   - 데이터 로드 → 전처리 → 학습(센서별 3개) → 추론(센서별 분기) → 대시보드용 JSON 출력까지 단일 명령으로 실행
   - PSC-01 통과 증빙용
7. `requirements.txt` 갱신 + lock — `pip install -r requirements.txt` clean install 검증 (PSC-03):
   - sktime, scikit-learn, fastapi, uvicorn, pandas, numpy, joblib 버전 마이너 자리까지 고정
8. 단위 테스트 `tests/test_registry.py`, `tests/test_router.py`:
   - registry 3-entry 로드 성공 (Current_U / Vib_Motor / Vib_TM 키 모두 존재)
   - dispatcher가 sensor key별 올바른 모델 선택 검증 (각 키 조회 시 해당 `model_id` 매칭)
   - 미존재 sensor 키 조회 시 `KeyError` 발생
   - router `predict` 1회 호출 시 반환 dict 키 7개 확인 (label, label_name, scores, model_id, sensor, latency_ms, n_chunks)
   - latency 측정 함수가 mean/p50/p95/p99 4개 키를 numeric으로 반환

### Deliverables
- `src/serve/model_registry.py`
- `src/serve/inference_router.py`
- `src/bench/latency.py`
- `backend/app.py` (또는 기존 스캐폴딩 확장)
- `frontend/` 최소 화면 1개 (기존 스캐폴딩 확장, model_key/sensor 표시)
- `scripts/run_pilot_e2e.py`
- `tests/test_registry.py`, `tests/test_router.py`
- `requirements.txt` (lock)
- `artifacts/latency_results.json` — mean/p50/p95/p99 + env 메타

### Acceptance Criteria
- `python scripts/run_pilot_e2e.py` 단일 명령으로 예외 없이 완료 (PSC-01)
- 브라우저에서 frontend 접속 → "추론" 버튼 클릭 → 결과 카드 1건 이상 표시 (사용된 model_key/sensor 표시 포함, PSC-01)
- 신규 Python 세션에서 `joblib.load` → 동일 입력 → 동일 출력 (PSC-02, Phase 4와 공유, 센서별)
- `pip install -r requirements.txt` clean install 후 `scripts/run_pilot_e2e.py` 성공 (PSC-03)
- `latency_results.json`의 mean/p50/p95/p99이 numeric 출력 + env 메타(OS, CPU, Python, sktime 버전) 동반 (PSC-04)
- pytest `tests/test_registry.py`, `tests/test_router.py` 전체 PASS (3-entry / dispatcher / KeyError 케이스 포함)

### 검증 항목 (V접두사 ↔ PSC 매핑)
- **V5-1** ↔ **PSC-01**: end-to-end 단일 명령 + 대시보드 1건 표시 (model_key/sensor 식별 가능)
- **V5-2** ↔ **PSC-03**: clean install 후 추론 성공
- **V5-3** ↔ **PSC-04**: latency mean/p50/p95/p99 numeric + env 메타 동반
- **V5-4** ↔ **PSC-06**: ModelRegistry가 `registry_manifest.json` 3-entry를 로드해 sensor key dispatcher로 정확히 라우팅

### 의존성
- Phase 4 (`models/pilot_{current_u,vib_motor,vib_tm}.joblib`, `registry_manifest.json` 3-entry, `config.json` 3-key)

### 예상 소요
8h (백엔드 2.5h + 프론트엔드 2.5h + 추론 라우터·레지스트리(3-entry/dispatcher) 1.5h + latency 측정 0.5h + e2e 검증 1h)

### 본 데이터 단계 이전 항목
- 9개 모델 라우팅 — tuple key `(vehicle, sensor)` 9개 매핑으로 확장 (파일럿 3 → 본 데이터 9, dispatcher 인터페이스는 동일하게 `registry[key].predict(signal)` 유지, key shape만 `sensor` → `(vehicle, sensor)` tuple로 확장)
- 정량 NFR 검증 (≤ 100ms, ≤ 200MB) — 본 PLAN은 측정만, 통과/실패 판정 없음
- 시각화 위젯 확장 (confusion matrix, per-class recall 차트 등) — 본 PLAN은 라벨/점수/메타 카드 1개만
- 대시보드 인증/권한 — 본 PLAN은 단일 dev 환경 가정

### 리스크/블로커
- 기존 FastAPI/Vite 스캐폴딩 부재 시 환경 셋업 추가 시간 발생 → 8h 예상에 1h 버퍼 포함
- 추론 latency가 매우 큰 값으로 나와도 **PRD §4-2에 따라 PSC-04는 numeric 출력만 요구**하므로 통과 가능
- CSV 경로에서 sensor 메타 추출 실패 시 dispatcher가 KeyError를 던지므로 path_indexer 규칙(센서 디렉터리명) 재사용으로 사전 정규화 필요

---

## Phase 6 — 파일럿 결과 정리 + 본 데이터 단계 전환 체크리스트 (OPTIONAL)

### 목표
파일럿 단계 종료를 선언하고, PRD §7 "본 데이터 단계 전환 조건"을 체크리스트로 정리한다. 학기말 발표 자료의 핵심 서사 ("feasibility 검증 완료 + 본 데이터 단계 계획")를 확보한다.

### 작업
1. **센서별 모델 카드 3개** `docs/model_cards/pilot_{sensor}.md` × 3 (`pilot_current_u.md` / `pilot_vib_motor.md` / `pilot_vib_tm.md`):
   - 학습 데이터 분포 (sample_index 기준 센서별 부모 CSV/chunk 수, 센서별 group_key 분포)
   - hyperparameters (num_kernels, alphas, class_weight, seed)
   - 성능 (val_acc, val_macro_f1, per-class recall) — non-binding으로 표기
   - 알려진 한계 (차종 합산·단일 컬럼·hold-out 1회)
   - 본 데이터 단계 권고 (vehicle outer loop 추가 → 9-모델 분리, 36-grid, StratifiedGroupKFold)
2. **파일럿 결과 보고서** `docs/pilot_report.md`:
   - PSC-01~06 통과 증빙표 (각 항목별 PASS + 근거 산출물 경로)
   - **센서별 metric 표 3개** (Current_U / Vib_Motor / Vib_TM 각각 Acc, Macro-F1, per-class recall, latency, disk_size)
   - 본 데이터 단계 전환 체크리스트 (PRD §7 항목 4개 + 활성화될 변경사항 6개)
3. **인터페이스 스키마 명세** `docs/config_schema.md`:
   - `splits_{sensor}.json` v1.0 × 3 (Phase 3에서 고정)
   - `config.json` v1.0 (Phase 4에서 고정, `model_assignments` 키 3개)
   - `registry_manifest.json` v1.0 (Phase 4에서 고정, `models` 배열 길이 3)
   - JSON Schema 형식으로 명세
4. **README** 업데이트 (프로젝트 루트):
   - 설치·실행 명령 (`scripts/run_pilot_e2e.py`)
   - 디렉터리 트리
   - 산출물 위치 표 (센서별 모델 3개 명시)
   - 본 데이터 단계 전환 시 변경 지점 요약
5. **발표 슬라이드 초안** `docs/presentation_outline.md`:
   - 5분 발표 흐름: 문제 → 파일럿 결정 근거(센서 분리/차종 비분리) → end-to-end 데모(dispatcher) → 본 데이터 단계 계획

### Deliverables
- `docs/model_cards/pilot_current_u.md`, `docs/model_cards/pilot_vib_motor.md`, `docs/model_cards/pilot_vib_tm.md`
- `docs/pilot_report.md`
- `docs/config_schema.md`
- `README.md` (프로젝트 루트, 갱신)
- `docs/presentation_outline.md`

### Acceptance Criteria
- PSC-01~06 6개 항목 모두 PASS 증빙이 `docs/pilot_report.md`에서 추적 가능
- README만 보고 신규 작업자가 30분 내 `scripts/run_pilot_e2e.py` 재현 가능 (자체 dry-run)
- 본 데이터 단계 전환 체크리스트 4개 항목 모두 명시

### 검증 항목 (V접두사 ↔ PSC 매핑)
- **V6-1** ↔ **PSC-01~06 종합**: 6개 항목 통과 증빙표 완성
- **V6-2**: 인터페이스 스키마 명세가 Phase 3·4에서 고정한 형식과 100% 일치

### 의존성
- Phase 3, 4, 5 모두 완료

### 예상 소요
4h (모델 카드 1h + 결과 보고서 1.5h + 스키마 명세 0.5h + README 0.5h + 발표 슬라이드 0.5h)

### 본 데이터 단계 이전 항목
- 모델 카드 9개 작성 → 본 PLAN은 센서별 카드 3개만 (vehicle outer loop 도입 시 9개로 확장)
- v2.0 PNG 파이프라인과의 정량 벤치마크 보고서 → 본 데이터 단계로 이전

### 리스크/블로커
- 본 Phase는 OPTIONAL이지만, 학기말 발표 자료 확보 및 본 데이터 단계 진입 게이트로 작용하므로 일정 가능 시 수행 권장

---

## 진행 상황 보드 (체크박스)

- [x] Phase 0 — 환경 셋업 (인계)
- [x] Phase 1 — 데이터 인덱싱 + CSVLoader (인계)
- [x] Phase 2 — 전처리 모듈 (인계, pytest 19 PASS)
- [ ] Phase 3 — GroupShuffleSplit 센서별(3) 단일 hold-out
- [ ] Phase 4 — MiniRocket + Ridge 센서별 모델 3개 학습
- [ ] Phase 5 — sensor key dispatcher 추론 + 대시보드 연동 (PSC-01·03·04 충족)
- [ ] Phase 6 — 결과 정리(센서별 카드 3개) + 본 데이터 단계 전환 체크리스트 (OPTIONAL)

---

## 핵심 인터페이스 스키마 (사전 고정 요약)

본 PLAN의 핵심 결정은 **3개 인터페이스 스키마를 본 데이터 단계와 호환 형식으로 사전 고정**해 코드 변경을 최소화하는 것이다.

| 파일 | 스키마 v1.0 정의 위치 | 파일럿 → 본 데이터 변경 지점 |
|---|---|---|
| `artifacts/splits_{sensor}.json` × 3 | Phase 3 작업 3 | 파일럿 3개(센서별) → 본 데이터 9개(`(vehicle, sensor)` 조합). 각 파일의 `folds` 배열 길이 1 → 3+ (`StratifiedGroupKFold`로 알고리즘만 교체) |
| `artifacts/registry_manifest.json` | Phase 4 작업 5 | `models` 배열 길이 **3**(파일럿) → 9(본 데이터, vehicle별·sensor별 모델) |
| `artifacts/config.json` | Phase 4 작업 6 | `model_assignments` 키 **3개**(파일럿, `ALL-{sensor}`) → 9개(본 데이터, `{vehicle}-{sensor}` 매핑) |

---

## 리스크/블로커 종합

| 리스크 | 영향 Phase | 대응 |
|---|---|---|
| 센서별 group_key 분포 불균형(예: REDUC=8 vs DEMAG=3) → 특정 센서에서 5 클래스 중 한쪽 쏠림 | Phase 3 | seed 회전(최대 10회) → 그래도 실패 시 PRD §2-3에 따라 "동작만 보장" 정책 적용, 사유 로깅 후 PSC-01 부분 통과 처리 |
| 누수 차단 메커니즘의 본격 검증 한계 (group 수 자체가 적어 통계적 검증 불가) | Phase 3 | **본 데이터 단계로 이전 명시** — 파일럿은 코드 경로 동작만 증빙 (센서별 3회 검증) |
| 센서별 모델 분류 성능 매우 낮음 (특히 차종 간 신호 차이 미흡수) | Phase 4 | PRD에 따라 성능은 평가 대상이 아님 (PSC-01만 동작 충족) |
| 기존 FastAPI/Vite 스캐폴딩 부재 | Phase 5 | 8h 예상에 1h 버퍼 포함, 최소 화면만 구현 (model_key/sensor 표시) |
| dispatcher의 미존재 sensor key 처리 누락 시 런타임 오류 | Phase 5 | 단위 테스트로 KeyError 케이스 명시적 검증, path_indexer 규칙 재사용한 sensor 정규화 |
| v2.0 PNG 파이프라인 splits 공유 협의 | (보류) | 본 데이터 단계로 이전 — 파일럿은 동시 진행 안 함 |
| 데드라인 D-27 제약 | Phase 3·4·5 | Phase 3·4는 1주 내, Phase 5는 2주 내 완료 목표. Phase 6는 OPTIONAL |

---

## 변경 이력

| 버전 | 날짜 | 내용 |
|---|---|---|
| v1.0 | 2026-05-17 | 최초 작성 — PRD `prd_260517_모터감속기_파일럿_feasibility.md` (v1.1) 기반 파일럿 단계 7개 Phase 도출 (DONE 3 + TODO 4). 핵심 변경점: (1) 직전 PLAN의 11개 Phase 구조(9슬롯 × 36-grid, StratifiedGroupKFold n_splits=3, v2.0 PNG 벤치마크, 엣지 NFR 정량 검증)를 모두 본 데이터 단계로 이전, (2) Phase 0/1/2는 기존 산출물 인계 명시 (재구현 금지), (3) Phase 3 신규 — `GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` + seed 회전 최대 10회 + 누수 차단 assertion, (4) Phase 4 신규 — 단일 통합 모델 1개(`Current_U` × `rms`) MiniRocket + RidgeClassifierCV(`class_weight='balanced'`) 학습, joblib `compress=('zlib', 6)` 직렬화, (5) Phase 5 신규 — ModelRegistry 1-entry + InferenceRouter + FastAPI 백엔드 + Vite/TS/Tailwind 최소 화면, PSC-01·03·04 본 Phase에서 충족, latency 측정은 numeric 출력만 요구(수치 목표 없음), (6) Phase 6 OPTIONAL — 통합 모델 카드 1개 + 파일럿 결과 보고서 + 본 데이터 단계 전환 체크리스트, (7) 3개 인터페이스 스키마(`splits.json`/`config.json`/`registry_manifest.json`) 모두 v1.0으로 본 데이터 단계 호환 형식 사전 고정, (8) 각 Phase에 "본 데이터 단계 이전 항목" 섹션 신설하여 OOS 명시, (9) 진행 상황 보드 갱신 (Phase 0~2 체크 완료), (10) 리스크/블로커 종합 표 추가. 직전 PLAN `plan_260516_모터감속기_시계열_miniROCKET_분류기.md`는 삭제하지 않고 deprecation 노트만 추가하여 본 데이터 단계 복귀 자산으로 보존. |
| v1.1 | 2026-05-17 | PRD v1.2(센서별 모델 3개 분리·차종 비분리) 반영 — (1) 헤더 근거 PRD v1.1→v1.2, §0 핵심 원칙 모델 구성 갱신, (2) Phase 3 splits_{sensor}.json 3개로 분리, sha256 3개, 누수 차단/사후 검증 센서별 적용, 예상 3h→3.5h, (3) Phase 4 함수 train_pilot_per_sensor로 변경, 센서 outer 루프 추가, 모델 3개(models/pilot_{sensor}.joblib) 저장, pilot_metrics.json 센서별 dict, registry_manifest.json models 배열 길이 1→3, config.json model_assignments 키 1→3, 예상 5h→6h, (4) Phase 5 sensor key dispatcher 패턴 명시(predict(signal, sensor_name)→registry[sensor_name].predict(signal)), ModelRegistry 3-entry 로드/조회 테스트, 본 데이터 단계 (vehicle, sensor) tuple 키 확장 경로 명시, (5) Phase 6 모델 카드 1개→3개(docs/model_cards/pilot_{sensor}.md), (6) 인터페이스 스키마 요약 표 갱신, (7) 일정 요약 표/진행 보드/리스크 종합 표 갱신. PSC-01~06 통과 기준·정량 성능 목표 미부여·NFR 정량 본 데이터 단계 이전 원칙은 v1.0 그대로 유지. |
