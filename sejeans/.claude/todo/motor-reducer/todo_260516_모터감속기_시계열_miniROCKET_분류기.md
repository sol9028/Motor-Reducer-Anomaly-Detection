# TODO: 모터 감속기 시계열 MiniROCKET 분류 파이프라인

> 근거 PLAN: `.claude/plan/motor-reducer/plan_260516_모터감속기_시계열_miniROCKET_분류기.md` (Rev. 2026-05-16-02)
> 작성일: 2026-05-16 | 마지막 갱신: 2026-05-16 (Rev. 02 동기화) | 데드라인: 2026-06-13
> 담당: 박세진(C321027)
> 총 예상 소요: 55h (4주)

---

## 상태 마커
- `[ ]` 미시작 / `[-]` 진행 중 / `[x]` 완료

---

## Immediate Tasks (최우선 3개) — Rev. 02 반영

1. [x] **Phase 2-0 (신규)**: 81개 CSV 행 수 sweep & chunk 수 집계 → `artifacts/csv_row_sweep.csv` 생성, PLAN `<TBD>` 자리 갱신 — 2026-05-17 완료. 분포 `{2000:21, 4000:24, 6000:9, 8000:6, 10000:21}`, 총 chunk 225 (9슬롯 × 5클래스 × 5 chunk 완전 균형). PLAN "데이터 양상" 섹션 갱신.
2. [x] **Phase 2-X (신규)**: `src/preprocess/chunker.py` 구현 — `split_into_chunks(signal_df, chunk_size=2000)` + group_key 상속 + chunk_id 부여 — 2026-05-17 완료. 모듈 + `src/preprocess/__init__.py` + `scripts/verify_chunker.py` (V2-1/V2-2 + 실측 CSV E2E) ALL PASS.
3. [x] **Phase 1-9 (신규)**: `csv_loader.py` shape assert 완화 (PLAN Rev. 02 반영, 예상 5분) — 2026-05-17 완료. `_EXPECTED_ROWS=10_000` → `_ALLOWED_ROW_COUNTS={2000,4000,6000,8000,10000}` frozenset 으로 교체, 허용 집합 이탈 시에만 WARN. `verify_phase1.py` V1-2 도 동기화. `scripts/verify_phase1.py` ALL PASS 재확인.

---

## Phase 0 — 환경 셋업 및 의존성 고정 (3h, 목표 05-17)

### 작업
- [x] 0-1. `src/requirements.txt` 작성 — sktime==0.26.*, scikit-learn==1.4.*, pandas==2.1.*, numpy==1.26.*, joblib==1.4.*, numba>=0.59,<0.61 (Python 3.11 venv 기준)
- [x] 0-2. MiniRocket import 경로 검증 — `from sktime.transformations.panel.rocket import MiniRocket` (notebooks/00_smoke_test.ipynb 통합)
- [x] 0-3. 공통 시드 모듈 `src/common/seed.py` — `set_global_seed(42)` (random/np/PYTHONHASHSEED)
- [x] 0-4. 로깅 설정 모듈 `src/common/logger.py` — INFO/WARN 분기 + 파일 로테이션
- [x] 0-5. Colab/로컬 양쪽에서 smoke test (`MiniRocket().fit_transform(np.random.randn(2,1,2000))` — Rev. 02 반영, 기존 10000 입력도 재검증 완료)
- [x] 0-6. `notebooks/00_smoke_test.ipynb` 작성
- [x] 0-7. `docs/env_setup.md` 1페이지 작성

### 검증
- [x] V0-1. `pip install -r requirements.txt` clean install 성공 (2026-05-16, Python 3.11.0 venv `.venv/`)
- [x] V0-2. smoke test MiniRocket 변환 shape `(2, 9996)` 확인 (2026-05-16, sktime 0.26.1) [재검증 2026-05-16: chunk_size=2000 PASS, 입력 (2,1,2000) → output (2,9996) — Rev. 02 반영]
- [x] V0-3. `set_global_seed(42)` 2회 호출 후 `np.random.rand()` 동일값 확인 (NFR-03)

### 환경 결정 사항 (2026-05-16)
- Python 3.11.0 venv `.venv/` 사용 (Python 3.14는 sktime 0.26.* 미지원)
- `pandas==2.2.*` → `pandas==2.1.*` 변경 (sktime 0.26.x가 `pandas<2.2.0` 요구)
- `numba>=0.59,<0.61` 추가 (MiniRocket soft dependency, 미설치 시 ModuleNotFoundError)
- 설치된 버전: sktime 0.26.1 / sklearn 1.4.2 / pandas 2.1.4 / numpy 1.26.4 / numba 0.60.0

---

## Phase 1 — 데이터 인덱싱 및 CSVLoader (5h, 목표 05-19)

### 작업
- [x] 1-1. `src/data/path_indexer.py` — `build_sample_index(root) -> pd.DataFrame` (csv_path, vehicle, fault_class, date, sensor, group_key)
- [x] 1-2. 폴더 트리 스캔 시 9개(차종×센서) × 5개(고장) = 45개 leaf 디렉터리 발견 확인
- [x] 1-3. `src/data/csv_loader.py` — `CSVLoader.load(path) -> (meta_df, signal_df)`
- [x] 1-4. signal_df 4개 컬럼(`peak_freq_bin, band_start, band_end, rms`) 분리 구현
- [x] 1-5. `src/data/label_map.py` — `LABEL_MAP = {NORMAL:0, ECC10:1, ECC20:2, DEMAG:3, REDUC:4}` 고정
- [x] 1-6. 인덱스 통계 출력 — (차종 × 센서 × 고장)별 샘플 수 분포 표 (`_print_distribution` + 노트북 pivot_table 3종)
- [x] 1-7. `artifacts/sample_index.parquet` 캐시 생성 (81행, parquet 라운드트립 OK)
- [x] 1-8. `notebooks/01_data_index_eda.ipynb` 작성 (분포 + 결측 폴더 점검, nbconvert 실행 통과)
- [x] 1-9. **(신규, Rev. 02 반영)** `csv_loader.py` shape assert 완화 — 기존 `shape == (10000, 4)` 강제를 `shape[1] == 4 and shape[0] in {2000,4000,6000,8000,10000}` 가변 길이 허용으로 변경. 2000행 chunk 분할 책임은 Phase 2 `chunker.py`로 이관. (2026-05-17 완료, `verify_phase1.py` ALL PASS)

### 검증
- [x] V1-1. `(vehicle, sensor, fault_class)` unique 조합 수 = 45 (2026-05-16, `scripts/verify_phase1.py` 및 노트북 셀에서 PASS)
- [x] V1-2. 임의 CSV 1개 `signal_df.shape[1] == 4 and shape[0] in {2000,4000,6000,8000,10000}` (Rev. 02 기준 완화 — csv_loader.py 패치 작업 필요, Step 1-9 참조), `meta_df`에 vehicle/sensor/fault_class/group_key/zsplit 포함 (2026-05-16, dtype=float32 동시 확인)
- [x] V1-3. 결측 폴더 발견 시 WARN 로그 + 인덱스 제외 동작 (2026-05-16, `__GHOST__` 가짜 fault_class → WARN 3줄 emit)

---

## Phase 2 — 전처리 모듈 (Chunking·Shape·NaN·Panel 변환) (4h, 목표 05-20) — Rev. 02 재편

### 작업
- [x] 2-0. **(신규, Rev. 02)** 81개 CSV 행 수 sweep & chunk 수 집계 — 전체 81 CSV를 순회해 행 수 분포 `{2000:?, 4000:?, 6000:?, 8000:?, 10000:?}` 산출 + chunk 분할 시 총 샘플 수 및 (vehicle, sensor, fault_class) 슬라이스별 chunk 수 표 생성. `artifacts/csv_row_sweep.csv` 저장, PLAN "데이터 양상" 섹션 `<TBD>` 갱신. (2026-05-17 완료, `scripts/sweep_csv_rows.py`, 분포 `{2000:21, 4000:24, 6000:9, 8000:6, 10000:21}`, 총 chunk 225)
- [x] 2-X-1. **(신규, Rev. 02)** `src/preprocess/chunker.py` 구현 — `split_into_chunks(signal_df, chunk_size=2000) -> List[pd.DataFrame]` (2026-05-17 완료)
- [x] 2-X-2. **(신규)** chunker 가드: `assert len(signal_df) % chunk_size == 0` (배수 아니면 명확한 에러 메시지) (2026-05-17 완료, V2-2-chunker PASS)
- [x] 2-X-3. **(신규)** chunker — 부모 CSV의 `group_key`를 모든 chunk가 동일 상속 (메타 컬럼 보존) (2026-05-17 완료, `.attrs["group_key"]` 슬롯 사용)
- [x] 2-X-4. **(신규)** chunker — 각 chunk에 0-indexed `chunk_id` 부여 (2026-05-17 완료, `.attrs["chunk_id"]` 슬롯 사용)
- [x] 2-1. `src/preprocess/shape_guard.py` — `enforce_length(arr, target=2000)` (Rev. 02 반영, chunker 뒤단 sanity check 역할로 격하; pad/truncate + 카운터, 정상 시 no-op) — 2026-05-17 완료. `ShapeGuardStats` 싱글톤(pad/truncate/noop 카운터 + history) + WARN 로그 + dtype 보존.
- [x] 2-2. `src/preprocess/nan_handler.py` — `clean_signal(arr)` (선형 보간 → 실패 시 0 대체, 결측 비율 반환) — 2026-05-17 완료. `np.interp` 기반 선형 보간(가장자리 nearest-neighbor 확장) + 전 구간 NaN fallback(zeros, ratio=1.0) + `(cleaned, nan_ratio)` 튜플 반환.
- [x] 2-3. `src/preprocess/panel_formatter.py` — `to_sktime_panel(...)` shape `(N, 1, 2000)` (Rev. 02 반영, 기존 10000) — 2026-05-17 완료. DataFrame/1D ndarray 양쪽 입력 지원, 내부에서 `enforce_length` 호출 → `(N,1,target)` float32.
- [x] 2-4. 헬퍼 `select_column(signal_df, col_name)` 구현 — 2026-05-17 완료. `panel_formatter.py` 내부에 배치(panel 변환 직전 단일 채널 추출 관심사 일치). 4개 유효 컬럼 화이트리스트 검증 + 1D float32 ndarray 반환.
- [x] 2-5. `tests/test_preprocess.py` (pytest) — 짧음/긴/NaN + chunker 분할 4종 케이스 (Rev. 02 반영) — 2026-05-17 완료. 4종 코어 + panel_formatter 추가 검증 합쳐 **19 tests ALL PASS** (0.59s). `tests/conftest.py` 로 sys.path 부트스트랩, `_reset_shape_stats` autouse fixture 로 카운터 격리.

### 검증
- [x] V2-0. **(신규, Rev. 02)** 81 CSV sweep 결과의 chunk 총합이 (vehicle, sensor, fault_class) 슬라이스별 표로 정리됨 + PLAN 상단 `<TBD>` 갱신 + 모든 CSV 길이가 {2000,4000,6000,8000,10000} 집합 소속 — 2026-05-17 PASS, 81/81 CSV 모두 허용 집합 소속, 총 chunk 225, 9슬롯 모두 5클래스 × 5 chunk 균형
- [x] V2-1-chunker. **(신규, Rev. 02)** 10000행 CSV 1개 → 5개 chunk 반환, 모든 chunk가 동일 `group_key` 상속, `chunk_id`는 0~4 0-indexed (2026-05-17 PASS, `scripts/verify_chunker.py`)
- [x] V2-2-chunker. **(신규, Rev. 02)** 4000행 → 2 chunk, 2000행 → 1 chunk, 3000행(배수 아님) → `AssertionError` 발생 (2026-05-17 PASS, AssertionError 메시지 명시적 출력)
- [x] V2-1. 길이 1,500 입력(예외 케이스) → `enforce_length(arr, target=2000)` 출력 (2000,) + pad 카운터 +1 (FR-02, Rev. 02 반영) — 2026-05-17 PASS (`tests/test_preprocess.py::TestEnforceLengthShort::test_pad_short_input_to_target`)
- [x] V2-2. NaN 5% 입력 → 보간 후 NaN 0개, 결측 비율 0.05 로그 (FR-03) — 2026-05-17 PASS (`tests/test_preprocess.py::TestCleanSignalNaN::test_interpolate_5pct_nan`, ratio == 0.05 ± 1e-6)
- [x] V2-3. panel shape `(N, 1, 2000)`, dtype float32 (NFR-02, Rev. 02 반영) — 2026-05-17 PASS (`tests/test_preprocess.py::TestPanelFormatter::test_to_sktime_panel_from_dataframes`, panel.shape=(3,1,2000), dtype=float32)

---

## Phase 3 — Split 설계 (StratifiedGroupKFold + splits.json 공유) (4h, 목표 05-22)

### 작업
- [ ] 3-1. `src/split/group_split.py` — `make_splits(sample_index, n_splits=3, val_ratio=0.15, test_ratio=0.15, seed=42)` (Rev. 02 반영, 7 → 3)
- [ ] 3-2. 2단계 전략 구현 — `StratifiedGroupKFold(n_splits=3)`로 test fold 1개 분리 → 잔여에서 val 분리 (Rev. 02 반영). 근거: (vehicle, sensor, class)별 최소 group 수 3 (DEMAG, NORMAL) → 7-fold 불가
- [ ] 3-3. (차종 × 센서)별 독립 split — 9개 모델 슬라이스
- [ ] 3-4. `artifacts/splits.json` 저장 (스키마: `{model_key: {train:[...], val:[...], test:[...]}}`)
- [ ] 3-5. `artifacts/splits.sha256` 무결성 해시 생성
- [ ] 3-6. `docs/split_contract.md` — v2.0 파이프라인과의 공유 계약서
- [ ] 3-7. v2.0 팀과 splits.json 공유 협의
- [ ] 3-8. 누수 검사 `assert_no_group_leakage(splits)` 구현
- [ ] 3-9. `tests/test_split.py` — 누수·비율·재현성 테스트

### 검증
- [ ] V3-1. 9개 모델 splits에서 group_key 교집합 = ∅ (FR-12)
- [ ] V3-2. 동일 seed 2회 실행 시 splits.json sha256 동일 (NFR-03)
- [ ] V3-3. fault_class 비율 train/val/test에서 ±2%p 이내
- [ ] V3-4. v2.0 측 동일 splits.json 로드 시 sample 수 정확 일치

---

## Phase 4 — 단일 (차종 × 센서 × 컬럼) 학습 프로토타입 (5h, 목표 05-24)

### 작업
- [ ] 4-1. `src/train/single_run.py` — `train_single(vehicle, sensor, column, splits, seed=42) -> dict`
- [ ] 4-2. 절차 구현: split 로드 → CSV 로드 → **chunker로 2,000행 분할 (Rev. 02 신규)** → preprocess → panel `(N,1,2000)` → MiniRocket(num_kernels=10000) → RidgeClassifierCV(alphas=logspace(-3,3,10)). **`class_weight='balanced'` 권고 제거(Rev. 02)** — 기본값 `class_weight=None`, sweep 결과 미세 불균형 슬롯에서만 fallback 옵션으로 활성화
- [ ] 4-3. 학습 시간/메모리 프로파일링 (`time.perf_counter`, `tracemalloc`)
- [ ] 4-4. 결과 dict 스키마 정의 — `{val_acc, val_macro_f1, val_recall_per_class, train_time_sec, peak_mem_mb}`
- [ ] 4-5. `notebooks/04_prototype_single.ipynb` — 결과 시각화 (IONIQ × Current_U × rms)
- [ ] 4-6. `artifacts/prototype_metrics.json` baseline 저장
- [ ] 4-7. `docs/prototype_findings.md` 1페이지 요약

### 검증
- [ ] V4-1. val macro-F1 > 0.5 (랜덤 0.2 대비 유의)
- [ ] V4-2. 단일 학습 ≤ 10분 (T4 또는 로컬 CPU 기준)
- [ ] V4-3. 동일 seed 2회 실행 시 val metric 소수점 6자리 동일 (FR-10)

---

## Phase 5 — 36회 Grid Search 자동화 + best_column 선택 (8h, 목표 05-28) **[핵심]**

### 작업
- [ ] 5-1. `src/train/grid_search.py` — `class GridSearchRunner` 클래스 골격 작성
- [ ] 5-2. 입력 인터페이스 — sample_index, splits.json, columns=['peak_freq_bin','band_start','band_end','rms']
- [ ] 5-3. 외부 루프(9개 차종×센서) × 내부 루프(4개 컬럼) = 36회 구현
- [ ] 5-4. tqdm 진행률 표시 (total=36) + 중간 실패 시 재시도 1회 → `status=failed` 기록 후 계속 진행
- [ ] 5-5. **체크포인트 정책** — `artifacts/grid/{vehicle}_{sensor}_{column}/`에 model.joblib, metrics.json, predictions.parquet 저장
- [ ] 5-6. 재실행 시 완료된 셀 스킵 로직 구현
- [ ] 5-7. **병렬화** — MiniRocket `n_jobs=-1`, 외부 루프 직렬
- [ ] 5-8. **메모리 가드** — 슬라이스 학습 후 `del transformer, classifier; gc.collect()`
- [ ] 5-9. `select_best_column(grid_results) -> dict` — val_macro_f1 argmax (동률 우선순위 rms > peak_freq_bin > band_end > band_start)
- [ ] 5-10. `artifacts/config.json` 작성 (스키마: version, seed, models[model_key])
- [ ] 5-11. `artifacts/grid_summary.csv` 36행 통합 결과표 생성
- [ ] 5-12. `scripts/run_grid_search.py` CLI 진입점 (`--config configs/grid.yaml`)
- [ ] 5-13. `configs/grid.yaml` 작성 (컬럼 리스트·시드·체크포인트 경로)
- [ ] 5-14. `logs/grid_search_YYYYMMDD.log` 로깅 설정

### 검증
- [ ] V5-1. 완료된 학습 = 36건, grid_summary.csv 행 수 36 (FR-04)
- [ ] V5-2. 9개 (차종 × 센서) 각각 best_column 정확히 1개 선정 (FR-05)
- [ ] V5-3. 총 학습 시간 Colab T4 ≤ 6시간 (NFR-06)
- [ ] V5-4. 학습 중 OOM 0건
- [ ] V5-5. 체크포인트 재실행 테스트 — 절반 완료 상태 재실행 시 완료분 스킵 확인

---

## Phase 6 — ModelRegistry · InferenceRouter 구현 (5h, 목표 05-30)

### 작업
- [ ] 6-1. `src/serve/model_registry.py` — `class ModelRegistry` 골격
- [ ] 6-2. `load(config_path)` 메서드 — config.json 읽고 9개 모델 joblib 로드
- [ ] 6-3. `get(vehicle, sensor)` — `(transformer, classifier, best_column)` 반환, 미존재 시 KeyError (FR-13)
- [ ] 6-4. `size_mb()` 메서드 — 메모리 상주 모델 총 크기
- [ ] 6-5. **직렬화 정책** — joblib `compress=('zlib', 6)`, 각 모델 별도 파일 (총 9개)
- [ ] 6-6. `src/serve/inference_router.py` — `class InferenceRouter` 골격
- [ ] 6-7. `predict(csv_path) -> dict` — 로드→preprocess→best_column→MiniRocket→predict→`{label, label_name, scores, model_key, best_column, inference_time_ms}`
- [ ] 6-8. **워밍업** — Registry 로드 직후 더미 입력 1회 predict
- [ ] 6-9. `tests/test_registry.py` — 9개 키 조회 + 미존재 키 KeyError
- [ ] 6-10. `tests/test_router.py` — end-to-end 추론
- [ ] 6-11. `artifacts/registry_manifest.json` — 모델 파일 sha256 + 크기

### 검증
- [ ] V6-1. `len(registry) == 9`
- [ ] V6-2. 미존재 키 조회 시 KeyError 발생 (FR-13)
- [ ] V6-3. 단일 CSV → 라벨 반환 end-to-end 정상
- [ ] V6-4. 모델 9개 합산 디스크 크기 ≤ 200MB (NFR-02 1차 확인)

---

## Phase 7 — 평가 (9모델 metric · classification_report · CM) (4h, 목표 06-01)

### 작업
- [ ] 7-1. `src/eval/evaluate.py` — `evaluate_all_models(registry, splits) -> pd.DataFrame`
- [ ] 7-2. 모델별 metric — accuracy, macro-F1, weighted-F1, per-class precision/recall/f1, 5×5 confusion matrix
- [ ] 7-3. 결함 클래스 Recall 집계 — ECC10/ECC20/DEMAG/REDUC 4클래스 각각 ≥ 0.85 PASS/FAIL 컬럼
- [ ] 7-4. `notebooks/07_evaluation.ipynb` — 9개 CM heatmap (3×3 grid) + per-model bar chart
- [ ] 7-5. `eval_report.md` 자동 생성기 작성
- [ ] 7-6. `artifacts/eval/eval_results.csv` (9행, 각 metric + PASS/FAIL) 출력
- [ ] 7-7. `artifacts/eval/confusion_matrices.png` (3×3 그리드) 저장
- [ ] 7-8. `artifacts/eval/classification_reports/` 9개 txt 저장
- [ ] 7-9. `docs/eval_report.md` — PRD 성공 지표 연결표 포함

### 검증
- [ ] V7-1. 9개 모델 모두 test acc ≥ 0.90 (NFR-05)
- [ ] V7-2. 결함 4클래스 Recall ≥ 0.85 (NFR-04)
- [ ] V7-3. macro-F1 ≥ 0.85 (NFR-05)
- [ ] V7-4. 미달 케이스 발견 시 원인 가설 docs/eval_report.md에 기록

---

## Phase 8 — 엣지 NFR 검증 (≤100ms · ≤200MB) (4h, 목표 06-03) **[핵심]**

### 작업
- [ ] 8-1. `src/bench/latency.py` — `measure_inference_latency(router, samples=100) -> dict`
- [ ] 8-2. 워밍업 5회 → 본 측정 100회 → 평균/중앙값/p95/p99 산출
- [ ] 8-3. 9개 모델 각각 독립 측정 + 통합 평균
- [ ] 8-4. 측정 환경 CPU only (Colab CPU 런타임)
- [ ] 8-5. `src/bench/disk_size.py` — joblib 합산 + `pympler.asizeof` 메모리 상주 측정
- [ ] 8-6. **압축 옵션 비교** — `compress=0/3/6/9` 디스크 크기 vs 로드 시간 trade-off 표
- [ ] 8-7. fallback 계획 문서화 — 100ms 초과 시 `num_kernels=5000` / 200MB 초과 시 transformer sparse 직렬화
- [ ] 8-8. `scripts/run_bench.py` CLI 작성
- [ ] 8-9. `artifacts/bench/latency_results.csv` (모델별 mean/p50/p95/p99) 출력
- [ ] 8-10. `artifacts/bench/disk_results.csv` 출력
- [ ] 8-11. `docs/bench_report.md` — NFR PASS/FAIL + 환경 표

### 검증
- [ ] V8-1. 9개 모델 단일 추론 평균 ≤ 100ms (NFR-01)
- [ ] V8-2. 9개 모델 joblib 총 크기 ≤ 200MB (NFR-02)
- [ ] V8-3. p95 ≤ 150ms
- [ ] V8-4. 미달 시 fallback 적용 결과 문서화

---

## Phase 9 — v2.0 PNG 파이프라인 벤치마크 비교 (5h, 목표 06-07) **[핵심]**

### 작업
- [ ] 9-1. v2.0 팀과 splits.json 공유 및 sha256 확인
- [ ] 9-2. v2.0 측 평가 결과 CSV 수령 — `artifacts/v2_eval_results.csv` (동일 스키마)
- [ ] 9-3. `src/bench/compare.py` — 두 파이프라인 결과 join 후 비교표 생성
- [ ] 9-4. 비교표 컬럼 — model_key, v2_acc, mini_acc, Δacc, v2_macro_f1, mini_macro_f1, Δf1, v2_size_mb, mini_size_mb, v2_latency_ms, mini_latency_ms
- [ ] 9-5. (차종 × 센서)별 우세 모델 표시 (Δ > 0 → MiniROCKET 우세)
- [ ] 9-6. 9개 슬롯 운영 채택 권고 작성
- [ ] 9-7. 시각화 — 9개 모델 × 2 파이프라인 metric 막대 그래프
- [ ] 9-8. 시각화 — latency vs accuracy 산점도
- [ ] 9-9. `artifacts/bench/comparison_table.csv` (9행) 저장
- [ ] 9-10. `artifacts/bench/comparison_chart.png` 저장
- [ ] 9-11. `docs/benchmark.md` — 비교표 + 모델 선택 권고 (FR-14)
- [ ] 9-12. `docs/split_sha_log.md` — splits.json 해시 일치 증빙

### 검증
- [ ] V9-1. splits.json sha256 v2.0 측과 100% 일치
- [ ] V9-2. 9개 model_key 모두 양 파이프라인 join (누락 0)
- [ ] V9-3. 권고 결정 9개 슬롯 모두 명시
- [ ] V9-4. benchmark.md가 PRD §4 "v2.0 대비 벤치마크" 지표 충족

---

## Phase 10 — 문서화 (README · 모델 카드 · config 스키마) (3h, 목표 06-10)

### 작업
- [ ] 10-1. `README.md` 작성 — 설치·데이터 경로·학습 명령·추론 예제·디렉터리 트리·산출물 위치
- [ ] 10-2. 모델 카드 9개 작성 — `docs/model_cards/{vehicle}_{sensor}.md` (학습 분포, best_column, 시드, 성능, 한계, 결함 Recall)
- [ ] 10-3. `docs/config_schema.md` — config.json / splits.json / registry_manifest.json JSON Schema 명세
- [ ] 10-4. `docs/api_reference.md` — ModelRegistry · InferenceRouter 공개 API 시그니처
- [ ] 10-5. `docs/presentation_outline.md` — 5분 발표 흐름

### 검증
- [ ] V10-1. README만 보고 신규 작업자가 30분 내 학습/추론 명령 재현 가능 (dry-run)
- [ ] V10-2. 9개 모델 카드 누락 0
- [ ] V10-3. 모든 산출물 경로가 README에 링크됨
- [ ] V10-4. PRD §4 성공 지표 11개 항목이 docs 어딘가에서 모두 검증 가능 (추적표 포함)

---

## 예비 Phase — 발표 자료 · 버퍼 (5h, 목표 06-13)

- [ ] B-1. 발표 슬라이드 최종본 작성 (10장 내외)
- [ ] B-2. 데모 영상/스크립트 준비
- [ ] B-3. Q&A 예상 질문 정리
- [ ] B-4. 버퍼 시간 — 미해결 이슈 처리

---

## 진행 상황 보드 (Phase 단위)

- [x] Phase 0 — 환경 셋업
- [x] Phase 1 — CSVLoader
- [x] Phase 2 — 전처리 모듈
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
| v1.0 | 2026-05-16 | PLAN v1.0(2026-05-16) 기반 최초 작성 — Phase 0~10 + 예비 Phase, 총 작업 95개 + 검증 항목 35개 + Phase 보드 11개 |
| v1.1 | 2026-05-16 | Phase 1 완료 — 인덱스/로더 구축(`src/data/{path_indexer,csv_loader,label_map}.py`), `artifacts/sample_index.parquet` 캐시(81행, 45 leaf, group_key 27), `notebooks/01_data_index_eda.ipynb` nbconvert 실행 통과, `scripts/verify_phase1.py`로 V1-1/V1-2/V1-3 자동 PASS |
| v1.2 | 2026-05-16 | PLAN Rev. 2026-05-16-02 동기화 — (1) 윈도우 10000 → 2000 전역 갱신(Phase 0/2/4 description), (2) Phase 2-0(81 CSV sweep) 및 2-X-1~4(chunker.py) 신규 작업 5건 추가, (3) Phase 1-9 csv_loader shape assert 완화 신규 작업 추가, (4) V0-2 chunk_size=2000 재검증 비고 추가, (5) V1-2 가변 길이 허용 기준 완화, (6) Phase 2 V2-0/V2-1-chunker/V2-2-chunker 신규 검증 3건 추가 + 기존 V2-1/V2-3 target 2000으로 갱신, (7) Phase 3 n_splits 7 → 3 갱신 + 근거 명시, (8) Phase 4 class_weight='balanced' fallback 옵션화 + chunker 절차 삽입, (9) Immediate Tasks를 Rev. 02 신규 항목 3개로 교체. 신규 작업 8건 / 신규 검증 3건. |
| v1.3 | 2026-05-17 | Phase 2-1 ~ 2-5 완료 — `src/preprocess/shape_guard.py`(`enforce_length` + `ShapeGuardStats` 카운터), `src/preprocess/nan_handler.py`(`clean_signal` 선형 보간 + ratio), `src/preprocess/panel_formatter.py`(`to_sktime_panel` + `select_column` 헬퍼) 신규 + `__init__.py` 재export. `tests/test_preprocess.py` 19 cases ALL PASS (0.59s, pytest 8.4.2). V2-1/V2-2/V2-3 모두 PASS. Phase 보드 Phase 2 완료 마킹. requirements.txt 에 `pytest>=8.0,<9.0` 추가. |
