# 환경 셋업 가이드 (Phase 0)

> 근거: PLAN Phase 0 / PRD NFR-03 (재현성)

## 1. Python 권장 버전

- **3.10** 또는 **3.11** (sktime 0.26 호환 범위)
- 3.12 는 numba/일부 sktime 의존성 빌드 이슈 가능 — 권장 X

## 2. 설치

로컬:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r src/requirements.txt
```

Colab:

```python
!pip install -q -r src/requirements.txt
# 설치 후 런타임 재시작 권장 (numpy 1.26 강제 교체로 인한 ABI 충돌 방지)
```

## 3. 의존성 정책

마이너 버전까지 고정한다. 메이저 변경 시 `from sktime.transformations.panel.rocket import MiniRocket`
경로가 깨질 수 있으므로 업그레이드 전 smoke test 필수.

| 패키지 | 핀 | 사유 |
|---|---|---|
| sktime | 0.26.* | MiniRocket import 경로 안정 구간 |
| scikit-learn | 1.4.* | RidgeClassifierCV API 안정 |
| pandas | 2.2.* | parquet I/O |
| numpy | 1.26.* | sktime 0.26 + 일부 numba 의존성과 호환 |
| joblib | 1.4.* | 모델 직렬화 (compress=('zlib', 6)) |

## 4. Smoke Test 실행

```bash
jupyter notebook notebooks/00_smoke_test.ipynb
```

또는 Colab 에서 동일 노트북 업로드 후 위에서부터 순차 실행.

검증 통과 조건:

- MiniRocket().fit_transform 결과 shape `(2, ~9996)` (커널 수 약 10,000 근방)
- `set_global_seed(42)` 두 번 호출 후 `np.random.rand()` 동일 값
- INFO / WARN 로그가 각각 콘솔과 `logs/smoke.log` 에 적절히 출력
