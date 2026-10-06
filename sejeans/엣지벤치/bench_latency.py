"""라즈베리파이(엣지) 추론 지연시간 벤치마크 — MiniRocket + LogisticRegression.

이 파일은 **단독 실행**을 전제로 한다. dashboard_streamlit/src 패키지에 의존하지
않으므로 이 파일과 .joblib 만 파이로 복사하면 돌아간다.

지원 번들 두 종류를 자동 판별한다:
  1) 파일럿  : dashboard_streamlit/models/pilot_*.joblib
               {sensor, minirocket, classifier, label_map, meta}   1채널 x 2000
  2) v3 융합 : 본격/모터_학습_v3.py 가 저장한 model_registry_*.joblib
               {(차종, "FUSED15"): {ch_mean, ch_std, mr, sc, lr}}  15채널 x 2000

실행:
  python bench_latency.py --models ./models                    # 1스레드
  python bench_latency.py --models ./models --threads 4        # 4코어
  python bench_latency.py --models ./models --batch 1,8,32 --reps 30

측정값의 의미
-------------
- MiniRocket 은 numba JIT 이다. **첫 호출에 커널 컴파일 비용**(파이4에서 수십 초)이
  한 번 붙는다. 이걸 지연시간에 넣으면 완전히 틀린 숫자가 나오므로 warmup 후에만 잰다.
  대신 컴파일 시간은 `jit_warmup_s` 로 따로 보고한다 — 실차 배포에서는
  '시동 후 첫 진단까지의 대기시간' 이 되므로 버릴 숫자가 아니다.
- 평균만 보면 안 된다. 실시간 판정은 꼬리지연(p95/p99)이 마감을 넘기는지가 기준이다.
- 파이4는 발열로 클럭이 떨어진다. 시작/종료 시점의 온도·스로틀 플래그를 함께 남긴다.
"""
from __future__ import annotations

import os
import sys

# ---- numpy/numba import 전에 스레드 수를 고정해야 한다 (import 후엔 안 먹는다) ----
def _early_thread_arg() -> int:
    for i, a in enumerate(sys.argv):
        if a == "--threads" and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return 1


_THREADS = _early_thread_arg()
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = str(_THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402


# ------------------------------------------------------------------ 환경 정보
def pi_thermal() -> dict:
    """파이의 온도/스로틀 상태. 파이가 아니면 빈 dict."""
    out = {}
    try:
        t = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True,
                           text=True, timeout=5).stdout.strip()
        out["temp"] = t.replace("temp=", "")
    except Exception:
        pass
    try:
        g = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                           text=True, timeout=5).stdout.strip()
        out["throttled"] = g.replace("throttled=", "")
        # 0x0 이 아니면 전압강하/과열로 클럭이 깎였다는 뜻 -> 측정치 신뢰도 하락
    except Exception:
        pass
    try:
        f = subprocess.run(["vcgencmd", "measure_clock", "arm"], capture_output=True,
                           text=True, timeout=5).stdout.strip()
        hz = int(f.split("=")[1])
        out["arm_mhz"] = round(hz / 1e6)
    except Exception:
        pass
    return out


def rss_mb() -> float:
    """현재 프로세스 상주 메모리(MB). 실패하면 -1."""
    try:  # Linux
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    try:  # Windows fallback
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        return -1.0


def env_info(threads: int) -> dict:
    import sklearn
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "threads_requested": threads,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    for mod in ("sktime", "numba", "joblib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as e:
            info[mod] = f"<{e}>"
    try:
        with open("/proc/device-tree/model", encoding="utf-8") as f:
            info["board"] = f.read().strip("\x00").strip()
    except Exception:
        pass
    if info["machine"] not in ("x86_64", "AMD64", "aarch64", "arm64"):
        info["WARNING"] = (
            f"machine={info['machine']} — 32비트 ARM 으로 보인다. numba/llvmlite "
            "휠이 없어 정상 측정이 어렵다. 64비트 Raspberry Pi OS 를 쓸 것."
        )
    return info


# ------------------------------------------------------------------ 번들 적재
class Model:
    """번들 두 형태를 같은 인터페이스로 감싼다."""

    def __init__(self, name: str, mr, clf, sc=None, ch_mean=None, ch_std=None):
        self.name = name
        self.mr = mr
        self.clf = clf
        self.sc = sc
        self.ch_mean = ch_mean
        self.ch_std = ch_std
        self.n_ch = int(len(ch_mean)) if ch_mean is not None else 1

    def set_threads(self, n: int) -> None:
        # sktime MiniRocket 은 transform 시점에 self.n_jobs 를 읽는다.
        # 학습 때 -1(전 코어)로 저장돼 있으므로 여기서 덮어써야 스레드 비교가 성립한다.
        if hasattr(self.mr, "n_jobs"):
            self.mr.n_jobs = n

    # --- 단계별로 나눠 재기 위해 분리 ---
    def pre(self, X: np.ndarray) -> np.ndarray:
        if self.ch_mean is None:
            return np.nan_to_num(X, nan=0.0).astype(np.float32, copy=False)
        return ((X - self.ch_mean[None, :, None])
                / self.ch_std[None, :, None]).astype(np.float32)

    def transform(self, Xn: np.ndarray) -> np.ndarray:
        F = self.mr.transform(Xn)
        if hasattr(F, "to_numpy"):
            F = F.to_numpy()
        return np.asarray(F, dtype=np.float32)

    def head(self, F: np.ndarray) -> np.ndarray:
        if self.sc is not None:
            F = self.sc.transform(F)
        return self.clf.predict_proba(F)


def load_models(models_dir: Path, slots_per_registry: int = 1) -> list[Model]:
    """models_dir 의 *.joblib 를 모두 적재한다.

    registry 하나에는 차종별 모델이 3개(IONIQ/KONA/NIRO) 들어 있는데, 세 모델은
    채널·커널 수가 같아 지연시간이 사실상 동일하다(학습 PC 측정에서 23.46/23.59/
    23.89 ms). 그래서 기본은 registry 당 1슬롯만 잰다. 전부 재려면 --slots 0.
    """
    import joblib

    models: list[Model] = []
    for p in sorted(models_dir.glob("*.joblib")):
        try:
            obj = joblib.load(p)
        except Exception as e:
            print(f"  !! 적재 실패 {p.name}: {type(e).__name__}: {e}")
            print("     -> 학습 PC 와 sktime/scikit-learn 버전이 다르다. "
                  "requirements-pi.txt 확인.")
            continue
        if isinstance(obj, dict) and "minirocket" in obj and "classifier" in obj:
            # 파일럿 번들 (1채널)
            models.append(Model(f"{p.stem}", obj["minirocket"], obj["classifier"]))
            continue
        if isinstance(obj, dict) and obj and all(
            isinstance(v, dict) and "mr" in v for v in obj.values()
        ):
            # v3 레지스트리 (슬롯 여러 개)
            items = list(obj.items())
            if slots_per_registry > 0:
                items = items[:slots_per_registry]
                if len(obj) > len(items):
                    print(f"  {p.name}: 슬롯 {len(obj)}개 중 {len(items)}개만 측정 "
                          f"(--slots 0 으로 전부)")
            for key, m in items:
                slot = "_".join(str(x) for x in (key if isinstance(key, tuple) else (key,)))
                models.append(Model(f"{p.stem}::{slot}", m["mr"], m["lr"],
                                    sc=m.get("sc"), ch_mean=m.get("ch_mean"),
                                    ch_std=m.get("ch_std")))
            continue
        print(f"  ! 형식을 모르겠는 번들 건너뜀: {p.name}")
    return models


# ------------------------------------------------------------------ 벤치
def make_input(n_ch: int, batch: int, length: int, real: np.ndarray | None) -> np.ndarray:
    """(batch, n_ch, length) float32 입력.

    MiniRocket 의 연산량은 입력 값에 의존하지 않는다(커널 컨볼루션 + PPV 카운트가
    고정 길이). 실데이터가 없으면 난수로 재도 지연시간은 사실상 같다.
    """
    if real is not None and real.shape[1] >= n_ch and real.shape[2] == length:
        idx = np.arange(batch) % real.shape[0]
        return np.ascontiguousarray(real[idx, :n_ch, :], dtype=np.float32)
    rng = np.random.default_rng(42)
    return rng.standard_normal((batch, n_ch, length), dtype=np.float32)


def bench_one(model: Model, batch: int, reps: int, warmup: int,
              length: int, real: np.ndarray | None) -> dict:
    X = make_input(model.n_ch, batch, length, real)

    # --- JIT 컴파일 + 캐시 예열 (여기 시간은 지연시간에 포함하지 않는다) ---
    t0 = time.perf_counter()
    for _ in range(max(1, warmup)):
        model.head(model.transform(model.pre(X)))
    jit_s = time.perf_counter() - t0

    pre_ms, tf_ms, head_ms, tot_ms = [], [], [], []
    for _ in range(reps):
        a = time.perf_counter()
        Xn = model.pre(X)
        b = time.perf_counter()
        F = model.transform(Xn)
        c = time.perf_counter()
        model.head(F)
        d = time.perf_counter()
        pre_ms.append((b - a) * 1e3)
        tf_ms.append((c - b) * 1e3)
        head_ms.append((d - c) * 1e3)
        tot_ms.append((d - a) * 1e3)

    tot = np.array(tot_ms)
    return {
        "model": model.name,
        "n_channels": model.n_ch,
        "batch": batch,
        "reps": reps,
        "jit_warmup_s": round(jit_s, 2),
        "total_ms_mean": round(float(tot.mean()), 2),
        "total_ms_p50": round(float(np.percentile(tot, 50)), 2),
        "total_ms_p95": round(float(np.percentile(tot, 95)), 2),
        "total_ms_p99": round(float(np.percentile(tot, 99)), 2),
        "total_ms_max": round(float(tot.max()), 2),
        "per_chunk_ms_p50": round(float(np.percentile(tot, 50)) / batch, 2),
        "throughput_chunks_per_s": round(batch / (tot.mean() / 1e3), 2),
        "pre_ms_p50": round(float(np.percentile(pre_ms, 50)), 3),
        "minirocket_ms_p50": round(float(np.percentile(tf_ms, 50)), 2),
        "clf_ms_p50": round(float(np.percentile(head_ms, 50)), 3),
    }


def md_table(rows: list[dict]) -> str:
    cols = ["model", "n_channels", "batch", "total_ms_p50", "total_ms_p95",
            "per_chunk_ms_p50", "throughput_chunks_per_s",
            "minirocket_ms_p50", "clf_ms_p50", "jit_warmup_s"]
    head = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def main() -> None:
    ap = argparse.ArgumentParser(description="엣지 추론 지연시간 벤치마크")
    ap.add_argument("--models", required=True, help=".joblib 들이 있는 폴더")
    ap.add_argument("--threads", type=int, default=1,
                    help="사용할 코어 수 (기본 1). 파이4는 1 과 4 를 각각 재서 비교할 것")
    ap.add_argument("--batch", default="1", help="배치 크기 목록, 예: 1,8,32")
    ap.add_argument("--reps", type=int, default=20, help="측정 반복 횟수")
    ap.add_argument("--warmup", type=int, default=3, help="예열 횟수(JIT 컴파일용)")
    ap.add_argument("--length", type=int, default=2000, help="청크 길이 (CHUNK_SIZE)")
    ap.add_argument("--input", default=None,
                    help="실데이터 npz/npy (선택). 없으면 난수 — 지연시간은 동일하다")
    ap.add_argument("--slots", type=int, default=1,
                    help="registry 당 측정할 슬롯(차종) 수. 0 이면 전부 (기본 1)")
    ap.add_argument("--out", default="bench_result.json")
    args = ap.parse_args()

    batches = [int(b) for b in args.batch.split(",") if b.strip()]
    models_dir = Path(args.models)
    if not models_dir.is_dir():
        sys.exit(f"!! 폴더 없음: {models_dir}")

    info = env_info(args.threads)
    print("=" * 74)
    print("엣지 추론 지연시간 벤치마크")
    for k, v in info.items():
        print(f"  {k:20s} {v}")
    thermal_before = pi_thermal()
    if thermal_before:
        print(f"  {'thermal_before':20s} {thermal_before}")
    print("=" * 74)

    real = None
    if args.input:
        p = Path(args.input)
        arr = np.load(p, allow_pickle=False)
        if hasattr(arr, "files"):  # npz
            arr = arr[arr.files[0]]
        real = np.asarray(arr, dtype=np.float32)
        if real.ndim == 2:
            real = real[:, None, :]
        print(f"실데이터 입력 {p.name} shape={real.shape}")

    rss0 = rss_mb()
    t0 = time.perf_counter()
    models = load_models(models_dir, args.slots)
    load_s = time.perf_counter() - t0
    rss1 = rss_mb()
    if not models:
        sys.exit("!! 적재된 모델이 없다")
    print(f"\n모델 {len(models)}개 적재  {load_s:.2f}s   "
          f"RSS {rss0:.0f} -> {rss1:.0f} MB\n")

    rows = []
    for m in models:
        m.set_threads(args.threads)
        for b in batches:
            print(f"  [{m.name}] batch={b} 측정 중...", flush=True)
            r = bench_one(m, b, args.reps, args.warmup, args.length, real)
            r["threads"] = args.threads
            rows.append(r)
            print(f"      p50 {r['total_ms_p50']}ms  p95 {r['total_ms_p95']}ms  "
                  f"청크당 {r['per_chunk_ms_p50']}ms  "
                  f"({r['throughput_chunks_per_s']} chunk/s)")

    thermal_after = pi_thermal()
    result = {
        "env": info,
        "thermal_before": thermal_before,
        "thermal_after": thermal_after,
        "model_load_seconds": round(load_s, 2),
        "rss_mb_after_load": round(rss1, 1),
        "rows": rows,
    }
    out = Path(args.out)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("\n" + md_table(rows))
    if thermal_after:
        print(f"\n종료 시점 {thermal_after}")
        if thermal_after.get("throttled", "0x0") != "0x0":
            print("  [!] 스로틀 발생 — 발열/전원 문제로 수치가 낮게 나왔다. "
                  "방열판·정품 전원 확인 후 재측정할 것.")
    print(f"\n결과 저장: {out.resolve()}")


if __name__ == "__main__":
    main()
