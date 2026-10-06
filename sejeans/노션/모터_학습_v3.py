"""
모터-감속기 학습 파이프라인 v3 — 15채널 융합 (기존 4피처 + rpm)

3차 발표의 12채널 early fusion 에 rpm 을 더한 구조.
  3센서 × 5피처 = 15채널, 차종당 모델 1개 (총 3개)

ver2 노트북 대비 변경점
1) 입력 컬럼   : peak_freq -> peak_freq_bin (v2 추출기 컬럼명)
2) 채널        : 12 -> 15 (센서마다 rpm 추가)
3) 그룹 키     : group_key -> timestamp (주행 단위 누수 방지)   << 중요
4) rpm 단독 진단: rpm 만으로 얼마나 맞히는지 별도 측정          << 중요
5) parquet/csv 자동 감지

실행:
  python 모터_학습_v3.py --data "C:\\...\\특징데이터" --out "C:\\...\\모델"
  python 모터_학습_v3.py --data ... --out ... --no-rpm     (12채널, 비교용)
  python 모터_학습_v3.py --data ... --out ... --per-sensor (센서별 9슬롯)
  python 모터_학습_v3.py --data ... --out ... --limit 50   (리허설)

--- 3차 발표 결과와 직접 비교하면 안 되는 이유 ---
세 가지가 동시에 바뀌었다:
  (a) 그룹키 group_key -> timestamp : 같은 주행이 train/test 로 갈리던 누수를 막았다.
      이것만으로도 수치는 떨어진다. 성능이 나빠진 게 아니라 원래 그 정도였던 것이다.
  (b) 추출기 구버전 -> v2 : LUT 역변환 + rownorm 으로 입력 자체가 다르다.
  (c) rpm 채널 추가
따라서 3차 발표 0.83~0.87 과의 차이를 rpm 탓으로 돌릴 수 없다.
rpm 효과만 보려면 --no-rpm 을 같은 데이터로 한 번 더 돌려 비교할 것.
"""

import os
import re
import sys
import glob
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Windows 콘솔이 cp949 라 한글/기호에서 죽는 걸 막는다
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ===================== CONFIG =====================
CHUNK_SIZE   = 2000
BASE_COLS    = ["peak_freq_bin", "band_start", "band_end", "rms"]
RPM_COLS     = ["rpm"]
CLASS_ORDER  = ["DEMAG", "ECC10", "ECC20", "NORMAL", "REDUC"]
N_SPLITS     = 3
NUM_KERNELS  = 10000
RANDOM_STATE = 42

CHUNK_CAP_PER_CLASS = 4000
MINIROCKET_N_JOBS   = -1
LR_MAX_ITER         = 300
LR_CLASS_WEIGHT     = None

# 융합 시 채널 순서 (센서 고정 순서 × 피처)
SENSOR_ORDER = ["Current_U", "Vib_Motor", "Vib_TM"]

# 누수 방지용 그룹 단위. --group-by 로 바꾼다.
#   group_key = timestamp+seq+zsplit+sensor  (3차 발표에서 쓴 단위)
#     -> zsplit 이 키에 들어가 있어 한 주행이 여러 group 으로 쪼개진다.
#        zsplit001 이 train, zsplit002 가 test 로 갈릴 수 있는데
#        둘은 같은 주행의 연속 구간이다(rpm 이 2341 -> 2343 으로 이어진다).
#   timestamp = 주행 단위. 한 주행 전체가 train/test 한쪽에만 들어간다. 더 보수적.
GROUP_BY = "timestamp"
# ==================================================

FEATURE_COLS = list(BASE_COLS)     # main 에서 rpm 여부에 따라 확정


# ------------------------------------------------------------ 입출력
def _is_index_file(p):
    n = Path(p).name.lower()
    return ("인덱스" in Path(p).name) or n.startswith("idx") or n == "index.parquet"


def find_data_files(root):
    ps = [p for p in sorted(glob.glob(os.path.join(root, "**", "*.parquet"),
                                      recursive=True)) if not _is_index_file(p)]
    if ps:
        return ps, "parquet"
    ps = [p for p in sorted(glob.glob(os.path.join(root, "**", "*.csv"),
                                      recursive=True)) if not _is_index_file(p)]
    return ps, "csv"


def load_index(root):
    """샘플인덱스: group_key -> vehicle/sensor/timestamp 등.

    v2 추출기는 행마다 반복되는 문자열 메타를 빼고 인덱스 한 벌로 분리했다.
    (IONIQ, Vib_TM 같은 값이 2000행마다 복사되면 용량의 30~40% 가 메타가 된다)
    """
    names = ["샘플인덱스.parquet", "샘플인덱스.csv", "idx.parquet", "index.parquet"]
    for name in names:
        for cand in [Path(root) / name, Path(root).parent / name]:
            if cand.exists():
                idx = (pd.read_parquet(cand) if cand.suffix == ".parquet"
                       else pd.read_csv(cand))
                if "group_key" not in idx.columns:
                    continue
                return idx.drop_duplicates("group_key").set_index("group_key"), cand
    return None, None


_TS = re.compile(r"(\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})")


def meta_from_path(path, df, idx):
    """(vehicle, sensor, fault_class, timestamp) 를 인덱스 -> 행 -> 경로 순으로 찾는다."""
    gk = str(df["group_key"].iloc[0]) if "group_key" in df.columns else None

    if idx is not None and gk in idx.index:
        r = idx.loc[gk]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        return (str(r["vehicle"]), str(r["sensor"]), str(r["fault_class"]),
                str(r["timestamp"]))

    def col(c):
        return str(df[c].iloc[0]) if c in df.columns else None

    veh, sen, fc = col("vehicle"), col("sensor"), col("fault_class")
    ts = col("timestamp")

    # 경로 구조: .../{vehicle}/{fault_class}/{date}/{sensor}/파일명
    #   parts[-1]=파일명  parts[-2]=sensor  parts[-3]=date  parts[-4]=fault_class
    #   parts[-5]=vehicle
    parts = Path(path).parts
    if len(parts) >= 5:
        if veh is None:
            veh = parts[-5]
        if fc is None:
            fc = parts[-4]
        if sen is None:
            sen = parts[-2]
    if sen is None:
        # 파일명 끝의 센서명 (…_Vib_TM.parquet)
        stem = Path(path).stem
        for s in ("Current_U", "Vib_Motor", "Vib_TM"):
            if stem.endswith(s):
                sen = s
                break
    if ts is None:
        m = _TS.search(Path(path).name) or (_TS.search(gk) if gk else None)
        ts = m.group(1) if m else (gk or Path(path).stem)
    return str(veh), str(sen), str(fc), str(ts)


def _feats_from_df(df):
    """FEATURE_COLS 를 (n, C) float32 로. rpm 의 조각경계 NaN 은 앞뒤로 메운다."""
    feats = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    if np.isnan(feats).any():
        feats = pd.DataFrame(feats).ffill().bfill().to_numpy(dtype=np.float32)
        feats = np.nan_to_num(feats, nan=0.0)
    return feats


def load_fused_chunks(entries, fmt, idx):
    """차종 1개의 3센서를 15채널로 묶어 청크 생성 (early fusion).

    entries: [(path, sensor, fault_class, timestamp, zsplit), ...]

    3차 발표에서 late fusion(센서별 예측 후 확률 합산) 대비
    early fusion(한 번에 입력)이 macro-F1 0.63 -> 0.83 으로 크게 좋았다.
    센서 간 결합 패턴(예: DEMAG 는 진동+전류 조합에서만 보이는 경우)을
    late fusion 은 원리상 볼 수 없기 때문이다.

    같은 (timestamp, zsplit) 의 3센서는 같은 구간을 동시에 잰 것이라
    시간축이 정렬돼 있다. 길이가 다르면 짧은 쪽에 맞춰 자른다.
    """
    reader = pd.read_parquet if fmt == "parquet" else pd.read_csv
    by_seg = defaultdict(dict)
    meta = {}
    for path, sensor, fc, ts, zs in entries:
        by_seg[(ts, zs)][sensor] = path
        # 융합은 센서를 합치므로 group_key 모드에서도 sensor 는 빼고 묶는다
        grp = ts if GROUP_BY == "timestamp" else f"{ts}_{zs}"
        meta[(ts, zs)] = (fc, grp)

    Xs, ys, gs = [], [], []
    n_skip_sensor = 0
    for seg, smap in sorted(by_seg.items()):
        if not all(s in smap for s in SENSOR_ORDER):
            n_skip_sensor += 1
            continue
        mats = []
        for s in SENSOR_ORDER:
            try:
                df = reader(smap[s])
            except Exception:
                mats = None
                break
            if df.empty:
                mats = None
                break
            mats.append(_feats_from_df(df))
        if not mats:
            continue

        L = min(len(m) for m in mats)                 # 짧은 쪽에 맞춤
        n = (L // CHUNK_SIZE) * CHUNK_SIZE
        if n == 0:
            continue
        fused = np.concatenate([m[:n] for m in mats], axis=1)   # (n, 15)
        C = fused.shape[1]
        chunks = fused.T.reshape(C, -1, CHUNK_SIZE).transpose(1, 0, 2)

        fc, grp = meta[seg]
        Xs.append(chunks)
        ys += [fc] * len(chunks)
        gs += [grp] * len(chunks)                     # 그룹 = GROUP_BY 단위

    if n_skip_sensor:
        print(f"    센서 누락으로 스킵한 구간 {n_skip_sensor}개")
    if not Xs:
        return None
    return np.concatenate(Xs, 0), np.array(ys), np.array(gs)


def scan_entries(root, limit=0):
    """융합용 스캔: 차종 -> [(path, sensor, class, timestamp, zsplit)]"""
    files, fmt = find_data_files(root)
    if not files:
        print(f"!! 파일 없음: {root}")
        sys.exit(1)
    idx, idx_path = load_index(root)
    print(f"{fmt} {len(files):,}개 발견"
          + (f" / 인덱스 {idx_path.name}" if idx_path else " / 인덱스 없음"))

    by_veh = defaultdict(list)
    classes = defaultdict(set)
    bad = 0
    for p in files:
        try:
            if fmt == "parquet":
                import pyarrow.parquet as pq
                have = set(pq.ParquetFile(p).schema.names)
                want = [c for c in ["group_key", "fault_class", "vehicle",
                                    "sensor", "timestamp", "zsplit"] if c in have]
                head = pd.read_parquet(p, columns=want).head(1)
            else:
                head = pd.read_csv(p, nrows=1)
            veh, sen, fc, ts = meta_from_path(p, head, idx)
            zs = _zsplit_of(p, head, idx)
        except Exception:
            bad += 1
            continue
        by_veh[veh].append((p, sen, fc, ts, zs))
        classes[veh].add(fc)
    if bad:
        print(f"  헤더 읽기 실패 {bad}건 스킵")
    if limit:
        by_veh = {k: v[:limit * 3] for k, v in by_veh.items()}
    return by_veh, classes, fmt, idx


_ZS = re.compile(r"(zsplit\d{3})")


def _zsplit_of(path, df, idx):
    gk = str(df["group_key"].iloc[0]) if "group_key" in df.columns else ""
    if idx is not None and gk in idx.index:
        r = idx.loc[gk]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        if "zsplit" in r.index:
            return str(r["zsplit"])
    if "zsplit" in df.columns:
        return str(df["zsplit"].iloc[0])
    m = _ZS.search(Path(path).name) or _ZS.search(gk)
    return m.group(1) if m else "z000"


def scan_slots(root, limit=0):
    files, fmt = find_data_files(root)
    if not files:
        print(f"!! 파일 없음: {root}")
        sys.exit(1)
    idx, idx_path = load_index(root)
    print(f"{fmt} {len(files):,}개 발견"
          + (f" / 인덱스 {idx_path.name}" if idx_path else " / 인덱스 없음"))

    reader = pd.read_parquet if fmt == "parquet" else pd.read_csv
    slots, slot_classes = defaultdict(list), defaultdict(set)
    bad = 0
    for p in files:
        try:
            if fmt == "parquet":
                import pyarrow.parquet as pq
                have = set(pq.ParquetFile(p).schema.names)
                want = [c for c in ["group_key", "fault_class", "vehicle",
                                    "sensor", "timestamp"] if c in have]
                head = pd.read_parquet(p, columns=want).head(1)
            else:
                head = pd.read_csv(p, nrows=1)
            veh, sen, fc, ts = meta_from_path(p, head, idx)
        except Exception:
            bad += 1
            continue
        slots[(veh, sen)].append(p)
        slot_classes[(veh, sen)].add(fc)
    if bad:
        print(f"  헤더 읽기 실패 {bad}건 스킵")
    if limit:
        slots = {k: v[:limit] for k, v in slots.items()}
    return slots, slot_classes, fmt, idx


def load_slot_chunks(paths, fmt, idx):
    """CSV/parquet 를 청크(고정 길이 다변량 샘플) 로 변환.

    반환 groups 는 timestamp(주행 세션). 같은 주행에서 나온 청크가
    train/test 로 갈리면 성능이 부풀려지므로 반드시 세션 단위로 묶는다.
    """
    reader = pd.read_parquet if fmt == "parquet" else pd.read_csv
    Xs, ys, gs = [], [], []
    miss = set()
    for p in paths:
        try:
            df = reader(p)
        except Exception:
            continue
        if df.empty:
            continue
        have = [c for c in FEATURE_COLS if c in df.columns]
        if len(have) != len(FEATURE_COLS):
            miss.update(set(FEATURE_COLS) - set(df.columns))
            continue

        feats = df[FEATURE_COLS].to_numpy(dtype=np.float32)
        # rpm 은 조각 경계에서 NaN 이 날 수 있다. 앞/뒤 값으로 메운다.
        if np.isnan(feats).any():
            f = pd.DataFrame(feats).ffill().bfill()
            feats = f.to_numpy(dtype=np.float32)
            feats = np.nan_to_num(feats, nan=0.0)

        n = (len(feats) // CHUNK_SIZE) * CHUNK_SIZE
        if n == 0:
            continue
        C = feats.shape[1]
        chunks = feats[:n].T.reshape(C, -1, CHUNK_SIZE).transpose(1, 0, 2)

        _, _, fc, ts = meta_from_path(p, df, idx)
        if GROUP_BY == "timestamp":
            grp = ts
        else:
            grp = (str(df["group_key"].iloc[0]) if "group_key" in df.columns
                   else f"{ts}_{_zsplit_of(p, df, idx)}")
        Xs.append(chunks)
        ys += [fc] * len(chunks)
        gs += [grp] * len(chunks)
    if miss:
        print(f"  !! 누락 컬럼: {sorted(miss)}")
    if not Xs:
        return None
    return np.concatenate(Xs, 0), np.array(ys), np.array(gs)


# ------------------------------------------------------------ 모델
def make_minirocket(n_channels):
    if n_channels == 1:
        from sktime.transformations.panel.rocket import MiniRocket
        return MiniRocket(num_kernels=NUM_KERNELS, random_state=RANDOM_STATE,
                          n_jobs=MINIROCKET_N_JOBS)
    from sktime.transformations.panel.rocket import MiniRocketMultivariate
    return MiniRocketMultivariate(num_kernels=NUM_KERNELS, random_state=RANDOM_STATE,
                                  n_jobs=MINIROCKET_N_JOBS)


def _fit_channel_stats(X):
    mean = X.mean(axis=(0, 2)).astype(np.float32)
    std = X.std(axis=(0, 2)).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def _apply_channel(X, mean, std):
    return ((X - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


def cap_per_class(X, y, g, cap, seed=RANDOM_STATE):
    """슬롯·클래스당 청크 수 제한.

    주의: 세션 단위로 뽑는다. 무작위로 뽑으면 특정 주행만 남아
    그 주행의 특성이 클래스 특성으로 둔갑할 수 있다.
    """
    if not cap:
        return X, y, g
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) <= cap:
            keep.append(idx)
            continue
        # 세션별로 균등하게 배분
        by = defaultdict(list)
        for i in idx:
            by[g[i]].append(i)
        per = max(1, cap // len(by))
        sel = []
        for s, ii in by.items():
            ii = np.array(ii)
            sel.append(rng.choice(ii, min(per, len(ii)), replace=False))
        sel = np.concatenate(sel)
        if len(sel) > cap:
            sel = rng.choice(sel, cap, replace=False)
        keep.append(sel)
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep], g[keep]


def build_features(X):
    mean, std = _fit_channel_stats(X)
    Xn = _apply_channel(X, mean, std)
    mr = make_minirocket(Xn.shape[1])
    mr.fit(Xn)
    F = np.asarray(mr.transform(Xn), dtype=np.float32)
    return mean, std, mr, F


def fit_lr(Ftr, ytr):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler().fit(Ftr)
    lr = LogisticRegression(max_iter=LR_MAX_ITER, class_weight=LR_CLASS_WEIGHT)
    lr.fit(sc.transform(Ftr), ytr)
    return sc, lr


def _proba_aligned(sc, lr, F):
    p = lr.predict_proba(sc.transform(F))
    full = np.zeros((len(F), len(CLASS_ORDER)), dtype=float)
    for j, c in enumerate(lr.classes_):
        full[:, CLASS_ORDER.index(str(c))] = p[:, j]
    return full


def slot_proba(model, X):
    Xn = _apply_channel(X, model["ch_mean"], model["ch_std"])
    F = np.asarray(model["mr"].transform(Xn), dtype=np.float32)
    return _proba_aligned(model["sc"], model["lr"], F)


# ---------------------------------------------------- rpm 지름길 진단
def rpm_shortcut_check(X, y, g, rpm_chs):
    """rpm '값 하나만' 으로 클래스를 맞힐 수 있는지 확인한다.

    왜 필요한가:
      샘플데이터에서 NIRO 는 rpm 만으로 100% 분류됐다. 자석이 열화돼서 rpm 이
      높은 게 아니라, DEMAG 측정을 그날 빠르게 달렸을 뿐이다(클래스당 세션 1개).
      모델이 고장이 아니라 '측정 조건' 을 외우면 실제 도로에서 무너진다.
      (고속주행 정상차 -> DEMAG 오경보 / 저속 열화차 -> 놓침)

    판정:
      높을수록 나쁘다. 세션 단위로 나눠서 재기 때문에,
      클래스별 rpm 대역이 실제로 분리돼 있을 때만 점수가 높게 나온다.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score
    from sklearn.linear_model import LogisticRegression

    # 융합이면 rpm 채널이 3개(센서마다) 있지만 값은 같다. 평균 하나로 본다.
    v = X[:, rpm_chs, :].mean(axis=(1, 2)).reshape(-1, 1)
    labels = sorted(set(map(str, y)))

    # 클래스별 rpm 범위는 CV 없이도 볼 수 있다. 겹침이 없으면 그 자체가 정답표다.
    rng = {c: (float(np.percentile(v[y == c], 5)),
               float(np.percentile(v[y == c], 95))) for c in labels}

    min_groups = min(len(set(g[y == c])) for c in labels)
    ns = min(N_SPLITS, min_groups)
    if ns < 2:
        return None, rng                                  # 점수는 못 내도 범위는 반환

    sgkf = StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=RANDOM_STATE)
    f1s = []
    for tr, va in sgkf.split(v, y, g):
        m = LogisticRegression(max_iter=200).fit(v[tr], y[tr])
        f1s.append(f1_score(y[va], m.predict(v[va]),
                            average="macro", labels=labels, zero_division=0))
    return float(np.mean(f1s)), rng


# ------------------------------------------------------------ 학습
def train_slot(slot_key, X, y, g, rpm_chs):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

    info = dict(slot=list(slot_key), n_chunks=int(len(y)),
                classes=sorted(set(map(str, y))))
    if len(set(y)) < 2:
        info["status"] = "skipped: 단일 클래스"
        return None, info

    X, y, g = cap_per_class(X, y, g, CHUNK_CAP_PER_CLASS)
    labels = sorted(set(map(str, y)))
    info["n_chunks"] = int(len(y))
    info["classes"] = labels
    info["n_sessions"] = int(len(set(g)))
    info["n_sessions_per_class"] = {c: int(len(set(g[y == c]))) for c in labels}

    # rpm 지름길 진단 (rpm 채널이 있을 때만)
    if rpm_chs:
        score, rng = rpm_shortcut_check(X, y, g, rpm_chs)
        if score is not None:
            info["rpm_only_macro_f1"] = round(score, 3)
        info["rpm_range_by_class"] = {k: [round(a), round(b)]
                                      for k, (a, b) in rng.items()}

    mean, std, mr, F = build_features(X)

    min_groups = min(len(set(g[y == c])) for c in labels)
    n_splits = min(N_SPLITS, min_groups)
    info["min_groups_per_class"] = int(min_groups)

    if n_splits >= 2:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=RANDOM_STATE)
        f1s, accs, cm_total = [], [], None
        for tr, va in sgkf.split(np.zeros(len(y)), y, g):
            sc, lr = fit_lr(F[tr], y[tr])
            pred = np.array(CLASS_ORDER)[_proba_aligned(sc, lr, F[va]).argmax(1)]
            f1s.append(f1_score(y[va], pred, average="macro",
                                labels=labels, zero_division=0))
            accs.append(accuracy_score(y[va], pred))
            cm = confusion_matrix(y[va], pred, labels=labels)
            cm_total = cm if cm_total is None else cm_total + cm
        info.update(status="ok", n_splits=int(n_splits), labels=labels,
                    macro_f1_mean=float(np.mean(f1s)),
                    macro_f1_std=float(np.std(f1s)),
                    acc_mean=float(np.mean(accs)), acc_std=float(np.std(accs)),
                    confusion_matrix=cm_total.tolist())
    else:
        info.update(status=f"trained (CV 생략: 세션 부족 min_groups={min_groups})",
                    labels=labels)

    sc, lr = fit_lr(F, y)
    model = dict(ch_mean=mean, ch_std=std, mr=mr, sc=sc, lr=lr,
                 feature_cols=list(FEATURE_COLS),
                 classes=list(map(str, lr.classes_)))
    return model, info


# ------------------------------------------------------------ 리포트
def report(eval_rows, use_rpm, out_dir, tag="rpm"):
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    trained = [r for r in eval_rows if "macro_f1_mean" in r]
    # 진단은 CV 여부와 무관하게 '데이터가 있는' 모든 슬롯에서 본다.
    # (CV 를 못 돌린 슬롯이야말로 세션이 부족하다는 뜻이라 진단이 더 필요하다)
    diagd = [r for r in eval_rows if r.get("n_chunks")]

    P("\n" + "=" * 78)
    P("학습 요약")
    P("=" * 78)
    P(f"  설정: {tag}   누수방지 그룹 = {GROUP_BY}"
      + ("  (3차 발표와 동일 단위)" if GROUP_BY == "group_key"
         else "  (주행 전체 = 더 보수적)"))
    P("")
    if trained:
        df = pd.DataFrame([{
            "slot": f'{r["slot"][0]}·{r["slot"][1]}',
            "chunks": r["n_chunks"],
            "그룹수": r.get("n_sessions", "-"),
            "macro_F1": f'{r["macro_f1_mean"]:.3f}±{r["macro_f1_std"]:.3f}',
            "acc": f'{r["acc_mean"]:.3f}',
            "rpm단독F1": r.get("rpm_only_macro_f1", "-"),
        } for r in trained])
        P(df.to_string(index=False))
        P("")
        P(f"  전체 macro-F1 평균 {np.mean([r['macro_f1_mean'] for r in trained]):.3f}")
    for r in eval_rows:
        if "macro_f1_mean" not in r:
            P(f'  [스킵] {r["slot"]}: {r.get("status")}')

    # ---- 세션 수 점검 ----
    P("\n" + "=" * 78)
    P(f"[진단 1] 클래스당 그룹 수 ({GROUP_BY} 기준)")
    P("=" * 78)
    P("  이게 적으면 CV 가 불안정하고, 1이면 CV 자체가 생략된다.")
    P("")
    worst, seen = None, 0
    for r in diagd:
        d = r.get("n_sessions_per_class", {})
        if not d:
            continue
        seen += 1
        mn = min(d.values())
        worst = mn if worst is None else min(worst, mn)
        flag = "" if mn >= 10 else "   << 부족"
        P(f'  {r["slot"][0]:6s}·{r["slot"][1]:10s} '
          + " ".join(f"{k}:{v}" for k, v in sorted(d.items())) + flag)
    P("")
    if seen == 0:
        P("  판정: 측정 불가 — 세션 정보가 있는 슬롯이 없다.")
    elif worst >= 10:
        P(f"  판정: 통과 — 최소 {worst}세션. 세션 단위 CV 가 성립한다.")
    else:
        P(f"  판정: 주의 — 최소 {worst}세션뿐인 조합이 있다.")
        P("        그 슬롯은 '고장' 이 아니라 '그날의 주행' 을 외웠을 수 있다.")
        P("        세션이 1개면 CV 자체가 생략되므로 성능 수치도 나오지 않는다.")

    # ---- rpm 지름길 ----
    if use_rpm:
        P("\n" + "=" * 78)
        P("[진단 2] rpm 지름길 — rpm 값 하나만으로 얼마나 맞히는가")
        P("=" * 78)
        P("  높을수록 나쁘다. rpm 은 '다른 신호를 해석하는 기준자' 여야 하고,")
        P("  'rpm 이 높으면 DEMAG' 같은 규칙이 되면 실제 도로에서 무너진다.")
        P("")
        risky, measured = [], 0
        for r in diagd:
            v = r.get("rpm_only_macro_f1")
            if v is None:
                continue
            measured += 1
            full = r.get("macro_f1_mean")
            flag = ""
            if v >= 0.60:
                flag = "   << 위험"
                risky.append(r["slot"])
            elif v >= 0.40:
                flag = "   << 주의"
            tail = (f'/ 전체 {full:.3f}  (설명비율 {v/full:.0%})'
                    if full else "/ 전체 -   (CV 생략된 슬롯)")
            P(f'  {r["slot"][0]:6s}·{r["slot"][1]:10s} '
              f'rpm단독 {v:.3f} {tail}{flag}')
        P("")
        if measured == 0:
            P("  판정: 측정 불가 — 세션이 부족해 rpm 진단을 돌릴 수 없었다.")
            P("        세션 수가 늘어나면 자동으로 측정된다.")
        elif risky:
            P(f"  판정: {len(risky)}개 슬롯에서 rpm 단독 F1 이 0.60 이상이다.")
            P("        --no-rpm 으로 다시 돌려 비교할 것.")
            P("        rpm 을 빼도 성능이 비슷하면 rpm 은 지름길이었을 뿐이다.")
        else:
            P("  판정: 통과 — rpm 만으로는 못 맞힌다. 보조 정보로 쓰이고 있다.")

        shown = [r for r in diagd if r.get("rpm_range_by_class")]
        if shown:
            P("")
            P("  클래스별 rpm 범위(5~95%) — 겹치지 않으면 그 자체가 정답표가 된다")
            for r in shown[:3]:
                P(f'  {r["slot"][0]}·{r["slot"][1]}')
                for c, (a, b) in sorted(r["rpm_range_by_class"].items()):
                    P(f"      {c:8s} {a:6d} ~ {b:6d}")

    # ---- DEMAG ----
    P("\n" + "=" * 78)
    P("[진단 3] DEMAG-NORMAL  — 이번 프로젝트의 핵심 과제")
    P("=" * 78)
    if not trained:
        P("  CV 를 돌린 슬롯이 없어 혼동행렬이 없다. (세션 부족)")
    for r in trained:
        labs = r.get("labels", [])
        if "DEMAG" not in labs or "NORMAL" not in labs:
            continue
        cm = np.array(r["confusion_matrix"])
        di, ni = labs.index("DEMAG"), labs.index("NORMAL")
        dn, nd = int(cm[di, ni]), int(cm[ni, di])
        rec = cm[di, di] / cm[di].sum() if cm[di].sum() else 0
        P(f'  {r["slot"][0]:6s}·{r["slot"][1]:10s} '
          f'DEMAG recall {rec:.3f}  D→N {dn:4d}  N→D {nd:4d}')

    P("\n" + "=" * 78)
    path = Path(out_dir) / f"학습리포트_{tag}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"리포트 저장: {path}")


# ------------------------------------------------------------ main
def main():
    global FEATURE_COLS

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="특징데이터 폴더")
    ap.add_argument("--out", required=True, help="모델/리포트 저장 폴더")
    ap.add_argument("--no-rpm", action="store_true", help="rpm 제외(비교용)")
    ap.add_argument("--per-sensor", action="store_true",
                    help="융합하지 않고 센서별 9슬롯으로 학습")
    ap.add_argument("--group-by", choices=["timestamp", "group_key"],
                    default="timestamp",
                    help="누수 방지 단위. timestamp=주행 전체(보수적, 기본) / "
                         "group_key=3차 발표와 동일(zsplit 단위)")
    ap.add_argument("--limit", type=int, default=0, help="파일 수 제한(리허설)")
    ap.add_argument("--cap", type=int, default=CHUNK_CAP_PER_CLASS)
    args = ap.parse_args()

    use_rpm = not args.no_rpm
    fuse = not args.per_sensor
    FEATURE_COLS = BASE_COLS + (RPM_COLS if use_rpm else [])
    globals()["GROUP_BY"] = args.group_by

    n_ch = len(FEATURE_COLS) * (len(SENSOR_ORDER) if fuse else 1)
    if use_rpm:
        k = FEATURE_COLS.index("rpm")
        rpm_chs = ([k + j * len(FEATURE_COLS) for j in range(len(SENSOR_ORDER))]
                   if fuse else [k])
    else:
        rpm_chs = []

    globals()["CHUNK_CAP_PER_CLASS"] = args.cap

    t0 = time.time()
    print("=" * 78)
    print("모터-감속기 학습 v3")
    if fuse:
        print(f"  {n_ch}채널 융합(early fusion): {len(SENSOR_ORDER)}센서 x "
              f"{len(FEATURE_COLS)}피처")
    else:
        print(f"  센서별 학습: {n_ch}채널 x 9슬롯")
    print(f"  피처 {FEATURE_COLS}")
    print(f"  그룹(누수방지) = {GROUP_BY}   커널 {NUM_KERNELS}   캡 {args.cap}")
    print("=" * 78)

    try:
        import sktime  # noqa: F401
    except ImportError:
        print("!! sktime 없음.  pip install -U sktime")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    import joblib
    tag = (("fuse" if fuse else "per") + ("_rpm" if use_rpm else "_norpm")
           + ("" if args.group_by == "timestamp" else "_gk"))
    reg_path = Path(args.out) / f"model_registry_{tag}.joblib"
    eval_path = Path(args.out) / f"eval_results_{tag}.json"

    # ---- 슬롯 구성 ----
    if fuse:
        by_veh, classes, fmt, idx = scan_entries(args.data, args.limit)
        print(f"\n차종 {len(by_veh)}개")
        for v in sorted(by_veh):
            print(f"  {v:6s} 파일 {len(by_veh[v]):5d}  클래스 {sorted(classes[v])}")
        keys = sorted(by_veh)
        loader = lambda k: load_fused_chunks(by_veh[k], fmt, idx)
        skey = lambda k: [k, "FUSED15" if use_rpm else "FUSED12"]
    else:
        slots, slot_classes, fmt, idx = scan_slots(args.data, args.limit)
        print(f"\n슬롯(차종×센서) {len(slots)}개")
        for k in sorted(slots, key=lambda x: (str(x[0]), str(x[1]))):
            cls = sorted(slot_classes[k])
            flag = "" if len(cls) >= 2 else "   [!] 단일 클래스 -> 스킵"
            print(f"  {str(k[0]):6s}·{str(k[1]):10s} 파일 {len(slots[k]):5d} "
                  f"클래스 {cls}{flag}")
        keys = sorted(slots, key=lambda x: (str(x[0]), str(x[1])))
        loader = lambda k: load_slot_chunks(slots[k], fmt, idx)
        skey = lambda k: list(k)

    registry, eval_rows = {}, []
    for i, k in enumerate(keys, 1):
        print(f"\n[{i}/{len(keys)}] {k} 학습 중...")
        data = loader(k)
        if data is None:
            eval_rows.append(dict(slot=skey(k), n_chunks=0,
                                  status="skipped: 청크 없음"))
            continue
        X, y, g = data
        print(f"    청크 {len(y):,}개 / 그룹 {len(set(g))}개({GROUP_BY}) "
              f"/ 채널 {X.shape[1]}")
        model, info = train_slot(skey(k), X, y, g, rpm_chs)
        eval_rows.append(info)
        if "macro_f1_mean" in info:
            msg = f'    macro-F1 {info["macro_f1_mean"]:.3f}'
            if "rpm_only_macro_f1" in info:
                msg += f'   (rpm단독 {info["rpm_only_macro_f1"]:.3f})'
            print(msg)
        else:
            print(f'    {info.get("status")}')
        if model is not None:
            registry[tuple(skey(k))] = model
            joblib.dump(registry, reg_path)
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(eval_rows, f, ensure_ascii=False, indent=2, default=str)

    joblib.dump(registry, reg_path)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, ensure_ascii=False, indent=2, default=str)

    report(eval_rows, use_rpm, args.out, tag)
    print(f"\n모델 {len(registry)}개 저장: {reg_path}")
    print(f"소요 {(time.time()-t0)/60:.1f}분")
    if use_rpm:
        print("\n다음: --no-rpm 으로 한 번 더 돌려 비교하세요.")
        print("      같은 데이터·같은 그룹키에서 rpm 만 뺀 것이라")
        print("      그 차이만이 rpm 의 순수 효과입니다.")
        print("      (3차 발표 0.83~0.87 과는 그룹키·추출기가 달라 직접 비교 불가)")


if __name__ == "__main__":
    main()
