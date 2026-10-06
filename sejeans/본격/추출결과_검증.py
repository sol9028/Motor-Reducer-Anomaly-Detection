"""
모터_특징추출_v2.py 결과 검증

3가지 확정 조치가 "의도한 대로 작동했는지"만 확인한다.
모델 성능은 여기서 보지 않는다 (그건 학습 코드에서 세션 단위로 봐야 한다).

  (1) LUT 역변환  -> rms 분포가 밝기 기반의 뭉개짐에서 벗어났는가
  (2) rownorm     -> band_start 포화가 풀렸는가        << 가장 중요
  (3) rpm         -> 컬럼이 실제로 채워졌고 split 별로 변하는가

실행:  python 추출결과_검증.py
      python 추출결과_검증.py --sample 500     (읽을 CSV 수)
"""

import io
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

WORK_DIR   = Path(r"C:\Users\jaesung\Desktop\매녀졸프")
CSV_DIR    = WORK_DIR / "특징데이터"
INDEX_PATH = WORK_DIR / "샘플인덱스.parquet"
REPORT     = WORK_DIR / "추출결과_검증리포트.txt"

BASE4 = ["peak_freq_bin", "band_start", "band_end", "rms"]
EXT   = [f"band_ratio_{i}" for i in range(8)] + \
        ["centroid", "entropy", "rolloff", "log_flatness"]

_LINES = []


def P(s=""):
    print(s)
    _LINES.append(s)


def flush():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with io.open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(_LINES))
    print(f"\n리포트 저장: {REPORT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500, help="읽을 CSV 수")
    args = ap.parse_args()

    t0 = time.time()
    P("=" * 78)
    P("추출결과 검증")
    P("=" * 78)

    if not CSV_DIR.exists():
        P(f"!! 경로 없음: {CSV_DIR}")
        flush()
        sys.exit(1)

    files = sorted(CSV_DIR.rglob("*.parquet"))
    fmt = "parquet"
    if not files:
        files = sorted(CSV_DIR.rglob("*_timeseries.csv"))
        fmt = "csv"
    P(f"{fmt} 총 {len(files):,}개 발견")
    if not files:
        P("!! 출력 파일이 없다. 추출이 실패했거나 경로가 다르다.")
        flush()
        sys.exit(1)

    total_bytes = sum(p.stat().st_size for p in files)
    P(f"총 용량 {total_bytes/2**30:.2f} GB "
      f"(파일당 평균 {total_bytes/len(files)/1024:.0f} KB)")

    import shutil
    du = shutil.disk_usage(CSV_DIR)
    P(f"드라이브 여유 {du.free/2**30:,.1f} GB / 전체 {du.total/2**30:,.1f} GB\n")

    # 균등 샘플링 (앞쪽만 읽으면 특정 차종/클래스에 쏠린다)
    if len(files) > args.sample:
        pick = np.linspace(0, len(files) - 1, args.sample).astype(int)
        files = [files[i] for i in pick]
    P(f"검증 표본 {len(files):,}개\n")

    reader = pd.read_parquet if fmt == "parquet" else pd.read_csv
    dfs = []
    bad = 0
    for p in files:
        try:
            dfs.append(reader(p))
        except Exception as e:
            bad += 1
            if bad == 1:
                P(f"  읽기 실패 예: {p.name} -> {e!r}")
    if bad:
        P(f"읽기 실패 {bad}건")
    if not dfs:
        flush()
        sys.exit(1)
    df = pd.concat(dfs, ignore_index=True)

    # 슬림메타면 vehicle/sensor/timestamp 가 행에 없다. 인덱스에서 붙인다.
    need = {"vehicle", "sensor", "timestamp"}
    if not need.issubset(df.columns):
        ip = INDEX_PATH if INDEX_PATH.exists() else INDEX_PATH.with_suffix(".csv")
        if ip.exists():
            idx = (pd.read_parquet(ip) if ip.suffix == ".parquet"
                   else pd.read_csv(ip))
            cols = ["group_key"] + [c for c in
                                    ["vehicle", "sensor", "timestamp", "date", "zsplit"]
                                    if c in idx.columns and c not in df.columns]
            df = df.merge(idx[cols], on="group_key", how="left")
            P(f"샘플인덱스 join 완료 ({ip.name})")
        else:
            P(f"!! 인덱스 없음: {ip}  -> 일부 항목을 건너뛴다")
    P(f"총 {len(df):,}행 / 컬럼 {len(df.columns)}개")
    P(f"컬럼: {list(df.columns)}\n")

    has_sensor = "sensor" in df.columns
    has_veh = "vehicle" in df.columns

    has_ext = all(c in df.columns for c in EXT)
    has_rpm = "rpm" in df.columns

    # ---------------------------------------------------------- (2) 포화
    P("=" * 78)
    P("[2] rownorm : band_start 포화 해소 여부   << 가장 중요")
    P("=" * 78)
    P("  기존 코드에서 Vib_Motor 94.6% / Vib_TM 80.7% 가 상수 0 이었다.")
    P("  5% 미만이면 해소, 30% 이상이면 rownorm 이 작동하지 않은 것이다.")
    P("")
    P(f"  {'센서':12s}{'band_start=0':>14s}{'band_end=max':>14s}"
      f"{'NaN(무효)':>12s}{'고유값수':>10s}")
    ok_sat = True
    groups = df.groupby("sensor") if has_sensor else [("전체", df)]
    for s, g in groups:
        bs = g["band_start"]
        be = g["band_end"]
        z = float((bs == 0).mean() * 100)
        mx = float((be == be.max()).mean() * 100)
        nn = float(bs.isna().mean() * 100)
        nu = int(bs.dropna().nunique())
        P(f"  {str(s):12s}{z:13.1f}%{mx:13.1f}%{nn:11.1f}%{nu:10d}")
        if z >= 30:
            ok_sat = False
    P("")
    P(f"  판정: {'통과 - 포화 해소됨' if ok_sat else '실패 - 여전히 포화. USE_ROWNORM 확인'}")
    P("")

    # ---------------------------------------------------------- (1) LUT
    P("=" * 78)
    P("[1] LUT 역변환 : 에너지 값이 살아있는가")
    P("=" * 78)
    P("  기존 밝기방식은 0~1 min-max 강제라 항상 [0,1] 을 꽉 채웠다.")
    P("  LUT 는 실제 scalar 라 이미지마다 범위가 다른 게 정상이다.")
    P("")
    r = df["rms"]
    P(f"  rms  min {r.min():.4f}  p25 {r.quantile(.25):.4f}  "
      f"median {r.median():.4f}  p75 {r.quantile(.75):.4f}  max {r.max():.4f}")
    P(f"  rms  고유값 {r.nunique():,}개 / NaN {r.isna().sum():,}개")
    P("")
    P("  클래스별 rms 중앙값 (DEMAG 와 NORMAL 이 갈리는지)")
    if has_sensor:
        piv = df.pivot_table(index="fault_class", columns="sensor",
                             values="rms", aggfunc="median")
        P(f"    {'':8s}" + "".join(f"{c:>12s}" for c in piv.columns))
        for cls in piv.index:
            P(f"    {cls:8s}" + "".join(f"{piv.loc[cls, c]:12.4f}"
                                        for c in piv.columns))
    else:
        for cls, g in df.groupby("fault_class"):
            P(f"    {cls:8s}{g['rms'].median():12.4f}")
    P("")
    deg = float((r == 0).mean() * 100)
    P(f"  rms==0 비율 {deg:.2f}%   "
      f"{'통과' if deg < 5 else '주의 - 에너지가 0 으로 뭉갠 구간이 많다'}")
    P("")

    # ---------------------------------------------------------- (3) rpm
    P("=" * 78)
    P("[3] rpm 컬럼")
    P("=" * 78)
    if not has_rpm:
        P("  rpm 컬럼 없음. ATTACH_RPM 확인.")
    else:
        v = df["rpm"]
        fill = float(v.notna().mean() * 100)
        P(f"  채움율 {fill:.1f}%  범위 {v.min():.0f} ~ {v.max():.0f}  "
          f"중앙 {v.median():.0f}")
        # 같은 group_key 안에서 rpm 이 변해야 정상 (split 마다 다르므로)
        nuniq = df.groupby("group_key")["rpm"].nunique()
        multi = float((nuniq > 1).mean() * 100)
        P(f"  group 내 rpm 고유값 평균 {nuniq.mean():.2f}개 / "
          f"2개 이상인 group {multi:.1f}%")
        P("")
        if fill < 50:
            P("  판정: 실패 - 매칭율이 낮다. 라벨링 경로/파일명 규칙 확인")
        elif multi < 10:
            P("  판정: 주의 - group 내 rpm 이 거의 상수다.")
            P("        split 단위 부착이 안 되고 세션 값 하나만 들어갔을 수 있다.")
        else:
            P("  판정: 통과 - split 단위로 rpm 이 부착됐다")
        if has_veh:
            P("")
            P("  차종별 rpm 중앙값")
            for veh, g in df.groupby("vehicle"):
                P(f"    {veh:8s} {g['rpm'].median():8.0f}  "
                  f"(p10 {g['rpm'].quantile(.1):.0f} ~ "
                  f"p90 {g['rpm'].quantile(.9):.0f})")
    P("")

    # ---------------------------------------------------------- 확장특징
    if has_ext:
        P("=" * 78)
        P("[4] 확장 특징 : 값이 정상 범위인가")
        P("=" * 78)
        br = [f"band_ratio_{i}" for i in range(8)]
        s = df[br].sum(axis=1)
        P(f"  band_ratio 합계  median {s.median():.4f}  "
          f"(1.0 이어야 정상)  범위 {s.min():.4f}~{s.max():.4f}")
        P(f"  NaN 포함 행 {int(df[EXT].isna().any(axis=1).sum()):,} / {len(df):,}")
        P("")
        P("  DEMAG vs NORMAL 대역비율 중앙값 (재분배가 보이는지)")
        sub = df[df["fault_class"].isin(["NORMAL", "DEMAG"])]
        if len(sub):
            t = sub.groupby("fault_class")[br].median()
            for cls in t.index:
                P(f"    {cls:8s}" + "".join(f"{t.loc[cls, c]*100:7.1f}" for c in br))
            if len(t) == 2:
                d = (t.loc["DEMAG"] - t.loc["NORMAL"]) * 100
                P(f"    {'차이(%p)':8s}" + "".join(f"{d[c]:7.1f}" for c in br))
        P("")
        for c in ["centroid", "entropy", "rolloff", "log_flatness"]:
            x = df[c].astype("float64")
            P(f"  {c:13s} median {x.median():10.4f}  "
              f"std {x.std():10.4f}  NaN {int(x.isna().sum()):,}")
        P("")
        # float16 저장이라 언더플로우(0 으로 뭉갬)가 없었는지 확인
        f16 = [c for c in EXT + ["rms"]
               if c in df.columns and df[c].dtype == np.float16]
        if f16:
            worst = max(((c, float((df[c] == 0).mean() * 100)) for c in f16),
                        key=lambda t: t[1])
            P(f"  float16 컬럼 {len(f16)}개 / 0 비율 최대 "
              f"{worst[0]} {worst[1]:.2f}%  "
              f"{'통과' if worst[1] < 5 else '주의 - 언더플로우 의심'}")
            P("")

    # ---------------------------------------------------------- 구성
    P("=" * 78)
    P("[5] 데이터 구성 (세션 단위 검증이 가능한지)")
    P("=" * 78)
    P("  group_key 는 timestamp+seq+zsplit+sensor 다.")
    P("  세션 단위 CV 를 하려면 timestamp 를 group 으로 써야 한다.")
    P("  (같은 timestamp 의 zsplit 은 같은 주행이므로 train/test 에 갈리면 안 된다)")
    P("")
    ip = INDEX_PATH if INDEX_PATH.exists() else INDEX_PATH.with_suffix(".csv")
    if ip.exists():
        # 인덱스는 전체 그룹을 담고 있으니 표본이 아니라 전수로 셀 수 있다
        idx = pd.read_parquet(ip) if ip.suffix == ".parquet" else pd.read_csv(ip)
        P(f"  (샘플인덱스 전수 {len(idx):,}그룹 기준)")
        P("")
        classes = ["NORMAL", "ECC10", "ECC20", "DEMAG", "REDUC"]
        P(f"  {'차종':8s}" + "".join(f"{c:>9s}" for c in classes))
        for veh, g in idx.groupby("vehicle"):
            row = f"  {veh:8s}"
            for cls in classes:
                n = g[g["fault_class"] == cls]["timestamp"].nunique()
                row += f"{n:9d}"
            P(row)
        P("")
        P("  ^ 클래스당 고유 세션(timestamp) 수.")
        P("    10 미만이면 세션 단위 CV 가 불안정하다.")
        mn = (idx.groupby(["vehicle", "fault_class"])["timestamp"]
                 .nunique().min())
        P("")
        if mn >= 10:
            P(f"  판정: 통과 - 최소 {mn}세션. 세션 단위 CV 가능,")
            P("        확장특징 채택 여부를 제대로 판정할 수 있다.")
        else:
            P(f"  판정: 주의 - 최소 {mn}세션뿐인 조합이 있다.")
            P("        해당 조합은 CV 결과를 신뢰하기 어렵다.")
    else:
        P("  인덱스가 없어 세션 수를 셀 수 없다.")
    P("")

    P("=" * 78)
    P(f"소요 {time.time()-t0:.1f}s")
    P("=" * 78)
    flush()


if __name__ == "__main__":
    main()
