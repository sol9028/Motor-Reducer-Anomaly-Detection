#!/usr/bin/env bash
# 파이4 전체 스윕: 1/2/4 코어 x 배치 1,8,32
#
# 스레드 수는 numpy/numba import 전에 정해져야 해서 프로세스를 따로 띄운다.
#
# 반복 횟수를 배치별로 다르게 준다. 총 소요시간은 reps x batch x 청크당지연 이라
# 모든 배치에 같은 reps 를 주면 batch=32 하나가 전체 시간을 잡아먹는다.
# (48채널 기준 파이4에서 batch=32, reps=30 이면 그 칸 하나가 20분이다.)
set -euo pipefail
cd "$(dirname "$0")"

MODELS="${1:-./models}"
PY="${PYTHON:-python}"

run() {  # run <threads> <batch> <reps> <suffix>
  echo ""
  echo "---- threads=$1 batch=$2 reps=$3 ----"
  "$PY" bench_latency.py \
    --models "$MODELS" \
    --threads "$1" \
    --batch "$2" \
    --reps "$3" \
    --warmup 3 \
    --out "bench_pi4_t$1_b$2.json"
}

for T in 1 2 4; do
  echo ""
  echo "################ threads=$T ################"
  run "$T" 1  30   # 단일 청크 지연 — 실시간 판정의 핵심 지표. 표본 많이.
  run "$T" 8  10
  run "$T" 32 5    # 처리량 확인용. 배치가 크니 5회면 충분하다.
done

echo ""
echo "완료. bench_pi4_t*_b*.json 을 PC 로 가져가서 비교하면 된다."
