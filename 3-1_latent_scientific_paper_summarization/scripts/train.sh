#!/usr/bin/env bash
# 모델을 학습한다. 단일 문단 요약 학습 단계와 다중 문헌 요약 활용 학습 단계를 한 번에 이어서 실행한다.
#   RECIPE  ours: 원문 복원 + 재진술 + 질의응답, oracle: ours + 오라클 배분, noqa: 원문 복원 + 재진술
#   NG      GPU 수 (1, 2, 4, 8)
#   BS      GPU당 배치 크기, 문장 복원/문단/다중 문헌 순서 (기본 4,4,4)
# 예: NG=4 RECIPE=ours bash scripts/train.sh   (결과: runs/ours/multidoc/best.pt)
# 뒤에 붙인 인자는 python -m scisum.train.run에 그대로 넘어간다.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH} TOKENIZERS_PARALLELISM=false
RECIPE=${RECIPE:-ours}; NG=${NG:-1}; RUNS=${RUNS:-runs}; BS=${BS:-4,4,4}
[ "$(tr -cd , <<< "$BS")" = ",," ] || { echo "BS는 문장 복원,문단,다중 문헌 순서로 세 값 (예: 4,4,4)" >&2; exit 2; }

accelerate launch --num_processes "$NG" --mixed_precision no --main_process_port "${PORT:-29500}" -m scisum.train.run \
  --recipe "$RECIPE" --out_dir "$RUNS/$RECIPE" --batch_sizes "$BS" "$@"
