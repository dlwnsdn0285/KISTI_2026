#!/usr/bin/env bash
# 단일 문단 요약 학습 단계에 쓸 학습 데이터(재진술, 단일 사실 QA, 복합 QA, 요지 QA)를 Qwen3.8-27B로 생성한다.
# 먼저 vLLM 서버를 띄워야 한다:
#   vllm serve Qwen/Qwen3.8-27B --reasoning-parser qwen3 --enable-prefix-caching --port 8000
# 서버 주소가 다르면 BASE_URL로 지정한다. 중간에 멈춰도 다시 실행하면 이어서 생성한다.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.${PYTHONPATH:+:$PYTHONPATH}
BASE_URL=${BASE_URL:-http://localhost:8000/v1}; MODEL=${MODEL:-Qwen/Qwen3.8-27B}

python -m pipeline.synthesize.paraphrase_qa --base_url "$BASE_URL" --model "$MODEL"   # 추론 끔
python -m pipeline.synthesize.gist_qa --base_url "$BASE_URL" --model "$MODEL"         # 추론 xhigh
python -m pipeline.synthesize.build_training_data                                     # data/synth/{paraphrase,qa}.json
