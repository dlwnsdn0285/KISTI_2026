#!/usr/bin/env bash
# 표 9의 베이스라인을 실행한다. 학습하지 않은 생성 모델(Qwen3.5-9B)에 논문마다 아래 텍스트를 넣고 답한 뒤 같은 방식으로 채점한다.
#   full        문헌 전문
#   titleabs    제목 + 초록 (SurveyGen 방식)
#   llmlingua2  LLMLingua-2로 전문을 제목 + 초록 길이까지 줄인 것
#   keynote     생성 모델이 전문을 읽고 쓴 요점 요약 (DeepSurvey 방식)
# llmlingua2와 keynote 입력을 만들 때 GPU 하나를 쓴다. 답 생성과 채점 설정은 scripts/evaluate.sh와 같다.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.${PYTHONPATH:+:$PYTHONPATH}

python -m baselines.titleabs     # data/baselines/titleabs_test.jsonl
python -m baselines.llmlingua2   # data/baselines/llmlingua2_test.jsonl
python -m baselines.keynote      # data/baselines/keynote_test.jsonl

bash scripts/evaluate.sh text full
for name in titleabs llmlingua2 keynote; do
  bash scripts/evaluate.sh text "$name" "data/baselines/${name}_test.jsonl"
done
