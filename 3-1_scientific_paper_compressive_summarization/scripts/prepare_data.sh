#!/usr/bin/env bash
# 데이터를 준비한다: IM-SciQA 분할, 논문 원문 내려받기, 부분 단위 나누기, 문단 중요도, 슬롯 배분 순서로 실행한다.
# GPU와 LLM 서버는 필요 없다.
#   bash scripts/prepare_data.sh             실험에 쓴 분할을 그대로 쓴다
#   SPLIT=new bash scripts/prepare_data.sh   분할을 새로 계산한다 (실험에 쓴 분할과 조금 다르다)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:.${PYTHONPATH:+:$PYTHONPATH}

if [ "${SPLIT:-reported}" = reported ]; then
  python -m pipeline.parse.split_imsciqa --reported   # data/imsciqa/ (실험에 쓴 분할)
  python -m pipeline.parse.fetch_pmc --from_split     # data/raw/pmc_bioc/ (분할에 든 문헌만, 이어받기 가능)
else
  python -m pipeline.parse.fetch_pmc                  # 분할에 문헌 제목이 필요해 전체 문헌을 먼저 받는다
  python -m pipeline.parse.split_imsciqa
fi
python -m pipeline.parse.build_units                                 # data/units/
python -m pipeline.allocate.build_tiers                              # data/alloc/tiers_{train,test}.json
python -m pipeline.allocate.build_alloc                              # data/alloc/{ratio16,oracle16}_{train,test}.json
