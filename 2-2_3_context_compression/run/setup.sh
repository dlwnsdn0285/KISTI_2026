#!/usr/bin/env bash
# 모델 다운로드(PISCO, Mistral-7B) + 공개 데이터 manifest (HotpotQA, 2Wiki, MuSiQue, MultiHop-RAG, NQ)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(conda info --base)/envs/pisco-ref/bin/python}

$PY scripts/03_load_reference.py fetch
$PY scripts/09c_prepare_2hop_fixed_context.py          # HotpotQA / 2Wiki / MuSiQue 평가
[ -f data/nq_dpr/biencoder-nq-dev.json.gz ] || \
  wget -P data/nq_dpr https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
$PY scripts/09d_prepare_nq_gold_fixed_context.py       # NQ 평가
$PY scripts/10b_prepare_multihoprag.py                 # MultiHop-RAG 평가
$PY scripts/10b_prepare_multihoprag_forced.py
$PY scripts/10a_prepare_musique_hops.py                # MuSiQue 하위 질문 (오류 유형 분석)
$PY scripts/11_prepare_fusion_train.py                 # 학습: HotpotQA train 8,000
$PY scripts/19_prepare_full_manifests.py               # OURS 입력 (*_full)
$PY scripts/24_prepare_chunk6_manifests.py             # Baseline-6 memory (*_chunk6)
