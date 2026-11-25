#!/usr/bin/env bash
# 영문 벤치마크 추론: 원문 / PISCO / Decoder LoRA / Baseline-6 memory / OURS
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(conda info --base)/envs/pisco-ref/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
M=data/manifests
OURS=artifacts/step19/ckpt/rep_full/gen.pt
DLORA=artifacts/step12/ckpt/decoder_lora/lora.pt
predict() { $PY scripts/08_run_eval.py predict "$@"; }

# 원문 vs PISCO 격차
for S in nq_gold1_dist4_n1000 hotpotqa_2hop_gold2_dist3_n1000 2wiki_2hop_gold2_dist3_n1000 \
         musique_2hop_gold2_dist3_n1000 multihoprag_2ev_yesno_forced_n882; do
  predict --system raw --manifest $M/$S.json
  predict --system pisco --manifest $M/$S.json
done
# MuSiQue 오류 유형 (하위 질문)
for S in musique_2hop_hop1_n1000 musique_2hop_hop2_n1000; do
  predict --system pisco --manifest $M/$S.json
done
# OURS / Decoder LoRA / Baseline-6 memory
for S in musique_2hop_gold2_dist3_n1000 multihoprag_2ev_yesno_forced_n882 nq_gold1_dist4_n1000; do
  predict --system fusion --checkpoint $OURS --manifest $M/${S}_full.json
  predict --system fusion --checkpoint $DLORA --manifest $M/$S.json
  predict --system pisco --manifest $M/${S}_chunk6.json
done
