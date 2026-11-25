#!/usr/bin/env bash
# KISTI multi-evidence QA: 인덱스 → 데이터 구축 → 추론(k = 5, 10, 20, 40) → Qwen2.5-7B 판정
# KISTI 데이터·모델 경로는 README의 "경로 설정"에 있는 스크립트 상수에서 바꿉니다.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(conda info --base)/envs/pisco-ref/bin/python}
KPY=${KPY:-$(conda info --base)/envs/pisco-kisti/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
M=data/manifests
OURS=artifacts/step19/ckpt/rep_full/gen.pt
DLORA=artifacts/step12/ckpt/decoder_lora/lora.pt
predict() { $PY scripts/08_run_eval.py predict "$@"; }

# 1) 검색 인덱스 (≤120토큰 청크, e5) + 단일 근거 질문 검색
KISTI_CORPUS=sample KISTI_CHUNKER=v2 $KPY scripts/26_kisti_embed.py
$KPY scripts/26_kisti_prepare_m120.py retrieve --index v2
$PY scripts/26_kisti_prepare_m120.py manifest --ko --index v2
$PY scripts/28_kisti_multi_evidence.py single_full

# 2) 조합형 multi-evidence QA (Qwen2.5-32B, GPU 여러 개: GEN_GPUS) + 문서 수 k manifest
$KPY scripts/28_kisti_multi_evidence.py pairs
CUDA_VISIBLE_DEVICES=${GEN_GPUS:-0,1,2,3} $KPY scripts/28_kisti_multi_evidence.py cgen
$KPY scripts/28_kisti_multi_evidence.py cretrieve
$PY scripts/28_kisti_multi_evidence.py cmanifest
$PY scripts/24_prepare_chunk6_manifests.py kisti_compA_gold2_dist3_ko kisti_compB_gold2_dist3_ko
$KPY scripts/29_kisti_topk.py retrieve
$PY scripts/29_kisti_topk.py manifest

# 3) 추론: k = 5에서 다섯 방법, k = 10·20·40에서 원문 / PISCO / OURS
for T in A B; do
  S=$M/kisti_comp${T}_gold2_dist3_ko
  predict --system fusion --checkpoint $DLORA --manifest $S.json
  predict --system pisco --manifest ${S}_chunk6.json
  for D in 3 8 18 38; do
    S=$M/kisti_comp${T}_gold2_dist${D}_ko
    predict --system raw --manifest $S.json
    predict --system pisco --manifest $S.json
    predict --system fusion --checkpoint $OURS --manifest ${S}_full.json
  done
done

# 4) 판정 + paired bootstrap 집계 (artifacts/step28, 29, 31, 32)
for T in A B; do
  $KPY scripts/28_score.py judge --set comp${T}_gold
  $KPY scripts/31_c6_compare.py judge --set comp${T}_gold
  $KPY scripts/32_lora_compare.py judge_dm --set comp${T}_gold
  for K in 5 10 20 40; do $KPY scripts/29_kisti_topk.py judge --set comp${T}_k$K; done
done
$PY scripts/28_score.py score
$PY scripts/31_c6_compare.py score
$PY scripts/32_lora_compare.py score_dm
$PY scripts/29_kisti_topk.py score
