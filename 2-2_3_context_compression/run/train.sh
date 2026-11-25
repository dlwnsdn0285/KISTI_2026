#!/usr/bin/env bash
# teacher 라벨·memory 캐시 → 추가 블록 생성기(OURS) → 비교군 Decoder LoRA. SMOKE=1이면 몇 step만 학습
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(conda info --base)/envs/pisco-ref/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}

$PY scripts/11_teacher_labels.py                       # 원문 Mistral-7B 답변 = KD 라벨
$PY scripts/11_precompute_memory.py                    # PISCO memory 캐시
$PY scripts/19_train_full_gen.py --variant text --no-residual --out artifacts/step19/ckpt/rep_full ${SMOKE:+--smoke}
$PY scripts/12_train_lora.py --target decoder --out artifacts/step12/ckpt/decoder_lora ${SMOKE:+--smoke}
