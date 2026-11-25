#!/usr/bin/env bash
# 응답시간: 원문 / 매 질의 압축(online) / OURS(memory 캐시 + 추가 블록), KISTI 질문, 문서 N = 5 … 320
# run/kisti.sh의 인덱스(1단계)가 먼저 있어야 합니다. 모든 N을 같은 GPU에서 측정합니다.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$(conda info --base)/envs/pisco-ref/bin/python}
KPY=${KPY:-$(conda info --base)/envs/pisco-kisti/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0}

$KPY scripts/27_latency_scaling.py prepare
$PY scripts/27_latency_scaling.py run --ns 5 10 20 40 80 160 320
$PY scripts/27_latency_scaling.py summarize           # artifacts/step27/{results.json,curves.csv}
$KPY scripts/27_plot_latency.py                       # artifacts/step27/{acceleration,absolute}.{png,pdf}
