#!/usr/bin/env bash
# Serve Qwen3-8B with vLLM (one GPU, about 20GB). Stop with Ctrl-C.
# Usage: GPU=0 PORT=8000 VLLM=.venv-vllm/bin/vllm bash serve_llm.sh
set -euo pipefail
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
VLLM="${VLLM:-vllm}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$VLLM" serve Qwen/Qwen3-8B \
  --host 127.0.0.1 --port "$PORT" \
  --served-model-name qwen/qwen3-8b \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.8 \
  --max-num-seqs 32
