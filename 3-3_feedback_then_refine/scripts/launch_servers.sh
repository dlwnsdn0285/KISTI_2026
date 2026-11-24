#!/bin/bash
# Run on the GPU server (8x NVIDIA L40S 46GB). Uses 3 of 8 GPUs (+3 more with judge2/refiner2):
#   GPU 0,1 : judge  (Llama-3.3-70B-Instruct, pre-quantized FP8, TP=2)   -> :8101
#   GPU 2   : refiner/feedback (Qwen3-30B-A3B-Instruct-2507, FP8)       -> :8102
#   GPU 2   : (remaining memory) bge-large retriever + DeBERTa NLI in the orchestrator
#   GPU 3,4 : judge2   (optional second judge replica)                  -> :8103
#   GPU 5   : refiner2 (optional second refiner replica)                -> :8104
# usage: launch_servers.sh [judge|refiner|all|judge2|refiner2]
# PY_BIN: directory containing the `vllm` executable (default: taken from PATH)
PY=${PY_BIN:-$(dirname "$(command -v vllm)")}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LOG=${LOG_DIR:-$ROOT/logs}
mkdir -p $LOG
WHAT=${1:-all}

if [[ $WHAT == judge || $WHAT == all ]]; then
tmux new-session -d -s ftr_judge "CUDA_VISIBLE_DEVICES=0,1 $PY/vllm serve RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic \
  --served-model-name judge --tensor-parallel-size 2 \
  --max-model-len 12288 --gpu-memory-utilization 0.92 --max-num-seqs 64 \
  --host 127.0.0.1 --port 8101 2>&1 | tee $LOG/judge_server.log"
fi

if [[ $WHAT == refiner || $WHAT == all ]]; then
tmux new-session -d -s ftr_refiner "CUDA_VISIBLE_DEVICES=2 $PY/vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --served-model-name refiner judge_tmp \
  --max-model-len 16384 --gpu-memory-utilization 0.80 --max-num-seqs 32 \
  --host 127.0.0.1 --port 8102 2>&1 | tee $LOG/refiner_server.log"
fi

if [[ $WHAT == judge2 ]]; then   # second judge replica for throughput
tmux new-session -d -s ftr_judge2 "CUDA_VISIBLE_DEVICES=3,4 $PY/vllm serve RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic \
  --served-model-name judge --tensor-parallel-size 2 \
  --max-model-len 12288 --gpu-memory-utilization 0.92 --max-num-seqs 64 \
  --host 127.0.0.1 --port 8103 2>&1 | tee $LOG/judge2_server.log"
fi

if [[ $WHAT == refiner2 ]]; then   # second refiner replica for throughput
tmux new-session -d -s ftr_refiner2 "CUDA_VISIBLE_DEVICES=5 $PY/vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --served-model-name refiner judge_tmp \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 32 \
  --host 127.0.0.1 --port 8104 2>&1 | tee $LOG/refiner2_server.log"
fi
