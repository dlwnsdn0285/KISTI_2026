#!/bin/bash
# FtR main run (4 conditions, 5 loops) -> per-loop metrics. Needs the vLLM servers from launch_servers.sh.
# Extra replicas: JUDGE_URL=http://127.0.0.1:8101/v1,http://127.0.0.1:8103/v1 (comma-separated, round-robin)
cd "$(dirname "$0")"
mkdir -p ../logs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}   # bge retriever + DeBERTa NLI share the refiner GPU
PY=${PYTHON:-python}
$PY -u run_ftr.py --loops 5 --out ../results/run_main "$@" > ../logs/run.log 2>&1
$PY -u analyze.py --run ../results/run_main > ../logs/analyze.log 2>&1
echo DONE
