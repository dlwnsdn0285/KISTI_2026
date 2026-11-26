#!/usr/bin/env bash
# 전체 과정을 아주 작게 실행해 보는 스모크 테스트 (CPU에서 약 2분, 네트워크 필요).
# 실제 데이터와 실행 스크립트를 그대로 쓰고, 다음만 바꾼다.
#   데이터: IM-SciQA 학습 6 / 검증 2 / 평가 2문항과 그 논문 20편만 쓴다
#   학습 데이터 생성과 채점: 실제 LLM 서버 대신 모의 서버(tests/smoke/mock_llm_server.py)
#   학습과 답 생성: 작은 무작위 모델(tests/smoke/tiny_models.py)
# LLMLingua-2와 Keynote 베이스라인은 큰 모델이 필요해 빼고, 제목 + 초록 베이스라인만 실행한다.
#   bash tests/smoke/run_smoke.sh [작업 폴더]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
W=${1:-$(mktemp -d)}; PORT=${PORT:-8099}
mkdir -p "$W"; for d in src pipeline baselines scripts; do ln -sfn "$REPO/$d" "$W/$d"; done
cd "$W"
export PYTHONPATH=src:. CUDA_VISIBLE_DEVICES= CPU=1 TOKENIZERS_PARALLELISM=false
step () { echo; echo "=== $* ($(date +%H:%M:%S))"; }

step "1. 데이터: 분할 -> 축소 -> 원문 수집 -> 부분 단위 -> 문단 중요도 -> 배분"
python -m pipeline.parse.split_imsciqa --reported
python - <<'PY'
import json
for split, n in (("train", 6), ("valid", 2), ("test", 2)):
    rows = [json.loads(l) for l in open(f"data/imsciqa/{split}.jsonl")][:n]
    with open(f"data/imsciqa/{split}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    json.dump(sorted({p for r in rows for p in r["paper_ids"]}), open(f"data/imsciqa/{split}_paper_ids.json", "w"))
    print(f"[smoke] {split}: {len(rows)}문항")
PY
python -m pipeline.parse.fetch_pmc --from_split
python -m pipeline.parse.build_units
python -m pipeline.allocate.build_tiers
python -m pipeline.allocate.build_alloc

step "2. 학습 데이터 생성 (모의 LLM 서버)"
python "$REPO/tests/smoke/mock_llm_server.py" --port "$PORT" & SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
sleep 1
BASE_URL=http://127.0.0.1:$PORT/v1 bash scripts/synthesize.sh

step "3. 학습: 단일 문단 요약 학습 -> 다중 문헌 요약 활용 학습 (작은 모델)"
python "$REPO/tests/smoke/tiny_models.py" tiny
RECIPE=ours bash scripts/train.sh --encoder_id tiny/encoder --decoder_id tiny/decoder --lora_rank 4 --encoder_max_len 256 \
  --effective_batches 4,4,2 --batch_sizes 2,2,1 --limit 16 --n_valid 4 --log_every 1

step "4. 평가: 답 생성 + 채점"
JUDGE_URL=http://127.0.0.1:$PORT/v1 RECIPE=ours bash scripts/evaluate.sh

step "5. 베이스라인: 제목 + 초록 입력"
python -m baselines.titleabs
ANSWER_ARGS="--decoder_id tiny/decoder" JUDGE_URL=http://127.0.0.1:$PORT/v1 \
  bash scripts/evaluate.sh text titleabs data/baselines/titleabs_test.jsonl

step "확인"
python - <<'PY'
import json, os
def n(path): return sum(1 for _ in open(path))
units = os.listdir("data/units"); para = json.load(open("data/synth/paraphrase.json")); qa = json.load(open("data/synth/qa.json"))
types = {}
for m in qa.values():
    for items in m.values():
        for x in items: types[x["type"]] = types.get(x["type"], 0) + 1
checks = {
    "부분 단위 파일": len(units) > 0,
    "배분 파일 4개": all(os.path.exists(f"data/alloc/{a}_{g}.json") for a in ("ratio16", "oracle16") for g in ("train", "test")),
    "재진술": sum(len(m) for m in para.values()) > 0,
    "질의응답 3종 (atomic/complex/gist)": set(types) == {"atomic", "complex", "gist"},
    "학습 구간 3개 last.pt": all(os.path.exists(f"runs/ours/{p}/last.pt") for p in ("sentence", "paragraph", "multidoc")),
    "best.pt": os.path.exists("runs/ours/multidoc/best.pt"),
    "답 2문항": n("runs/eval/ours.jsonl") == 2,
    "채점 요약": json.load(open("runs/eval/ours.judged.summary.json"))["n_scored"] == 2,
    "베이스라인 채점": json.load(open("runs/eval/titleabs.judged.summary.json"))["n_scored"] == 2,
}
print(f"[smoke] 부분 단위 {len(units)}편, 재진술 {sum(len(m) for m in para.values())}개, 질의응답 {types}")
for k, ok in checks.items():
    print(f"[smoke] {'통과' if ok else '실패'}: {k}")
raise SystemExit(0 if all(checks.values()) else 1)
PY
echo "[smoke] 작업 폴더: $W"
