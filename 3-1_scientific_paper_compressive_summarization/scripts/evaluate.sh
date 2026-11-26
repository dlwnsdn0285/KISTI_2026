#!/usr/bin/env bash
# IM-SciQA 평가 문항에 답을 생성하고 gpt-oss-20b로 채점한다. 채점 서버 주소는 JUDGE_URL로 지정한다.
#   RECIPE=ours bash scripts/evaluate.sh                  학습한 모델 (runs/<RECIPE>/multidoc/best.pt)
#   bash scripts/evaluate.sh text full                    문헌 전문을 넣은 생성 모델
#   bash scripts/evaluate.sh text <이름> <텍스트 jsonl>    베이스라인 텍스트를 넣은 생성 모델
#   bash scripts/evaluate.sh judge <답 jsonl> ...          채점만
# NG는 GPU 수, PER_GPU는 GPU당 생성 프로세스 수다. CPU=1이면 GPU를 쓰지 않고, ANSWER_ARGS로 답 생성 인자를 더 넘길 수 있다.
# 중간에 멈춰도 다시 실행하면 끝난 문항은 건너뛴다.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src${PYTHONPATH:+:$PYTHONPATH} TOKENIZERS_PARALLELISM=false
NG=${NG:-1}; PER_GPU=${PER_GPU:-1}; RUNS=${RUNS:-runs}; OUT=${OUT:-$RUNS/eval}; JUDGE_URL=${JUDGE_URL:-http://localhost:8000/v1}
TEST=data/imsciqa/test.jsonl
mkdir -p "$OUT"

judge () { for f in "$@"; do python -m scisum.eval.judge --preds "$f" --base_url "$JUDGE_URL"; done; }

answers () {   # answers <출력 jsonl> <scisum.infer.answer 인자...>
  local out=$1; shift
  local n=$((NG * PER_GPU)) pids=()
  for i in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES=$([ "${CPU:-0}" = 1 ] || echo $((i % NG))) python -m scisum.infer.answer "$@" ${ANSWER_ARGS:-} --test_jsonl "$TEST" --shard "$i" --nshards "$n" \
      --out "$out.$i-of-$n" > "$out.$i-of-$n.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  # 조각을 합친다: 끊긴 줄은 버리고 같은 id는 하나만 남긴다
  python - "$out" $(seq -f "$out.%g-of-$n" 0 $((n - 1))) <<'PY'
import json, sys
rows = {}
for path in sys.argv[2:]:
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        rows[d["id"]] = d
with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in rows.values())
PY
  local got want; got=$(wc -l < "$out"); want=$(grep -c . "$TEST")
  [ "$got" -eq "$want" ] || { echo "답 $got / $want개: 각 조각 로그 $out.*-of-$n.log 확인" >&2; exit 1; }
}

case ${1:-latent} in
  judge) shift; judge "$@" ;;
  text)
    NAME=${2:?text 이름}; TEXTS=${3:-}
    answers "$OUT/$NAME.jsonl" --mode text ${TEXTS:+--text_jsonl "$TEXTS"}
    judge "$OUT/$NAME.jsonl" ;;
  latent)
    RECIPE=${RECIPE:-ours}
    case $RECIPE in ours|noqa) ALLOC=ratio16 ;; oracle) ALLOC=oracle16 ;; *) echo "RECIPE는 ours | oracle | noqa" >&2; exit 2 ;; esac
    answers "$OUT/$RECIPE.jsonl" --mode latent --ckpt "$RUNS/$RECIPE/multidoc/best.pt" --alloc "data/alloc/${ALLOC}_test.json"
    judge "$OUT/$RECIPE.jsonl" ;;
  *) echo "사용법: scripts/evaluate.sh [latent | text <이름> [텍스트 jsonl] | judge <답 jsonl>...]" >&2; exit 2 ;;
esac
