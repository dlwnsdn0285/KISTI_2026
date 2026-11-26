"""생성한 재진술과 질의응답을 단일 문단 요약 학습 단계에서 읽는 파일 두 개로 합친다.

단일 사실 QA와 복합 QA는 보충 질문이 끝까지 나온(문단의 사실을 다 묻은) 문단의 것만 쓰고, 같은 문단의 중복 질문은 하나만 남긴다.

입력: data/synth/raw/
출력: data/synth/paraphrase.json (재진술), data/synth/qa.json (단일 사실 QA, 복합 QA, 요지 QA)
실행: python -m pipeline.synthesize.build_training_data
"""
import os
import re
from collections import Counter, defaultdict

from pipeline.common.io import ALLOC_DIR, load_json, read_jsonl, save_json
from pipeline.synthesize.inputs import generator_papers, raw_path

OUT_DIR = os.path.join("data", "synth")


def clean_markdown(text: str) -> str:
    return re.sub(r"(?m)^#+\s*", "", text).replace("**", "").strip()


def paper_data(pid: str, stats: Counter):
    """(재진술 {idx: [text]}, QA {idx: [qa]}) 한 문헌분."""
    paraphrase, qa = {}, defaultdict(list)
    seen = defaultdict(set)

    def add(idx: str, kind: str, q: str, a: str) -> None:
        key = " ".join(q.lower().split())
        if q.strip() and a.strip() and key not in seen[idx]:
            seen[idx].add(key)
            qa[idx].append({"type": kind, "question": q.strip(), "answer": a.strip()})
            stats[kind] += 1

    for r in read_jsonl(raw_path("gist_qa", pid)):
        for pt in r["points"]:
            add(str(r["idx"]), "gist", pt["question"], pt["answer"])
    for r in read_jsonl(raw_path("paraphrase_qa", pid)):
        idx = str(r["idx"])
        if r["coverage_complete"]:
            for kind in ("atomic", "complex"):
                for x in r[kind]:
                    add(idx, kind, x["question"], x["answer"])
        else:
            stats["qa_dropped_incomplete_units"] += 1
        if r["paraphrase"]:
            paraphrase[idx] = [clean_markdown(r["paraphrase"])]
            stats["paraphrase"] += 1
    return paraphrase, dict(qa)


def coverage(paraphrase: dict, qa: dict, pids: list) -> None:
    alloc = load_json(os.path.join(ALLOC_DIR, "ratio16_train.json"))
    units = [(p, str(i)) for p in pids if p in alloc for i, _ in alloc[p]["units"]]
    has = Counter()
    for p, i in units:
        has["paraphrase"] += i in paraphrase.get(p, {})
        kinds = {x["type"] for x in qa.get(p, {}).get(i, [])}
        has["qa"] += bool(kinds)
        for k in kinds:
            has[k] += 1
    print(f"[synth] 학습 문헌 부분 단위 {len(units)}개 중: " +
          ", ".join(f"{k} {has[k] / max(len(units), 1):.1%}" for k in ("paraphrase", "qa", "gist", "atomic", "complex")))


def main() -> None:
    stats = Counter()
    paraphrase, qa = {}, {}
    pids = generator_papers()
    for pid in pids:
        pp, q = paper_data(pid, stats)
        if pp:
            paraphrase[pid] = pp
        if q:
            qa[pid] = q
    save_json(paraphrase, os.path.join(OUT_DIR, "paraphrase.json"))
    save_json(qa, os.path.join(OUT_DIR, "qa.json"))
    print(f"[synth] 문헌 {len(pids)}편: {dict(stats)}")
    coverage(paraphrase, qa, pids)


if __name__ == "__main__":
    main()
