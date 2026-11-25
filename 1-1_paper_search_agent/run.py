"""Entry point: take a survey title (or a file of titles) and write candidates and the final list.

Writes <out>/<qid>.json per query. Existing files are skipped, so a run can be resumed.

    python run.py --config config.yaml --backend s2 \
        --title "Graph Neural Networks for Recommender Systems" --cutoff 2024-06-01
    python run.py --config config.yaml --backend s2 --queries data/eval/queries.jsonl \
        --blocklist data/eval/blocklist.jsonl --out results/s2
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time

from psa import util

util.redact_install()


def parse_args():
    ap = argparse.ArgumentParser(description="서베이 제목 → 관련 논문 후보와 최종 목록")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--title", help="서베이 제목(또는 연구 질문) 하나")
    src.add_argument("--queries", help="queries.jsonl: 한 줄에 {qid, title, cutoff}")
    ap.add_argument("--cutoff", help="--title 과 함께: 이 날짜 이후 논문은 쓰지 않는다 (YYYY-MM-DD)")
    ap.add_argument("--config", default=None, help="config.yaml (기본값은 psa/util.py DEFAULTS)")
    ap.add_argument("--backend", choices=["s2", "datalake"], default="s2",
                    help="이웃 · 메타데이터를 읽을 저장소")
    ap.add_argument("--blocklist", default=None,
                    help="후보에서 뺄 논문 jsonl (doi/arxiv/corpusid/title). 없으면 차단 없음")
    ap.add_argument("--out", default="results/run", help="결과 폴더")
    ap.add_argument("--train", action="store_true",
                    help="검색 대신 확장 점수 식을 학습한다 (--queries, --gold 필요) → <out>/prior.json")
    ap.add_argument("--gold", default=None, help="--train 과 함께: gold.jsonl")
    args = ap.parse_args()
    if args.cutoff and not args.title:
        ap.error("--cutoff 는 --title 과 함께 쓴다 (목록 파일은 cutoff 필드)")
    if args.train and not (args.queries and args.gold):
        ap.error("--train 은 --queries 와 --gold 가 필요하다")
    return args


def queries_of(args):
    if args.title:
        qid = re.sub(r"[^a-z0-9]+", "_", args.title.lower()).strip("_")[:80]
        cutoff = dt.date.fromisoformat(args.cutoff) if args.cutoff else None
        rows = [{"qid": qid, "title": args.title, "cutoff": cutoff}]
    else:
        rows = util.load_queries(args.queries)
    return rows


def main():
    args = parse_args()
    cfg = util.load_config(args.config)
    # only S2 search and the vLLM server are reachable
    util.netguard_install({"api.semanticscholar.org", util.host_of(cfg["llm"]["url"])})

    from psa.expand import MiniLM, Relevance
    from psa.judge import VLLMJudge
    from psa.pipeline import Config, candidates_of, run_query
    from psa.search import JUDGE_PROMPT, QUERY_PROMPT, LLMQuerySearcher, SearchCache
    from psa.store import open_backend

    blocklist = util.Blocklist(args.blocklist)
    if args.train:
        from psa import train
        harvest_dir = os.path.join(args.out, "harvest")
        train.harvest(cfg, queries_of(args), args.backend, blocklist, harvest_dir)
        train.fit(harvest_dir, args.gold, os.path.join(args.out, "prior.json"))
        return 0
    run_cfg = Config.for_backend(args.backend, cfg["prior"])
    os.makedirs(args.out, exist_ok=True)
    env_path = os.path.join(args.out, "_environment.json")
    if not os.path.exists(env_path):
        with open(env_path, "w", encoding="utf-8") as fh:
            json.dump({"backend": args.backend, "prompts": [QUERY_PROMPT, JUDGE_PROMPT],
                       "llm": cfg["llm"], "embed": cfg["embed"],
                       "blocklist": args.blocklist, "blocklist_size": len(blocklist),
                       "config": run_cfg.__dict__,
                       "started": dt.datetime.now().isoformat(timespec="seconds")},
                      fh, ensure_ascii=False, indent=1)
    print(f"psa · 저장소 {args.backend} "
          f"· LLM {cfg['llm']['url']} · 차단 {len(blocklist):,}", flush=True)

    rows = queries_of(args)
    model = MiniLM(cfg["embed"]["model"], cfg["embed"]["device"], cfg["embed"]["batch_size"])
    for i, q in enumerate(rows, 1):
        path = os.path.join(args.out, f"{q['qid']}.json")
        if os.path.exists(path):
            print(f"[{i}/{len(rows)}] {q['qid']} 이미 있다 — 건너뛴다", flush=True)
            continue
        meter = util.Meter(q["qid"], backend=args.backend)
        store = open_backend(args.backend, meter=meter, config=cfg["store"])
        cache = SearchCache(cfg, meter=meter)
        searcher = LLMQuerySearcher(
            cache=cache, cfg=cfg, key_of=store.key_of,
            no_key_label="no_corpusid" if args.backend == "s2" else "no_doi", meter=meter)
        judge = VLLMJudge(url=cfg["llm"]["url"], model=cfg["llm"]["model"],
                          prompt_file=JUDGE_PROMPT, parallel=cfg["llm"]["parallel"],
                          meter=meter)
        relevance = Relevance(q["title"], model)
        start = time.perf_counter()
        error, trace = "", {}
        try:
            trace = run_query(q["title"], q["qid"], cutoff=q["cutoff"], store=store,
                              searcher=searcher, judge=judge, relevance=relevance,
                              blocklist=blocklist, cfg=run_cfg, meter=meter)
            error = trace.get("error", "")
        except Exception as exc:  # noqa: BLE001 - keep going if one query fails
            error = util.redact(f"{type(exc).__name__}: {exc}")
        wall = time.perf_counter() - start
        summary = meter.summary()
        summary["search_cache"] = cache.stats()
        summary["netguard_blocked"] = util.netguard_report()
        candidates = candidates_of(trace, args.backend) if trace else []
        keep_out = ("judged", "output", "cosine_all", "pool_ids", "collected_ids", "via",
                    "owner", "seed_ids")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"qid": q["qid"], "question": q["title"],
                       "cutoff": q["cutoff"].isoformat() if q["cutoff"] else None,
                       "system": "psa", "backend": args.backend,
                       "wall_sec": round(wall, 2),
                       "error": error, "candidates": candidates, "meter": summary,
                       "trace": {k: v for k, v in trace.items() if k not in keep_out}},
                      fh, ensure_ascii=False, default=str)
        st = summary.get("stage_wall_sec", {})
        print(f"[{i}/{len(rows)}] {q['qid']}: 후보 {len(candidates):,} · "
              f"출력 {sum(1 for c in candidates if c['in_output']):,} · {wall:.0f}s "
              f"(검색 {st.get('search', 0):.0f} / 확장 {st.get('expand', 0):.0f} / "
              f"판정 {st.get('judge', 0):.0f}) · 검색 캐시 "
              f"{summary['search_cache']['hit']}적중/{summary['search_cache']['api_call']}API"
              + (f" · 오류 {error}" if error else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
