"""IM-SciQA 논문 원문을 NCBI BioC-PMC API에서 내려받는다.

--from_split을 주면 data/imsciqa/에 나뉜 학습/검증/평가 문항의 논문만 받는다.
이미 받은 논문은 건너뛰고, 실패한 논문은 다음 실행에서 다시 받는다. 요청 제한에 걸리면 기다렸다가 다시 요청한다.

출력: data/raw/pmc_bioc/<PMCID>.json
실행: python -m pipeline.parse.fetch_pmc [--from_split]
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request

from pipeline.common.io import save_json, split_paper_ids
from pipeline.parse.imsciqa_source import all_paper_ids, load_source_rows

BIOC_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pid}/unicode"
OUT_DIR = os.path.join("data", "raw", "pmc_bioc")
MIN_INTERVAL = 0.34  # 초당 3회 이하


def norm_pmcid(pid: str) -> str:
    pid = str(pid).strip()
    return pid if pid.upper().startswith("PMC") else "PMC" + pid


def fetch_passages(pmcid: str, tries: int = 5) -> list:
    """요청 제한(429), 서버 오류(5xx), 연결 오류는 2, 4, 8, 16초 기다렸다가 다시 요청한다."""
    req = urllib.request.Request(BIOC_URL.format(pid=pmcid), headers={"User-Agent": "scisum-data/0.1"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                payload = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
        time.sleep(2 ** (attempt + 1))
    docs = payload[0]["documents"] if isinstance(payload, list) else payload["documents"]
    return docs[0].get("passages", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_split", action="store_true", help="data/imsciqa/ split에 든 문헌만 받는다")
    args = ap.parse_args()
    if args.from_split:
        pids = sorted({norm_pmcid(p) for s in ("train", "valid", "test") for p in split_paper_ids(s)})
    else:
        pids = [norm_pmcid(p) for p in all_paper_ids(load_source_rows())]
    todo = [p for p in pids if not os.path.exists(os.path.join(OUT_DIR, f"{p}.json"))]
    print(f"[fetch] 문헌 {len(pids)}편 중 {len(todo)}편 내려받기 -> {OUT_DIR}", flush=True)
    failed = 0
    for i, pid in enumerate(todo, 1):
        t0 = time.time()
        try:
            save_json({"pmcid": pid, "passages": fetch_passages(pid)}, os.path.join(OUT_DIR, f"{pid}.json"))
        except Exception as e:  # 네트워크 오류나 비공개 문헌: 기록만 하고 계속
            failed += 1
            print(f"[fetch] {pid} 실패: {type(e).__name__}: {e}", flush=True)
        time.sleep(max(0.0, MIN_INTERVAL - (time.time() - t0)))
        if i % 200 == 0:
            print(f"[fetch] {i}/{len(todo)} (실패 {failed})", flush=True)
    print(f"[fetch] 완료: 실패 {failed}편")


if __name__ == "__main__":
    main()
