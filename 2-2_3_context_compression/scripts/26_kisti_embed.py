"""Step 26 (KISTI): embed the KISTI paper corpus in PISCO-sized chunks for top-5 retrieval.

Restored from this server's lost "Step 23/25 (KISTI)" (see STEP_LOG Step 26); behaviour unchanged.

  # corpus: KISTI_CORPUS=full (default, ~481k papers) or KISTI_CORPUS=sample (KISTI_sample, 20,252 papers)
  # one worker per GPU (shard i of n); resumable, finished blocks are skipped
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g ~/anaconda3/envs/refrag/bin/python scripts/26_kisti_embed.py embed --shard $g --num-shards 4 &
  done
  ~/anaconda3/envs/refrag/bin/python scripts/26_kisti_embed.py papers   # paper metadata table (CPU)
  ~/anaconda3/envs/refrag/bin/python scripts/26_kisti_embed.py verify   # counts / sanity checks
  ~/anaconda3/envs/refrag/bin/python scripts/26_kisti_embed.py smoke    # 40 papers into OUT_DIR/_smoke

Corpus     (분류) 국내 논문 전문 텍스트 데이터셋 under the KISTI_CORPUS root
Units      same as kisti-rag1 get_paper_data: abstract (ko, else en) = section 0, then each body_text
           section ("\\n".join(text)) in order. A chunk never crosses a section.
Chunking   sentence spans (end punctuation + whitespace, or newlines) packed greedily up to
           MAX_TOKENS Mistral-7B-Instruct-v0.2 tokens (no special tokens), so PISCO (128-token
           encoder window) reads every chunk whole. A sentence longer than MAX_TOKENS is split
           as KISTI_CHUNKER says (v1: every MAX_TOKENS tokens; v2: equal pieces at word/clause
           boundaries, then short-chunk merge). Chunk text is an exact substring of the section
           text: (paper_id, section_idx, char_start, char_end) locates it. Chunks with fewer than
           MIN_CHARS non-space characters are dropped (kisti-rag1 remove_small_chunks threshold 20).
Embedding  intfloat/multilingual-e5-large-instruct, document side without instruction (as in
           kisti-rag1), normalized, stored float16. Chunk text only (no title prefix).
Output     OUT_DIR/blocks/s{shard}_b{block:05d}.{npy,parquet}: vectors [n,1024] fp16 and rows
           (chunk_id, paper_id, rel_path, section_idx, section, char_start, char_end, n_tokens, text)
           in the same order. A block = BLOCK_PAPERS consecutive papers of the shard. Files are
           written to a temp name and renamed, so a block exists only if complete.
"""

import os

# The HF tokenizers library parallelizes encode_batch over all cores (Rayon) in every process;
# with several chunking workers per GPU this saturated the shared server. Pin it to one thread.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import re
import sys
import time
from pathlib import Path

# KISTI_CORPUS=sample|full selects the corpus (an env var, so the spawned chunking workers see it too)
CORPORA = {
    "full": (Path("/data2/jwlee/data/KISTI/KISTI"), Path("/data3/jwlee/kisti_full_e5_m120")),
    "sample": (Path("/home/jwlee/kisti-rag1/data/KISTI_sample"), Path("/data3/jwlee/kisti_sample_e5_m120")),
}
_root, OUT_DIR = CORPORA[os.environ.get("KISTI_CORPUS", "full")]
# KISTI_CHUNKER=v1: a sentence over MAX_TOKENS is cut every MAX_TOKENS tokens (short tails become chunks).
# KISTI_CHUNKER=v2: such a sentence is split into equal-size pieces at word boundaries,
# preferring clause punctuation, and chunks under SHORT_TOKENS are merged into a neighbour when it fits.
CHUNKER = os.environ.get("KISTI_CHUNKER", "v1")
if CHUNKER == "v2":
    OUT_DIR = OUT_DIR.with_name(OUT_DIR.name + "_v2")
CORPUS = _root / "(분류) 국내 논문 전문 텍스트 데이터셋"
QA_ROOT = _root / "(분류) 국내 논문 QA 데이터셋"
MISTRAL_TOKENIZER = "/data2/jwlee/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/63a8b081895390a26e140280378bc85ec8bce07a/tokenizer.json"
E5_DIR = "/data2/jwlee/huggingface/hub/models--intfloat--multilingual-e5-large-instruct/snapshots/274baa43b0e13e37fafa6428dbc7938e62e5c439"
MAX_TOKENS = 120
MIN_CHARS = 20
BLOCK_PAPERS = 2000
BATCH_SIZE = 256
SENT_BOUNDARY = re.compile(r"(?<=[.?!。])\s+|\n+")
SPLIT_WINDOW = 20      # v2: a cut may move this many tokens away from the equal-split target
SHORT_TOKENS = 32      # v2: chunks shorter than this are merged into a neighbour when the result fits
CLAUSE_END = set(",;:)]")


def paper_list():
    """Deterministic order of all papers (relative paths journal/year/file)."""
    cache = OUT_DIR / "paper_list.json"
    if cache.exists():
        return json.loads(cache.read_text())
    paths = sorted(str(p.relative_to(CORPUS)) for p in CORPUS.glob("*/*/*.json"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(paths, ensure_ascii=False))
    os.replace(tmp, cache)
    return paths


def choose_language(d):
    if isinstance(d, dict):
        return d.get("ko") or d.get("en") or ""
    return d or ""


def sections(content):
    out = []
    abstract = choose_language(content.get("abstract", {}))
    out.append(("abstract", abstract))
    for s in content.get("body_text", []):
        if "text" in s:
            out.append((s.get("section", ""), "\n".join(s["text"])))
    return out


def sentence_spans(text):
    spans, start = [], 0
    for m in SENT_BOUNDARY.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def chunk_section(tok, text):
    """-> list of (char_start, char_end) packed to <= MAX_TOKENS Mistral tokens."""
    spans = [s for s in sentence_spans(text) if text[s[0]:s[1]].strip()]
    if not spans:
        return []
    encs = tok.encode_batch([text[a:b] for a, b in spans], add_special_tokens=False)
    pieces = []   # (start, end, n_tokens) with n_tokens <= MAX_TOKENS
    for (a, b), enc in zip(spans, encs):
        n = len(enc.ids)
        if n <= MAX_TOKENS:
            pieces.append((a, b, n))
            continue
        offs = enc.offsets
        if CHUNKER == "v1":
            for i in range(0, n, MAX_TOKENS):
                seg = offs[i:i + MAX_TOKENS]
                pieces.append((a + seg[0][0], a + seg[-1][1], len(seg)))
        else:
            pieces.extend(split_long_sentence(text, a, offs))
    def pack(group, budget):
        out, cur = [], []
        for p in group:
            if cur and sum(q[2] for q in cur) + p[2] > budget:
                out.append(cur)
                cur = []
            cur.append(p)
        out.append(cur)
        return out

    # The token count of joined sentences can exceed the sum of their counts by a few tokens,
    # so every packed chunk is re-tokenized and re-packed with a smaller budget until it fits.
    final, todo = [], [(g, MAX_TOKENS) for g in pack(pieces, MAX_TOKENS)]
    while todo:
        encs = tok.encode_batch([text[g[0][0]:g[-1][1]] for g, _ in todo], add_special_tokens=False)
        nxt = []
        for (g, budget), enc in zip(todo, encs):
            n = len(enc.ids)
            if n <= MAX_TOKENS or len(g) == 1:
                final.append((g[0][0], g[-1][1], n))
            else:
                nxt.extend((sub, budget - (n - MAX_TOKENS)) for sub in pack(g, budget - (n - MAX_TOKENS)))
        todo = nxt
    final = sorted(final)
    return merge_short(tok, text, final) if CHUNKER == "v2" else final


def split_long_sentence(text, a, offs):
    """v2: cut a sentence of n > MAX_TOKENS tokens into ceil(n / MAX_TOKENS) near-equal pieces.
    Cuts go before a token that starts a word (preceded by whitespace), preferring one right
    after clause punctuation, within SPLIT_WINDOW tokens of the equal-split target."""
    n = len(offs)

    def word_start(i):
        c = a + offs[i][0]
        return c > a and (text[c - 1].isspace() or text[c].isspace())

    def clause_end(i):
        c = a + offs[i][0]
        prev = text[a:c].rstrip()
        return bool(prev) and prev[-1] in CLAUSE_END

    cuts, s = [], 0
    while n - s > MAX_TOKENS:
        k = -(-(n - s) // MAX_TOKENS)
        target = s + -(-(n - s) // k)
        cands = [i for i in range(max(s + 1, target - SPLIT_WINDOW), min(s + MAX_TOKENS, target + SPLIT_WINDOW) + 1)
                 if i < n and word_start(i)]
        strong = [i for i in cands if clause_end(i)]
        cut = min(strong or cands or [min(target, s + MAX_TOKENS)], key=lambda i: abs(i - target))
        cuts.append(cut)
        s = cut
    out = []
    for lo, hi in zip([0] + cuts, cuts + [n]):
        start, end = a + offs[lo][0], a + offs[hi - 1][1]
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        out.append((start, end, hi - lo))
    return out


def merge_short(tok, text, chunks):
    """v2: merge a chunk shorter than SHORT_TOKENS into the shorter neighbour if the merged span fits."""
    chunks = list(chunks)
    i = 0
    while i < len(chunks):
        a, b, n = chunks[i]
        if n >= SHORT_TOKENS or len(chunks) == 1:
            i += 1
            continue
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(chunks)]
        merged = False
        for j in sorted(nbrs, key=lambda j: chunks[j][2]):
            lo, hi = min(i, j), max(i, j)
            span = (chunks[lo][0], chunks[hi][1])
            m = len(tok.encode(text[span[0]:span[1]], add_special_tokens=False).ids)
            if m <= MAX_TOKENS:
                chunks[lo:hi + 1] = [(span[0], span[1], m)]
                i = lo
                merged = True
                break
        if not merged:
            i += 1
    return chunks


def chunk_paper(tok, rel_path):
    content = json.loads((CORPUS / rel_path).read_text(encoding="utf8"))
    paper_id = content.get("doc_id") or Path(rel_path).stem
    rows = []
    for si, (name, text) in enumerate(sections(content)):
        if not text.strip():
            continue
        for a, b, _ in chunk_section(tok, text):
            piece = text[a:b]
            if len("".join(piece.split())) < MIN_CHARS:
                continue
            rows.append({"paper_id": paper_id, "rel_path": rel_path, "section_idx": si, "section": name,
                         "char_start": a, "char_end": b, "text": piece})
    # exact token counts of the final chunk strings (joined sentences may tokenize slightly differently)
    if rows:
        counts = [len(e.ids) for e in tok.encode_batch([r["text"] for r in rows], add_special_tokens=False)]
        for r, c in zip(rows, counts):
            r["n_tokens"] = c
    return rows


def _chunk_worker(args):
    rel_paths, = args
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(MISTRAL_TOKENIZER)
    out, failed = [], []
    for p in rel_paths:
        try:
            out.extend(chunk_paper(tok, p))
        except Exception as exc:  # unreadable JSON etc.: recorded, never silent
            failed.append({"rel_path": p, "error": f"{type(exc).__name__}: {exc}"})
    return out, failed


def run_embed(shard, num_shards, out_dir, papers=None, n_cpu=None):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from multiprocessing import get_context
    from sentence_transformers import SentenceTransformer

    n_cpu = n_cpu or int(os.environ.get("KISTI_CHUNK_WORKERS", "2"))   # chunking processes per GPU worker
    torch.set_num_threads(1)
    papers = papers if papers is not None else paper_list()
    mine = papers[shard::num_shards]
    blocks = [mine[i:i + BLOCK_PAPERS] for i in range(0, len(mine), BLOCK_PAPERS)]
    block_dir = out_dir / "blocks"
    block_dir.mkdir(parents=True, exist_ok=True)
    todo = [(bi, b) for bi, b in enumerate(blocks) if not (block_dir / f"s{shard}_b{bi:05d}.parquet").exists()]
    print(f"[shard {shard}] papers {len(mine)}, blocks {len(blocks)}, to do {len(todo)}", flush=True)
    if not todo:
        return
    model = SentenceTransformer(E5_DIR, device="cuda").half()
    ctx = get_context("spawn")
    with ctx.Pool(n_cpu) as pool:
        def submit(block):
            parts = [block[i::n_cpu] for i in range(n_cpu)]
            return pool.map_async(_chunk_worker, [(p,) for p in parts])
        pending = submit(todo[0][1])
        for k, (bi, block) in enumerate(todo):
            t0 = time.perf_counter()
            results = pending.get()
            if k + 1 < len(todo):
                pending = submit(todo[k + 1][1])   # chunk the next block while this one embeds
            order = {p: i for i, p in enumerate(block)}
            rows = [r for part, _ in results for r in part]
            rows.sort(key=lambda r: (order[r["rel_path"]], r["section_idx"], r["char_start"]))
            failed = [f for _, part in results for f in part]
            t1 = time.perf_counter()
            vecs = model.encode([r["text"] for r in rows], batch_size=BATCH_SIZE, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
            t2 = time.perf_counter()
            for j, r in enumerate(rows):
                r["chunk_id"] = f"s{shard}_b{bi:05d}_{j}"
            stem = block_dir / f"s{shard}_b{bi:05d}"
            np.save(f"{stem}.npy.tmp.npy", vecs)
            os.replace(f"{stem}.npy.tmp.npy", f"{stem}.npy")
            cols = ["chunk_id", "paper_id", "rel_path", "section_idx", "section", "char_start", "char_end", "n_tokens", "text"]
            table = pa.table({c: [r[c] for r in rows] for c in cols})
            pq.write_table(table, f"{stem}.parquet.tmp", compression="zstd")
            if failed:
                Path(f"{stem}.failed.json").write_text(json.dumps(failed, ensure_ascii=False))
            os.replace(f"{stem}.parquet.tmp", f"{stem}.parquet")   # parquet last = block complete
            print(f"[shard {shard}] block {bi} ({k + 1}/{len(todo)}): papers {len(block)}, chunks {len(rows)}, "
                  f"failed {len(failed)}, wait-chunk {t1 - t0:.0f}s, embed {t2 - t1:.0f}s "
                  f"({len(rows) / max(t2 - t1, 1e-9):.0f}/s)", flush=True)


def cmd_papers():
    """Paper metadata table: one row per paper, with has_qa."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    papers = paper_list()
    qa = {p.name for p in QA_ROOT.glob("*/*/*.json")}
    rows = []
    for rel in papers:
        try:
            c = json.loads((CORPUS / rel).read_text(encoding="utf8"))
        except Exception as exc:
            rows.append({"rel_path": rel, "paper_id": Path(rel).stem, "title": "", "journal": rel.split("/")[0],
                         "year": rel.split("/")[1], "keywords": "", "has_qa": Path(rel).name in qa, "error": str(exc)})
            continue
        rows.append({"rel_path": rel, "paper_id": c.get("doc_id") or Path(rel).stem,
                     "title": choose_language(c.get("title", {})), "journal": rel.split("/")[0], "year": rel.split("/")[1],
                     "keywords": choose_language(c.get("keywords", {})), "has_qa": Path(rel).name in qa, "error": ""})
    pq.write_table(pa.Table.from_pylist(rows), OUT_DIR / "papers.parquet", compression="zstd")
    print("papers", len(rows), "with QA", sum(r["has_qa"] for r in rows), "errors", sum(bool(r["error"]) for r in rows))


def cmd_verify(out_dir=OUT_DIR):
    import numpy as np
    import pyarrow.parquet as pq
    files = sorted((out_dir / "blocks").glob("*.parquet"))
    n_chunks = n_papers = over = 0
    for f in files:
        t = pq.read_table(f, columns=["paper_id", "n_tokens"])
        v = np.load(str(f).replace(".parquet", ".npy"), mmap_mode="r")
        if v.shape != (t.num_rows, 1024):
            sys.exit(f"{f.name}: vectors {v.shape} vs rows {t.num_rows}")
        n_chunks += t.num_rows
        n_papers += len(set(t.column("paper_id").to_pylist()))
        over += sum(n > 128 for n in t.column("n_tokens").to_pylist())
    failed = sum(len(json.loads(f.read_text())) for f in (out_dir / "blocks").glob("*.failed.json"))
    print(json.dumps({"blocks": len(files), "chunks": n_chunks, "papers_with_chunks": n_papers,
                      "chunks_over_128_tokens": over, "failed_papers": failed}, ensure_ascii=False))


def cmd_smoke():
    import numpy as np
    import pyarrow.parquet as pq
    papers = paper_list()
    smoke_dir = OUT_DIR / "_smoke"
    run_embed(0, 1, smoke_dir, papers=papers[::12000][:40], n_cpu=2)
    t = pq.read_table(smoke_dir / "blocks" / "s0_b00000.parquet").to_pylist()
    v = np.load(smoke_dir / "blocks" / "s0_b00000.npy")
    ok = 0
    for r in t:   # every chunk must be an exact substring of its section
        c = json.loads((CORPUS / r["rel_path"]).read_text(encoding="utf8"))
        ok += sections(c)[r["section_idx"]][1][r["char_start"]:r["char_end"]] == r["text"]
    nt = sorted(r["n_tokens"] for r in t)
    print(f"smoke: chunks {len(t)}, exact-substring {ok}/{len(t)}, tokens median {nt[len(nt)//2]} max {nt[-1]}, "
          f"norm {float(np.linalg.norm(v[:100].astype('float32'), axis=1).mean()):.4f}")
    for r in t[:3]:
        print(repr(r["text"][:150]), r["n_tokens"], r["section"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "papers", "verify", "smoke"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "embed":
        if os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
            sys.exit("Set CUDA_VISIBLE_DEVICES to exactly one assigned GPU.")
        run_embed(a.shard, a.num_shards, OUT_DIR)
    else:
        {"papers": cmd_papers, "verify": cmd_verify, "smoke": cmd_smoke}[a.cmd]()
