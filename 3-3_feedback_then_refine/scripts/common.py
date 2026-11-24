"""Shared components for the Feedback-then-Refine (FtR) experiment.

Verification is a faithful re-implementation of the ch.4 pipeline
(ch.4 verifier code: fidelity_analysis.py, nli-prob.py):
  - paragraphs: OCR lines with >= 400 chars, whitespace-collapsed
  - retrieval : BAAI/bge-large-en-v1.5 with the bge query instruction, top-5 cosine
  - LLM judge : same system prompt; labels supported / contradiction / unrelated
  - NLI       : DeBERTa-v3-large-mnli-fever-anli-ling-wanli, premise=paragraph, hypothesis=sentence
  - final     : supported iff LLM says supported AND any paragraph's NLI argmax is entailment
Only the LLM judge is swapped from gpt-5.4-mini to an open-source model served by vLLM.
"""
import asyncio
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

# Paths: everything is relative to the repository root unless overridden by env vars.
FTR_ROOT = Path(os.environ.get("FTR_ROOT", Path(__file__).resolve().parent.parent))
SRC = Path(os.environ.get("FTR_SRC", FTR_ROOT / "src_data"))  # surveys (reference_ocr/), human labels
POOL = Path(os.environ.get("FTR_POOL", SRC / "evidence-grounding"))  # ch.4 {supported,unsupported}.jsonl
CACHE = Path(os.environ.get("FTR_CACHE", FTR_ROOT / "cache"))

DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
TOP_K = 5
MIN_PARAGRAPH_CHARS = 400

JUDGE_URL = os.environ.get("JUDGE_URL", "http://127.0.0.1:8101/v1")
REFINER_URL = os.environ.get("REFINER_URL", "http://127.0.0.1:8102/v1")
JUDGE_NAME = os.environ.get("JUDGE_NAME", "judge")
REFINER_NAME = os.environ.get("REFINER_NAME", "refiner")

JUDGE_SYSTEM_PROMPT = (
    "You are a rigorous scientific citation verifier.\n"
    "Judge if the provided paragraphs support the citing sentence.\n"
    "Labels:\n"
    "- 'supported': Evidence exists (default choice if any connection exists).\n"
    "- 'contradiction': Evidence conflicts with the sentence.\n"
    "- 'unrelated': Absolutely no evidence (zero contextual overlap).\n\n"
    "Return strictly in JSON format with 'label' and 'reasoning' keys."
)


# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------
def survey_dir_of(row: dict) -> Path:
    """Map a row of unsupported.jsonl to its survey folder (with reference_ocr/)."""
    lf = row["provenance"][0]["llm_file"].split("/")
    result_set, cond, survey = lf[-4], lf[-3], lf[-2]
    root = "v2-final-0430-qwen80b" if result_set == "result-400-qwen" else "v2-final-0430"
    prefix = cond.split("-")[0]
    for cd in sorted(glob.glob(str(SRC / root / f"{prefix}-*"))):
        d = Path(cd) / survey
        if d.is_dir():
            return d
    raise FileNotFoundError(f"{root}/{cond}/{survey}")


def find_ocr_json(survey_dir: Path, ref_no: int):
    cands = list((survey_dir / "reference_ocr").glob(f"{ref_no}_*.json"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_size, reverse=True)
    return cands[0]


def load_paragraphs(json_path: Path):
    paragraphs = []
    try:
        data = json.load(open(json_path, encoding="utf-8"))
        for page in data.get("document", []):
            for line in page.get("content", "").split("\n"):
                line = line.strip()
                if len(line) >= MIN_PARAGRAPH_CHARS:
                    paragraphs.append(re.sub(r"\s+", " ", line))
    except Exception:
        pass
    return paragraphs


def load_reference_titles(survey_dir: Path) -> dict:
    titles = {}
    p = survey_dir / "references.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                titles[int(r["refNo"])] = r.get("title") or ""
    return titles


# ---------------------------------------------------------------------------
# Retrieval + NLI (GPU, synchronous, batched)
# ---------------------------------------------------------------------------
class Retriever:
    def __init__(self, device="cuda"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(DENSE_MODEL_NAME, device=device)
        self.model.half()
        self.cache = {}  # ocr path -> (paragraphs, normalized embeddings)
        (CACHE / "emb").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def emb_path(ocr_path: Path) -> Path:
        import hashlib
        return CACHE / "emb" / (hashlib.md5(str(ocr_path).encode()).hexdigest() + ".npy")

    def doc(self, ocr_path: Path):
        key = str(ocr_path)
        if key not in self.cache:
            paras = load_paragraphs(ocr_path)
            embs = None
            if paras:
                ep = self.emb_path(ocr_path)
                if ep.exists():
                    embs = np.load(ep).astype(np.float32)
                if embs is None or len(embs) != len(paras):
                    embs = self.model.encode(paras, batch_size=128, normalize_embeddings=True,
                                             show_progress_bar=False).astype(np.float32)
                    np.save(ep, embs.astype(np.float16))
            self.cache[key] = (paras, embs)
        return self.cache[key]

    def encode_queries(self, sentences):
        return self.model.encode([QUERY_INSTRUCTION + s for s in sentences], batch_size=64,
                                 normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    def topk(self, query_emb, ocr_path: Path, k=TOP_K):
        paras, embs = self.doc(ocr_path)
        if not paras:
            return []
        scores = embs @ query_emb
        idx = np.argsort(-scores)[:k]
        return [{"rank": i + 1, "text": paras[j], "score": round(float(scores[j]), 4)}
                for i, j in enumerate(idx)]


class NLI:
    def __init__(self, device=0):
        from transformers import pipeline
        self.pipe = pipeline("text-classification", model=NLI_MODEL_NAME, device=device,
                             batch_size=32)

    def __call__(self, pairs):
        """pairs: list of (premise_paragraph, hypothesis_sentence) -> list of labels."""
        if not pairs:
            return []
        outs = self.pipe([{"text": p, "text_pair": h} for p, h in pairs], truncation=True)
        labels = []
        for o in outs:
            l = o["label"].lower()
            labels.append("entailment" if "entail" in l else
                          "contradiction" if "contrad" in l else "neutral")
        return labels


# ---------------------------------------------------------------------------
# LLM calls (async, OpenAI-compatible vLLM servers)
# ---------------------------------------------------------------------------
def parse_json(text: str) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


class LLM:
    def __init__(self, base_url, model, concurrency=48, max_tokens=768):
        """base_url may list several replicas separated by commas (round-robin)."""
        from openai import AsyncOpenAI
        self.clients = [AsyncOpenAI(base_url=u.strip(), api_key="EMPTY", timeout=900)
                        for u in base_url.split(",")]
        self.model = model
        self.sem = asyncio.Semaphore(concurrency * len(self.clients))
        self.max_tokens = max_tokens
        self._rr = 0

    async def json_call(self, system, user, temperature=0.0, retries=3, validate=None):
        """JSON-mode call; retries (with a little sampling) when parsing or `validate` fails."""
        for attempt in range(retries):
            try:
                self._rr = (self._rr + 1) % len(self.clients)
                client = self.clients[self._rr]
                async with self.sem:
                    r = await client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": user}],
                        temperature=temperature if attempt == 0 else 0.3,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"})
                out = parse_json(r.choices[0].message.content)
                if out and (validate is None or validate(out)):
                    return out
            except Exception as e:  # noqa: BLE001
                err = str(e)
                await asyncio.sleep(2 * (attempt + 1))
        return {}

    async def gather(self, coros):
        return await asyncio.gather(*coros)


def normalize_judge_label(lbl: str) -> str:
    l = (lbl or "").strip().lower()
    if l.startswith("support"):
        return "supported"
    if l.startswith("contrad"):
        return "contradiction"
    if l.startswith("unrel") or l.startswith("unsupport"):
        return "unrelated"
    return "error"


async def judge_one(judge: LLM, sentence: str, paragraphs: list) -> dict:
    joined = "\n\n".join(f"[{p['rank']}] {p['text']}" for p in paragraphs)
    user = f"Sentence: {sentence}\n\nRetrieved Paragraphs:\n{joined}"
    out = await judge.json_call(JUDGE_SYSTEM_PROMPT, user)
    return {"label": normalize_judge_label(out.get("label")),
            "reasoning": out.get("reasoning", "")}


def combine(llm_label: str, nli_labels: list) -> str:
    nli_sup = any(l == "entailment" for l in nli_labels)
    return "supported" if (llm_label == "supported" and nli_sup) else "unsupported"
