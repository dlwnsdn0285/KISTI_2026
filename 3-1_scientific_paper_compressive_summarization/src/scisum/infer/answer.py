"""IM-SciQA 평가 문항에 답을 생성한다 (greedy, 최대 256토큰).

--mode latent: 학습한 모델로 논문을 잠재 표현으로 압축해 넣는다.
--mode text: 학습하지 않은 생성 모델에 논문 텍스트(기본은 전문, --text_jsonl을 주면 베이스라인 텍스트)를 넣는다.
답은 한 문항씩 저장하며, 다시 실행하면 답한 문항은 건너뛴다.

실행: python -m scisum.infer.answer --mode latent --ckpt runs/ours/multidoc/best.pt --alloc data/alloc/ratio16_test.json --out runs/eval/ours.jsonl
"""
import argparse
import json
import os
from typing import Dict

import torch

from .. import prompts
from ..data.datasets import PaperUnits, question_items, read_jsonl
from ..data.units import full_text, load_paper
from ..model.decoder import LatentDecoder
from ..model.summarizer import load_checkpoint


def done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except (ValueError, KeyError):
                continue   # 중간에 끊긴 줄
    return ids


def latent_inputs(model, row: dict) -> tuple:
    """문헌마다 [sum k, H] 잠재 표현과 그 길이."""
    papers = [model.encode_paper(p["texts"], p["counts"]) for p in row["papers"]]
    return papers, [int(z.shape[0]) for z in papers]


def text_inputs(decoder: LatentDecoder, row: dict, texts: Dict[str, str], units_dir: str) -> tuple:
    """문헌마다 자연어 텍스트와 그 토큰 수."""
    papers = [texts[pid] if pid in texts else full_text(load_paper(pid, units_dir)) for pid in row["paper_ids"]]
    return papers, [len(decoder.tokenizer(p, add_special_tokens=False)["input_ids"]) for p in papers]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["latent", "text"], default="latent")
    ap.add_argument("--ckpt", help="latent: 2단계 체크포인트")
    ap.add_argument("--alloc", help="latent: 평가 문헌 배분 (ratio16_test / oracle16_test)")
    ap.add_argument("--decoder_id", default="Qwen/Qwen3.5-9B", help="text: 생성 모델")
    ap.add_argument("--text_jsonl", default=None, help="text: 문헌별 텍스트 {paper_id, text} (없으면 전문)")
    ap.add_argument("--test_jsonl", default="data/imsciqa/test.jsonl")
    ap.add_argument("--units_dir", default="data/units")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    units = PaperUnits(args.units_dir)
    rows = question_items(args.test_jsonl, units, args.alloc if args.mode == "latent" else None)[args.shard::args.nshards]
    skip = done_ids(args.out)
    rows = [r for r in rows if r["id"] not in skip]
    if args.mode == "latent":
        model, _ = load_checkpoint(args.ckpt, device=device)
        model.eval()
        decoder, build = model.decoder, lambda r: latent_inputs(model, r)
    else:
        decoder = LatentDecoder.from_pretrained(args.decoder_id, lora_rank=0, grad_checkpointing=False).to(device).eval()
        texts = {x["paper_id"]: x["text"] for x in read_jsonl(args.text_jsonl)} if args.text_jsonl else {}
        build = lambda r: text_inputs(decoder, r, texts, args.units_dir)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.exists(args.out) and os.path.getsize(args.out):
        with open(args.out, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":   # 이전 실행이 줄 중간에 끊겼으면 새 줄에서 이어 쓴다 (끊긴 줄은 읽을 때 건너뜀)
                with open(args.out, "a", encoding="utf-8") as g:
                    g.write("\n")
    with open(args.out, "a", encoding="utf-8") as f, torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            inputs = [build(r) for r in batch]
            preds = decoder.generate([prompts.qa_segments(r["question"], papers) for r, (papers, _) in zip(batch, inputs)],
                                     args.max_new_tokens)
            for r, (_, lens), pred in zip(batch, inputs, preds):
                rec = {"id": r["id"], "question": r["question"], "answer": r["answer"], "paper_ids": r["paper_ids"],
                       "paper_tokens": lens, "pred": pred}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()


if __name__ == "__main__":
    main()
