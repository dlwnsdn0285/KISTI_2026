"""학습 전체를 한 번에 실행한다: 단일 문단 요약 학습 단계 -> 다중 문헌 요약 활용 학습 단계.

단일 문단 요약 학습 단계는 문장 복원(sentence)으로 시작해 문단 학습(paragraph)으로 이어지고,
다중 문헌 요약 활용 학습 단계(multidoc)가 뒤따른다. 모델은 한 번만 올리고 구간마다 이어서 학습한다.
--recipe: ours(원문 복원 + 재진술 + 질의응답), oracle(+ 오라클 배분), noqa(원문 복원 + 재진술).
구간마다 <out_dir>/<구간>/last.pt를 저장하며, 다시 실행하면 끝난 구간은 건너뛰고 중단된 곳부터 이어서 학습한다.

실행: accelerate launch --num_processes 4 -m scisum.train.run --recipe ours --out_dir runs/ours
"""
import argparse
import copy
import functools
import gc
import os
from typing import Callable, List, NamedTuple

import torch

from ..data.datasets import PaperUnits, paragraph_items, question_items, read_json, sentence_items, split_valid
from ..model.summarizer import LatentSummarizer
from .objectives import multi_paper_qa, sentence_reconstruction, single_paragraph
from .trainer import Trainer

RECIPES = {"ours": ("ratio16", True), "oracle": ("oracle16", True), "noqa": ("ratio16", False)}


class Phase(NamedTuple):
    name: str
    objective: Callable
    train: list
    valid: list
    effective_batch: int
    batch_size: int
    epochs: int
    eval_every: int


def build_phases(a: argparse.Namespace) -> List[Phase]:
    """구간별 학습 항목, 손실, 배치 설정. 유효 배치와 에폭은 보고 설정이 기본값."""
    alloc2, use_qa = RECIPES[a.recipe]
    units = PaperUnits(a.units_dir)
    train_papers, valid_papers = read_json(a.train_papers), read_json(a.valid_papers)
    qa = a.qa if use_qa else None
    multidoc_alloc = a.multidoc_alloc or os.path.join(a.alloc_dir, f"{alloc2}_train.json")
    paragraph_alloc = os.path.join(a.alloc_dir, "ratio16_train.json") if a.paragraph_alloc is None else a.paragraph_alloc
    sent = functools.partial(sentence_items, units=units, latents_per_sentence=a.latents_per_sentence)
    para = functools.partial(paragraph_items, units=units, alloc_path=paragraph_alloc, paraphrase_path=a.paraphrase, qa_path=qa)
    ques = functools.partial(question_items, units=units, alloc_path=multidoc_alloc)
    eb = a.effective_batches
    return [
        Phase("sentence", sentence_reconstruction, sent(train_papers), split_valid(sent(valid_papers), a.n_valid, a.seed),
              eb[0], a.batch_sizes[0], 1, 500),
        Phase("paragraph", functools.partial(single_paragraph, use_qa=use_qa), para(train_papers),
              split_valid(para(valid_papers), a.n_valid, a.seed), eb[1], a.batch_sizes[1], 1, 500),
        Phase("multidoc", multi_paper_qa, ques(a.train_jsonl), ques(a.valid_jsonl), eb[2], a.batch_sizes[2], 2, 0),
    ]


def phase_args(a: argparse.Namespace, p: Phase) -> argparse.Namespace:
    """Trainer가 읽는 구간별 설정."""
    args = copy.copy(a)
    args.out_dir = os.path.join(a.out_dir, p.name)
    world = int(os.environ.get("WORLD_SIZE", "1"))   # accelerate launch가 정하는 GPU(프로세스) 수
    args.effective_batch, args.epochs, args.eval_every = p.effective_batch, p.epochs, p.eval_every
    args.batch_size = max(1, min(p.batch_size, p.effective_batch // world))   # 마이크로배치는 유효 배치 / GPU 수를 넘지 않게
    return args


def ints(text: str) -> List[int]:
    return [int(x) for x in text.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="ours")
    ap.add_argument("--out_dir", required=True)
    # 모델
    ap.add_argument("--encoder_id", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--decoder_id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--lora_rank", type=int, default=32)
    ap.add_argument("--encoder_max_len", type=int, default=2048)
    # 데이터
    ap.add_argument("--units_dir", default="data/units")
    ap.add_argument("--train_papers", default="data/imsciqa/train_paper_ids.json")
    ap.add_argument("--valid_papers", default="data/imsciqa/valid_paper_ids.json")
    ap.add_argument("--train_jsonl", default="data/imsciqa/train.jsonl")
    ap.add_argument("--valid_jsonl", default="data/imsciqa/valid.jsonl")
    ap.add_argument("--alloc_dir", default="data/alloc")
    ap.add_argument("--paragraph_alloc", default=None, help="1단계 배분 (기본 <alloc_dir>/ratio16_train.json)")
    ap.add_argument("--multidoc_alloc", default=None, help="2단계 배분 (기본 레시피에 따라 ratio16 / oracle16)")
    ap.add_argument("--paraphrase", default="data/synth/paraphrase.json")
    ap.add_argument("--qa", default="data/synth/qa.json")
    ap.add_argument("--latents_per_sentence", type=int, default=2)
    # 학습
    ap.add_argument("--effective_batches", type=ints, default=[128, 32, 8], help="sentence,paragraph,multidoc 유효 배치")
    ap.add_argument("--batch_sizes", type=ints, default=[4, 4, 4], help="sentence,paragraph,multidoc GPU당 마이크로배치")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_valid", type=int, default=1000, help="1단계 검증 항목 수 (2단계는 검증 문항 전부)")
    ap.add_argument("--ckpt_every", type=int, default=500, help="resume_state.pt 저장 간격(step)")
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="구간별 학습 항목 수 상한 (점검용)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    model = LatentSummarizer.build(a.encoder_id, a.decoder_id, a.lora_rank, a.encoder_max_len)
    for p in build_phases(a):
        args = phase_args(a, p)
        if p.name == "multidoc":
            args.n_valid = 0
        last = os.path.join(args.out_dir, "last.pt")
        if os.path.exists(last):   # 끝난 구간: 가중치만 읽고 다음 구간으로
            model.load_trainable_state_dict(torch.load(last, map_location="cpu", weights_only=False))
            print(f"[run] {p.name}: 이미 끝남 ({last}), 건너뜀", flush=True)
            continue
        print(f"[run] {p.name}: 학습 시작 (항목 {len(p.train)}개, 유효 배치 {p.effective_batch}, {p.epochs} epoch)", flush=True)
        trainer = Trainer(args, p.objective, p.train, p.valid, model=model)
        trainer.fit()
        model = trainer.unwrapped
        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"[run] 완료: {os.path.join(a.out_dir, 'multidoc', 'best.pt')}", flush=True)


if __name__ == "__main__":
    main()
