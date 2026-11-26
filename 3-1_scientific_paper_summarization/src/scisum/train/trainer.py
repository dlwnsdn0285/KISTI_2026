"""모든 학습 구간이 함께 쓰는 학습 루프 (여러 GPU는 DDP).

유효 배치 크기는 고정이며, GPU 수와 마이크로배치 크기에 맞춰 누적 횟수가 정해진다.
매 step에 보는 항목은 GPU 수와 상관없이 같다. 학습률은 3% warmup 후 cosine으로 줄인다.
검증 손실이 가장 낮은 가중치는 best.pt, 마지막 가중치는 last.pt로 저장하고, resume_state.pt가 있으면 이어서 학습한다.
"""
import contextlib
import json
import os
import random
import time
from typing import Callable, List

import torch
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from ..model.summarizer import LatentSummarizer
from .loss import reduce_sums, scale_to_token_mean
from .optim import MasterAdamW

WARMUP_RATIO, WEIGHT_DECAY, MAX_GRAD_NORM = 0.03, 0.01, 1.0


def epoch_steps(n_items: int, epoch: int, seed: int, eb: int, world: int, rank: int, batch_size: int) -> List[List[List[int]]]:
    """이 GPU가 에폭 하나에서 맡을 [step][마이크로배치][항목 인덱스]."""
    order = torch.randperm(n_items, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    per_rank = eb // world
    steps = []
    for s in range(n_items // eb):
        mine = order[s * eb + rank * per_rank: s * eb + (rank + 1) * per_rank]
        steps.append([mine[i:i + batch_size] for i in range(0, per_rank, batch_size)])
    return steps


class Trainer:
    def __init__(self, args, objective: Callable, train_items: list, valid_items: list, model: LatentSummarizer):
        """args: out_dir, effective_batch, batch_size, epochs, lr, eval_every, ckpt_every, log_every, limit, seed (train/run.py가 구간별로 채움).
        model: 앞 구간에서 이어받은 모델 (학습 대상은 인코더, 프로젝터, 잠재 표현용 LoRA)."""
        self.args, self.objective = args, objective
        self.train_items = train_items[: args.limit] if args.limit else train_items
        self.valid_items = valid_items
        self.acc = Accelerator(mixed_precision="no")   # bf16 가중치를 직접 쓰므로 accelerate 설정 파일의 fp16/bf16을 따르지 않는다
        world = self.acc.num_processes
        if args.effective_batch % (world * args.batch_size):
            raise ValueError(f"유효 배치 {args.effective_batch}가 GPU 수 {world} x batch_size {args.batch_size}로 나누어떨어지지 않음")
        self.accum = args.effective_batch // (world * args.batch_size)
        random.seed(args.seed + self.acc.process_index)
        torch.manual_seed(args.seed)

        self.params = model.trainable_parameters()
        self.optimizer = MasterAdamW(self.params, lr=args.lr, weight_decay=WEIGHT_DECAY)
        self.steps_per_epoch = len(self.train_items) // args.effective_batch
        if self.steps_per_epoch == 0:
            raise ValueError(f"학습 항목 {len(self.train_items)}개가 유효 배치 {args.effective_batch}보다 적음")
        total = self.steps_per_epoch * args.epochs
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, int(total * WARMUP_RATIO), total)
        self.model = self.acc.prepare(model)
        self.step, self.best = 0, float("inf")
        os.makedirs(args.out_dir, exist_ok=True)
        self._maybe_resume()
        self.log({"event": "start", "train_items": len(self.train_items), "valid_items": len(valid_items), "gpus": world,
                  "batch_size": args.batch_size, "accum": self.accum, "effective_batch": args.effective_batch, "total_steps": total})

    @property
    def unwrapped(self) -> LatentSummarizer:
        return self.acc.unwrap_model(self.model)

    def log(self, rec: dict):
        if self.acc.is_main_process:
            rec = {"step": self.step, "time": round(time.time()), **rec}
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            with open(os.path.join(self.args.out_dir, "log.jsonl"), "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()

    # ---- 체크포인트
    def save(self, name: str):
        self.acc.wait_for_everyone()
        if self.acc.is_main_process:
            torch.save(self.unwrapped.trainable_state_dict(), os.path.join(self.args.out_dir, name))

    def save_resume(self):
        self.acc.wait_for_everyone()
        if self.acc.is_main_process:
            state = {"model": self.unwrapped.trainable_state_dict(), "optimizer": self.optimizer.state_dict(),
                     "scheduler": self.scheduler.state_dict(), "step": self.step, "best": self.best,
                     "effective_batch": self.args.effective_batch, "epochs": self.args.epochs}
            tmp = os.path.join(self.args.out_dir, "resume_state.pt.tmp")
            torch.save(state, tmp)
            os.replace(tmp, os.path.join(self.args.out_dir, "resume_state.pt"))
        self.acc.wait_for_everyone()

    def _maybe_resume(self):
        path = os.path.join(self.args.out_dir, "resume_state.pt")
        if not os.path.exists(path):
            return
        st = torch.load(path, map_location="cpu", weights_only=False)
        if (st["effective_batch"], st["epochs"]) != (self.args.effective_batch, self.args.epochs):
            raise ValueError("resume_state.pt의 유효 배치나 에폭 수가 지금 설정과 다름")
        self.unwrapped.load_trainable_state_dict(st["model"])
        self.optimizer.load_state_dict(st["optimizer"])
        self.scheduler.load_state_dict(st["scheduler"])
        self.step, self.best = st["step"], st["best"]

    # ---- 학습과 검증
    def train_step(self, micro_batches: List[List[int]]) -> tuple:
        """누적 마이크로배치 전부 forward/backward 후 토큰 평균 gradient로 optimizer step 한 번."""
        self.model.train()
        nll_sum, n_tok = 0.0, 0
        for j, mb in enumerate(micro_batches):
            sync = j == len(micro_batches) - 1
            with contextlib.nullcontext() if sync else self.acc.no_sync(self.model):
                nll = self.model(self.objective, [self.train_items[i] for i in mb])
                self.acc.backward(nll.sum())
            nll_sum += float(nll.detach().sum()); n_tok += nll.numel()
        nll_sum, n_tok = reduce_sums(nll_sum, n_tok, self.acc)
        scale_to_token_mean(self.params, n_tok, self.acc.num_processes)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.params, MAX_GRAD_NORM))
        self.optimizer.step(); self.scheduler.step(); self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        return nll_sum, n_tok, grad_norm

    @torch.no_grad()
    def evaluate(self) -> float:
        """검증 항목 전체의 토큰 평균 NLL (GPU마다 나눠 계산 후 합산)."""
        if not self.valid_items:
            return float("nan")
        model = self.unwrapped
        model.eval()
        mine = self.valid_items[self.acc.process_index::self.acc.num_processes]
        nll_sum, n_tok = 0.0, 0
        for i in range(0, len(mine), self.args.batch_size):
            nll = model(self.objective, mine[i:i + self.args.batch_size])
            nll_sum += float(nll.sum()); n_tok += nll.numel()
        nll_sum, n_tok = reduce_sums(nll_sum, n_tok, self.acc)
        return nll_sum / max(n_tok, 1.0)

    def validate_and_keep_best(self, epoch: int):
        loss = self.evaluate()
        improved = loss < self.best
        if improved:
            self.best = loss
            self.save("best.pt")
        self.log({"event": "valid", "epoch": epoch, "valid_loss": loss, "best": self.best, "saved_best": improved})

    def fit(self):
        a = self.args
        window = [0.0, 0.0]
        for epoch in range(self.step // self.steps_per_epoch, a.epochs):
            steps = epoch_steps(len(self.train_items), epoch, a.seed, a.effective_batch, self.acc.num_processes,
                                self.acc.process_index, a.batch_size)
            for s in range(self.step - epoch * self.steps_per_epoch, len(steps)):   # resume: 끝난 step은 건너뛴다
                nll_sum, n_tok, grad_norm = self.train_step(steps[s])
                window[0] += nll_sum; window[1] += n_tok
                if self.step % a.log_every == 0:
                    self.log({"event": "train", "epoch": epoch, "loss": window[0] / max(window[1], 1.0), "tokens": int(window[1]),
                              "grad_norm": grad_norm, "lr": self.scheduler.get_last_lr()[0]})
                    window = [0.0, 0.0]
                if a.eval_every and self.step % a.eval_every == 0:
                    self.validate_and_keep_best(epoch)
                if a.ckpt_every and self.step % a.ckpt_every == 0:
                    self.save_resume()
            self.validate_and_keep_best(epoch)
            self.save_resume()
        self.save("last.pt")
        self.log({"event": "done", "best_valid_loss": self.best})
