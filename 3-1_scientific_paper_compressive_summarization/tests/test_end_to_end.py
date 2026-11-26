"""작은 모델로 학습 전체(문장 복원 -> 문단 학습 -> 다중 문헌 학습)와 답 생성을 끝까지 실행해 본다."""
import json
import os
import subprocess
import sys

import torch

from scisum import prompts
from scisum.data.datasets import PaperUnits, paragraph_items, question_items, sentence_items
from scisum.eval.judge import M3SCIQA_JUDGE_PROMPT, parse_score
from scisum.train.objectives import multi_paper_qa, single_paragraph

SRC = os.path.join(os.path.dirname(__file__), "..", "src")


def _data(tiny, name):
    return os.path.join(tiny["data"], name)


def test_datasets(tiny):
    units = PaperUnits(_data(tiny, "units"))
    sents = sentence_items(["P1"], units)
    assert len(sents) == 16 and all(s["k"] == 2 for s in sents)   # 초록 4문장 + 문단 3개 x 4문장
    paras = paragraph_items(["P1", "P2"], units, _data(tiny, "alloc.json"), _data(tiny, "paraphrase.json"), _data(tiny, "qa.json"))
    assert len(paras) == 8 and paras[0]["text"].startswith("Paper P1\n") and paras[0]["k"] == 2
    assert paras[1]["text"].startswith("Introduction\n") and len(paras[1]["qa"]) == 2 and len(paras[1]["paraphrases"]) == 2
    assert not paras[2]["text"].startswith("OTHER")
    rows = question_items(_data(tiny, "qa_rows.jsonl"), units, _data(tiny, "alloc.json"))
    assert [len(p["texts"]) for p in rows[0]["papers"]] == [4, 4] and rows[0]["papers"][0]["counts"] == [2, 3, 1, 3]


def test_paragraph_rows(model, tiny, monkeypatch):
    units = PaperUnits(_data(tiny, "units"))
    items = paragraph_items(["P1"], units, _data(tiny, "alloc.json"), _data(tiny, "paraphrase.json"), _data(tiny, "qa.json"))[:2]
    seen = []
    real = model.decoder.target_nll
    monkeypatch.setattr(model.decoder, "target_nll", lambda rows: seen.append(rows) or real(rows))
    model.train()
    with torch.no_grad():
        single_paragraph(model, items)
        single_paragraph(model, items, use_qa=False)
    # 부분 단위 -1: 원문 복원 / 부분 단위 0: 원문 복원 + 재진술 + QA 2개
    assert len(seen[0]) == 1 + 4 and len(seen[1]) == 1 + 2
    latents = [s for row in seen[0] for kind, s in row if kind == "latent"]
    assert [z.shape[0] for z in latents] == [2, 3, 3, 3, 3] and latents[1] is latents[4]   # 같은 문단은 같은 잠재 표현을 공유


def test_multidoc_forward_backward_and_generate(model, tiny):
    units = PaperUnits(_data(tiny, "units"))
    items = question_items(_data(tiny, "qa_rows.jsonl"), units, _data(tiny, "alloc.json"))[:2]
    model.train()
    model.zero_grad(set_to_none=True)
    nll = model(multi_paper_qa, items)
    assert nll.dim() == 1 and torch.isfinite(nll).all()
    nll.sum().backward()
    assert model.encoder.model.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert model.projector.net[0].weight.grad.abs().sum() > 0
    # lora_B (초기값 0)의 gradient. 작은 모델은 전체 어텐션 층이 마지막 층뿐이라, 잠재 표현 위치의 Q는 뒤 토큰에 영향이 없어
    # q_proj의 gradient는 0이고 K/V만 정답 토큰까지 이어진다
    _, k_b, v_b = model.decoder.lora_params[1::2]
    assert k_b.grad.abs().sum() > 0 and v_b.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.decoder.model.parameters() if not p.requires_grad)
    model.eval()
    with torch.no_grad():
        papers = [model.encode_paper(p["texts"], p["counts"]) for p in items[0]["papers"]]
        assert [z.shape[0] for z in papers] == [9, 9]
        preds = model.decoder.generate([prompts.qa_segments(it["question"], papers) for it in items], max_new_tokens=4)
    assert len(preds) == 2 and all(isinstance(p, str) for p in preds)


def test_judge_prompt_and_parse():
    msg = M3SCIQA_JUDGE_PROMPT.format(q="Q?", a="A.", p="P.")
    assert '{"selection": "0.5"}' in msg and msg.endswith("<CANDIDATE> P. </CANDIDATE>")
    assert parse_score('reasoning... {"selection": "0.5"}') == 0.5
    assert parse_score("1") == 1.0 and parse_score("") is None


def _run(module, *args):
    env = {**os.environ, "PYTHONPATH": SRC, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "-m", module, *map(str, args)], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return r


def test_training_run_and_answer_cli(tiny, tmp_path):
    """학습 한 번 실행으로 문장 복원 -> 문단 학습 -> 다중 문헌 학습이 이어지고, 다시 실행하면 끝난 구간을 건너뛴다."""
    d = lambda n: _data(tiny, n)
    run = ["--recipe", "ours", "--out_dir", tmp_path / "run", "--encoder_id", tiny["encoder"], "--decoder_id", tiny["decoder"],
           "--lora_rank", 4, "--encoder_max_len", 256, "--units_dir", d("units"), "--log_every", 1,
           "--train_papers", d("paper_ids.json"), "--valid_papers", d("paper_ids.json"),
           "--train_jsonl", d("qa_rows.jsonl"), "--valid_jsonl", d("qa_rows.jsonl"),
           "--paragraph_alloc", d("alloc.json"), "--multidoc_alloc", d("alloc.json"),
           "--paraphrase", d("paraphrase.json"), "--qa", d("qa.json"),
           "--effective_batches", "4,4,2", "--batch_sizes", "2,2,1", "--limit", 8, "--n_valid", 4]
    _run("scisum.train.run", *run)
    for phase in ("sentence", "paragraph", "multidoc"):
        assert os.path.exists(tmp_path / "run" / phase / "last.pt"), phase
    s2 = tmp_path / "run" / "multidoc"
    log = [json.loads(line) for line in open(s2 / "log.jsonl")]
    assert [r["event"] for r in log].count("train") == 4 and log[0]["accum"] == 2
    assert all(os.path.exists(s2 / f) for f in ("best.pt", "last.pt", "resume_state.pt"))
    # 다시 실행: 세 구간 모두 끝났으므로 학습 없이 끝난다
    r = _run("scisum.train.run", *run)
    assert r.stdout.count("건너뜀") == 3
    assert sum(1 for line in open(s2 / "log.jsonl") if '"train"' in line) == 4

    out = tmp_path / "preds.jsonl"
    for shard in (0, 1):
        _run("scisum.infer.answer", "--ckpt", s2 / "best.pt", "--alloc", d("alloc.json"), "--test_jsonl", d("qa_rows.jsonl"),
             "--units_dir", d("units"), "--out", out, "--max_new_tokens", 4, "--shard", shard, "--nshards", 2)
    _run("scisum.infer.answer", "--ckpt", s2 / "best.pt", "--alloc", d("alloc.json"), "--test_jsonl", d("qa_rows.jsonl"),
         "--units_dir", d("units"), "--out", out, "--max_new_tokens", 4)   # 다시 실행: 끝난 id는 건너뛴다
    preds = [json.loads(line) for line in open(out)]
    assert sorted(p["id"] for p in preds) == ["q0", "q1", "q2", "q3"] and preds[0]["paper_tokens"] == [9, 9]

    over = tmp_path / "titleabs.jsonl"
    over.write_text(json.dumps({"paper_id": "P1", "text": "Paper P1\nWe study lung development."}) + "\n")
    out_t = tmp_path / "text.jsonl"
    _run("scisum.infer.answer", "--mode", "text", "--decoder_id", tiny["decoder"], "--text_jsonl", over, "--test_jsonl", d("qa_rows.jsonl"),
         "--units_dir", d("units"), "--out", out_t, "--max_new_tokens", 4)
    rows = {r["id"]: r for r in map(json.loads, open(out_t))}
    assert rows["q1"]["paper_tokens"][0] < rows["q1"]["paper_tokens"][1]   # P1은 짧은 대체 텍스트, P2는 전문
