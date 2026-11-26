"""테스트용 작은 무작위 모델과 가짜 데이터 (CPU에서 실행)."""
import json
import os
import sys

import pytest
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "smoke"))
from tiny_models import tiny_decoder, tiny_encoder  # noqa: E402

TEXT = ("The expression of TGF-beta1 mRNA in preterm infants correlated positively with maturity. "
        "Maternal factors reduced the expression. These results suggest that TGF-beta1 may impair normal fetal lung development.")


def _fake_data(root):
    """문헌 3편(P1, P2 학습/검증용, P3 평가용)의 부분 단위, 배분, 재진술, QA, IM-SciQA 문항."""
    os.makedirs(os.path.join(root, "units"), exist_ok=True)
    alloc, para, qa = {}, {}, {}
    for n, pid in enumerate(["P1", "P2", "P3"]):
        paper = {"paper_id": pid, "title": f"Paper {pid}", "abstract": "We study lung development. " + TEXT,
                 "paragraphs": [{"idx": i, "section": ["Introduction", "OTHER", "Discussion"][i], "text": f"Paragraph {i} of {pid}. " + TEXT}
                                for i in range(3)]}
        json.dump(paper, open(os.path.join(root, "units", f"{pid}.json"), "w"))
        alloc[pid] = {"B": 9, "units": [[-1, 2], [0, 3], [1, 1], [2, 3]]}
        para[pid] = {"0": ["Rewritten paragraph zero.", "Another rewrite."]}
        qa[pid] = {"0": [{"type": "atomic", "question": "What correlated with maturity?", "answer": "TGF-beta1 mRNA expression."},
                         {"type": "gist", "question": "What do the results suggest?", "answer": "TGF-beta1 may impair lung development."}]}
    for name, obj in [("alloc.json", alloc), ("paraphrase.json", para), ("qa.json", qa), ("paper_ids.json", ["P1", "P2"])]:
        json.dump(obj, open(os.path.join(root, name), "w"))
    rows = [{"id": f"q{i}", "question": f"Question {i} about both papers?", "answer": f"Answer {i}.", "paper_ids": ["P1", "P2"] if i % 2 else ["P2", "P3"]}
            for i in range(4)]
    with open(os.path.join(root, "qa_rows.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return root


@pytest.fixture(scope="session")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    enc, dec = str(root / "encoder"), str(root / "decoder")
    tiny_encoder(enc)
    tiny_decoder(dec)
    return {"encoder": enc, "decoder": dec, "data": _fake_data(str(root / "data"))}


@pytest.fixture(scope="session")
def model(tiny):
    from scisum.model.summarizer import LatentSummarizer
    torch.manual_seed(0)
    m = LatentSummarizer.build(tiny["encoder"], tiny["decoder"], lora_rank=4, encoder_max_len=256, dtype=torch.float32,
                               grad_checkpointing=False)
    return m
