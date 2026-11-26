"""손실이 정답 토큰만으로 계산되고, GPU 수나 마이크로배치 크기와 무관하게 같은지 확인한다."""
import torch
import torch.nn.functional as F

from scisum import prompts
from scisum.train.loss import reduce_sums, scale_to_token_mean
from scisum.train.trainer import epoch_steps


def test_only_target_tokens_are_supervised(model):
    dec = model.decoder
    z = torch.randn(4, dec.embed.weight.shape[1]) * 0.05
    answer = "TGF-beta1 may impair lung development."
    segs = [prompts.qa_segments("What do the results suggest?", [z], answer)]
    _, _, labels, latent = dec.collate(segs, left_pad=False)
    target_ids = dec.tokenizer(answer + prompts.END_OF_TURN, add_special_tokens=False)["input_ids"]
    assert labels[0][labels[0] != -100].tolist() == target_ids          # 지시문, 질문, 잠재 표현 위치는 모두 -100
    assert labels[0][-len(target_ids):].tolist() == target_ids and int(latent.sum()) == 4
    with torch.no_grad():
        nll = dec.target_nll(segs)
        embeds, attn, labels, latent = dec.collate(segs, left_pad=False)
        dec.positions.mask = latent
        logits = dec.model(inputs_embeds=embeds, attention_mask=attn).logits[0, :-1]
        ref = F.cross_entropy(logits, labels[0, 1:], ignore_index=-100, reduction="none")[labels[0, 1:] != -100]
    assert nll.shape == (len(target_ids),) and torch.allclose(nll, ref, atol=1e-4)


def _toy_items():
    torch.manual_seed(0)
    return [(torch.randn(n, 5), torch.randint(0, 7, (n,))) for n in (3, 9, 1, 6, 4, 2, 8, 5)]


def _grads(w, fn):
    w.grad = None
    fn()
    return w.grad.clone()


def test_eb_token_mean_is_split_invariant():
    items = _toy_items()
    w = torch.randn(5, 7, requires_grad=True)
    nll = lambda batch: torch.cat([F.cross_entropy(x @ w, y, reduction="none") for x, y in batch])
    n_tok = sum(len(y) for _, y in items)

    full = _grads(w, lambda: nll(items).mean().backward())           # 유효 배치를 통째로 토큰 평균

    def accumulate(micro_batches, world=1):
        nll(micro_batches[0]).sum().backward()
        for mb in micro_batches[1:]:
            nll(mb).sum().backward()
        scale_to_token_mean([w], n_tok, world)
    one_gpu = _grads(w, lambda: accumulate([items[i:i + 1] for i in range(8)]))
    uneven = _grads(w, lambda: accumulate([items[:5], items[5:]]))
    # GPU 2장: 각자 NLL 합을 누적하고, DDP는 두 gradient를 평균한다
    g0 = _grads(w, lambda: [nll(mb).sum().backward() for mb in (items[:2], items[2:4])])
    g1 = _grads(w, lambda: [nll(mb).sum().backward() for mb in (items[4:6], items[6:])])
    w.grad = (g0 + g1) / 2
    scale_to_token_mean([w], n_tok, world_size=2)
    two_gpu = w.grad.clone()
    for g in (one_gpu, uneven, two_gpu):
        assert torch.allclose(g, full, atol=1e-6)
    naive = _grads(w, lambda: [(nll(items[i:i + 1]).mean() / 8).backward() for i in range(8)])   # 마이크로배치 평균의 평균
    assert not torch.allclose(naive, full, atol=1e-4)


def test_reduce_sums_without_accelerator():
    assert reduce_sums(6.0, 3) == (6.0, 3.0)


def test_each_step_sees_the_same_items_for_any_split():
    n, eb = 37, 8
    ref = [sorted(sum(mbs, [])) for mbs in epoch_steps(n, 0, 1, eb, 1, 0, 8)]
    assert len(ref) == n // eb
    for world, bs in [(1, 2), (2, 4), (2, 1), (4, 2)]:
        per_rank = [epoch_steps(n, 0, 1, eb, world, r, bs) for r in range(world)]
        for s in range(len(ref)):
            assert all(len(mb) == bs for r in range(world) for mb in per_rank[r][s])
            assert sorted(i for r in range(world) for mb in per_rank[r][s] for i in mb) == ref[s]
    assert epoch_steps(n, 0, 1, eb, 1, 0, 8) != epoch_steps(n, 1, 1, eb, 1, 0, 8)   # 에폭마다 순서가 바뀐다
