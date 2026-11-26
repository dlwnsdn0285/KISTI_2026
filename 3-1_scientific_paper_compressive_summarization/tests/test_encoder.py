"""인코더의 슬롯 입력(구간별 평균 임베딩), 양방향 어텐션, 패딩과 무관한 결과를 확인한다."""
import torch

from conftest import TEXT
from scisum.model.encoder import bidirectional_mask, chunk_mean_slots


def test_chunk_mean_slots_split_contiguous_chunks():
    x = torch.arange(10, dtype=torch.float32)[:, None]
    # linspace(0, 10, 4).round() = [0, 3, 7, 10] -> 청크 [0..2], [3..6], [7..9]
    assert chunk_mean_slots(x, 3).squeeze(1).tolist() == [1.0, 4.5, 8.0]
    assert chunk_mean_slots(x, 1).item() == 4.5
    short = chunk_mean_slots(x[:2], 5)   # 토큰보다 슬롯이 많으면 빈 청크는 전체 평균
    assert short.shape == (5, 1) and 0.5 in short.squeeze(1).tolist()


def test_bidirectional_mask_shape_and_padding():
    m = bidirectional_mask([3, 5], 5)
    assert m.shape == (2, 1, 5, 5) and m.dtype == torch.bool
    assert m[0, 0, 0].tolist() == [True, True, True, False, False]   # 첫 토큰도 뒤 위치를 본다 (인과 마스크 없음)
    assert m[1].all()


def test_encoder_inputs_and_outputs(model):
    enc = model.encoder
    texts, counts = ["Short text.", TEXT], [2, 5]
    embeds, mask, slot_pos = enc.build_inputs(texts, counts)
    n0 = len(enc.token_ids(texts[0], 2))
    assert slot_pos[0].nonzero().flatten().tolist() == [n0, n0 + 1]
    assert slot_pos.sum(1).tolist() == counts
    tok = enc.model.get_input_embeddings()(torch.tensor(enc.token_ids(texts[0], 2)))
    assert torch.allclose(embeds[0, n0:n0 + 2], chunk_mean_slots(tok, 2))
    out = enc(texts, counts)
    assert [tuple(z.shape) for z in out] == [(2, enc.hidden_size), (5, enc.hidden_size)]


def test_encoder_is_bidirectional(model):
    """마지막 단어만 바꿔도 첫 토큰의 출력이 바뀐다 (인과 어텐션이면 같아야 한다)."""
    enc = model.encoder
    with torch.no_grad():
        outs = []
        for t in ["alpha beta gamma delta", "alpha beta gamma omega"]:
            embeds, mask, _ = enc.build_inputs([t], [2])
            outs.append(enc.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state[0, 0])
    assert not torch.allclose(outs[0], outs[1])


def test_latents_do_not_depend_on_batch_padding(model):
    with torch.no_grad():
        alone = model.encoder(["Short text."], [2])[0]
        batched = model.encoder(["Short text.", TEXT], [2, 6])[0]
    assert torch.allclose(alone, batched, atol=1e-5)


def test_truncation_leaves_room_for_slots(model):
    enc = model.encoder
    long = " ".join(["word"] * 1000)
    assert len(enc.token_ids(long, 16)) == enc.max_len - 16
