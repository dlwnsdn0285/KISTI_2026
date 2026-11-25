"""Fusion module (Step 11): a small question-conditioned transformer over the cached
per-document PISCO memories, inserted between `compress` and `build_decoder_inputs`.

    memory  [k, 8, H]  (frozen PISCO compressor, cached offline per document)
    question hidden states [Lq, H] (frozen decoder, one middle layer)
          -> fusion -> memory' [k, 8, H] = memory + up(f(down(memory), down(question)))

`up` is zero-initialised, so an untrained module is the identity and the
system equals PISCO bitwise. Everything else in PISCO stays frozen.

Two variants with identical parameters, data and objective:
    cross_doc=True   memory tokens of every document attend to each other and to the question
    cross_doc=False  memory tokens attend only to their own document and to the question;
                     question tokens attend only to the question (they cannot relay)
The difference between the two isolates cross-document fusion.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

MAX_DOCS = 16


@dataclass
class FusionConfig:
    d_model: int = 1024
    n_layers: int = 2
    n_heads: int = 8
    d_ff: int = 4096
    dropout: float = 0.0
    cross_doc: bool = True
    q_layer: int = 16          # decoder hidden-state layer used as the question representation
    max_q_tokens: int = 64


class FusionModule(nn.Module):
    def __init__(self, hidden_size, n_mem, cfg: FusionConfig):
        super().__init__()
        self.cfg, self.hidden_size, self.n_mem = cfg, hidden_size, n_mem
        d = cfg.d_model
        self.down = nn.Linear(hidden_size, d)
        self.q_down = nn.Linear(hidden_size, d)
        self.type_emb = nn.Embedding(2, d)          # 0 memory, 1 question
        self.slot_emb = nn.Embedding(n_mem, d)
        self.doc_emb = nn.Embedding(MAX_DOCS, d)
        self.q_pos = nn.Embedding(cfg.max_q_tokens, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(d)
        self.up = nn.Linear(d, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def attention_mask(self, k, lq, q_valid):
        """Boolean mask [B, T, T], True = NOT allowed (PyTorch convention). T = k*n_mem + lq."""
        b = q_valid.size(0)
        t_mem = k * self.n_mem
        t = t_mem + lq
        allowed = torch.ones(b, t, t, dtype=torch.bool, device=q_valid.device)
        if not self.cfg.cross_doc:
            doc_of = torch.arange(t_mem, device=q_valid.device) // self.n_mem
            same_doc = doc_of[:, None] == doc_of[None, :]
            allowed[:, :t_mem, :t_mem] = same_doc
            allowed[:, t_mem:, :t_mem] = False        # question tokens cannot read memory
        allowed[:, :, t_mem:] &= q_valid[:, None, :]  # nobody attends to padded question positions
        return ~allowed

    def forward(self, memory, q_hidden, q_valid):
        """memory [B, k, n_mem, H]; q_hidden [B, Lq, H]; q_valid [B, Lq] bool -> memory' [B, k, n_mem, H]."""
        b, k, n_mem, h = memory.shape
        lq = q_hidden.size(1)
        if k > MAX_DOCS or n_mem != self.n_mem or lq > self.cfg.max_q_tokens:
            raise ValueError(f"unsupported shapes: k={k}, n_mem={n_mem}, Lq={lq}")
        dev = memory.device
        x_mem = self.down(memory.float()).view(b, k * n_mem, -1)
        x_mem = x_mem + self.type_emb.weight[0] \
            + self.doc_emb(torch.arange(k, device=dev)).repeat_interleave(n_mem, 0)[None] \
            + self.slot_emb(torch.arange(n_mem, device=dev)).repeat(k, 1)[None]
        x_q = self.q_down(q_hidden.float()) + self.type_emb.weight[1] + self.q_pos(torch.arange(lq, device=dev))[None]
        x = torch.cat([x_mem, x_q], dim=1)
        mask = self.attention_mask(k, lq, q_valid)
        mask = mask.repeat_interleave(self.cfg.n_heads, 0)   # [B*heads, T, T]
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        delta = self.up(self.final_ln(x[:, :k * n_mem])).view(b, k, n_mem, h)
        return memory + delta.to(memory.dtype)

    # ----------------------------------------------------------- persistence

    def save(self, path, extra=None):
        torch.save({"config": asdict(self.cfg), "hidden_size": self.hidden_size, "n_mem": self.n_mem,
                    "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        module = cls(ckpt["hidden_size"], ckpt["n_mem"], FusionConfig(**ckpt["config"]))
        module.load_state_dict(ckpt["state_dict"])
        return module.to(device), ckpt.get("extra", {})


# ------------------------------------------------------------------ helpers

def encode_questions(model, tokenizer, questions, layer, max_tokens):
    """Frozen decoder hidden states of the question tokens (layer `layer`), [B, Lq, H] and valid mask."""
    enc = tokenizer(questions, return_tensors="pt", padding="longest", truncation=True, max_length=max_tokens,
                    add_special_tokens=True)
    device = model.decoder.device
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    if "decoder_adapter" in model.adapter_keys:
        model.decoder.set_adapter("decoder_adapter")
    with torch.no_grad():
        out = model.decoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    hidden = out.hidden_states[layer]
    # left padding (tokenizer padding_side="left"): keep as is, the mask marks valid positions
    return hidden, mask.bool()


def fuse(wrapper, fusion, questions, memory_per_question):
    """Inference: list of [k, 8, H] -> list of fused [k, 8, H] (same dtype), one batch."""
    cfg = fusion.cfg
    memory = torch.stack(memory_per_question)                 # [B, k, 8, H]
    q_hidden, q_valid = encode_questions(wrapper.model, wrapper.tokenizer, questions, cfg.q_layer, cfg.max_q_tokens)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        fused = fusion(memory, q_hidden, q_valid)
    return [m.to(memory.dtype) for m in fused]


def build_training_batch(wrapper, questions, answers, k):
    """Prompt (with MEM placeholders) + answer + EOS, right padded.

    Returns input_ids [B, L], attention_mask [B, L], labels [B, L] (-100 outside the answer),
    mem_blocks (per row: k [first, last] slot positions).
    """
    tok = wrapper.tokenizer
    wrapper.model.generation_top_k = k
    rows, labels, blocks = [], [], []
    for q, a in zip(questions, answers):
        prompt_ids = tok(wrapper.model.blend_prompt_and_memory_tokens(query=q), add_special_tokens=False)["input_ids"]
        answer_ids = tok(a, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        blocks.append(wrapper._check_slot_layout(prompt_ids, k))
        rows.append(prompt_ids + answer_ids)
        labels.append([-100] * len(prompt_ids) + answer_ids)
    length = max(len(r) for r in rows)
    pad = tok.pad_token_id
    input_ids = torch.tensor([r + [pad] * (length - len(r)) for r in rows])
    attention_mask = torch.tensor([[1] * len(r) + [0] * (length - len(r)) for r in rows])
    labels = torch.tensor([l + [-100] * (length - len(l)) for l in labels])
    return input_ids, attention_mask, labels, blocks


def embed_with_memory(model, input_ids, mem_blocks, memory):
    """Differentiable version of replace_embeddings: memory [B, k, n_mem, H] written into the slots."""
    embeds = model.decoder.get_input_embeddings()(input_ids).clone()
    for i, blocks in enumerate(mem_blocks):
        for j, (first, last) in enumerate(blocks):
            embeds[i, first:last + 1] = memory[i, j].to(embeds.dtype)
    return embeds


# ============================================================ Step 13: appended memory block

class FusionAppend(nn.Module):
    """A SIXTH memory block built from the five: learned query tokens attend over all cached
    memory tokens and the question; the five original blocks are passed to the decoder unchanged.

    init: block = per-slot mean of the k document memories + up(...), with `up` zero-initialised,
    so the appended block starts as an "average document" rather than noise.
    """

    def __init__(self, hidden_size, n_mem, cfg: FusionConfig):
        super().__init__()
        self.cfg, self.hidden_size, self.n_mem = cfg, hidden_size, n_mem
        d = cfg.d_model
        self.queries = nn.Parameter(torch.randn(n_mem, d) * 0.02)
        self.down = nn.Linear(hidden_size, d)
        self.q_down = nn.Linear(hidden_size, d)
        self.type_emb = nn.Embedding(3, d)          # 0 memory, 1 question, 2 output queries
        self.slot_emb = nn.Embedding(n_mem, d)
        self.doc_emb = nn.Embedding(MAX_DOCS, d)
        self.q_pos = nn.Embedding(cfg.max_q_tokens, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(cfg.n_layers)])
        self.final_ln = nn.LayerNorm(d)
        self.up = nn.Linear(d, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, memory, q_hidden, q_valid):
        """memory [B, k, n_mem, H]; q_hidden [B, Lq, H]; q_valid [B, Lq] -> new block [B, n_mem, H]."""
        b, k, n_mem, h = memory.shape
        lq = q_hidden.size(1)
        dev = memory.device
        x_out = (self.queries + self.type_emb.weight[2] + self.slot_emb(torch.arange(n_mem, device=dev)))[None].expand(b, -1, -1)
        x_mem = self.down(memory.float()).view(b, k * n_mem, -1) + self.type_emb.weight[0] \
            + self.doc_emb(torch.arange(k, device=dev)).repeat_interleave(n_mem, 0)[None] \
            + self.slot_emb(torch.arange(n_mem, device=dev)).repeat(k, 1)[None]
        x_q = self.q_down(q_hidden.float()) + self.type_emb.weight[1] + self.q_pos(torch.arange(lq, device=dev))[None]
        x = torch.cat([x_out, x_mem, x_q], dim=1)
        t = x.size(1)
        allowed = torch.ones(b, t, t, dtype=torch.bool, device=dev)
        allowed[:, :, n_mem + k * n_mem:] &= q_valid[:, None, :]
        mask = (~allowed).repeat_interleave(self.cfg.n_heads, 0)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        delta = self.up(self.final_ln(x[:, :n_mem]))
        return (memory.float().mean(dim=1) + delta).to(memory.dtype)

    def save(self, path, extra=None):
        torch.save({"kind": "append", "config": asdict(self.cfg), "hidden_size": self.hidden_size, "n_mem": self.n_mem,
                    "state_dict": self.state_dict(), "extra": extra or {}}, path)


class FusionConst(nn.Module):
    """Control for FusionAppend: the sixth block is a learned constant (no information from the
    documents or the question). Isolates the effect of 'one more slot' on decoder behaviour."""

    def __init__(self, hidden_size, n_mem, cfg: FusionConfig):
        super().__init__()
        self.cfg, self.hidden_size, self.n_mem = cfg, hidden_size, n_mem
        self.block = nn.Parameter(torch.zeros(n_mem, hidden_size))

    def forward(self, memory, q_hidden=None, q_valid=None):
        return self.block[None].expand(memory.size(0), -1, -1).to(memory.dtype)

    def save(self, path, extra=None):
        torch.save({"kind": "const", "config": asdict(self.cfg), "hidden_size": self.hidden_size, "n_mem": self.n_mem,
                    "state_dict": self.state_dict(), "extra": extra or {}}, path)


APPEND_KINDS = {"append": FusionAppend, "const": FusionConst}


def load_append(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    module = APPEND_KINDS[ckpt["kind"]](ckpt["hidden_size"], ckpt["n_mem"], FusionConfig(**ckpt["config"]))
    module.load_state_dict(ckpt["state_dict"])
    return module.to(device), ckpt.get("extra", {})


def append_block(wrapper, module, questions, memory_per_question):
    """Inference: list of [k, 8, H] -> list of [k+1, 8, H] with the new block appended last."""
    cfg = module.cfg
    memory = torch.stack(memory_per_question)
    q_hidden, q_valid = encode_questions(wrapper.model, wrapper.tokenizer, questions, cfg.q_layer, cfg.max_q_tokens)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        block = module(memory, q_hidden, q_valid)
    return [torch.cat([m, blk[None].to(m.dtype)], dim=0) for m, blk in zip(memory_per_question, block)]
