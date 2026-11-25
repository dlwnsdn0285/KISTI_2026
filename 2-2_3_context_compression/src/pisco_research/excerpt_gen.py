"""Step 17: a dedicated generator for ONE extra memory block built from a raw excerpt + question
(+ optionally the existing document memories), with PISCO's compressor and decoder frozen.

The generator is a separate LoRA adapter ("excerpt_adapter") on the shared backbone, initialised
from PISCO's encoder adapter and trained on its own. Input layout (PISCO's query-conditioned
encoder format, L = EXCERPT_MAX_LEN + 8 text positions, left padded):

    [memory vectors of the k documents]   (variant textmem only; prepended as embeddings)
    <ENC><s>\\nQuery:\\n{question}Document:\\n{excerpt}</s>  <MEM0..MEM7>

The last-layer hidden states at the 8 MEM positions are the block. The decoder then sees
[k document memories, block] in k+1 slots, exactly as in Steps 13-15.
"""

import torch

ADAPTER = "excerpt_adapter"
EXCERPT_MAX_LEN = 128


def add_excerpt_adapter(model):
    """Add the adapter (same LoRA shape as PISCO's) and copy the encoder adapter's weights into it."""
    from peft import LoraConfig
    if ADAPTER in getattr(model.decoder, "peft_config", {}):
        return
    # same target modules as the existing encoder adapter ("all-linear" would also match LoRA submodules)
    targets = sorted({n.split(".lora_A.")[0].rsplit(".", 1)[-1] for n, _ in model.decoder.named_parameters()
                      if ".lora_A.encoder_adapter." in n})
    cfg = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32, target_modules=targets, lora_dropout=0.0)
    model.decoder.add_adapter(cfg, ADAPTER)
    params = dict(model.decoder.named_parameters())
    n_src = sum(".encoder_adapter." in n for n in params)
    n_new = sum(f".{ADAPTER}." in n for n in params)
    if n_src != n_new:
        raise RuntimeError(f"excerpt adapter has {n_new} tensors, encoder adapter {n_src} (targets {targets})")
    copied = 0
    with torch.no_grad():
        for name, p in params.items():
            if ".encoder_adapter." in name:
                target = name.replace(".encoder_adapter.", f".{ADAPTER}.")
                if target in params:
                    params[target].copy_(p)
                    copied += 1
    if copied == 0:
        raise RuntimeError("no encoder adapter weights copied into the excerpt adapter")
    return copied


def adapter_params(model):
    return {n: p for n, p in model.decoder.named_parameters() if "lora_" in n and f".{ADAPTER}." in n}


def excerpt_token_inputs(tok, questions, excerpts, max_len=EXCERPT_MAX_LEN):
    texts = [tok.enc_token + tok.bos_token + "\nQuery:\n" + q + "Document:\n" + e + tok.eos_token
             for q, e in zip(questions, excerpts)]
    # Steps 17-18 (max_len 128): fixed-length padding as in PISCO. Step 19 (full five documents): pad to the longest row.
    padding = "max_length" if max_len <= EXCERPT_MAX_LEN else "longest"
    enc = tok(texts, return_tensors="pt", padding=padding, max_length=max_len + 8, truncation=True, add_special_tokens=False)
    mem = torch.tensor(tok.mem_token_ids)[None].repeat(len(texts), 1)
    ids = torch.cat([enc["input_ids"], mem], dim=1)
    mask = torch.cat([enc["attention_mask"], torch.ones_like(mem)], dim=1)
    return ids, mask


def generate_block(model, tok, questions, excerpts, memory=None, max_len=EXCERPT_MAX_LEN):
    """-> block [B, 8, H] (dtype of the decoder). memory: [B, k, 8, H] or None (variant text).

    Call with the right grad context; the caller sets adapters afterwards for the decoder pass.
    """
    device = model.decoder.device
    ids, mask = excerpt_token_inputs(tok, questions, excerpts, max_len)
    ids, mask = ids.to(device), mask.to(device)
    model.decoder.set_adapter(ADAPTER)
    embeds = model.decoder.get_input_embeddings()(ids)
    prefix = 0
    if memory is not None:
        b, k, n_mem, h = memory.shape
        mem_flat = memory.reshape(b, k * n_mem, h).to(embeds.dtype)
        embeds = torch.cat([mem_flat, embeds], dim=1)
        mask = torch.cat([torch.ones(b, k * n_mem, dtype=mask.dtype, device=device), mask], dim=1)
        prefix = k * n_mem
    out = model.decoder(inputs_embeds=embeds, attention_mask=mask, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[-1]
    is_mem = torch.isin(ids, torch.tensor(tok.mem_token_ids, device=device))
    block = hidden[:, prefix:][is_mem].reshape(ids.size(0), len(tok.mem_token_ids), hidden.size(-1))
    return block


# ============================================================ Step 18: residual generator

class MemProj(torch.nn.Module):
    """Per-dimension affine map for memory vectors prepended to the generator input.

    PISCO memories live on a different scale than token embeddings (the Step 17 textmem run started at
    val 0.997 vs 0.340 for text). scale is initialised to (mean token-embedding norm / mean memory norm)
    so the prefix enters the generator at the norm of ordinary tokens.
    """

    def __init__(self, hidden_size, init_scale=1.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.full((hidden_size,), float(init_scale)))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        return x * self.scale.to(x.dtype) + self.bias.to(x.dtype)


def init_memproj(model, memory_sample):
    emb = model.decoder.get_input_embeddings().weight.float()
    ratio = emb.norm(dim=-1).mean() / memory_sample.float().norm(dim=-1).mean()
    return MemProj(emb.size(1), ratio.item()).to(model.decoder.device)


def generate_block_res(model, tok, questions, excerpts, memory=None, memproj=None, max_len=EXCERPT_MAX_LEN):
    """generate_block with an optional MemProj on the memory prefix."""
    if memory is not None and memproj is not None:
        memory = memproj(memory.float()).to(memory.dtype)
    return generate_block(model, tok, questions, excerpts, memory, max_len)


def token_ce(logits, labels):
    """Per-row list of per-token cross-entropies at label positions (labels: -100 elsewhere)."""
    lg = logits[:, :-1].float()
    lb = labels[:, 1:]
    ce = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1), reduction="none", ignore_index=-100)
    ce = ce.view(lb.shape)
    return [ce[i][lb[i] != -100] for i in range(lb.size(0))]
