"""Raw path: the original Mistral backbone reading the same evidence as plain text.

Loaded from the pinned backbone snapshot as a fresh AutoModelForCausalLM —
NOT a PISCO object with adapters disabled (that object has resized
embeddings and extra tokens).

Prompt rendering mirrors PISCO's (same system text and user template from
modelling_pisco.py L1048-1049); only the "Background" block differs:
PISCO places memory slots there, Raw places the documents' used_text joined
by RAW_DOC_SEPARATOR, in manifest order, without numbering.
"""

import torch

# Copied verbatim from the pinned modelling_pisco.py L1048 (checked at runtime by the caller).
SYSTEM_PROMPT = ("You are a helpful assistant. Your task is to extract relevant information from "
                 "provided documents and to answer to questions as briefly as possible.")
RAW_DOC_SEPARATOR = "\n"
MAX_CONTEXT_TOKENS = 32768  # Mistral-7B-Instruct-v0.2 context window; longer prompts are rejected


class RawBackbone:
    def __init__(self, backbone_dir):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_dir, use_fast=True, padding_side="left")
        # attn_implementation left unset, as in the PISCO decoder load (resolves to sdpa, Step 03)
        self.model = AutoModelForCausalLM.from_pretrained(backbone_dir, torch_dtype=torch.bfloat16).to("cuda")
        self.model.eval()

    def build_prompt(self, question, used_texts):
        user = f"Background:\n{RAW_DOC_SEPARATOR.join(used_texts)}\n\nQuestion:{question}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, question, used_texts, max_new_tokens=128):
        """One question -> new token IDs [T] and decoded text (batch size 1, greedy)."""
        prompt = self.build_prompt(question, used_texts)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        if enc["input_ids"].size(1) > MAX_CONTEXT_TOKENS:
            raise ValueError(f"prompt of {enc['input_ids'].size(1)} tokens exceeds {MAX_CONTEXT_TOKENS}")
        input_ids = enc["input_ids"].to("cuda")
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, attention_mask=enc["attention_mask"].to("cuda"),
                                      do_sample=False, max_new_tokens=max_new_tokens,
                                      pad_token_id=self.tokenizer.bos_token_id)  # same pad id as the PISCO decoder
        new_ids = out[0, input_ids.size(1):].cpu()   # with input_ids, generate returns prompt + new tokens
        return {"prompt": prompt, "input_length": input_ids.size(1), "output_ids": new_ids,
                "text": self.tokenizer.decode(new_ids, skip_special_tokens=True)}
