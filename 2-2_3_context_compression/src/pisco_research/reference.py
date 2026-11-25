"""Research Base (path B): an explicit wrapper around the pinned HF PISCO model.

It calls the official sub-functions of `modelling_pisco.COCOM` in the same
order as the official API, split into four roles so each can later be
replaced or extended on its own:

    prepare_encoder_inputs  text -> encoder ids/mask        (official prepare_encoder_inputs)
    compress                encoder ids -> memory [N,8,H]   (official compress -> compr_decoder)
    build_decoder_inputs    question + memory -> embeds     (official blend_prompt_and_memory_tokens,
                                                             tokenizer call of L1006, replace_emb)
    generate                embeds -> output ids + text     (decoder.generate as in L1013-1025)

What it adds on top of the official code (no change to any computation):
  * input validation for the silent failure modes found in Step 06
    (reserved token strings, regrouping, wrong slot count, prompt truncation)
  * a fixed compression batch policy: one compress call per question
    (memory differs numerically across compression batch sizes, Step 06)
  * the real generated token IDs (the official API returns strings only)

The model object carries mutable state (`generation_top_k`, the active LoRA
adapter). Calls must therefore be sequential; never share one instance across
threads.
"""

from dataclasses import dataclass

import torch

RESERVED_STRINGS = ("<MEM", "<ENC>", "<AE>", "<SEP>", "<s>", "</s>")
DOC_MAX_LENGTH = 128        # hardcoded by the official compress_documents / generate_from_text (L983, L1031)
DECODER_MAX_LENGTH = 2048   # truncation limit of the official decoder tokenizer call (L1006)
COMPRESSION_POLICY = "per_question"  # each question's k documents form one compress batch


@dataclass
class DecoderInputs:
    prompts: list            # Q chat-formatted prompts with k x [MEM0..7, SEP] placeholders
    input_ids: torch.Tensor  # [Q, L] int64, left-padded with pad id (= BOS id 1)
    attention_mask: torch.Tensor  # [Q, L] int64, 0 on left padding
    inputs_embeds: torch.Tensor   # [Q, L, H] bf16, memory written into the MEM placeholder positions
    mem_blocks: list         # per question: k [first, last] positions of each document's memory


@dataclass
class GenerationOutput:
    output_ids: torch.Tensor  # [Q, T] int64, newly generated tokens only (right-padded after EOS)
    texts: list               # decoded with skip_special_tokens=True, as the official API returns


class PiscoReference:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.decoder_tokenizer
        self.n_mem = model.n_mem_tokens                       # 8 = doc_max_length // compr_rate
        self.hidden_size = model.decoder.config.hidden_size   # 4096
        self.slot_len = self.n_mem + 1 if model.sep else self.n_mem

    # ------------------------------------------------------------ validation

    def validate_documents(self, docs_per_question):
        """docs_per_question: list (one per question) of lists of document strings."""
        if not docs_per_question:
            raise ValueError("no questions")
        ks = {len(docs) for docs in docs_per_question}
        if len(ks) != 1 or 0 in ks:
            raise ValueError(f"every question needs the same non-zero number of documents, got {sorted(ks)}")
        for q_idx, docs in enumerate(docs_per_question):
            for d_idx, text in enumerate(docs):
                hits = [s for s in RESERVED_STRINGS if s in text]
                if hits:
                    # a literal "<MEM3>" would be tokenized as an extra memory slot
                    raise ValueError(f"question {q_idx} document {d_idx} contains reserved strings {hits}")
        return ks.pop()

    def validate_memory(self, questions, memory_per_question):
        if len(memory_per_question) != len(questions):
            raise ValueError(f"{len(memory_per_question)} memory blocks for {len(questions)} questions")
        shapes = {tuple(m.shape) for m in memory_per_question}
        if len(shapes) != 1:
            raise ValueError(f"memory shapes differ across questions: {sorted(shapes)}")
        k, n_mem, hidden = shapes.pop()
        if (n_mem, hidden) != (self.n_mem, self.hidden_size):
            raise ValueError(f"memory per document must be [{self.n_mem}, {self.hidden_size}], got [{n_mem}, {hidden}]")
        if any(m.dtype != self.model.decoder.dtype for m in memory_per_question):
            raise ValueError(f"memory dtype must be {self.model.decoder.dtype}")
        return k

    # ---------------------------------------------------------------- roles

    def prepare_encoder_inputs(self, texts):
        """N document strings -> {'input_ids': [N, 139] int64, 'attention_mask': [N, 139] float32} on CPU.

        139 = 131 (left pad + <ENC> + <s> + text + </s>) + 8 MEM tokens.
        """
        return self.model.prepare_encoder_inputs(texts, max_length=DOC_MAX_LENGTH)

    def compress(self, texts):
        """N documents compressed in ONE batch -> memory [N, 8, H] bf16 on the decoder device."""
        enc = self.prepare_encoder_inputs(texts)
        device = self.model.decoder.device
        with torch.no_grad():
            memory = self.model.compress(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        if tuple(memory.shape) != (len(texts), self.n_mem, self.hidden_size):
            raise RuntimeError(f"unexpected memory shape {tuple(memory.shape)}")
        if self.model.decoder.active_adapters() != ["encoder_adapter"]:
            raise RuntimeError(f"compression ran with adapters {self.model.decoder.active_adapters()}")
        return memory, enc

    def compress_per_question(self, docs_per_question):
        """Apply COMPRESSION_POLICY: one compress call per question -> list of [k, 8, H]."""
        self.validate_documents(docs_per_question)
        results = [self.compress(docs) for docs in docs_per_question]
        return [m for m, _ in results], [enc for _, enc in results]

    def build_decoder_inputs(self, questions, memory_per_question):
        """Questions + per-question memory [k, 8, H] -> DecoderInputs (memory written into MEM slots)."""
        k = self.validate_memory(questions, memory_per_question)
        # Same mutable field the official API sets (L1001); blend_prompt and replace_emb read it.
        self.model.generation_top_k = k
        prompts = [self.model.blend_prompt_and_memory_tokens(query=q) for q in questions]

        lengths = [len(ids) for ids in self.tokenizer(prompts, add_special_tokens=False)["input_ids"]]
        if max(lengths) > DECODER_MAX_LENGTH:
            raise ValueError(f"prompt of {max(lengths)} tokens would be truncated at {DECODER_MAX_LENGTH}")
        enc = self.tokenizer(prompts, return_tensors="pt", padding="longest", add_special_tokens=False,
                             truncation=True, max_length=DECODER_MAX_LENGTH)
        device = self.model.decoder.device
        input_ids, attention_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        mem_blocks = [self._check_slot_layout(row, k) for row in input_ids.tolist()]

        flat_memory = torch.cat(memory_per_question)          # [Q*k, 8, H], question-major order
        with torch.no_grad():
            inputs_embeds = self.model.replace_emb(flat_memory, input_ids)
        return DecoderInputs(prompts, input_ids, attention_mask, inputs_embeds, mem_blocks)

    def generate(self, decoder_inputs, max_new_tokens=128):
        """DecoderInputs -> GenerationOutput, using the same decoder.generate call as L1013-1022."""
        if "decoder_adapter" in self.model.adapter_keys:
            self.model.decoder.set_adapter("decoder_adapter")
        with torch.no_grad():
            output_ids = self.model.decoder.generate(
                inputs_embeds=decoder_inputs.inputs_embeds,
                attention_mask=decoder_inputs.attention_mask,
                generation_config=self.model.generation_config,
                max_new_tokens=max_new_tokens)
        texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return GenerationOutput(output_ids=output_ids, texts=texts)

    def answer(self, questions, docs_per_question, max_new_tokens=128):
        """End to end: per-question compression, decoder input construction, generation."""
        if len(questions) != len(docs_per_question):
            raise ValueError(f"{len(questions)} questions vs {len(docs_per_question)} document lists")
        memory, encoder_inputs = self.compress_per_question(docs_per_question)
        decoder_inputs = self.build_decoder_inputs(questions, memory)
        output = self.generate(decoder_inputs, max_new_tokens=max_new_tokens)
        return {"compression_policy": COMPRESSION_POLICY, "encoder_inputs": encoder_inputs,
                "memory": memory, "decoder_inputs": decoder_inputs, "output": output}

    # --------------------------------------------------------------- helpers

    def _check_slot_layout(self, row, k):
        """One decoder row must hold exactly k contiguous [MEM0..MEM7(, SEP)] blocks."""
        mem_ids = self.tokenizer.mem_token_ids
        mem_positions = [i for i, t in enumerate(row) if t in mem_ids]
        if len(mem_positions) != k * self.n_mem:
            raise ValueError(f"{len(mem_positions)} MEM placeholders, expected {k * self.n_mem}")
        start = row.index(mem_ids[0])
        blocks = []
        for j in range(k):
            first = start + j * self.slot_len
            if row[first:first + self.n_mem] != mem_ids:
                raise ValueError(f"memory block {j} is not MEM0..MEM{self.n_mem - 1} at position {first}")
            blocks.append([first, first + self.n_mem - 1])
        return blocks
