"""Helpers for the experiments that use real Qwen3-0.6B weights on the CPU.

We run the model with Hugging Face `transformers`, not mini-sglang's engine: the engine's
attention and sampling kernels (FlashInfer, sgl-kernel) need an NVIDIA GPU. The loop in
`generate()` is written by hand to mirror mini-sglang's Engine: one prefill pass over the
prompt, then one decode pass per token that reuses the KV cache, with greedy sampling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen3-0.6B"
torch.manual_seed(0)


def _local_path() -> str:
    """Return the local copy of the model, downloading it only if it is missing.

    Loading by repo name makes `transformers` call the Hugging Face API on every run, and
    anonymous API calls get rate-limited (HTTP 429). A local path needs no network.
    """
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(MODEL, local_files_only=True)
    except Exception:
        return snapshot_download(MODEL)  # first run: ~1.5 GB download


def load():
    """Load tokenizer + model (float32 on CPU, ~2.4 GB RAM). Takes ~10 s."""
    t0 = time.perf_counter()
    path = _local_path()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    print(f"loaded {MODEL} in {time.perf_counter() - t0:.1f} s")
    return tok, model


def chat_ids(tok, user: str, system: str | None = None) -> torch.Tensor:
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    return tok.apply_chat_template(
        msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
    )


@dataclass
class Result:
    tokens: list = field(default_factory=list)
    prefill_s: float = 0.0  # time to first token
    decode_s: list = field(default_factory=list)  # time of each later token
    cache: DynamicCache | None = None

    @property
    def per_token_ms(self) -> float:
        return 1000 * sum(self.decode_s) / max(len(self.decode_s), 1)


@torch.inference_mode()
def generate(
    model,
    ids: torch.Tensor,
    max_new_tokens: int,
    cache=None,
    use_cache=True,
    ignore_eos=False,
    eos_id=None,
) -> Result:
    """Greedy generation with a hand-written prefill/decode loop.

    ids:   the tokens to feed now (the whole prompt, or only the part not in `cache`).
    cache: an existing KV cache to extend (a reused prefix), or None to start empty.
    use_cache=False recomputes attention over the whole sequence at every step.
    ignore_eos=True keeps generating to max_new_tokens (like SamplingParams.ignore_eos).
    eos_id overrides the model config's end-of-sequence token (the tiny model's config has none).
    """
    eos = eos_id if eos_id is not None else model.config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    res = Result()
    if use_cache and cache is None:
        cache = DynamicCache()
    seq = ids

    t0 = time.perf_counter()
    out = model(ids, past_key_values=cache, use_cache=use_cache)  # prefill
    for i in range(max_new_tokens):
        next_tok = int(out.logits[0, -1].argmax())  # greedy sampling, like Sampler on argmax
        if i == 0:
            res.prefill_s = time.perf_counter() - t0
        else:
            res.decode_s.append(time.perf_counter() - t0)
        res.tokens.append(next_tok)
        if (next_tok in eos and not ignore_eos) or i == max_new_tokens - 1:
            break
        t0 = time.perf_counter()
        step = torch.tensor([[next_tok]])
        if use_cache:  # decode: feed ONE token; attention reads the cached K/V
            out = model(step, past_key_values=cache, use_cache=True)
        else:  # no cache: feed the WHOLE sequence again
            seq = torch.cat([seq, step], dim=1)
            out = model(seq, use_cache=False)
    res.cache = cache
    return res
