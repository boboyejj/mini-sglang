"""Step 3: test the trained tiny Qwen3.

1. Validation loss and perplexity on 100 batches of held-out stories (more than during training).
2. Stories from new opening lines, two decoding styles: greedy and sampled.
3. Speed and size, compared with Qwen3-0.6B.

Run:  python training/test_model.py
"""

import math
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = [
    "Once upon a time, a little rabbit wanted to fly.",
    "Sara and her brother went to the beach.",
    "The old robot sat alone in the garden.",
]


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    path = os.path.join(HERE, "tiny-qwen3")
    tok = AutoTokenizer.from_pretrained(path)
    model = Qwen3ForCausalLM.from_pretrained(path).eval()
    print(f"loaded {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M-parameter model from {path}\n")

    val = np.fromfile(os.path.join(HERE, "data", "val.bin"), dtype=np.uint16)
    rng, losses = np.random.default_rng(99), []
    for _ in range(100):
        starts = rng.integers(0, len(val) - 257, size=16)
        x = torch.from_numpy(np.stack([val[s : s + 256] for s in starts]).astype(np.int64))
        losses.append(model(input_ids=x, labels=x).loss.item())
    loss = float(np.mean(losses))
    print(f"validation loss (100 batches, {100 * 16 * 256:,} tokens): {loss:.3f}")
    print(f"perplexity: {math.exp(loss):.1f}  (random guessing: {len(tok):,})\n")

    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        kw = dict(max_new_tokens=100, eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        greedy = model.generate(ids, do_sample=False, **kw)
        t0 = time.perf_counter()
        sampled = model.generate(ids, do_sample=True, temperature=0.8, top_k=50, **kw)
        dt = time.perf_counter() - t0
        n = sampled.shape[1] - ids.shape[1]
        print(f"PROMPT: {p}")
        print(f"  greedy : {tok.decode(greedy[0, ids.shape[1]:], skip_special_tokens=True)!r}")
        print(f"  sampled: {tok.decode(sampled[0, ids.shape[1]:], skip_special_tokens=True)!r}")
        print(f"  ({n} tokens in {dt:.2f} s = {n / dt:.0f} tokens/s)\n")


if __name__ == "__main__":
    main()
