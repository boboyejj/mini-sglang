"""Experiment 10: prefix caching with real K/V tensors.

Concept: with causal attention, the K/V of a token depends only on the tokens BEFORE it.
So the K/V computed for a shared system prompt is valid for every request that starts with
it. This is the guarantee mini-sglang's radix cache relies on. Here we compute the shared
prefix once, reuse a copy of its cache for each question, and check that the answers are
the same as computing every prompt from scratch.

Run:  python experiments/10_prefix_reuse_real.py
Try:  change one word near the START of the system prompt for one question only. What
      is the longest prefix you can still reuse?
"""

import copy
import time

import _model as M

SYSTEM = (
    "You are a friendly assistant for a cooking website. Keep answers to one sentence. "
    "Always mention the cooking time, the main ingredient, and one tip for beginners. "
) * 4
QUESTIONS = [
    "How do I make pancakes?",
    "How long should I boil an egg?",
    "What is a quick vegetarian dinner?",
]
NEW_TOKENS = 24


def common_prefix_len(seqs) -> int:
    n = 0
    while all(len(s) > n for s in seqs) and len({s[n] for s in seqs}) == 1:
        n += 1
    return n


def main() -> None:
    tok, model = M.load()
    prompts = [M.chat_ids(tok, q, system=SYSTEM) for q in QUESTIONS]
    shared = common_prefix_len([p[0].tolist() for p in prompts])
    print(f"prompt lengths: {[p.shape[1] for p in prompts]}, shared prefix: {shared} tokens\n")
    M.generate(model, prompts[0][:, :8], max_new_tokens=1)  # warm-up, not timed

    # compute the shared prefix's K/V once (this is what the radix tree stores)
    t0 = time.perf_counter()
    prefix = M.generate(model, prompts[0][:, :shared], max_new_tokens=1).cache
    prefix.crop(shared)
    print(f"prefix prefill (once): {1000 * (time.perf_counter() - t0):.0f} ms\n")

    for q, ids in zip(QUESTIONS, prompts):
        scratch = M.generate(model, ids, NEW_TOKENS)
        reused = M.generate(model, ids[:, shared:], NEW_TOKENS, cache=copy.deepcopy(prefix))
        print(f"Q: {q}")
        print(
            f"  prefill from scratch: {1000 * scratch.prefill_s:5.0f} ms ({ids.shape[1]} tokens)"
            f" | with reused prefix: {1000 * reused.prefill_s:5.0f} ms"
            f" ({ids.shape[1] - shared} tokens)"
        )
        print(f"  same answer: {scratch.tokens == reused.tokens}")
        print(f"  A: {tok.decode(reused.tokens, skip_special_tokens=True)!r}\n")
    print(
        "Question: we feed ids[shared:] after the cached prefix. mini-sglang matches at most"
        "\nlen(prompt) - 1 tokens, so at least one token is always fed. Why must it be?"
    )


if __name__ == "__main__":
    main()
