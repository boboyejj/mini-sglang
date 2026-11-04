"""Experiment 8: why a KV cache exists at all.

Concept: without a cache, every decode step re-runs the model over the WHOLE sequence,
so step n costs about as much as a prefill of n tokens. With a cache, each step feeds only
the newest token and reads the stored K/V. The output is identical; only the cost differs.
Everything in mini-sglang's kvcache/ and scheduler/cache.py exists to manage this memory.

Run:  python experiments/08_kv_cache_on_off.py
Try:  halve the prompt length and see which decode time changes.
"""

import _model as M

NEW_TOKENS = 32
# a longer prompt makes the difference easier to see
QUESTION = "Here is some context. " * 40 + "Now list three primary colors."


def main() -> None:
    tok, model = M.load()
    ids = M.chat_ids(tok, QUESTION)
    print(f"prompt: {ids.shape[1]} tokens, generating {NEW_TOKENS} tokens\n")

    M.generate(model, ids[:, :8], max_new_tokens=2)  # warm-up pass, not timed
    with_cache = M.generate(model, ids, NEW_TOKENS, use_cache=True, ignore_eos=True)
    without = M.generate(model, ids, NEW_TOKENS, use_cache=False, ignore_eos=True)

    for name, r in [("with KV cache", with_cache), ("without cache", without)]:
        total = r.prefill_s + sum(r.decode_s)
        first, last = 1000 * r.decode_s[0], 1000 * r.decode_s[-1]
        print(
            f"{name}: total {total:5.1f} s | decode step 1: {first:5.0f} ms,"
            f" last step: {last:5.0f} ms"
        )
    same = with_cache.tokens == without.tokens
    print(f"\nidentical output tokens: {same}")
    print(f"output: {tok.decode(with_cache.tokens, skip_special_tokens=True)!r}")
    print(
        "\nQuestion: without a cache, one decode step costs about as much as a prefill of the"
        "\nwhole sequence. Why? The cache removes that cost; what does it cost instead?"
        "\n(Hint: run experiment 9.)"
    )


if __name__ == "__main__":
    main()
