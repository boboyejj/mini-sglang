"""Experiment 7: real generation with Qwen3-0.6B, timing prefill vs decode.

Concept: prefill processes the whole prompt in ONE forward pass and produces the first
token (time to first token, TTFT). Decode then produces one token per forward pass
(time per output token, TPOT). A server tunes for both. See study guide Modules 0 and 4.

Run:  python experiments/07_generate_cpu.py
Try:  make the prompt 10x longer (repeat the question) and see which number grows.
"""

import _model as M

QUESTION = "Explain in two sentences why the sky is blue."


def main() -> None:
    tok, model = M.load()
    ids = M.chat_ids(tok, QUESTION)
    res = M.generate(model, ids, max_new_tokens=48)
    print(f"\nprompt: {ids.shape[1]} tokens  ->  generated: {len(res.tokens)} tokens")
    print(f"answer: {tok.decode(res.tokens, skip_special_tokens=True)!r}\n")
    print(f"prefill (all {ids.shape[1]} prompt tokens, 1 pass): {1000 * res.prefill_s:7.0f} ms")
    print(f"decode  (1 token per pass), average:     {res.per_token_ms:7.0f} ms")
    per_prompt_token = 1000 * res.prefill_s / ids.shape[1]
    print(f"prefill cost per prompt token:           {per_prompt_token:7.1f} ms")
    print(
        "\nQuestion: prefill handles many tokens per pass far more cheaply per token than"
        "\ndecode. Why? (Hint: one big matrix multiply vs. many small ones.) This is why"
        "\nmini-sglang batches many requests into one decode step."
    )


if __name__ == "__main__":
    main()
