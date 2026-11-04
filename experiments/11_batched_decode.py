"""Experiment 11: why batching makes decode cheap.

Concept: a decode step reads all the model's weights to produce ONE token per request
(experiment 7). If B requests share the step, the weights are read once for all of them,
so step time grows far slower than B and throughput (tokens/s) rises. This is why
mini-sglang's DecodeManager puts every running request into the same decode step.

Each batch here is B copies of the same prompt, so no padding is needed. After a
prefill, we time DECODE_STEPS decode steps and repeat the measurement REPEATS times.

Run:  python experiments/11_batched_decode.py
Try:  a longer prompt (more cached tokens to attend over), or BATCH_SIZES up to 128.
"""

import statistics
import time

import _model as M
import torch
from transformers import DynamicCache

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
DECODE_STEPS = 8
REPEATS = 3


@torch.inference_mode()
def time_decode(model, ids: torch.Tensor, batch: int) -> float:
    """Median ms per decode step for `batch` copies of the prompt."""
    runs = []
    for _ in range(REPEATS):
        cache = DynamicCache()
        out = model(ids.repeat(batch, 1), past_key_values=cache, use_cache=True)  # prefill
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        t0 = time.perf_counter()
        for _ in range(DECODE_STEPS):
            out = model(nxt, past_key_values=cache, use_cache=True)  # one token per request
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        runs.append(1000 * (time.perf_counter() - t0) / DECODE_STEPS)
    return statistics.median(runs)


def main() -> None:
    tok, model = M.load()
    ids = M.chat_ids(tok, "Write a short poem about the ocean.")
    print(f"torch threads: {torch.get_num_threads()} | prompt: {ids.shape[1]} tokens\n")
    time_decode(model, ids, 1)  # warm-up

    print(
        f"{'batch':>5} {'ms/step':>8} {'step vs B=1':>11} {'tokens/s':>9} {'throughput gain':>15}"
    )
    base_ms = None
    for b in BATCH_SIZES:
        ms = time_decode(model, ids, b)
        base_ms = base_ms or ms
        tps = 1000 * b / ms
        print(
            f"{b:>5} {ms:>8.0f} {ms / base_ms:>10.2f}x {tps:>9.1f} {tps / (1000 / base_ms):>14.1f}x"
        )
    print(
        "\nQuestion: why does throughput eventually stop growing as the batch gets larger?"
        "\n(Hint: at some batch size the matrix multiplies, not the weight reads, dominate.)"
    )


if __name__ == "__main__":
    main()
