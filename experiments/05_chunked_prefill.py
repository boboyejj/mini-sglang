"""Experiment 5: chunked prefill splits a long prompt across several batches.

Concept: each prefill batch may compute at most `prefill_budget` tokens
(--max-extend-tokens on the server). A longer prompt becomes a ChunkedReq that continues
in the next batch. This caps how much work (and activation memory) one step can take.

Run:  python experiments/05_chunked_prefill.py
Try:  other budgets; or make the short requests arrive at step 0 together with the long one.
"""

import _common as C

LONG_PROMPT = 64
SHORT_PROMPT = 6


def run(budget: int) -> None:
    print(f"\n=== prefill_budget = {budget} ===")
    m = C.make_managers(num_pages=512)
    arrivals = {
        0: [
            C.user_msg(0, range(1, SHORT_PROMPT + 1), max_tokens=8),
            C.user_msg(1, range(11, SHORT_PROMPT + 11), max_tokens=8),
        ],
        2: [
            C.user_msg(2, range(100, 100 + LONG_PROMPT), max_tokens=3)
        ],  # long prompt arrives later
    }
    logs, _ = C.run_sim(m, arrivals, prefill_budget=budget)
    biggest = max(log.tokens_computed for log in logs)
    paused = sum(1 for log in logs if log.phase == "prefill" and log.step >= 2)
    print(f"largest step: {biggest} tokens | prefill steps while reqs 0,1 were decoding: {paused}")


def main() -> None:
    for budget in (64, 16):
        run(budget)
    print(
        "\nQuestion: with budget 16 no step computes more than 16 tokens, but requests 0 and 1"
        "\nstill wait for every chunk. Look at Scheduler._schedule_next_batch: why?"
    )


if __name__ == "__main__":
    main()
