"""Experiment 6: how many requests can run at once in a small KV pool?

Concept: PrefillAdder only admits a request if its whole future fits: uncached prompt +
max_tokens + what running requests may still need. Nothing is ever preempted, but if
requests stop early (EOS), much of that reserved memory was never needed.

Run:  python experiments/06_admission_memory.py
Try:  change POOL_TOKENS, or MAX_TOKENS while keeping STOPS_AFTER fixed.
"""

import _common as C

POOL_TOKENS = 100  # tiny KV pool (a real GPU has hundreds of thousands of slots)
NUM_REQS = 8
PROMPT = 10
STOPS_AFTER = 5  # every request "samples" EOS as its 5th output token


def run(max_tokens: int) -> None:
    print(f"\n=== max_tokens = {max_tokens}: each request reserves {PROMPT + max_tokens} slots ===")
    m = C.make_managers(num_pages=POOL_TOKENS, max_running_req=NUM_REQS)
    msgs = [
        C.user_msg(i, range(100 * i + 1, 100 * i + 1 + PROMPT), max_tokens) for i in range(NUM_REQS)
    ]
    eos = dict.fromkeys(range(NUM_REQS), STOPS_AFTER)
    logs, _ = C.run_sim(m, {0: msgs}, eos_after=eos, eos_id=7, verbose=False)
    peak_running = max(log.running for log in logs)
    peak_busy = max(log.kv_in_use for log in logs)
    print(f"steps to finish all {NUM_REQS} requests: {len(logs)}")
    print(f"most requests running at once:        {peak_running}")
    print(f"most KV slots actually busy:          {peak_busy} of {POOL_TOKENS}")


def main() -> None:
    for max_tokens in (5, 20, 60):
        run(max_tokens)
    print(
        "\nEvery request really generates only 5 tokens, yet a larger max_tokens means fewer"
        "\nrequests run together and the whole workload takes more steps. Question: what would it take to"
        "\nadmit more requests safely? (Hint: what if memory runs out mid-decode?)"
    )


if __name__ == "__main__":
    main()
