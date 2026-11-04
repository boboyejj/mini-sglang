"""Experiment 3: how much prefill work does the radix cache save for chat traffic?

Concept: many chat requests start with the same system prompt. The radix cache keeps its
K/V after the first request, so later requests only compute their own question.
Uses Qwen3-0.6B's real tokenizer (small text files; no model weights needed).

Run:  python experiments/03_prefix_sharing_chat.py
Try:  make the system prompt longer, or give every user a different system prompt.
"""

import _common as C

SYSTEM = (
    "You are a helpful assistant for a cooking website. Answer briefly, list ingredients "
    "with quantities, and always mention any common allergens in the recipe."
)
QUESTIONS = [
    "How do I make pancakes?",
    "What is a quick vegetarian dinner?",
    "How long should I boil an egg?",
    "Can I freeze cooked rice?",
]


def build_prompts():
    tok = C.load_tokenizer()
    texts = [f"System: {SYSTEM}\nUser: {q}\nAssistant:" for q in QUESTIONS]
    if tok is None:  # fallback: fake IDs with the same shape (shared prefix + own tail)
        return [list(range(1, 41)) + [100 + 10 * i + j for j in range(8)] for i in range(4)]
    return [tok(t)["input_ids"] for t in texts]


def run(cache_type: str, prompts) -> int:
    m = C.make_managers(num_pages=512, cache_type=cache_type)
    # one request every 6 steps, so each arrives after the previous one finished
    arrivals = {6 * i: [C.user_msg(i, p, max_tokens=4)] for i, p in enumerate(prompts)}
    logs, hits = C.run_sim(m, arrivals, verbose=False)
    computed = sum(log.tokens_computed for log in logs if log.phase == "prefill")
    print(f"\ncache_type={cache_type}")
    for uid, p in enumerate(prompts):
        print(f"  request {uid}: prompt {len(p):>3} tokens, reused from cache {hits[uid]:>3}")
    print(f"  prefill tokens computed: {computed} of {sum(len(p) for p in prompts)}")
    return computed


def main() -> None:
    prompts = build_prompts()
    naive = run("naive", prompts)
    radix = run("radix", prompts)
    print(
        f"\nradix cache computed {naive - radix} fewer prefill tokens ({1 - radix / naive:.0%} less work)"
    )


if __name__ == "__main__":
    main()
