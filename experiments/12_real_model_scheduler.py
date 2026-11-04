"""Experiment 12 (capstone): mini-sglang's real scheduler serving the real model.

Concept: everything together. mini-sglang's real PrefillManager, DecodeManager,
CacheManager (radix prefix cache) and TableManager decide every batch, exactly as in
experiments 1-6. But instead of fake tokens, each batch runs Qwen3-0.6B, and the K/V
lives in a paged pool indexed by mini-sglang's page_table, like MHAKVCache on a GPU:

    page_table[row, pos] -> KV slot   (decided by mini-sglang's CacheManager)
    pool[layer, slot]    -> K and V   (written and read by this CPU "engine")

A cached prefix found by the radix tree is read straight from its old slots, never
recomputed. At the end, every answer is checked against plain generation from scratch.
The engine runs requests in a batch one at a time (no padding), so this shows
correctness and scheduling, not batched speed; experiment 11 covers that.

Run:  python experiments/12_real_model_scheduler.py
Try:  PREFILL_BUDGET = 512 (no chunking), or cache_type="naive" in make_managers.
"""

import time

import _common as C
import _model as M
import torch
from minisgl.scheduler.prefill import ChunkedReq
from transformers import DynamicCache

PREFILL_BUDGET = 64  # tokens per prefill batch (mini-sglang: --max-prefill-length)
MAX_TOKENS = 20
SYSTEM = (
    "You are a friendly assistant for a cooking website. Keep answers to one sentence. "
    "Always mention the cooking time, the main ingredient, and one tip for beginners. "
) * 4
ARRIVALS = {  # step -> questions arriving at that step
    0: ["How do I make pancakes?"],
    4: ["How long should I boil an egg?", "What is a quick vegetarian dinner?"],
    8: ["How do I cook rice?"],
}


class PagedKVEngine:
    """A CPU stand-in for mini-sglang's Engine + MHAKVCache, using Hugging Face layers."""

    def __init__(self, model, managers: C.Managers, num_slots: int):
        cfg = model.config
        shape = (cfg.num_hidden_layers, num_slots, cfg.num_key_value_heads, cfg.head_dim)
        self.k, self.v = torch.zeros(shape), torch.zeros(shape)  # the KV pool
        self.model, self.m, self.layers = model, managers, cfg.num_hidden_layers
        self.step_ms: list = []

    @torch.inference_mode()
    def __call__(self, batch) -> dict:
        t0, sampled = time.perf_counter(), {}
        for r in batch.reqs:
            row, old, new = r.table_idx, r.cached_len, r.device_len
            old_slots = self.m.table.page_table[row, :old].long()  # K/V already stored here
            new_slots = self.m.table.page_table[row, old:new].long()  # this step writes here
            cache = DynamicCache()
            if old > 0:  # gather the request's cached K/V from the pool (incl. shared prefix)
                for layer in range(self.layers):
                    k = self.k[layer, old_slots].permute(1, 0, 2).unsqueeze(0)
                    v = self.v[layer, old_slots].permute(1, 0, 2).unsqueeze(0)
                    cache.update(k, v, layer)
            ids = r.input_ids[old:new].long().unsqueeze(0)  # only the uncached tokens
            out = self.model(ids, past_key_values=cache, use_cache=True)
            for layer in range(self.layers):  # store new K/V at out_loc, like store_kv
                self.k[layer, new_slots] = cache.layers[layer].keys[0, :, old:].permute(1, 0, 2)
                self.v[layer, new_slots] = cache.layers[layer].values[0, :, old:].permute(1, 0, 2)
            if not isinstance(r, ChunkedReq):  # a chunk's logits are mid-prompt: no sampling
                sampled[r.uid] = int(out.logits[0, -1].argmax())
        self.step_ms.append(1000 * (time.perf_counter() - t0))
        return sampled


def main() -> None:
    tok, model = M.load()
    eos = model.config.eos_token_id
    m = C.make_managers(num_pages=1024, max_running_req=8, max_seq_len=512)
    engine = PagedKVEngine(model, m, num_slots=1024)

    prompts, arrivals, uid = {}, {}, 0
    for step, questions in ARRIVALS.items():
        for q in questions:
            prompts[uid] = (q, M.chat_ids(tok, q, system=SYSTEM)[0])
            arrivals.setdefault(step, []).append(C.user_msg(uid, prompts[uid][1], MAX_TOKENS))
            uid += 1
    print(f"{len(prompts)} requests, prompts of {[len(p) for _, p in prompts.values()]} tokens")
    print(f"prefill budget {PREFILL_BUDGET} tokens/step, max_tokens {MAX_TOKENS}\n")

    logs, hits = C.run_sim(m, arrivals, prefill_budget=PREFILL_BUDGET, eos_id=eos, engine=engine)
    print(f"\n{len(logs)} steps, engine time {sum(engine.step_ms) / 1000:.1f} s")

    print("\nchecking every answer against plain generation from scratch:")
    computed = sum(log.tokens_computed for log in logs if log.phase == "prefill")
    for uid, (q, ids) in prompts.items():
        ref = M.generate(model, ids.unsqueeze(0), MAX_TOKENS).tokens
        got = logs.generated[uid]
        print(
            f"  req {uid}: reused {hits[uid]:>3} of {len(ids)} prompt tokens | identical: {got == ref}"
        )
        print(f"     Q: {q}\n     A: {tok.decode(got, skip_special_tokens=True)!r}")
    total = sum(len(p) for _, p in prompts.values())
    print(f"\nprefill tokens computed: {computed} of {total} ({1 - computed / total:.0%} saved)")
    print(
        "\nQuestion: request 1 reused request 0's prefix while request 0 was still decoding."
        "\nWhat stops CacheManager from evicting or overwriting those slots meanwhile?"
    )


if __name__ == "__main__":
    main()
