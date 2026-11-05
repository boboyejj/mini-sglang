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

Two workloads:
  --model qwen3-0.6b (default): 4 chat requests sharing a long system prompt (as above).
  --model tiny: our 5.2M-parameter TinyStories model (training/tiny-qwen3) serving all 44
                TinyStories evaluation openings, 4 arriving per step, up to 16 running at
                once. Shared prefixes ("Once upon a time, there...") are found by the radix
                cache on their own.

Run:  python experiments/12_real_model_scheduler.py [--model tiny]
Try:  --prefill-budget 512 (no chunking), --cache naive, or --slots 1024 (memory pressure).
"""

import argparse
import time

import _common as C
import _model as M
import torch
from minisgl.scheduler.prefill import ChunkedReq
from transformers import DynamicCache

PREFILL_BUDGET = 64  # tokens per prefill batch for the Qwen workload (--max-prefill-length)
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


def load_workload(args, tok):
    """Returns (prompts {uid: (label, ids)}, arrivals {step: [UserMsg]})."""
    prompts, arrivals = {}, {}
    if args.model == "tiny":
        import yaml
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            "roneneldan/TinyStories", "Evaluation prompts.yaml", repo_type="dataset"
        )
        with open(path, encoding="utf-8") as f:
            openings = yaml.safe_load(f)
        for uid, text in enumerate(openings):
            prompts[uid] = (
                text[:60].replace("\n", " ") + "...",
                tok(text, return_tensors="pt").input_ids[0],
            )
            arrivals.setdefault(uid // 4, []).append(
                C.user_msg(uid, prompts[uid][1], args.max_tokens)
            )
    else:
        uid = 0
        for step, questions in ARRIVALS.items():
            for q in questions:
                prompts[uid] = (q, M.chat_ids(tok, q, system=SYSTEM)[0])
                arrivals.setdefault(step, []).append(
                    C.user_msg(uid, prompts[uid][1], args.max_tokens)
                )
                uid += 1
    return prompts, arrivals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen3-0.6b", "tiny"], default="qwen3-0.6b")
    ap.add_argument("--prefill-budget", type=int, default=None, help="tokens per prefill step")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--max-running", type=int, default=None, help="max concurrent requests")
    ap.add_argument("--slots", type=int, default=None, help="KV pool size in tokens")
    ap.add_argument("--cache", choices=["radix", "naive"], default="radix")
    args = ap.parse_args()
    tiny = args.model == "tiny"
    args.prefill_budget = args.prefill_budget or (128 if tiny else PREFILL_BUDGET)
    args.max_tokens = args.max_tokens or (64 if tiny else MAX_TOKENS)
    args.max_running = args.max_running or (16 if tiny else 8)
    args.slots = args.slots or (4096 if tiny else 1024)

    if tiny:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained("training/tiny-qwen3")
        model = AutoModelForCausalLM.from_pretrained("training/tiny-qwen3").eval()
        eos = tok.eos_token_id  # the tiny model's config has no eos_token_id
    else:
        tok, model = M.load()
        eos = model.config.eos_token_id
    m = C.make_managers(
        num_pages=args.slots,
        max_running_req=args.max_running,
        max_seq_len=512,
        cache_type=args.cache,
    )
    engine = PagedKVEngine(model, m, num_slots=args.slots)
    prompts, arrivals = load_workload(args, tok)
    lens = [len(p) for _, p in prompts.values()]
    print(
        f"{args.model}: {len(prompts)} requests, prompts {min(lens)}-{max(lens)} tokens"
        f" (total {sum(lens):,}); prefill budget {args.prefill_budget} tokens/step,"
        f" max_tokens {args.max_tokens}, up to {args.max_running} running, {args.slots:,} KV slots,"
        f" {args.cache} cache\n"
    )

    t0 = time.perf_counter()
    logs, hits = C.run_sim(
        m, arrivals, prefill_budget=args.prefill_budget, eos_id=eos, engine=engine, verbose=not tiny
    )
    wall = time.perf_counter() - t0
    generated = sum(len(v) for v in logs.generated.values())
    if tiny:  # 44 requests make ~100 steps: print a sample of the step log
        for log in logs[:12] + logs[12::15]:
            chunk = f" chunked={log.chunked_uids}" if log.chunked_uids else ""
            print(
                f"step {log.step:>3} {log.phase:<7} reqs={len(log.uids):>2}{chunk:<14}"
                f" tokens_computed={log.tokens_computed:<4} running={log.running:>2} kv_busy={log.kv_in_use}"
            )
    decode = [len(log.uids) for log in logs if log.phase == "decode"]
    print(
        f"\n{len(logs)} steps ({len(logs) - len(decode)} prefill, {len(decode)} decode);"
        f" decode batch size avg {sum(decode) / max(len(decode), 1):.1f}, max {max(decode, default=0)}"
    )
    print(
        f"generated {generated:,} tokens in {wall:.1f} s ({generated / wall:.0f} tokens/s;"
        f" engine time {sum(engine.step_ms) / 1000:.1f} s)"
    )

    print("\nchecking every answer against plain generation from scratch:")
    computed = sum(log.tokens_computed for log in logs if log.phase == "prefill")
    t0 = time.perf_counter()
    same = 0
    for uid, (q, ids) in prompts.items():
        ref = M.generate(model, ids.unsqueeze(0), args.max_tokens, eos_id=eos).tokens
        got = logs.generated[uid]
        same += got == ref
        if not tiny or uid < 3 or got != ref:
            print(
                f"  req {uid}: reused {hits[uid]:>3} of {len(ids)} prompt tokens | identical: {got == ref}"
            )
            print(f"     Q: {q}\n     A: {tok.decode(got, skip_special_tokens=True)!r}")
    ref_time = time.perf_counter() - t0
    print(
        f"identical answers: {same} of {len(prompts)}"
        f" (plain generation, one request at a time: {ref_time:.1f} s)"
    )
    total = sum(len(p) for _, p in prompts.values())
    print(
        f"prefill tokens computed: {computed:,} of {total:,} ({1 - computed / total:.0%} saved by the"
        f" {args.cache} cache); requests with a cache hit: {sum(h > 0 for h in hits.values())}"
    )
    if not tiny:
        print(
            "\nQuestion: request 1 reused request 0's prefix while request 0 was still decoding."
            "\nWhat stops CacheManager from evicting or overwriting those slots meanwhile?"
        )


if __name__ == "__main__":
    main()
