"""Experiment 13: a KV-cache bookkeeping bug in CacheManager.cache_req, and a fix.

Found while serving the tiny model through experiment 12 with a small KV pool. When two
requests that share a prefix are prefilled before either is in the radix tree, each computes
its own copy of the shared part. When the second is inserted, cache_req frees its duplicate
slots "to avoid a memory leak", but the request keeps decoding and its page_table still
points at them. With a large pool the freed slots are not reused in time and answers stay
correct (a latent bug); with a small pool another request overwrites them and the answer
changes. The fix points the request at the tree's copy before freeing its duplicate.

This script runs the 44-request tiny-model workload with the original and the fixed
cache_req (patched in at runtime; mini-sglang's source is not modified) at three pool sizes,
and counts identical answers and forward passes that read K/V from freed slots.
Run: python experiments/13_cache_req_bug.py   (~1 min)
"""

import argparse
import importlib.util
import sys

sys.path.insert(0, "experiments")  # noqa: E402
import _common as C
import _model as M
from minisgl.scheduler.cache import CacheManager

spec = importlib.util.spec_from_file_location("e12", "experiments/12_real_model_scheduler.py")
e12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e12)
from transformers import AutoModelForCausalLM, AutoTokenizer


def cache_req_fixed(self, req, *, finished):
    insert_ids = req.input_ids[: req.cached_len]
    page_indices = self.page_table[req.table_idx, : req.cached_len]
    old_handle = req.cache_handle
    cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
    self.unlock(old_handle)
    duplicate = page_indices[old_handle.cached_len : cached_len].clone()
    if cached_len > old_handle.cached_len:  # FIX: point the request at the tree's copy first
        tree = new_handle.get_matched_indices()
        self.page_table[req.table_idx, old_handle.cached_len : cached_len] = tree[
            old_handle.cached_len : cached_len
        ]
    self._free(duplicate)
    if finished:
        self._free(page_indices[new_handle.cached_len :])
    else:
        req.cache_handle = new_handle
        self.lock(new_handle)


tok = AutoTokenizer.from_pretrained("training/tiny-qwen3")
model = AutoModelForCausalLM.from_pretrained("training/tiny-qwen3").eval()
args = argparse.Namespace(model="tiny", max_tokens=64)
prompts, _ = e12.load_workload(args, tok)
refs = {
    u: M.generate(model, ids.unsqueeze(0), 64, eos_id=tok.eos_token_id).tokens
    for u, (_, ids) in prompts.items()
}
original = CacheManager.cache_req
for label, fn in [("original cache_req", original), ("fixed cache_req   ", cache_req_fixed)]:
    CacheManager.cache_req = fn
    for slots in (4096, 1024, 768):
        m = C.make_managers(
            num_pages=slots, max_running_req=16, max_seq_len=512, cache_type="radix"
        )
        eng = e12.PagedKVEngine(model, m, num_slots=slots)
        freed_reads = []

        def checked(batch, m=m, eng=eng, freed_reads=freed_reads):
            free = set(m.cache.free_slots.tolist())
            for r in batch.reqs:
                if free.intersection(m.table.page_table[r.table_idx, : r.cached_len].tolist()):
                    freed_reads.append(r.uid)
            return eng(batch)

        _, arrivals = e12.load_workload(args, tok)
        logs, hits = C.run_sim(
            m, arrivals, prefill_budget=128, eos_id=tok.eos_token_id, engine=checked, verbose=False
        )
        same = sum(logs.generated[u] == refs[u] for u in prompts)
        print(
            f"{label} slots={slots:>4}: identical {same}/44 | passes reading freed slots: {len(freed_reads)}"
        )
CacheManager.cache_req = original
