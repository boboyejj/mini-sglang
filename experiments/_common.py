"""Shared helpers that let mini-sglang's scheduler code run on a CPU-only machine.

The real scheduler copies data through *pinned* CPU memory (``pin_memory``), which needs
an NVIDIA driver. Importing this module replaces those calls with plain CPU tensors, so
the bookkeeping logic (cache, admission, chunking) runs unchanged. No model is executed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ---- CPU shim: make pin_memory a no-op -------------------------------------------------
torch.Tensor.pin_memory = lambda self, *a, **k: self  # type: ignore[method-assign]
_orig_empty, _orig_tensor = torch.empty, torch.tensor


def _empty(*args, pin_memory: bool = False, **kwargs):
    return _orig_empty(*args, **kwargs)


def _tensor(*args, pin_memory: bool = False, **kwargs):
    return _orig_tensor(*args, **kwargs)


torch.empty = _empty  # type: ignore[assignment]
torch.tensor = _tensor  # type: ignore[assignment]

import minisgl.core as core  # noqa: E402
from minisgl.core import SamplingParams  # noqa: E402
from minisgl.message import UserMsg  # noqa: E402
from minisgl.scheduler.cache import CacheManager  # noqa: E402
from minisgl.scheduler.decode import DecodeManager  # noqa: E402
from minisgl.scheduler.prefill import PrefillManager  # noqa: E402
from minisgl.scheduler.table import TableManager  # noqa: E402


@dataclass
class Managers:
    cache: CacheManager
    table: TableManager
    decode: DecodeManager
    prefill: PrefillManager


def make_managers(
    num_pages: int = 64,
    page_size: int = 1,
    max_running_req: int = 8,
    max_seq_len: int = 256,
    cache_type: str = "radix",
) -> Managers:
    """Build the scheduler's managers exactly as Scheduler.__init__ does, but on CPU."""
    core._GLOBAL_CTX = None  # allow calling this more than once per process
    core.set_global_ctx(core.Context(page_size=page_size))
    page_table = torch.zeros((max_running_req + 1, max_seq_len), dtype=torch.int32)
    cache = CacheManager(num_pages, page_size, page_table, cache_type)
    table = TableManager(max_running_req, page_table)
    decode = DecodeManager(page_size)
    prefill = PrefillManager(cache, table, decode)
    return Managers(cache, table, decode, prefill)


def user_msg(uid: int, input_ids, max_tokens: int = 4) -> UserMsg:
    ids = torch.as_tensor(input_ids, dtype=torch.int32)
    return UserMsg(uid=uid, input_ids=ids, sampling_params=SamplingParams(max_tokens=max_tokens))


def load_tokenizer():
    """Qwen3-0.6B's tokenizer: only its small text files, no model weights.

    Returns None if they cannot be downloaded (then experiments use fake token IDs).
    """
    try:
        import os

        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        files = ["tokenizer_config.json", "vocab.json", "merges.txt", "config.json"]
        paths = [hf_hub_download("Qwen/Qwen3-0.6B", f) for f in files]
        return AutoTokenizer.from_pretrained(os.path.dirname(paths[0]))
    except Exception as e:  # network blocked, etc.
        print(f"(tokenizer unavailable: {type(e).__name__}; falling back to fake token IDs)")
        return None


def print_tree(cache: CacheManager, decode=None) -> None:
    """Print the radix tree and the memory counters."""

    def fmt(t: torch.Tensor) -> str:
        v = t.tolist()
        return str(v) if len(v) <= 8 else f"[{v[0]}, {v[1]}, ... {v[-1]}] ({len(v)} tokens)"

    def walk(node, depth):
        for child in node.children.values():
            print("    " * depth + f"- tokens={fmt(child._key)} ref={child.ref_count}")
            walk(child, depth + 1)

    print("  radix tree:")
    walk(cache.prefix_cache.root_node, 1)
    info = cache.prefix_cache.size_info
    print(
        f"  free={len(cache.free_slots) * cache.page_size} evictable={info.evictable_size}"
        f" protected={info.protected_size} available={cache.available_size}"
    )


# ---- A CPU-only copy of Scheduler.normal_loop ------------------------------------------
@dataclass
class StepLog:
    step: int
    phase: str
    uids: list
    chunked_uids: list
    tokens_computed: int
    running: int
    kv_in_use: int


def run_sim(
    m: Managers,
    arrivals: dict,
    prefill_budget: int = 64,
    eos_after: dict | None = None,
    eos_id: int = 0,
    verbose: bool = True,
    max_steps: int = 1000,
):
    """Drive the real PrefillManager / DecodeManager / CacheManager with fake tokens.

    arrivals:  {step: [UserMsg, ...]} - requests that arrive at the start of that step.
    eos_after: {uid: n} - that request "samples" EOS as its n-th output token.
    Returns (step logs, {uid: prefix-cache hit length at admission}).
    """
    from minisgl.scheduler.prefill import ChunkedReq

    eos_after = eos_after or {}
    logs, hits, outputs = [], {}, {}
    step = 0
    while step < max_steps:
        for msg in arrivals.get(step, []):
            m.prefill.add_one_req(msg)
        if not (m.prefill.runnable or m.decode.runnable):
            if any(s > step for s in arrivals):
                step += 1
                continue
            break

        # 1. schedule (prefill first), 2. allocate pages  -- Scheduler._schedule_next_batch
        batch = m.prefill.schedule_next_batch(prefill_budget) or m.decode.schedule_next_batch()
        if batch is None:
            raise RuntimeError(f"step {step}: requests are waiting but none can be admitted")
        m.cache.allocate_paged(batch.reqs)
        for r in batch.reqs:
            if batch.is_prefill and r.uid not in hits:
                hits[r.uid] = r.cached_len
        tokens = sum(r.extend_len for r in batch.reqs)

        # 3. "forward pass"                               -- Engine.forward_batch
        for r in batch.reqs:
            r.complete_one()
        m.decode.filter_reqs(batch.reqs)

        # 4. process results                              -- Scheduler._process_last_data
        with m.cache.lazy_free_region():
            for r in batch.reqs:
                if isinstance(r, ChunkedReq):
                    continue
                n = outputs[r.uid] = outputs.get(r.uid, 0) + 1
                token = eos_id if eos_after.get(r.uid) == n else 1000 + n
                r.append_host(torch.tensor([token], dtype=torch.int32))
                finished = (not r.can_decode) or token == eos_id
                if finished:
                    m.decode.remove_req(r)
                    m.table.free(r.table_idx)
                    m.cache.cache_req(r, finished=True)
                elif batch.is_prefill:
                    m.cache.cache_req(r, finished=False)

        kv_in_use = m.cache.num_pages * m.cache.page_size - m.cache.available_size
        log = StepLog(
            step,
            batch.phase,
            [r.uid for r in batch.reqs],
            [r.uid for r in batch.reqs if isinstance(r, ChunkedReq)],
            tokens,
            len(m.decode.running_reqs),
            kv_in_use,
        )
        logs.append(log)
        if verbose:
            chunk = f" chunked={log.chunked_uids}" if log.chunked_uids else ""
            print(
                f"step {step:>3} {batch.phase:<7} reqs={log.uids}{chunk} "
                f"tokens_computed={tokens:<4} running={log.running} kv_busy={kv_in_use}"
            )
        step += 1
    m.cache.check_integrity()  # free + cached must equal the pool size
    return logs, hits
