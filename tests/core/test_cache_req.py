"""Regression test for CacheManager.cache_req prefix deduplication (sgl-project/mini-sglang#168).

Two requests share a prefix and are both prefilled before either is in the radix cache, so
each computes its own copy of the shared part. When the second is inserted, cache_req frees
its duplicate slots. A request that is still running must then point at the cache's copy;
otherwise it keeps reading slots that another request can be given and overwrite.
"""

from __future__ import annotations

import minisgl.core as core
import pytest
import torch
from minisgl.core import Req, SamplingParams
from minisgl.scheduler.cache import CacheManager

NUM_PAGES = 16
SHARED = [1, 2, 3, 4]


@pytest.fixture(autouse=True)
def reset_global_ctx():
    old_ctx = core._GLOBAL_CTX
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = old_ctx


def _prefill_two_requests_sharing_a_prefix():
    core.set_global_ctx(core.Context(page_size=1))
    page_table = torch.zeros((3, 32), dtype=torch.int32)
    cm = CacheManager(NUM_PAGES, 1, page_table, type="radix")
    reqs = []
    for row, tail in enumerate([[5, 6], [7, 8]]):
        ids = torch.tensor(SHARED + tail, dtype=torch.int32)
        handle = cm.prefix_cache.match_prefix(ids[:-1]).cuda_handle  # empty cache: no match
        cm.lock(handle)
        reqs.append(
            Req(
                input_ids=ids,
                table_idx=row,
                cached_len=0,
                output_len=4,
                uid=row,
                sampling_params=SamplingParams(),
                cache_handle=handle,
            )
        )
    for r in reqs:  # what allocate_paged does for page_size 1 (it needs CUDA pinned memory)
        page_table[r.table_idx, : r.device_len] = cm._page_to_token(cm._allocate(r.device_len))
    for r in reqs:
        r.complete_one()  # prefill done: cached_len = 6
    return cm, page_table, reqs


def test_running_request_never_points_at_freed_slots():
    cm, page_table, (a, b) = _prefill_two_requests_sharing_a_prefix()
    b_shared_before = page_table[b.table_idx, : len(SHARED)].clone()
    cm.cache_req(a, finished=False)
    cm.cache_req(b, finished=False)  # SHARED is already cached by a: b's copy is a duplicate

    free = set(cm.free_slots.tolist())
    b_slots = page_table[b.table_idx, : b.cached_len].tolist()
    assert not free.intersection(b_slots), "running request still points at freed slots"
    # b now shares a's (the cache's) slots for the prefix, and its duplicates were freed
    assert (
        page_table[b.table_idx, : len(SHARED)].tolist()
        == page_table[a.table_idx, : len(SHARED)].tolist()
    )
    assert set(b_shared_before.tolist()) <= free


def test_slots_are_accounted_for_after_both_finish():
    cm, _, (a, b) = _prefill_two_requests_sharing_a_prefix()
    cm.cache_req(a, finished=False)
    cm.cache_req(b, finished=False)
    cm.cache_req(a, finished=True)
    cm.cache_req(b, finished=True)
    cm.check_integrity()  # free slots + cached tokens == pool size: nothing leaked or double-freed
    assert len(set(cm.free_slots.tolist())) == len(cm.free_slots), "a slot was freed twice"
