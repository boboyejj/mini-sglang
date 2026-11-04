"""Experiment 2: watch the radix prefix cache share, lock and evict KV slots.

Run:  python experiments/02_radix_prefix_sharing.py
"""

import minisgl.core as core
import torch
from minisgl.scheduler.cache import CacheManager

PAGE_SIZE = 1
NUM_PAGES = 16  # a tiny KV pool: 16 token slots


def show_tree(cm: CacheManager) -> None:
    def walk(node, depth):
        for child in node.children.values():
            print(
                "    " * depth
                + f"- tokens={child._key.tolist()} slots={child.value.tolist()} ref={child.ref_count}"
            )
            walk(child, depth + 1)

    print("  radix tree:")
    walk(cm.prefix_cache.root_node, 1)
    info = cm.prefix_cache.size_info
    print(
        f"  free slots={len(cm.free_slots)}  evictable={info.evictable_size}"
        f"  protected={info.protected_size}  available={cm.available_size}"
    )


def main() -> None:
    core.set_global_ctx(core.Context(page_size=PAGE_SIZE))
    page_table = torch.empty((1,))
    cm = CacheManager(NUM_PAGES, PAGE_SIZE, page_table, type="radix")
    tree = cm.prefix_cache

    print("1) Request A computes 6 tokens and is cached")
    a_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int32)
    a_slots = cm._allocate(len(a_ids))
    tree.insert_prefix(a_ids, a_slots)
    show_tree(cm)

    print("\n2) Request B shares the first 4 tokens: match splits A's node")
    b_ids = torch.tensor([1, 2, 3, 4, 9, 9], dtype=torch.int32)
    handle = tree.match_prefix(b_ids).cuda_handle
    print(
        f"  matched {handle.cached_len} tokens -> reused slots {handle.get_matched_indices().tolist()}"
    )
    show_tree(cm)

    print("\n3) B locks its prefix while running: those tokens become protected")
    cm.lock(handle)
    show_tree(cm)

    print("\n4) Ask for 12 slots (all that is available): only unlocked leaves are evicted")
    got = cm._allocate(12)  # try 13: eviction fails, the locked prefix is protected
    print(f"  allocated {len(got)} slots")
    show_tree(cm)

    print("\n5) B finishes and unlocks: its prefix is evictable again")
    cm.unlock(handle)
    show_tree(cm)


if __name__ == "__main__":
    main()
