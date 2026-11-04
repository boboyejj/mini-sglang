"""Experiment 4: which cached prefix gets evicted when memory runs out?

Concept: when the KV pool is full, the radix cache frees unlocked leaf nodes, least
recently used first (each node keeps a timestamp of its last match).

Run:  python experiments/04_eviction_lru.py
Try:  touch conversation B instead of A, or lock A before allocating (cm.lock(handle)).
"""

import time

import _common as C
import torch

CONVERSATIONS = {
    "A": [1, 2, 3, 4],
    "B": [5, 6, 7, 8],
    "C": [9, 10, 11, 12],
}


def main() -> None:
    m = C.make_managers(num_pages=12)  # room for exactly the 3 conversations
    cm, tree = m.cache, m.cache.prefix_cache

    print("1) Cache three conversations, oldest first: A, B, C")
    for name, ids in CONVERSATIONS.items():
        slots = cm._allocate(len(ids))
        tree.insert_prefix(torch.tensor(ids, dtype=torch.int32), slots)
        time.sleep(0.01)  # make timestamps clearly different
    C.print_tree(cm)

    print("\n2) A new request matches conversation A again (A becomes most recently used)")
    tree.match_prefix(torch.tensor(CONVERSATIONS["A"] + [99], dtype=torch.int32))

    print("\n3) Someone needs 4 new slots, but the pool is full -> evict one conversation")
    cm._allocate(4)
    remaining = [child._key.tolist() for child in tree.root_node.children.values()]
    for name, ids in CONVERSATIONS.items():
        print(f"  {name}: {'kept' if ids in remaining else 'EVICTED'}")
    C.print_tree(cm)
    print("\nQuestion: B was evicted, not A, even though A was cached first. Why?")


if __name__ == "__main__":
    main()
