"""Experiment 1: follow one request's counters from arrival to finish.

Concept: every Req is tracked by cached_len (K/V already stored), device_len (tokens on
the device) and max_device_len (prompt + output limit). See core.py and study guide Module 0.

Run:  python experiments/01_request_counters.py
Try:  change PROMPT_LEN / MAX_TOKENS, or set PAGE_SIZE = 4 and watch which slots get used.
"""

import _common as C

PROMPT_LEN = 10
MAX_TOKENS = 3
PAGE_SIZE = 1


def show(step: str, req, m: C.Managers, allocated: bool) -> None:
    row = m.table.page_table[req.table_idx]
    print(
        f"{step:<22} cached_len={req.cached_len:<3} device_len={req.device_len:<3} "
        f"extend_len={req.extend_len:<3} remain_len={req.remain_len:<3} can_decode={req.can_decode}"
    )
    print(f"{'':<22} K/V stored in slots   {row[: req.cached_len].tolist()}")
    if allocated:
        print(f"{'':<22} slots to write next   {row[req.cached_len : req.device_len].tolist()}")


def main() -> None:
    m = C.make_managers(num_pages=32, page_size=PAGE_SIZE)
    m.prefill.add_one_req(C.user_msg(0, range(1, PROMPT_LEN + 1), max_tokens=MAX_TOKENS))

    batch = m.prefill.schedule_next_batch(prefill_budget=64)
    req = batch.reqs[0]
    m.cache.allocate_paged(batch.reqs)  # done in Scheduler._prepare_batch
    show("admitted (prefill)", req, m, allocated=True)

    step = 0
    while True:
        req.complete_one()  # done in Engine.forward_batch right after the forward pass
        req.append_host(C.torch.tensor([1000 + step], dtype=C.torch.int32))  # fake sampled token
        if not req.can_decode:
            show(f"after forward #{step}", req, m, allocated=False)
            print("\nrequest finished: remain_len hit 0 (max_tokens reached)")
            break
        m.cache.allocate_paged([req])  # next decode step needs one more slot
        show(f"after forward #{step}", req, m, allocated=True)
        step += 1

    print(f"\ntokens generated: {len(req.input_ids) - PROMPT_LEN}  (MAX_TOKENS = {MAX_TOKENS})")
    print("Question: why does the last forward pass not need a new KV slot?")


if __name__ == "__main__":
    main()
