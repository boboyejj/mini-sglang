# Mini-SGLang Study Guide

Nov 4, 2026 · Boboye

> Exported from the live doc: https://claude.ai/artifact/EGxfmastshdsorZ2BLt3rB (the doc is the
> editable version; this file is a snapshot).

## How to use this guide

Work through four modules in order — scheduler, admission, KV cache, engine — reading the named files, then do the CPU exercises and check yourself with the questions at the end. The core path is about 1,000 lines of Python.

- **Goal:** be able to trace one request from arrival to its last token, and explain prefix caching, chunked prefill and overlap scheduling in your own words.
- **Prerequisites:** Python and basic PyTorch tensors; a rough idea of a transformer's attention and what a KV cache stores.
- **Code:** [boboyejj/mini-sglang](https://github.com/boboyejj/mini-sglang), branch `learn/playground` — a free sandbox for notes and experiments. All paths below are relative to `python/minisgl/`.
- **Setup:** `cd /home/user/mini-sglang && source .venv/bin/activate` (Python 3.12, project installed with dev extras).
- **No GPU in the cloud session:** you can read everything and run the CPU-only parts (cache bookkeeping, scheduler logic). Serving a model needs an NVIDIA GPU.

| Module | Files | Lines | Time |
| --- | --- | --- | --- |
| 0. Data structures | `core.py`, `scheduler/table.py` | ~150 | 30 min |
| 1. Scheduler loop | `scheduler/scheduler.py`, `scheduler/decode.py` | ~300 | 1 h |
| 2. Admission + chunked prefill | `scheduler/prefill.py` | ~160 | 45 min |
| 3. KV cache | `scheduler/cache.py`, `kvcache/radix_cache.py`, `kvcache/mha_pool.py` | ~450 | 1.5 h |
| 4. Engine | `engine/engine.py`, `engine/graph.py`, `engine/sample.py` | ~480 | 1 h |

Times are rough estimates for a first careful read.

## Big picture

Mini-sglang runs as separate processes, and a request makes one loop through them; this guide studies the scheduler and engine box in detail.

```text
A request makes one loop through four kinds of process
(control messages over ZeroMQ; tensors between GPUs over NCCL)

 +-------------------------+      +-------------------+      +==============================+
 | API Server              | ---> | Tokenizer         | ---> | Scheduler, rank 0            |
 | HTTP, OpenAI-compatible |      | text to token IDs |      | batches requests, runs Engine|
 +-------------------------+      +-------------------+      +==============================+
              ^                                                 |                     |
              | streamed text            next token per request |                     | broadcast
              |                                                 v                     v (multi-GPU)
              |                   +-------------------+      +------------------------------+
              +------------------ | Detokenizer       |      | Schedulers, ranks 1..N       |
                                  | token IDs to text |      | same steps on their own GPU  |
                                  +-------------------+      +------------------------------+
```

The API server sends the prompt to the tokenizer, which passes token IDs to the rank 0 scheduler. With several GPUs, rank 0 broadcasts each request so every rank runs the same batches. Each generated token goes to the detokenizer and streams back to the user. Start-up code: `server/launch.py`; full description: `docs/structures.md`.

## Module 0: Core data structures

Everything the scheduler does is bookkeeping on three counters per request and two GPU tables. Learn these first.

**Read:** `core.py` (`Req`, `Batch`, `Context`), `scheduler/table.py`.

**`Req` counters** — example: a 10-token prompt allowed 5 output tokens.

| Field | Meaning | At arrival | After prefill | After 1st decode |
| --- | --- | --- | --- | --- |
| `cached_len` | tokens whose K/V is stored | 0 | 10 | 11 |
| `device_len` | tokens on the GPU | 10 | 11 | 12 |
| `max_device_len` | prompt + output limit | 15 | 15 | 15 |
| `extend_len` | tokens to compute this step | 10 | 1 | 1 |

- `complete_one()` sets `cached_len = device_len`, then adds 1 to `device_len`. It runs on the CPU right after the forward pass is launched.
- `can_decode` is `remain_len > 0`. A request stops when this is false or it samples EOS.
- If the prefix cache matched part of the prompt, `cached_len` starts above 0 and prefill computes only the rest.

**Two GPU tables**, both shaped `[max_running_req + 1, max_seq_len]`, one row per running request:

- `token_pool[row, pos]` = token ID at that position.
- `page_table[row, pos]` = slot index where that token's K/V lives in the KV pool.
- A request's row is its `table_idx`, handed out by `TableManager`. The extra last row belongs to a dummy request used to pad batches.

**`Batch`** = a list of `Req` plus a phase (`prefill` or `decode`). The scheduler fills `input_ids`, `positions` and `out_loc` (where to write this step's K/V); the attention backend fills `attn_metadata`.

**`Context`** is a process-wide global (`get_global_ctx()`). During a forward pass it holds the current batch, the KV pool, the page table and the attention backend, so model layers read them without passing arguments.

## Module 1: The scheduler loop

The scheduler runs one batch per loop iteration, always preferring prefill, and processes each batch's results one iteration late so the CPU work hides behind GPU work.

**Read:** `scheduler/scheduler.py` (all of it), `scheduler/decode.py`.

**One iteration of `overlap_loop`:**

1. `receive_msg` — new `UserMsg`s go to `prefill_manager.add_one_req` (just a waiting list; nothing is reserved yet). Blocks only when there is no work at all.
2. `_schedule_next_batch` — a prefill batch if anything is waiting, otherwise one decode step for every running request.
3. `_prepare_batch` — pad for CUDA graphs, allocate KV pages, build `positions`, compute `out_loc = page_table[row, positions]`, prepare attention metadata.
4. `_forward` — gather `input_ids` from `token_pool`, call `engine.forward_batch`, write each sampled token back into `token_pool[row, device_len]` on the GPU.
5. `_process_last_data` — for the **previous** batch: wait for its tokens to reach the CPU, send `DetokenizeMsg`s, finish requests on EOS or length, cache prompt prefixes after prefill.

**Why overlap works:** `complete_one()` advances the counters on the CPU as soon as a batch is launched, and the sampled token never leaves the GPU for the next step. So the CPU can plan batch N+1 before it has seen batch N's tokens. `normal_loop` (set `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`) does the same steps without the overlap — compare the two to see exactly what changes.

**Decode side:** `DecodeManager.running_reqs` is a set of requests that can still decode. Every decode batch includes all of them, sorted by `uid` so every tensor-parallel rank builds the same order (commit #113).

**Things to notice:**

- `finished_reqs` guards against freeing a request twice: with overlap, a request can finish in batch N while already scheduled in batch N+1.
- Freed KV slots are collected in `lazy_free_region()` and returned in one `torch.cat`, not one at a time.
- Prefill-first is a fixed policy (see the `TODO` at `_schedule_next_batch`). Long prefills delay decode steps. Chunked prefill (Module 2) caps each step's size, but decodes still wait for every chunk (experiment 5).

## Module 2: Admission and chunked prefill

A waiting request is admitted only if a table row is free and its whole future — uncached prompt plus every output token — fits in memory, so running requests never have to be preempted.

**Read:** `scheduler/prefill.py`.

**`PrefillAdder._try_allocate_one`, step by step:**

1. Stop if `table_manager` has no free row (limit: `max_running_req`).
2. `cache_manager.match_req` looks up the prompt in the prefix cache. It matches `input_ids[:len-1]`: the last token is always recomputed, because its logits are needed to sample the first output token.
3. Memory check: `extend_len + output_len + reserved_size <= available_size`. `reserved_size` starts at `decode_manager.inflight_tokens`, the memory running requests will still need.
4. Lock the matched prefix so it cannot be evicted, then check memory again (locking makes those tokens non-evictable, which shrinks `available_size`).
5. Take a table row and copy the matched token IDs and KV slot indices into it.

**Chunked prefill** (`_add_one_req`): each prefill batch has a token budget, `max_extend_tokens`. A prompt that doesn't fit is split: the first part becomes a `ChunkedReq`, which is never sampled (`can_decode` is always `False`). It goes back to the front of `pending_list` and continues in the next prefill batch from its new `cached_len`. This caps how many tokens one step computes, and so its activation memory. Running decodes still wait: prefill comes first, so every chunk runs before the next decode batch (experiment 5).

**Things to notice:**

- Admission is first-come, first-served: the loop `break`s at the first request that doesn't fit, so a big request blocks smaller ones behind it.
- The memory estimate assumes every request uses its full `max_tokens`. That is safe but conservative — fewer requests run at once than memory could actually hold.
- Pages are not allocated here, only reserved in the estimate. Real allocation happens in `_prepare_batch` (Module 3).

## Module 3: The KV cache

The KV cache has three layers: one big GPU tensor that stores K and V, a free list of pages, and a radix tree that remembers which token sequences already have K/V so other requests can reuse them.

**Read:** `kvcache/mha_pool.py`, `scheduler/cache.py`, `kvcache/radix_cache.py` (compare with `kvcache/naive_cache.py`, which caches nothing).

| Layer | Class | Job |
| --- | --- | --- |
| Storage | `MHAKVCache` | Tensor `[2, layers, pages, page_size, kv_heads, head_dim]`; `store_kv` writes K/V at `out_loc` |
| Allocation | `CacheManager` | `free_slots` list of pages; `allocate_paged`, `_free`, eviction when short |
| Reuse | `RadixPrefixCache` | Token sequences → KV slot indices; match, insert, lock, evict |

**Sizing.** At startup the engine loads the model, measures free memory, and turns `memory_ratio` of it into pages (`engine/engine.py`, `_determine_num_pages`). Bytes per page = 2 × head_dim × kv_heads × page_size × dtype bytes × layers.

**Allocation.** `allocate_paged` finds, for each request, the pages between `cached_len` and `device_len` it does not yet own, takes them from `free_slots` (evicting from the radix tree if short), and writes their slot indices into `page_table`. `available_size` = free pages + evictable cached tokens.

**The radix tree** (`RadixTreeNode`):

- Each node holds a run of token IDs (key) and their KV slot indices (value). Children are indexed by their first token (or first page of tokens when `page_size > 1`).
- `match_prefix` walks down from the root. If a match ends inside a node, `split_at` cuts the node in two so the match ends on a node boundary.
- `insert_prefix` walks the same way and adds one new leaf for the unmatched remainder.
- `ref_count` is the lock. Locking a handle adds 1 to every node from it up to the root, moving those tokens from `evictable_size` to `protected_size`.
- `evict` pops leaves with `ref_count == 0`, oldest `timestamp` first (LRU). A parent that becomes an unlocked leaf joins the queue.
- Everything is page-aligned: the unaligned tail of a sequence is never inserted, and is freed when the request finishes.

**When is a request cached?** `cache_req` runs twice in a request's life: after its prefill (`finished=False`, so others can reuse the prompt while it decodes) and when it finishes (`finished=True`). Read the comment block in `cache_req` carefully: if two requests computed the same prefix at the same time, the second one's duplicate slots are freed so they do not leak.

**Example.** Request A = "You are helpful. Hi" is cached. Request B = "You are helpful. Bye" arrives: `match_prefix` splits A's node after "You are helpful.", returns that node, and B only computes "Bye".

## Module 4: The engine

The engine owns everything on one GPU — model, KV pool, page table, attention backend, sampler, CUDA graphs — and its `forward_batch` turns a prepared batch into one sampled token per request.

**Read:** `engine/engine.py`, `engine/graph.py`, `engine/sample.py`; then skim `layers/attention.py` and one model, e.g. `models/qwen3.py`.

**Startup order in `Engine.__init__`:**

1. Set up tensor-parallel info and the global `Context`.
2. Build the model on the `meta` device (no memory), then load real weights with `load_state_dict`.
3. Size and create the KV pool from leftover memory (+1 dummy page).
4. Create the page table (+1 dummy row), pointing the dummy row at the dummy page.
5. Pick the attention backend: `trtllm` on Blackwell (sm100), FlashAttention + FlashInfer on Hopper (sm90), FlashInfer otherwise.
6. Capture CUDA graphs for decode.

**`forward_batch`:**

1. `ctx.forward_batch(batch)` makes the batch visible to every layer through the global context.
2. Decode batches up to `max_graph_bs` replay a captured CUDA graph; prefill and larger batches call `model.forward()` normally.
3. Each attention layer stores this step's K/V at `batch.out_loc`, then attends over the request's slots listed in `page_table`.
4. `complete_one()` advances every request's counters.
5. The sampler picks tokens: `argmax` when every request is greedy, FlashInfer top-k/top-p sampling otherwise.
6. Tokens are copied to the CPU without blocking; an event records when the copy is done.

**CUDA graphs.** Launching hundreds of small kernels per decode step costs real CPU time. A CUDA graph records the whole decode forward once per batch size (1, 2, 4, then steps of 8 up to the max) and replays it with one launch. Batches are padded up to the next captured size with the dummy request, which is why the dummy row and page exist.

**Things to notice:**

- Prefill never uses CUDA graphs: its token count varies too much to capture.
- The engine knows nothing about requests' lifetimes; the scheduler owns all bookkeeping. This split is the main design idea to take away.

## Hands-on exercises (CPU only)

The `experiments/` folder on `learn/playground` holds six beginner scripts that run mini-sglang's real scheduler and KV-cache code on a CPU, with fake tokens in place of the model. All six are verified to run in the cloud session; `experiments/README.md` explains the CPU shim.

| # | Script | What you see | Module |
| --- | --- | --- | --- |
| 1 | `01_request_counters.py` | `cached_len` / `device_len` and KV slots from prefill to finish | 0 |
| 2 | `02_radix_prefix_sharing.py` | Node splitting, locking and eviction in the radix tree | 3 |
| 3 | `03_prefix_sharing_chat.py` | A shared system prompt cuts prefill work by 60% (real Qwen3 tokenizer) | 3 |
| 4 | `04_eviction_lru.py` | Least-recently-used eviction | 3 |
| 5 | `05_chunked_prefill.py` | A 64-token prompt split into 16-token chunks | 1, 2 |
| 6 | `06_admission_memory.py` | Larger `max_tokens` means fewer requests run at once | 2 |

Each script ends with a question, and its docstring suggests a variation to try. Run them from the repo root with the virtual environment active:

```bash
python experiments/01_request_counters.py
```

Then try these extensions. E1 and E2 use experiment 2, which caches request A (6 tokens), matches request B against it (4 shared tokens), locks B's prefix, fills the pool, then unlocks.

- [ ] **E1 — Read the output.** For each step, explain why `free`, `evictable`, `protected` and `available` change the way they do.
- [ ] **E2 — Protection.** Change `_allocate(12)` to `_allocate(13)`. Predict the error before running it, then explain which line in `radix_cache.py` raised it.
- [ ] **E3 — LRU order.** Insert three unrelated sequences, call `match_prefix` on the first one again, then allocate enough to force one eviction. Which sequence is evicted, and why? (Hint: `timestamp`.)
- [ ] **E4 — Paging.** Set `PAGE_SIZE = 4` and `NUM_PAGES = 4`. `_allocate` now counts pages, so build A's 6 tokens on `cm._page_to_token(cm._allocate(2))`, which gives 8 slots. How many of A's tokens land in the tree, and who must free the rest?
- [ ] **E5 — Counter trace.** Without running anything, write the `cached_len` / `device_len` values for a 10-token prompt with `max_tokens=3` and a prefix-cache hit of 4 tokens, through every step until it finishes.
- [ ] **E6 — Chunking by hand.** With `max_extend_tokens = 8`, trace `PrefillManager.schedule_next_batch` for prompts of 5 and 20 tokens arriving together. How many prefill batches run, and which requests are `ChunkedReq` in each?
- [ ] **E7 — Write a test.** Add a pytest in `tests/core/` that checks `CacheManager.cache_req` frees duplicate slots when two requests insert the same prefix. Run it with `python -m pytest --no-cov tests/core`.

Needs a GPU (save for later): serve `Qwen/Qwen3-0.6B` with `python -m minisgl --model Qwen/Qwen3-0.6B`, compare throughput with and without `MINISGL_DISABLE_OVERLAP_SCHEDULING=1`, and send two requests sharing a long system prompt to watch the prefix cache hit.

## Self-check questions

Try to answer each question before reading its answer; if you can answer all ten, you understand the core of mini-sglang.

| # | Question | Answer |
| --- | --- | --- |
| 1 | Why does `match_req` match only `input_ids[:len-1]`? | The last prompt token must be recomputed: its logits are what the first output token is sampled from. |
| 2 | What does `out_loc` hold, and where does it come from? | The KV slot each new token's K/V is written to; `page_table[table_idx, positions]`, after `allocate_paged` fills the table. |
| 3 | Why can the scheduler plan batch N+1 before it has seen batch N's tokens? | `complete_one()` updates counters on the CPU at launch, and sampled tokens are written to `token_pool` on the GPU, where the next step reads them. |
| 4 | Why is a request freed only if it is not in `finished_reqs`? | With overlap, a request finishing in batch N may already be in batch N+1; its second finish must not free its resources again. |
| 5 | What stops a running request's prefix from being evicted? | Its handle is locked: `ref_count > 0` on every node up to the root, so those tokens count as protected, not evictable. |
| 6 | Which nodes can `evict` remove, and in what order? | Leaves with `ref_count == 0`, oldest `timestamp` first; parents left as unlocked leaves join the queue. |
| 7 | What is a `ChunkedReq`, and why is it never sampled? | The first part of a prompt larger than the prefill token budget; its logits are mid-prompt, so sampling from them would be meaningless. |
| 8 | Why does admission count `output_len` and `inflight_tokens`? | It reserves memory for every request's full future, so running requests never run out of KV slots and nothing needs preempting. |
| 9 | Why do the page table and KV pool each have one extra entry? | They belong to a dummy request used to pad decode batches to a captured CUDA-graph size. |
| 10 | Why do decode batches use CUDA graphs but prefill does not? | Decode batches have one token per request and a few fixed sizes, so they can be captured; prefill token counts vary too much. |

## Glossary and next topics

| Term | Meaning |
| --- | --- |
| Prefill | Computing K/V for a prompt's tokens all at once; produces the first output token. |
| Decode | Generating one token per request per step, reusing stored K/V. |
| KV cache | Stored key and value vectors for past tokens, so attention doesn't recompute them. |
| Page | A fixed-size group of `page_size` KV slots; the unit of allocation. |
| Prefix cache / radix cache | A tree of token sequences whose K/V is still in memory, shared across requests. |
| Chunked prefill | Splitting a long prompt across several prefill batches under a token budget. |
| Overlap scheduling | Preparing the next batch on the CPU while the GPU runs the current one. |
| CUDA graph | A recorded sequence of GPU kernels replayed with a single launch. |
| TP rank | One GPU's process in tensor parallelism; each holds a slice of the model's weights. |
| `table_idx` | A request's row in `token_pool` and `page_table`. |

**Next topics, once the core is solid:**

- Attention backends: `attention/fi.py` and `attention/fa.py`, especially `prepare_metadata`.
- Tensor parallelism: `distributed/` and `layers/linear.py` (how weights are split and outputs all-reduced).
- The front end: `server/api_server.py`, `tokenizer/`, and `scheduler/io.py` (ZeroMQ messaging between processes).
- Mixture-of-experts: `moe/` and `models/qwen3_moe.py`.
- The project's own docs: `docs/structures.md` and `docs/features.md`.
