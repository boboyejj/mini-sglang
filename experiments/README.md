# Beginner experiments (CPU only)

These scripts run mini-sglang's **real** scheduler and KV-cache code on a CPU-only machine
(like this cloud session). No GPU and no model weights are needed: the "forward pass" is
replaced by fake sampled tokens, and `_common.py` turns `pin_memory` (which needs an NVIDIA
driver) into a no-op. Everything else (admission, chunking, prefix matching, eviction,
freeing) is the code that runs in production.

Pair them with the study guide (`docs/study_guide.md`).

## Setup

```bash
cd /home/user/mini-sglang && source .venv/bin/activate
```

Run each script from the repository root, e.g. `python experiments/01_request_counters.py`.

## The experiments

| # | Script | Concept | Study guide module |
|---|--------|---------|--------------------|
| 1 | `01_request_counters.py` | A request's `cached_len` / `device_len` from prefill to finish | 0. Data structures |
| 2 | `02_radix_prefix_sharing.py` | Matching, node splitting, locking and eviction in the radix tree | 3. KV cache |
| 3 | `03_prefix_sharing_chat.py` | How much prefill a shared system prompt saves (real Qwen3 tokenizer) | 3. KV cache |
| 4 | `04_eviction_lru.py` | Least-recently-used eviction | 3. KV cache |
| 5 | `05_chunked_prefill.py` | Splitting a long prompt under a token budget | 1 + 2. Scheduler, admission |
| 6 | `06_admission_memory.py` | How reserving `max_tokens` limits concurrency | 2. Admission |

Each script ends with a question. Read its docstring's **Try** line for a variation, predict
the output, then run it.

## Experiments with real model weights (7–13)

**Why download weights?** Experiments 1–6 simulate the bookkeeping with fake tokens. They
show *where* K/V lives and *when* it is reused, but not *what* it is or *why* it matters.
With real weights you can watch real K/V tensors being built, measure what caching saves,
and check that reusing a cached prefix gives exactly the same answer.

**Which model?** [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B): 0.6B parameters,
about 1.5 GB to download and about 2.4 GB of RAM in float32. It is the smallest model
mini-sglang supports, and the one its README uses in examples.

**How we use them.** mini-sglang's own engine can't run here: its attention and sampling
kernels (FlashInfer, sgl-kernel) need an NVIDIA GPU. So these experiments run the same
weights with Hugging Face `transformers` on the CPU, about 6–10 tokens per second.
`_model.py` drives the model with a hand-written loop that mirrors mini-sglang's `Engine`:
one prefill pass over the prompt, then one decode pass per token that reuses the KV cache,
with greedy (`argmax`) sampling.

| # | Script | Concept | Result here |
|---|--------|---------|-------------|
| 7 | `07_generate_cpu.py` | Prefill vs decode: time to first token vs time per token | 23-token prefill ~350 ms; each decode step ~90 ms |
| 8 | `08_kv_cache_on_off.py` | Why a KV cache exists | 32 tokens: 4.1 s with cache vs 28.5 s without; identical output |
| 9 | `09_kv_cache_memory.py` | KV bytes per token; how mini-sglang sizes its pool | 112 KiB/token in bf16; a 24 GB GPU holds ~178k tokens |
| 10 | `10_prefix_reuse_real.py` | Prefix caching with real tensors (what the radix cache relies on) | 137 of ~153 tokens reused; ~2x faster prefill; identical answers |
| 11 | `11_batched_decode.py` | Batching decode: step time vs batch size | Batch 64: step only 5x slower than batch 1, so 12x the tokens/s |
| 12 | `12_real_model_scheduler.py` | Capstone: mini-sglang's real scheduler + radix cache driving the real model through a paged KV pool | 4 requests, chunking + prefix hits; every answer identical to plain generation; 68% of prefill saved |
| 12 | `12_real_model_scheduler.py --model tiny` | Our 5.2M TinyStories model served by the same scheduler: 44 requests, up to 16 running | 211 steps, decode batches of 14.4 on average; all 44 answers identical to plain generation |
| 13 | `13_cache_req_bug.py` | A KV bookkeeping bug in `CacheManager.cache_req` (found with experiment 12 under memory pressure; upstream issue sgl-project/mini-sglang#168), now fixed in the source | Original: 43/44 identical at 1,024 and 768 slots, 110-174 reads of freed slots; fixed: 44/44 and 0 |

After a fresh container start, the first load can take ~2 minutes (reading 1.5 GB from a cold disk); later loads take ~1–2 s. Timings vary with the machine.
Qwen3-0.6B is small, so expect simple, sometimes wrong answers: these experiments measure
the serving mechanics, not answer quality.

## Files

- `_model.py`: loads Qwen3-0.6B and runs the prefill/decode loop for experiments 7–12.
- `_common.py`: the CPU shim, `make_managers()` (builds the scheduler's managers like
  `Scheduler.__init__`), `run_sim()` (a copy of `Scheduler.normal_loop` with fake tokens),
  `load_tokenizer()` and `print_tree()`.

## Notes

- Experiment 3 downloads only Qwen3-0.6B's tokenizer files (a few MB). Without network
  access it falls back to fake token IDs.
- Experiments 7–12 download Qwen3-0.6B's weights (~1.5 GB) on first run into
  `~/.cache/huggingface`. The network policy must allow `huggingface.co` and `*.hf.co`.
