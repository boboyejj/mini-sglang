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

## Files

- `_common.py`: the CPU shim, `make_managers()` (builds the scheduler's managers like
  `Scheduler.__init__`), `run_sim()` (a copy of `Scheduler.normal_loop` with fake tokens),
  `load_tokenizer()` and `print_tree()`.

## Notes

- Experiment 3 downloads only Qwen3-0.6B's tokenizer files (a few MB). Without network
  access it falls back to fake token IDs.
- Model weights can't be downloaded here yet: they come from `*.hf.co` hosts
  (`cas-server.xethub.hf.co`, `us.aws.cdn.hf.co`) that the network policy blocks. Running
  the model also needs a GPU.
