# Training a tiny Qwen3 on TinyStories (CPU)

A 5.2M-parameter model with the same architecture as Qwen3-0.6B (grouped-query attention,
RoPE, RMSNorm, SwiGLU, QK-norm), trained from scratch for 47 minutes on this cloud
session's 4-core CPU. mini-sglang's `models/qwen3.py` implements this architecture, so
`tiny-qwen3/` could be served by mini-sglang on a GPU.

## Steps

```bash
python training/prepare_data.py   # download TinyStories, train tokenizer, encode (~1 min)
python training/train.py --check  # sanity checks (~2 min)
python training/train.py          # train 6,000 steps (~47 min); resumes from checkpoints/
python training/test_model.py     # validation loss + sample stories
```

| Item | Value |
|------|-------|
| Data | [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) V2-GPT4: first 100 MB of train (122,051 stories, 24.1M tokens); full validation file (27,630 stories, 5.4M tokens) |
| Tokenizer | Byte-level BPE, 8,192 tokens, trained on the training slice |
| Model | 4 layers, hidden 256, 4 query heads / 2 KV heads, head dim 64, tied embeddings: 5.2M parameters |
| Training | 6,000 steps × 16 sequences × 256 tokens = 24.6M tokens (~1 pass over the data); AdamW, peak LR 2e-3, 200-step warm-up, cosine decay |

## Results

Sanity checks: initial loss 9.02 (random guessing = ln 8192 = 9.01); one batch memorized
from 9.02 to 0.009 in 200 steps.

| Step | Minutes | Train loss | Val loss |
|------|---------|-----------|----------|
| 500 | 4.2 | 2.81 | 2.77 |
| 1,000 | 8.2 | 2.48 | 2.46 |
| 2,000 | 16.0 | 2.20 | 2.19 |
| 3,000 | 23.9 | 2.07 | 2.05 |
| 4,000 | 31.6 | 1.95 | 1.95 |
| 5,000 | 39.2 | 1.87 | 1.88 |
| 6,000 | 47.0 | 1.84 | 1.84 |

Final test (`test_model.py`, 409,600 held-out tokens): **validation loss 1.88, perplexity 6.6**.
Generation speed on CPU: ~300–365 tokens/s (Qwen3-0.6B: ~11 tokens/s here).

Sample (prompt in bold, sampled at temperature 0.8):

> **Once upon a time, a little rabbit wanted to fly.** He was very heavy and knew he needed
> to be strong. The rabbit went to his friend, a tall owl. … The wise owl thought very hard
> and decided to help the rabbit.

All checkpoint samples are in `samples.md`; the per-step log is in `log.jsonl`.
