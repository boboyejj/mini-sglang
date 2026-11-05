# Evaluating the tiny TinyStories model vs Qwen3-0.6B

Step 1 of the evaluation plan: run standard benchmarks with EleutherAI's
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (`lm_eval` 0.4.13).

```bash
./evaluation/run_step1.sh        # ~3 min for the tiny model, ~16 min for Qwen3-0.6B on 4 CPUs
```

Setup notes: `lm_eval` upgrades `huggingface-hub` past what transformers 4.57.3 accepts;
pin it with `uv pip install lm-eval "huggingface-hub>=0.34,<1.0"`. Both models are loaded
from local paths. `--limit` scores the **first** N questions of each benchmark (300; BLiMP 50
per sub-task × 67), so both models see identical questions.

## Results (accuracy, zero-shot)

| Benchmark | Tests | Chance | Tiny model (5.2M) | Qwen3-0.6B | Questions |
|-----------|-------|--------|-------------------|------------|-----------|
| BLiMP | Grammar: prefer the correct sentence of a minimal pair | 50% | **54.6%** ± 0.8 | **74.8%** ± 0.7 | 3,350 |
| LAMBADA | Predict a passage's last word | ~0% | 4.7% ± 1.2 | **42.7%** ± 2.9 | 300 |
| HellaSwag | Sensible ending of an everyday scene (4 options) | 25% | 28.7% ± 2.6 | **41.7%** ± 2.9 (50.7% norm.) | 300 |
| PIQA | Physically sensible solution (2 options) | 50% | 51.7% ± 2.9 | **68.0%** ± 2.7 | 300 |
| ARC-Easy | Grade-school science (4 options) | ~25% | 23.0% ± 2.4 | **60.3%** ± 2.8 | 300 |
| LAMBADA perplexity | Lower is better | — | 7,784,219 | 22.2 | 300 |

± is the harness's standard error. "norm." is `acc_norm`: the answer's log-probability
divided by its length, so long answers aren't penalized for having more tokens.

Runtime on this CPU: tiny model 2.5 min for everything; Qwen3-0.6B 16 min.

## Findings

- **The tiny model is at chance on everything except grammar.** HellaSwag, PIQA and ARC
  need world knowledge it never saw in children's stories. A benchmark only measures a model
  that is inside its range.
- **Its BLiMP average hides a split.** It scores ≥ 70% on 15 of 67 grammar categories
  (e.g. reflexives `principle_A_case_1` 100%, irregular past participles 88%) but ≤ 30% on 6,
  being *systematically wrong* on where "ever"-type words are allowed (`npi_present_1/2`
  8–10%). Below-chance means it learned a misleading rule, not nothing.
- **Perplexity only compares within the same kind of text.** 6.6 on TinyStories validation,
  7.8 million on LAMBADA's novels, whose words the tiny model has rarely or never seen.
- **Scale buys knowledge.** Qwen3-0.6B (~100x more parameters, trained on trillions of
  tokens) beats the tiny model everywhere, most on knowledge-heavy tasks (ARC +37 points,
  LAMBADA +38). Its HellaSwag score jumps from 41.7% to 50.7% with length normalization:
  scoring choices is itself a design decision.
