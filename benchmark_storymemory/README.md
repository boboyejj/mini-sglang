# StoryMemory: a small benchmark of in-story memory

**Question it answers:** after reading a short story, does a model remember who had what,
when and where? Built to evaluate our 5.2M-parameter TinyStories model and Qwen3-0.6B,
and to practise benchmark design.

## Format

```text
Lily had a red ball. Max had a kite. Sam had a cup.   <- facts
The sky was very clear. A dog barked far away.        <- 0, 5, 15 or 25 neutral fillers
The kite belonged to ___                               <- unfinished sentence
choices: Max / Lily / Sam       correct: Max
```

| Category | Unfinished sentence | Choices |
|---|---|---|
| name | "The kite belonged to" | the story's 3 children |
| object | "Then Max picked up his" | the story's 3 objects |
| time | "Max took a walk every" | morning / afternoon / evening |
| place | "Max had a lot of fun at the" | the story's 3 places |
| gender | "The one who found the stick was a little" | boy / girl |

- **standard**: 3 children (gender: 2); **easy**: 2 children, opening "Once upon a time,
  there were two friends, …". 50 items per category and variant: 500 items.
- **No-story baseline**: the same unfinished sentence without the story. The gap between
  the two scores is what the story contributes.
- **Scoring** (lm-eval `multiple_choice`): log P(" " + choice | context); highest wins;
  accuracy. Every choice is one token for both tokenizers, so no length normalization is needed.

```bash
python benchmark_storymemory/generate.py   # regenerate data (fixed seed)
python benchmark_storymemory/check.py      # quality checks
./benchmark_storymemory/run.sh             # lm_eval on both models (~4 min)
python benchmark_storymemory/analyze.py    # by category, distance, probability mass
python benchmark_storymemory/by_hand.py    # reproduce scores without the harness
```

## Design rules and the lessons behind them

| Rule | From |
|---|---|
| Wrong choices come from the same story | BLiMP: models prefer familiar words |
| All choices are 1 token for both tokenizers | BLiMP/HellaSwag: summed scores favor shorter answers |
| No-story baseline | Separates memory from guessing habits |
| Correct position balanced; gender answers balanced | Avoid shortcuts (v1 bug: 32 boy / 18 girl) |
| Fresh templated stories; no story setup appears in the training data | Contamination |
| Wordings piloted for probability mass (`pilot.py`) | v1: models put ~0% on the choices |
| Every marked answer re-derived from the story text (`check.py`) | v2 bug: 278/500 answers were wrong |

## Building it: what went wrong and how it was caught

1. **"a orange bike"**: grammar bug, caught by reading samples; `check.py` now tests it.
2. **Gender answers imbalanced** (v1): position was balanced, labels weren't. A model that
   always says "boy" scored 64% with no story. Caught by the no-story baseline.
3. **Implausible wordings** (v1): after "The child with the cup was", both models expected
   "happy", not a name (≤1.3% probability on all choices). `pilot.py` compared wordings by
   probability mass, choosing the one where the weaker model's mass was highest (not by
   accuracy, which would tune the benchmark to the models).
4. **Wrong answer key** (v2 rewrite): an off-by-one in position balancing mislabeled 278 of
   500 items; all automatic checks passed. Fixed, and `check.py` now verifies every answer.

## Results (v2)

| Model | Standard | No story | Easy | No story | Chance |
|---|---|---|---|---|---|
| Qwen3-0.6B | **87.2%** | 35.2% | **93.6%** | 49.6% | 33% / 50% |
| Tiny (5.2M) | 41.2% | 35.2% | 51.6% | 46.8% | 33% / 50% |

By distance (all categories, standard / easy):

| Model | d=0 | d=5 | d=15 | d=25 |
|---|---|---|---|---|
| Qwen3-0.6B | 89% / 88% | 82% / 95% | 85% / 95% | 93% / 97% |
| Tiny | 52% / 63% | 35% / 42% | 38% / 52% | 38% / 50% |

(v1, with the flawed wordings: Qwen3-0.6B 80.4%, tiny 34.8%.)

## Findings

- **Qwen3-0.6B remembers story details well and does not forget over ~200 tokens**: the
  story adds 44-52 points; accuracy at 25 filler sentences equals distance 0.
- **The tiny model's memory is very short**: above its baseline only when the facts come
  right before the question (52% vs 35%; easy 63% vs 47%), at chance after 5 sentences. Its
  clearest signal is gender (72% vs 48%).
- **Fixing the wordings changed results more than anything else** (Qwen +7 points; name
  54% to 72%): what a likelihood benchmark measures depends on whether the model finds the
  choices plausible at all.
- Scores reproduce exactly by hand (`by_hand.py`: 160 items, 0.0000 difference).

## Limitations

- **Small cells**: ~12 items per category and distance, so per-cell numbers vary by ±15
  points (see the tiny model's gender row). Generate more items for firmer per-cell results.
- **Distance is too short for Qwen3-0.6B** (max ~200 tokens); a longer range is needed to see
  it forget. It is capped here by the tiny model's 256-token training length.
- **Templated, list-like stories** are unlike natural text; the tiny model remembers better
  in natural stories, so it is under-measured.
- **Low probability mass remains** for the tiny model on time (2%) and name (4%): its
  results there are close to noise.
- **Time** repeats the fact's wording in the question ("took a walk every"), so it partly
  tests copying; **gender** relies on name-gender associations of the chosen names.
- 500 synthetic items from one template set; not a general measure of reading comprehension.
