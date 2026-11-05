# BLiMP deep dive

How BLiMP tests a model, how lm-evaluation-harness scores it, and what our two models'
scores really measure. Scripts: `blimp_by_hand.py` (reproduce the score), `blimp_analysis.py`
(token-count and word-frequency effects). Both read the saved answers in `results/`.

## 1. What BLiMP is

BLiMP (Benchmark of Linguistic Minimal Pairs; Warstadt et al., 2020) tests grammar with
**minimal pairs**: two sentences that differ in one small way, one grammatical, one not.

- 67 sub-tasks ("paradigms"), each 1,000 pairs, grouped into 12 phenomena such as
  subject-verb agreement, anaphor (`himself`) binding, NPI licensing (`ever`, `any`),
  island effects and irregular forms. We scored the first 50 pairs of each: 3,350 pairs.
- Pairs are generated automatically from templates and a fixed vocabulary, which is why
  some read oddly ("Paula references Robert.").
- No training or prompt: the model is never asked a question. Only its probabilities are used.

One raw item (`regular_plural_subject_verb_agreement_1`):

```text
sentence_good: "Paula references Robert."
sentence_bad:  "Paula reference Robert."
```

## 2. How the harness scores it

The task file (`lm_eval/tasks/blimp/_template_yaml`) says: multiple choice, empty context,
choices `[sentence_good, sentence_bad]`, correct answer 0, metric accuracy. So for each pair:

```text
score(s) = sum over the tokens of s of  log P(token | start token + earlier tokens)
correct  = score(good) > score(bad)
accuracy = fraction of pairs correct
```

Details that matter:

- The harness puts a space before each sentence (`" Paula references Robert."`).
- With an empty context, the sentence is conditioned on one start token: the tokenizer's
  BOS token, or its EOS token if there is none. For our tiny model that is
  `<|endoftext|>` (what precedes every story in training). For Qwen3-0.6B, which has no
  BOS token, it is `<|im_end|>`, the end-of-chat-turn marker: a harness default nobody chose.
- `blimp_by_hand.py` recomputes every pair in 3 sub-tasks for both models with these rules
  and matches the harness to within 0.0002 in log-probability, and exactly in accuracy.

## 3. Reading individual answers

Tiny model, Paula pair, per-token log-probabilities:

```text
good " Paula references Robert."   Paul -12.38  a -7.60  re -12.88  fere -10.72  nce -11.73  s -3.76  Robert -15.11  . -2.99   total -77.18
bad  " Paula reference Robert."    Paul -12.38  a -7.60  re -12.88  fere -10.72  nce -11.73           Robert -12.04  . -3.59   total -70.95
```

The tiny model has never seen "reference(s)" (0 times in our 19.7M-word training slice), so
the word is split into pieces. The grammatical sentence has one more piece (`s`), which costs
3.76, and it loses. Qwen3-0.6B has single tokens for both words and picks the right one.

Tiny model, `npi_present_1` pair: "Even Suzanne has **really** joked around." (good) vs
"Even Suzanne has **ever** joked around." (bad). Everything matches except one word:
`really` costs -9.11, `ever` only -7.54, because "ever" appears 10,785 times in TinyStories
("happily ever after") and "really" 3,196. Familiarity beats grammar, so the tiny model
scores 10% on this sub-task, worse than a coin flip.

## 4. What the scores are made of (all 3,350 pairs)

| Pairs where the grammatical sentence has… | Pairs (tiny) | Tiny model | Pairs (Qwen) | Qwen3-0.6B |
|---|---|---|---|---|
| fewer tokens | 431 | 97.0% | 255 | 91.4% |
| the same number of tokens | 2,454 | 56.2% | 2,950 | 74.5% |
| more tokens | 465 | 6.7% | 145 | 51.0% |

| Pairs whose differing words are, in TinyStories… | Pairs | Tiny model | Qwen3-0.6B |
|---|---|---|---|
| rarer in the grammatical sentence | 1,106 | 34.1% | 70.3% |
| more common in the grammatical sentence | 1,184 | 80.0% | 88.0% |

(Token counts depend on each model's tokenizer, hence different pair counts. The frequency
table excludes 1,060 pairs that differ only in word order or whose differing words are
equally frequent.)

## 5. Findings

- **For the tiny model, token count decides most outcomes.** When the sentences tokenize to
  different lengths, it picks the shorter one 93–97% of the time, right or wrong. Its 54.6%
  average mostly reflects how often BLiMP happens to make the grammatical sentence shorter.
- **On pairs with equal token counts, a fairer test of grammar, the tiny model scores 56.2%
  and Qwen3-0.6B 74.5%.** The tiny model has learned a little grammar; most of its apparent
  wins and losses are tokenization.
- **Word frequency matters for both models,** but far less for the larger one (18-point gap
  vs 46): a bigger model's preferences depend more on grammar than on familiarity.
- **Below-chance sub-tasks are systematic, not random.** They are where the ungrammatical
  sentence is shorter or uses a more familiar word.
- **General lessons:** a likelihood benchmark scores the model *and* its tokenizer;
  harness defaults (start token, leading space, summing vs averaging) are part of the
  result; and one average can hide very different behaviors underneath.

## 6. Re-scoring with length normalization

`blimp_rescore.py` re-scores the saved log-probabilities three ways (no model runs):

| Method | Tiny: overall | FEWER / same / MORE tokens | Qwen3-0.6B: overall | FEWER / same / MORE tokens |
|---|---|---|---|---|
| sum (standard) | 54.6% | 97.0 / 56.2 / 6.7 | **74.8%** | 91.4 / 74.5 / 51.0 |
| per token | **57.1%** | 56.8 / 56.2 / 61.9 | 71.6% | 23.9 / 74.5 / 97.2 |
| per character | 52.9% | 80.5 / 53.1 / 26.0 | 70.9% | 62.4 / 71.4 / 77.2 |

("FEWER / MORE" = the grammatical sentence has fewer / more tokens than the ungrammatical one.)

- **Per-token averaging removes the tiny model's length bias** (57–62% in every group) and
  raises its score to 57.1%, close to its equal-length score: its real grammar level.
- **For Qwen3-0.6B it overcorrects:** it now prefers the *longer* sentence (97% vs 24%),
  because Qwen's extra tokens are usually cheap, predictable pieces such as `s` or `'t`, and
  averaging rewards adding them. Its score drops to 71.6%.
- **Per-character** (lm-eval's `acc_norm` idea) helps neither model here.
- **No normalization is neutral.** The equal-token pairs are unchanged by summing or
  averaging, which makes them the most trustworthy comparison: 56.2% vs 74.5%. BLiMP
  officially uses the plain sum, because its pairs are designed to differ minimally.
