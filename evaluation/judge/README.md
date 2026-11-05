# Step 3: LLM-as-judge on TinyStories evaluation prompts

The TinyStories paper evaluated models by having GPT-4 grade their story continuations.
Here, Claude (this session) is the judge.

## Method

1. `generate.py`: the first 12 of the 44 openings in TinyStories' `Evaluation prompts.yaml`
   (each ends mid-sentence). Both models continue each one: same seed, temperature 0.8,
   top-k 50, up to 150 tokens, plain text completion (no chat template). The 24 results are
   shuffled under IDs S01-S24 (`stories.json`); the answer key is kept in `key.json`.
2. `rubric.md`: written before reading any story. Grammar, consistency, plot, creativity, 1-10.
3. `grades_round1.json`: the judge's scores and one-line reasons, given before opening the key.
4. `analyze.py`: unblinds and averages per model.

## Results (round 1)

| Model | Grammar | Consistency | Plot | Creativity | Overall | Left the story |
|---|---|---|---|---|---|---|
| Tiny (5.2M) | 6.0 | **3.2** | **3.7** | **3.8** | **4.2** | 0 / 12 |
| Qwen3-0.6B | **6.8** | 2.5 | 1.4 | 1.8 | 3.1 | 11 / 12 |

11 of Qwen3-0.6B's 12 continuations leave the story: 9 become multiple-choice quizzes
("Answer: \boxed{B}"), 2 become analysis essays. Its one real story (S09) scored 5.5, among
the best. The tiny model always writes a story (best 5.75, worst 2.25), with frequent name
switches (Alice becomes Anna or Lily) and logic gaps.

## Findings

- **The ranking flips relative to the benchmarks.** Qwen3-0.6B won every benchmark; here the
  tiny model wins because this task matches what it was trained for.
- **The evaluation must match how a model is used.** Qwen3-0.6B is a chat model trained on
  lots of question-answer data; given raw text with no chat template, it continues it like an
  exam. A fair comparison would prompt it the way it is meant to be used.
- **Blinding failed.** The quiz style, and Qwen always using the full 150 tokens, reveal the
  author. The judge could not avoid knowing, a common problem when comparing models.
- **Grammar is the least informative criterion:** both models write mostly grammatical
  sentences; the differences are in consistency and plot.

## Judge limitations

- One judge, one pass, 24 stories: averages over 12 stories per model are noisy.
- The judge designed the experiment and saw earlier outputs of both models.
- A second pass by the same judge in the same conversation would not test consistency: the
  first grades are still in its context.
