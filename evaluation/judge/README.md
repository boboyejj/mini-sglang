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

## Round 2: fair rematch + three independent judges

- `generate_chat.py`: Qwen3-0.6B continues the same 12 openings through its **chat template**
  ("Continue this children's story…", thinking off, same seeds and sampling). Its replies
  are short stories (41-86 tokens); it sometimes repeats the opening first.
- 36 blinded stories (tiny, Qwen raw, Qwen chat) were graded by **three fresh sub-agent
  judges** that saw only `rubric.md` and their own story order (`round2/stories_judge{A,B,C}.json`;
  B = A reversed, C = random). `analyze_round2.py` compares them; output in `round2/analysis.txt`.

| Model | Judge A | Judge B | Judge C | Mean | Grammar | Consistency | Plot | Creativity |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B, chat format | 5.33 | 5.29 | 5.25 | **5.29** | 8.3 | 5.1 | 4.4 | 3.4 |
| Tiny (5.2M) | 3.40 | 3.75 | 3.48 | 3.54 | 5.5 | 2.7 | 3.2 | 2.8 |
| Qwen3-0.6B, raw text | 2.58 | 2.48 | 2.58 | 2.55 | 5.9 | 1.7 | 1.2 | 1.4 |

**Judge reliability:**

| Check | Result |
|---|---|
| Agreement between judges (overall score per story) | correlation 0.95-0.98; mean difference 0.22-0.33 points; 60-72% of individual scores identical |
| Order effect | none consistent: the same story seen early vs late differs by -0.14 and 0.00 points |
| Round-1 judge vs independent judges (24 shared stories) | correlation 0.96, but round 1 scored **0.6 points higher** on average (3.66 vs 3.05) |

## Findings (both rounds)

- **Prompting matters more than the model here.** The same Qwen3-0.6B weights score 2.55 as
  raw text and 5.29 through the chat template, overtaking the tiny model (3.54). Round 1's
  ranking was an artifact of prompting a chat model like a base model.
- **Judges agreed closely with each other**, and story order made no consistent difference.
  But all four judges are the same underlying model with the same rubric: high agreement
  shows self-consistency, not that the grades are right. Shared biases are invisible to
  this test; a different judge model or human raters would be needed to find them.
- **Absolute scores are judge-specific; rankings are more robust.** The round-1 judge ranked
  stories almost identically (r = 0.96) but was systematically more lenient by 0.6 points.
  Compare models graded by the same judge, never scores from different judges.
- **Remaining caveats:** 12 stories per model is a small sample; the chat stories are shorter,
  which leaves fewer chances for mistakes (a possible length bias); and the judge's own
  writing style may favor fluent, polished text.
