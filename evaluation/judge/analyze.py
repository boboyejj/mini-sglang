"""Step 3c: unblind the grades and summarize them per model.

Run: python evaluation/judge/analyze.py [grades_file.json]
"""

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CRITERIA = ["grammar", "consistency", "plot", "creativity"]
QUIZ_MARKERS = ["Answer:", "A)", "A.", "\\boxed", "Options:"]


def main() -> None:
    gfile = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "grades_round1.json")
    grades = json.load(open(gfile))["grades"]
    key = json.load(open(os.path.join(HERE, "key.json")))
    stories = {s["id"]: s for s in json.load(open(os.path.join(HERE, "stories.json")))}

    print(f"{'model':<12} {'n':>3} " + " ".join(f"{c:>12}" for c in CRITERIA) + f" {'overall':>8} {'quiz drift':>11}")
    for model in ["tiny-qwen3", "qwen3-0.6b"]:
        ids = [s for s in grades if key[s]["model"] == model]
        means = [statistics.mean(grades[s][c] for s in ids) for c in CRITERIA]
        quiz = sum(any(m in stories[s]["continuation"] for m in QUIZ_MARKERS) for s in ids)
        print(f"{model:<12} {len(ids):>3} " + " ".join(f"{m:>12.1f}" for m in means)
              + f" {statistics.mean(means):>8.1f} {quiz:>8}/{len(ids)}")

    print("\nper story (id: model, overall, quiz-style?)")
    for s in sorted(grades):
        overall = statistics.mean(grades[s][c] for c in CRITERIA)
        quiz = any(m in stories[s]["continuation"] for m in QUIZ_MARKERS)
        print(f"  {s}: {key[s]['model']:<11} prompt {key[s]['prompt_id']:>2}  overall {overall:4.2f}"
              f"  {'QUIZ' if quiz else ''}")


if __name__ == "__main__":
    main()
