"""Break StoryMemory results down by variant, category, distance and fact position, and
report how much probability each model puts on the choices (a validity check).

Run: python benchmark_storymemory/analyze.py [results_dir]   (default: results, after run.sh)
"""

import collections
import glob
import json
import math
import statistics
import sys

CATS = ["name", "object", "time", "place", "gender"]
DIST = [0, 5, 15, 25]
RESULTS = sys.argv[1] if len(sys.argv) > 1 else "benchmark_storymemory/results"


def load(model, task_prefix):
    rows = {}
    for cat in CATS:
        files = glob.glob(f"{RESULTS}/{model}/*/samples_{task_prefix}{cat}_*.jsonl")
        for line in open(files[0]):
            r = json.loads(line)
            rows[r["doc"]["story_id"]] = r
    return rows


def main() -> None:
    for model in ["tiny-qwen3", "qwen3-0.6b"]:
        for variant, prefix in [("standard", "storymemory_"), ("easy", "storymemory_easy_")]:
            story = load(model, prefix)
            base = load(model, prefix + "nostory_")
            chance = "33%" if variant == "standard" else "50%"
            print(f"\n######## {model} / {variant} (chance {chance}; gender always 50%)")
            print(f"{'category':<8} {'no story':>9} " + " ".join(f"{'d=' + str(d):>6}" for d in DIST)
                  + f" {'all':>6} {'mass':>7}")
            for cat in CATS:
                ids = [k for k in story if story[k]["doc"]["category"] == cat]
                nb = statistics.mean(base[k]["acc"] for k in ids)
                by_d = [statistics.mean(story[k]["acc"] for k in ids if story[k]["doc"]["distance"] == d)
                        for d in DIST]
                allv = statistics.mean(story[k]["acc"] for k in ids)
                mass = statistics.median(sum(math.exp(float(x[0])) for x in story[k]["filtered_resps"])
                                         for k in ids)
                print(f"{cat:<8} {nb:>9.0%} " + " ".join(f"{v:>6.0%}" for v in by_d)
                      + f" {allv:>6.0%} {mass:>7.1%}")
            by_d = [statistics.mean(story[k]["acc"] for k in story if story[k]["doc"]["distance"] == d)
                    for d in DIST]
            print(f"{'ALL':<8} {statistics.mean(base[k]['acc'] for k in story):>9.0%} "
                  + " ".join(f"{v:>6.0%}" for v in by_d)
                  + f" {statistics.mean(story[k]['acc'] for k in story):>6.0%}")


if __name__ == "__main__":
    main()
