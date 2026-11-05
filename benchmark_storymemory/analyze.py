"""Break StoryMemory results down by category, distance and the asked-about fact's position.

Run: python benchmark_storymemory/analyze.py   (after run.sh)
"""

import collections
import glob
import json
import statistics

CATS = ["name", "object", "time", "place", "gender"]


def load(model, variant):
    rows = {}
    for cat in CATS:
        f = glob.glob(f"benchmark_storymemory/results/{model}/*/samples_storymemory_{variant}{cat}_*.jsonl")[0]
        for line in open(f):
            r = json.loads(line)
            rows[r["doc"]["story_id"]] = r
    return rows


def fact_position(doc):
    """Which of the story's facts (0 = first) the question asks about."""
    if doc["category"] == "gender":
        return None
    answer = doc["choices"][doc["gold"]]
    facts = doc["context"].split(". ")[:3]
    return next(i for i, f in enumerate(facts) if answer in f.split() or answer + "." in f)


def main() -> None:
    for model in ["tiny-qwen3", "qwen3-0.6b"]:
        story, base = load(model, ""), load(model, "nostory_")
        print(f"\n######## {model}")
        print(f"{'category':<8} {'no story':>9} {'d=0':>6} {'d=5':>6} {'d=15':>6} {'d=25':>6} {'all':>6}")
        for cat in CATS:
            ids = [k for k in story if story[k]["doc"]["category"] == cat]
            nb = statistics.mean(base[k]["acc"] for k in ids)
            by_d = [statistics.mean(story[k]["acc"] for k in ids if story[k]["doc"]["distance"] == d)
                    for d in (0, 5, 15, 25)]
            allv = statistics.mean(story[k]["acc"] for k in ids)
            print(f"{cat:<8} {nb:>9.0%} " + " ".join(f"{v:>6.0%}" for v in by_d) + f" {allv:>6.0%}")
        pos = collections.defaultdict(list)
        for r in story.values():
            p = fact_position(r["doc"])
            if p is not None:
                pos[p].append(r["acc"])
        print("asked-about fact was the " + ", ".join(
            f"{['1st', '2nd', '3rd'][p]}: {statistics.mean(v):.0%} (n={len(v)})" for p, v in sorted(pos.items())))
        picks = collections.Counter()
        for r in story.values():
            if r["doc"]["category"] == "name":
                lls = [float(x[0]) for x in r["filtered_resps"]]
                chosen = r["doc"]["choices"][lls.index(max(lls))]
                picks[fact_position({**r["doc"], "gold": r["doc"]["choices"].index(chosen)})] += 1
        print(f"name category: the name it picks belongs to the 1st/2nd/3rd fact: "
              f"{[picks[i] for i in range(3)]}")
    gender = [json.loads(l)["choices"][json.loads(l)["gold"]]
              for l in open("benchmark_storymemory/data/storymemory_gender.jsonl")]
    print(f"\ngender category correct answers: {dict(collections.Counter(gender))}")


if __name__ == "__main__":
    main()
