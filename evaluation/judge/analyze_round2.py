"""Step 3e: compare three independent judges (round 2) with each other and with round 1.

Measures:
  1. per-model averages from each judge (does the ranking hold across judges?)
  2. inter-judge agreement: correlation and mean absolute difference of overall scores
  3. order effect: does a story's position in a judge's list change its score?
     (judge B saw the reverse of judge A's order; judge C a random order)
  4. round-1 judge (main session) vs the independent judges on the 24 shared stories

Run: python evaluation/judge/analyze_round2.py
"""

import itertools
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.join(HERE, "round2")
CRITERIA = ["grammar", "consistency", "plot", "creativity"]
MODELS = ["tiny-qwen3", "qwen3-0.6b-raw", "qwen3-0.6b-chat"]
JUDGES = ["A", "B", "C"]


def overall(g):
    return statistics.mean(g[c] for c in CRITERIA)


def pearson(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def main() -> None:
    key = json.load(open(os.path.join(R2, "key.json")))
    ids = sorted(key)
    grades = {j: json.load(open(os.path.join(R2, f"grades_judge{j}.json")))["grades"] for j in JUDGES}
    orders = {j: [s["id"] for s in json.load(open(os.path.join(R2, f"stories_judge{j}.json")))]
              for j in JUDGES}
    for j in JUDGES:
        assert set(grades[j]) == set(ids), f"judge {j} is missing stories"

    print("1. average overall score (1-10) by model and judge")
    print(f"   {'model':<16}" + "".join(f"{'judge ' + j:>10}" for j in JUDGES) + f"{'mean':>8}")
    for m in MODELS:
        ms = [t for t in ids if key[t]["model"] == m]
        per = [statistics.mean(overall(grades[j][t]) for t in ms) for j in JUDGES]
        print(f"   {m:<16}" + "".join(f"{v:>10.2f}" for v in per) + f"{statistics.mean(per):>8.2f}")
    print("   per criterion (mean of the 3 judges):")
    for m in MODELS:
        ms = [t for t in ids if key[t]["model"] == m]
        vals = [statistics.mean(grades[j][t][c] for j in JUDGES for t in ms) for c in CRITERIA]
        print(f"   {m:<16}" + "".join(f"  {c[:5]} {v:.1f}" for c, v in zip(CRITERIA, vals)))

    print("\n2. agreement between judges on each story's overall score (36 stories)")
    for a, b in itertools.combinations(JUDGES, 2):
        x = [overall(grades[a][t]) for t in ids]
        y = [overall(grades[b][t]) for t in ids]
        mad = statistics.mean(abs(p - q) for p, q in zip(x, y))
        exact = statistics.mean(grades[a][t][c] == grades[b][t][c] for t in ids for c in CRITERIA)
        print(f"   {a} vs {b}: correlation {pearson(x, y):.2f}, mean |difference| {mad:.2f} points,"
              f" identical individual scores {exact:.0%}")
    for c in CRITERIA:
        rs = [pearson([grades[a][t][c] for t in ids], [grades[b][t][c] for t in ids])
              for a, b in itertools.combinations(JUDGES, 2)]
        print(f"   {c:<12} average pairwise correlation {statistics.mean(rs):.2f}")

    print("\n3. order effect: overall score of stories in the first vs second half of each list")
    for j in JUDGES:
        first, second = orders[j][:18], orders[j][18:]
        f = statistics.mean(overall(grades[j][t]) for t in first)
        s = statistics.mean(overall(grades[j][t]) for t in second)
        pos = list(range(36))
        r = pearson(pos, [overall(grades[j][t]) for t in orders[j]])
        print(f"   judge {j}: first half {f:.2f}, second half {s:.2f}, correlation with position {r:+.2f}")
    # A and B saw the same stories in opposite orders: same story early for one, late for the other
    early_a = [t for t in ids if orders["A"].index(t) < 18]
    d = statistics.mean(overall(grades["A"][t]) - overall(grades["B"][t]) for t in early_a)
    d2 = statistics.mean(overall(grades["B"][t]) - overall(grades["A"][t]) for t in ids if t not in early_a)
    print(f"   same story, seen early minus seen late (A vs B): {d:+.2f} and {d2:+.2f} points")

    r1 = json.load(open(os.path.join(HERE, "grades_round1.json")))["grades"]
    shared = [t for t in ids if key[t]["round1_id"]]
    x = [overall(r1[key[t]["round1_id"]]) for t in shared]
    y = [statistics.mean(overall(grades[j][t]) for j in JUDGES) for t in shared]
    print(f"\n4. round-1 judge vs mean of independent judges on {len(shared)} shared stories:")
    print(f"   correlation {pearson(x, y):.2f}, round-1 mean {statistics.mean(x):.2f},"
          f" independent mean {statistics.mean(y):.2f}")
    for m in MODELS[:2]:
        ms = [t for t in shared if key[t]["model"] == m]
        a = statistics.mean(overall(r1[key[t]["round1_id"]]) for t in ms)
        b = statistics.mean(statistics.mean(overall(grades[j][t]) for j in JUDGES) for t in ms)
        print(f"   {m:<16} round-1 {a:.2f}  independent {b:.2f}")


if __name__ == "__main__":
    main()
