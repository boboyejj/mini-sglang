"""Pilot candidate wordings before generating the benchmark.

For each category, each candidate (fact template, question stem) is tried on N_PILOT small
stories (3 children, distance 0) with both models. Reported per model:
  mass = median probability the model puts on all choices combined (want it high: the
         choices must be plausible next words, or the benchmark compares near-zero numbers)
  acc  = how often the correct choice has the highest probability
Run: python benchmark_storymemory/pilot.py
"""

import random
import statistics

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

N_PILOT = 24
GIRLS = "Lily Mia Amy Sara Sue Anna Emma Lucy Kate Jane Rose Zoe".split()
BOYS = "Tom Max Ben Sam Tim Jack Leo Bob Dan Joe Jim".split()
OBJECTS = "ball kite hat cup box book doll car drum bag bike boat".split()
TIMES = "morning afternoon evening".split()
PLACES = "park beach zoo farm lake shop forest school garden river".split()


def an(w):
    return "an" if w[0] in "aeiou" else "a"


def pos(g):
    return "his" if g == "boy" else "her"


# category -> list of (label, builder). A builder gets (rng) and returns (context, choices, gold)
def kids(rng, n=3):
    return rng.sample([(g, "girl") for g in GIRLS] + [(b, "boy") for b in BOYS], n)


def b_name(stem):
    def f(rng):
        k, o = kids(rng), rng.sample(OBJECTS, 3)
        t = rng.randrange(3)
        facts = " ".join(f"{n} had {an(x)} {x}." for (n, _), x in zip(k, o))
        return f"{facts} {stem.format(o=o[t])}", [n for n, _ in k], t
    return f


def b_object(stem):
    def f(rng):
        k, o = kids(rng), rng.sample(OBJECTS, 3)
        t = rng.randrange(3)
        facts = " ".join(f"{n} had {an(x)} {x}." for (n, _), x in zip(k, o))
        return f"{facts} {stem.format(n=k[t][0], p=pos(k[t][1]))}", o, t
    return f


def b_time(fact, stem):
    def f(rng):
        k, ts = kids(rng), rng.sample(TIMES, 3)
        t = rng.randrange(3)
        facts = " ".join(fact.format(n=n, t=x) for (n, _), x in zip(k, ts))
        return f"{facts} {stem.format(n=k[t][0])}", ts, t
    return f


def b_place(stem):
    def f(rng):
        k, ps = kids(rng), rng.sample(PLACES, 3)
        t = rng.randrange(3)
        facts = " ".join(f"{n} went to the {x}." for (n, _), x in zip(k, ps))
        return f"{facts} {stem.format(n=k[t][0])}", ps, t
    return f


def b_gender(stem):
    def f(rng):
        g, b = rng.choice(GIRLS), rng.choice(BOYS)
        finder, ans = rng.choice([(g, "girl"), (b, "boy")])
        a, c = (g, b) if rng.random() < 0.5 else (b, g)
        ch = ["boy", "girl"]
        return f"{a} and {c} went for a walk. {finder} found a big stick. {stem}", ch, ch.index(ans)
    return f


CANDIDATES = {
    "name": [("v1: The child with the {o} was", b_name("The child with the {o} was")),
             ("The {o} belonged to", b_name("The {o} belonged to")),
             ("Who had the {o}? It was", b_name("Who had the {o}? It was")),
             ("Everyone wanted to play with the {o}, so they asked",
              b_name("Everyone wanted to play with the {o}, so they asked"))],
    "object": [("v1: Then {n} picked up {p}", b_object("Then {n} picked up {p}")),
               ("Then {n} played with {p}", b_object("Then {n} played with {p}"))],
    "time": [("v1: {n} played outside in the {t} / {n} played outside in the",
              b_time("{n} played outside in the {t}.", "{n} played outside in the")),
             ("{n} played outside every {t} / {n} played outside every",
              b_time("{n} played outside every {t}.", "{n} played outside every")),
             ("{n} took a walk every {t} / {n} took a walk every",
              b_time("{n} took a walk every {t}.", "{n} took a walk every"))],
    "place": [("v1: At the end of the day, {n} came home from the",
               b_place("At the end of the day, {n} came home from the")),
              ("{n} had a lot of fun at the", b_place("{n} had a lot of fun at the"))],
    "gender": [("v1: The child who found the stick was a", b_gender("The child who found the stick was a")),
               ("The stick was found by a", b_gender("The stick was found by a")),
               ("The one who found the stick was a little", b_gender("The one who found the stick was a little"))],
}


@torch.no_grad()
def main() -> None:
    models = {"tiny": "training/tiny-qwen3",
              "qwen": snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)}
    loaded = {k: (AutoTokenizer.from_pretrained(p), AutoModelForCausalLM.from_pretrained(p).eval())
              for k, p in models.items()}
    for cat, cands in CANDIDATES.items():
        print(f"\n[{cat}]")
        for label, build in cands:
            items = [build(random.Random(100 + i)) for i in range(N_PILOT)]
            out = []
            for name, (tok, model) in loaded.items():
                mass, acc = [], []
                for ctx, choices, gold in items:
                    ids = tok(ctx, return_tensors="pt", add_special_tokens=False).input_ids
                    p = model(ids).logits[0, -1].softmax(-1)
                    pc = [p[tok(" " + c, add_special_tokens=False).input_ids[0]].item() for c in choices]
                    mass.append(sum(pc))
                    acc.append(pc.index(max(pc)) == gold)
                out.append(f"{name}: mass {statistics.median(mass):6.1%} acc {statistics.mean(acc):4.0%}")
            print(f"  {label:<70} " + " | ".join(out))


if __name__ == "__main__":
    main()
