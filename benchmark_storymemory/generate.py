"""StoryMemory: a small benchmark of whether a model remembers details from a short story.

Each item: a short story, then an unfinished last sentence; the model must pick the word
that completes it, choosing among 2-3 options that ALL appear in the story (so the answer
can't be guessed from word familiarity). Scored by likelihood, like HellaSwag/BLiMP.

Five categories, N_PER_CATEGORY items each:
  name     who had an object                "The child with the kite was" -> Max
  object   what someone had                 "Then Max picked up his"      -> kite
  color    the color of someone's thing     "Max's ball was"              -> blue
  place    where someone went               "...Max came home from the"   -> zoo
  gender   whether a boy or girl did it     "The child who found the stick was a" -> boy
Distance: 0, 2 or 5 neutral filler sentences between the facts and the question.

Output: data/storymemory_<category>.jsonl with fields
  context (story + question stem), stem (question stem only, for the no-story baseline),
  choices (list, each with no leading space), gold (index), category, distance, story_id.
Run: python benchmark_storymemory/generate.py
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
N_PER_CATEGORY = 50
SEED = 2026
DISTANCES = [0, 2, 5]  # filler sentences between the facts and the question

GIRLS = "Lily Mia Amy Sara Sue Anna Emma Lucy Kate Jane Rose Zoe".split()
BOYS = "Tom Max Ben Sam Tim Jack Leo Bob Dan Joe Jim".split()
OBJECTS = "ball kite hat cup box book doll car drum bag bike boat".split()
COLORS = "red blue green pink yellow black white brown purple orange".split()
PLACES = "park beach zoo farm lake shop forest school garden river".split()
FILLERS = [  # neutral: no names, objects, colors or places from the lists above
    "The sun was warm and bright.", "The birds sang in the trees.", "It was a nice day.",
    "The wind was soft and cool.", "Everyone felt happy.", "The sky was very clear.",
    "They laughed and talked a lot.", "Some clouds moved slowly.", "A dog barked far away.",
    "The grass was soft.", "They had a snack and a drink.", "Time went by quickly.",
]


def pick_kids(rng, n):
    kids = rng.sample([(g, "girl") for g in GIRLS] + [(b, "boy") for b in BOYS], n)
    return kids


def his_her(gender):
    return "his" if gender == "boy" else "her"


def a_an(word):
    return "an" if word[0] in "aeiou" else "a"


def fillers(rng, k):
    return " ".join(rng.sample(FILLERS, k))


def make_item(cat, i, rng):
    distance = DISTANCES[i % len(DISTANCES)]
    if cat == "gender":
        g, b = rng.choice(GIRLS), rng.choice(BOYS)
        finder_name, finder = rng.choice([(g, "girl"), (b, "boy")])
        a, c = (g, b) if rng.random() < 0.5 else (b, g)
        facts = f"{a} and {c} went for a walk. {finder_name} found a big stick."
        stem = "The child who found the stick was a"
        choices, answer = ["boy", "girl"], finder
    else:
        kids = pick_kids(rng, 3)
        target = rng.randrange(3)
        name, gender = kids[target]
        if cat in ("name", "object"):
            objs = rng.sample(OBJECTS, 3)
            facts = " ".join(f"{k} had {a_an(o)} {o}." for (k, _), o in zip(kids, objs))
            if cat == "name":
                stem, choices, answer = f"The child with the {objs[target]} was", [k for k, _ in kids], name
            else:
                stem, choices, answer = f"Then {name} picked up {his_her(gender)}", objs, objs[target]
        elif cat == "color":
            obj, cols = rng.choice(OBJECTS), rng.sample(COLORS, 3)
            facts = " ".join(f"{k} had {a_an(c)} {c} {obj}." for (k, _), c in zip(kids, cols))
            stem, choices, answer = f"{name}'s {obj} was", cols, cols[target]
        elif cat == "place":
            places = rng.sample(PLACES, 3)
            facts = " ".join(f"{k} went to the {p}." for (k, _), p in zip(kids, places))
            stem, choices, answer = f"At the end of the day, {name} came home from the", places, places[target]
    story = facts + (" " + fillers(rng, distance) if distance else "")
    order = choices[:]
    rng.shuffle(order)
    # balance the correct answer's position: rotate it to position i % len(order)
    want = i % len(order)
    order.remove(answer)
    order.insert(want, answer)
    return {
        "story_id": f"{cat}-{i:03d}", "category": cat, "distance": distance,
        "context": f"{story} {stem}", "stem": stem, "choices": order, "gold": want,
    }


def main() -> None:
    rng = random.Random(SEED)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    for cat in ["name", "object", "color", "place", "gender"]:
        items = [make_item(cat, i, rng) for i in range(N_PER_CATEGORY)]
        with open(os.path.join(HERE, "data", f"storymemory_{cat}.jsonl"), "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        print(f"{cat}: {len(items)} items")


if __name__ == "__main__":
    main()
