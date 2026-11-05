"""StoryMemory v2: does a model remember details from a short story?

Each item: a short story, then an unfinished last sentence; the model picks the word that
completes it from 2-3 options that ALL appear in the story, so word familiarity can't
decide. Scored by likelihood, like HellaSwag/BLiMP. Wordings were chosen with pilot.py so
that both test models put real probability on the choices.

Categories (N_PER_CATEGORY items each):
  name     who had an object             "The kite belonged to"                    -> Max
  object   what someone had              "Then Max picked up his"                  -> kite
  time     when someone took a walk      "Max took a walk every"                   -> afternoon
  place    where someone went            "Max had a lot of fun at the"             -> zoo
  gender   was it a boy or a girl        "The one who found the stick was a little" -> boy
Variants:
  standard: 3 children (3 choices; gender: 2 children, 2 choices)
  easy:     2 children (2 choices), opening "Once upon a time, there were two friends..."
Distance: 0, 5, 15 or 25 neutral filler sentences between the facts and the question.
Balanced: the correct answer's position, and (gender) the answer itself.

Output: data/storymemory_<category>.jsonl and data/storymemory_easy_<category>.jsonl
Run: python benchmark_storymemory/generate.py
"""

import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
N_PER_CATEGORY = 50
SEED = 2026
DISTANCES = [0, 5, 15, 25]
CATEGORIES = ["name", "object", "time", "place", "gender"]

GIRLS = "Lily Mia Amy Sara Sue Anna Emma Lucy Kate Jane Rose Zoe".split()
BOYS = "Tom Max Ben Sam Tim Jack Leo Bob Dan Joe Jim".split()
OBJECTS = "ball kite hat cup box book doll car drum bag bike boat".split()
TIMES = "morning afternoon evening".split()
PLACES = "park beach zoo farm lake shop forest school garden river".split()
FILLERS = [  # neutral: no names, objects, places or times of day from the lists above
    "The sun was warm and bright.", "The birds sang in the trees.", "It was a nice day.",
    "The wind was soft and cool.", "Everyone felt happy.", "The sky was very clear.",
    "They laughed and talked a lot.", "Some clouds moved slowly.", "A dog barked far away.",
    "The grass was soft.", "They had a snack and a drink.", "Time went by quickly.",
    "A little bird flew over the houses.", "The flowers smelled sweet.", "A cat slept in the sun.",
    "They sang a happy song.", "The air smelled fresh.", "A butterfly danced in the air.",
    "They counted the clouds in the sky.", "A small frog jumped into a puddle.",
    "The trees moved in the wind.", "They told funny jokes.", "A bee buzzed around a flower.",
    "Everyone smiled and waved.", "The road was quiet.", "A squirrel ran up a tree.",
    "They clapped their hands.", "The leaves were green and shiny.", "A cow said moo.",
    "They felt calm and safe.",
]


def a_an(word):
    return "an" if word[0] in "aeiou" else "a"


def his_her(gender):
    return "his" if gender == "boy" else "her"


def make_item(cat, i, rng, easy):
    n_kids = 2 if easy else 3
    distance = DISTANCES[i % len(DISTANCES)]
    if cat == "gender":
        g, b = rng.choice(GIRLS), rng.choice(BOYS)
        answer = ["boy", "girl"][(i // 2) % 2]  # balanced answers, crossed with positions
        finder = b if answer == "boy" else g
        first, second = (g, b) if rng.random() < 0.5 else (b, g)
        opening = f"Once upon a time, there were two friends, {first} and {second}. " if easy else ""
        facts = f"{opening}{first} and {second} went for a walk. {finder} found a big stick."
        stem = "The one who found the stick was a little"
        choices = ["boy", "girl"]
    else:
        kids = rng.sample([(x, "girl") for x in GIRLS] + [(x, "boy") for x in BOYS], n_kids)
        t = rng.randrange(n_kids)
        name, gender = kids[t]
        names = [k for k, _ in kids]
        opening = ""
        if easy:
            opening = f"Once upon a time, there were two friends, {names[0]} and {names[1]}. "
        if cat in ("name", "object"):
            objs = rng.sample(OBJECTS, n_kids)
            body = " ".join(f"{k} had {a_an(o)} {o}." for k, o in zip(names, objs))
            if cat == "name":
                stem, choices, answer = f"The {objs[t]} belonged to", names, name
            else:
                stem, choices, answer = f"Then {name} picked up {his_her(gender)}", objs, objs[t]
        elif cat == "time":
            times = rng.sample(TIMES, n_kids)
            body = " ".join(f"{k} took a walk every {x}." for k, x in zip(names, times))
            stem, choices, answer = f"{name} took a walk every", times, times[t]
        elif cat == "place":
            places = rng.sample(PLACES, n_kids)
            body = " ".join(f"{k} went to the {p}." for k, p in zip(names, places))
            stem, choices, answer = f"{name} had a lot of fun at the", places, places[t]
        facts = opening + body
    story = facts + (" " + " ".join(rng.sample(FILLERS, distance)) if distance else "")
    gold = i % len(choices)  # balance the correct answer's position
    order = choices[:]
    rng.shuffle(order)
    order.remove(answer)
    order.insert(gold, answer)
    prefix = "easy-" if easy else ""
    return {
        "story_id": f"{prefix}{cat}-{i:03d}", "category": cat, "variant": "easy" if easy else "standard",
        "distance": distance, "context": f"{story} {stem}", "stem": stem,
        "choices": order, "gold": gold,
    }


def main() -> None:
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    for easy in (False, True):
        rng = random.Random(SEED + easy)
        for cat in CATEGORIES:
            items = [make_item(cat, i, rng, easy) for i in range(N_PER_CATEGORY)]
            name = f"storymemory_{'easy_' if easy else ''}{cat}.jsonl"
            with open(os.path.join(HERE, "data", name), "w") as f:
                for it in items:
                    f.write(json.dumps(it) + "\n")
            print(f"{name}: {len(items)} items")


if __name__ == "__main__":
    main()
