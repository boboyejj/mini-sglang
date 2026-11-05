"""Quality checks for StoryMemory items. Run: python benchmark_storymemory/check.py

1. every choice is exactly 1 token (with a leading space) for both models' tokenizers
2. choices are distinct, and (except 'gender') every choice appears in the story
3. the correct answer's position and the distances are balanced
4. stories are unique, and their fact sentences don't appear verbatim in the training text
5. grammar spot-check: no "a" before a vowel
"""

import collections
import statistics
import glob
import json
import re

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer


def main() -> None:
    toks = {"tiny": AutoTokenizer.from_pretrained("training/tiny-qwen3"),
            "qwen": AutoTokenizer.from_pretrained(snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True))}
    rows = []
    for f in sorted(glob.glob("benchmark_storymemory/data/storymemory_*.jsonl")):
        rows += [json.loads(line) for line in open(f)]
    problems = []
    for r in rows:
        story = r["context"][: -len(r["stem"])]
        for c in r["choices"]:
            for name, t in toks.items():
                if len(t(" " + c, add_special_tokens=False).input_ids) != 1:
                    problems.append(f"{r['story_id']}: '{c}' is not 1 token for {name}")
            if r["category"] != "gender" and not re.search(rf"\b{c}\b", story):
                problems.append(f"{r['story_id']}: choice '{c}' not in story")
        if len(set(r["choices"])) != len(r["choices"]):
            problems.append(f"{r['story_id']}: duplicate choices")
        if re.search(r"\ba [aeiou]", r["context"]):
            problems.append(f"{r['story_id']}: 'a' before a vowel")
    print(f"items: {len(rows)}; problems found: {len(problems)}")
    for p in problems[:10]:
        print("  ", p)

    for cat in ["name", "object", "time", "place", "gender"]:
        rs = [r for r in rows if r["category"] == cat]
        gold = collections.Counter(r["gold"] for r in rs)
        dist = collections.Counter(r["distance"] for r in rs)
        print(f"{cat:<7} correct-answer positions {dict(sorted(gold.items()))}"
              f"  distances {dict(sorted(dist.items()))}"
              f"  unique stories {len({r['context'] for r in rs})}/{len(rs)}")

    print("context length (tiny tokenizer) by distance:")
    for d in sorted({r["distance"] for r in rows}):
        n = [len(toks["tiny"](r["context"]).input_ids) for r in rows if r["distance"] == d]
        print(f"  distance {d:>2}: avg {statistics.mean(n):.0f}, max {max(n)} tokens")
    vocab = {w.lower() for r in rows for w in r["choices"]} | {"park", "beach", "zoo", "farm", "lake",
             "shop", "forest", "school", "garden", "river", "noon", "night"}
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen", "benchmark_storymemory/generate.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    bad = [f for f in gen.FILLERS if set(re.findall(r"[a-z]+", f.lower())) & (vocab | {n.lower() for n in gen.GIRLS + gen.BOYS} | set(gen.OBJECTS))]
    print(f"fillers: {len(gen.FILLERS)} distinct; containing a name/object/place/time word: {bad or 'none'}")

    path = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt",
                           repo_type="dataset", local_files_only=True)
    with open(path, encoding="utf-8") as f:
        train = f.read(100_000_000)
    facts = {s for r in rows for s in re.split(r"(?<=\.) ", r["context"])[:3] if " had " in s or " went to " in s or " found " in s}
    hits = [s for s in facts if s in train]
    print(f"contamination: {len(hits)} of {len(facts)} distinct fact sentences appear verbatim in the"
          f" training slice{': ' + '; '.join(hits[:5]) if hits else ''}")


if __name__ == "__main__":
    main()
