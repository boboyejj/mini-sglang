"""Why does each model get BLiMP pairs right or wrong? Tests two non-grammar effects.

1. Token count: a sentence's score is a SUM of per-token log-probabilities, so a sentence
   that splits into more tokens tends to score lower, whatever its grammar.
2. Word frequency: a model may prefer a sentence because it contains a more familiar word.
   We count how often each pair's differing words appear in our TinyStories training slice.

Uses only the harness's saved answers (no model runs). Run: python evaluation/blimp_analysis.py
"""

import collections
import glob
import json
import re

from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

TOKENIZERS = {
    "tiny-qwen3": "training/tiny-qwen3",
    "qwen3-0.6b": snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True),
}


def load(name):
    rows = []
    for f in glob.glob(f"evaluation/results/{name}/*/samples_blimp_*.jsonl"):
        rows += [json.loads(line) for line in open(f)]
    return rows


def words(s):
    return re.findall(r"[a-z']+", s.lower())


def main() -> None:
    path = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-train.txt",
                           repo_type="dataset", local_files_only=True)
    with open(path, encoding="utf-8") as f:
        counts = collections.Counter(words(f.read(100_000_000)))
    total = sum(counts.values())
    print(f"TinyStories training slice: {total:,} words, {len(counts):,} distinct")
    for w in ["ever", "really", "references", "reference", "teenagers", "suzanne", "the"]:
        print(f"  {w!r:>13} appears {counts[w]:>9,} times")

    for name, tpath in TOKENIZERS.items():
        tok, rows = AutoTokenizer.from_pretrained(tpath), load(name)
        print(f"\n######## {name}: {len(rows)} pairs, accuracy {sum(r['acc'] for r in rows) / len(rows):.1%}")

        by_len = collections.defaultdict(list)
        for r in rows:
            g, b = (r["arguments"][k]["arg_1"] for k in ("gen_args_0", "gen_args_1"))
            d = len(tok(g, add_special_tokens=False).input_ids) - len(tok(b, add_special_tokens=False).input_ids)
            key = "good has FEWER tokens" if d < 0 else "same token count" if d == 0 else "good has MORE tokens"
            by_len[key].append(r["acc"])
        print("  1. token count of the grammatical vs ungrammatical sentence:")
        for key in ["good has FEWER tokens", "same token count", "good has MORE tokens"]:
            v = by_len[key]
            print(f"     {key:<22} {len(v):>5} pairs  accuracy {sum(v) / max(len(v), 1):6.1%}")

        by_freq = collections.defaultdict(list)
        for r in rows:
            g, b = words(r["doc"]["sentence_good"]), words(r["doc"]["sentence_bad"])
            only_g, only_b = set(g) - set(b), set(b) - set(g)
            if not only_g or not only_b:
                continue  # pairs that differ only in word order: no word-frequency contrast
            fg = min(counts[w] for w in only_g)
            fb = min(counts[w] for w in only_b)
            key = "good's words RARER" if fg < fb else "good's words MORE common" if fg > fb else "equal"
            by_freq[key].append(r["acc"])
        print("  2. TinyStories frequency of the words that differ (rarest word on each side):")
        for key in ["good's words RARER", "good's words MORE common"]:
            v = by_freq[key]
            print(f"     {key:<25} {len(v):>5} pairs  accuracy {sum(v) / max(len(v), 1):6.1%}")


if __name__ == "__main__":
    main()
