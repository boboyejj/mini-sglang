"""Re-score BLiMP three ways from the harness's saved log-probabilities (no model runs).

  sum       : total log-probability (what BLiMP and the harness use)
  per token : total / number of tokens  (removes the "fewer tokens wins" bias)
  per char  : total / number of characters (what lm-eval calls acc_norm for other tasks)

Run: python evaluation/blimp_rescore.py
"""

import collections
import glob
import json

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

TOKENIZERS = {
    "tiny-qwen3": "training/tiny-qwen3",
    "qwen3-0.6b": snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True),
}
METHODS = ["sum", "per token", "per char"]


def main() -> None:
    for name, path in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(path)
        rows = []
        for f in glob.glob(f"evaluation/results/{name}/*/samples_blimp_*.jsonl"):
            rows += [json.loads(line) for line in open(f)]
        acc = collections.defaultdict(list)  # (method, bucket) -> [0/1]
        per_task = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            texts = [r["arguments"][k]["arg_1"] for k in ("gen_args_0", "gen_args_1")]
            lps = [float(r["resps"][j][0][0]) for j in (0, 1)]
            ntok = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
            d = ntok[0] - ntok[1]
            bucket = "good FEWER tokens" if d < 0 else "same tokens" if d == 0 else "good MORE tokens"
            scores = {
                "sum": lps,
                "per token": [lp / n for lp, n in zip(lps, ntok)],
                "per char": [lp / len(t) for lp, t in zip(lps, texts)],
            }
            for m, (g, b) in scores.items():
                ok = float(g > b)
                acc[(m, "all")].append(ok)
                acc[(m, bucket)].append(ok)
                per_task[m][r["doc"]["UID"]].append(ok)

        print(f"\n######## {name} ({len(rows)} pairs)")
        print(f"{'method':<10} {'overall':>8} {'good FEWER':>11} {'same':>8} {'good MORE':>10}"
              f" {'tasks<=30%':>11}")
        for m in METHODS:
            row = [sum(acc[(m, b)]) / len(acc[(m, b)])
                   for b in ["all", "good FEWER tokens", "same tokens", "good MORE tokens"]]
            low = sum(sum(v) / len(v) <= 0.3 for v in per_task[m].values())
            print(f"{m:<10} " + " ".join(f"{x:>{w}.1%}" for x, w in zip(row, [8, 11, 8, 10]))
                  + f" {low:>11}")


if __name__ == "__main__":
    main()
