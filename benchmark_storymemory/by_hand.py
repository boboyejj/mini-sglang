"""Reproduce lm-eval's StoryMemory scores by hand for a sample of items.

score(choice) = log P(" " + choice | context); all choices are one token, so this is the
log-probability of a single next token. The highest-scoring choice is the model's answer.
Run: python benchmark_storymemory/by_hand.py
"""

import glob
import json

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {"tiny-qwen3": "training/tiny-qwen3",
          "qwen3-0.6b": snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)}


@torch.no_grad()
def main() -> None:
    for name, path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        rows = []
        for f in sorted(glob.glob(f"benchmark_storymemory/results/{name}/*/samples_storymemory_*.jsonl")):
            rows += [json.loads(line) for line in open(f)][:4]  # 4 items from each of the 20 tasks
        max_diff, agree = 0.0, 0
        for r in rows:
            ctx = r["arguments"]["gen_args_0"]["arg_0"]
            ids = tok(ctx, return_tensors="pt", add_special_tokens=False).input_ids
            logp = model(ids).logits[0, -1].log_softmax(-1)
            mine = [logp[tok(" " + c, add_special_tokens=False).input_ids[0]].item() for c in r["doc"]["choices"]]
            harness = [float(x[0]) for x in r["filtered_resps"]]
            max_diff = max(max_diff, max(abs(a - b) for a, b in zip(mine, harness)))
            agree += float(mine.index(max(mine)) == r["doc"]["gold"]) == r["acc"]
        print(f"{name}: {len(rows)} items | largest log-prob difference from harness {max_diff:.4f}"
              f" | same right/wrong verdict on {agree}/{len(rows)}")


if __name__ == "__main__":
    main()
