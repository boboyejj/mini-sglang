"""HellaSwag by hand: reproduce lm-eval's acc and acc_norm, then measure ending-length effects.

For each item: context = "<activity>: <scene start>", 4 candidate endings, 1 correct.
  score_i     = log P(ending_i | context) = sum over the ending's tokens (context not scored)
  acc         : correct if argmax score_i is the gold ending
  acc_norm    : correct if argmax score_i / len(ending_i in characters) is the gold ending

Part 1 recomputes the first ITEMS items with each model and compares with the harness.
Part 2 uses the harness's saved scores for all 300 items (no model runs).
Run: python evaluation/hellaswag_by_hand.py
"""

import glob
import json

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "tiny-qwen3": "training/tiny-qwen3",
    "qwen3-0.6b": snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True),
}
ITEMS = 10


@torch.no_grad()
def ending_logprob(model, tok, context: str, ending: str) -> float:
    ctx = tok(context, add_special_tokens=False).input_ids
    whole = tok(context + ending, add_special_tokens=False).input_ids
    cont = whole[len(ctx):]  # the ending's tokens, as tokenized after the context
    logp = model(torch.tensor([whole])).logits[0, :-1].log_softmax(-1)
    pos = torch.arange(len(ctx) - 1, len(whole) - 1)  # positions that predict the ending
    return logp[pos, torch.tensor(cont)].sum().item()


def main() -> None:
    for name, path in MODELS.items():
        f = glob.glob(f"evaluation/results/{name}/*/samples_hellaswag_*.jsonl")[0]
        rows = [json.loads(line) for line in open(f)]
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        print(f"\n######## {name}")

        # Part 1: recompute and compare with the harness
        max_diff, agree = 0.0, 0
        for i, r in enumerate(rows[:ITEMS]):
            ctx = r["arguments"]["gen_args_0"]["arg_0"]
            ends = [r["arguments"][f"gen_args_{j}"]["arg_1"] for j in range(4)]
            mine = [ending_logprob(model, tok, ctx, e) for e in ends]
            harness = [float(r["resps"][j][0][0]) for j in range(4)]
            max_diff = max(max_diff, max(abs(a - b) for a, b in zip(mine, harness)))
            chars = [len(c) for c in r["doc"]["choices"]]
            my_acc = float(np.argmax(mine) == r["doc"]["gold"])
            my_norm = float(np.argmax(np.array(mine) / chars) == r["doc"]["gold"])
            agree += (my_acc == r["acc"]) and (my_norm == r["acc_norm"])
            if i == 0:
                print(f"  item 0  context: {ctx!r}")
                for j, e in enumerate(ends):
                    gold = "  <- correct" if j == r["doc"]["gold"] else ""
                    print(f"    {j}: {e!r:<45} log P {mine[j]:7.2f}  chars {chars[j]:>3}"
                          f"  per char {mine[j] / chars[j]:6.3f}{gold}")
                print(f"    acc picks {int(np.argmax(mine))}, acc_norm picks"
                      f" {int(np.argmax(np.array(mine) / chars))}")
        print(f"  first {ITEMS} items: largest log-prob difference from harness {max_diff:.4f};"
              f" acc and acc_norm agree with harness on {agree}/{ITEMS}")

        # Part 2: length effects over all saved items
        picks = {"acc": [], "acc_norm": []}
        gold_shortest = gold_longest = 0
        for r in rows:
            lps = np.array([float(r["resps"][j][0][0]) for j in range(4)])
            chars = np.array([len(c) for c in r["doc"]["choices"]])
            for m, s in [("acc", lps), ("acc_norm", lps / chars)]:
                picks[m].append(int(np.argmax(s)))
            gold_shortest += r["doc"]["gold"] == int(np.argmin(chars))
            gold_longest += r["doc"]["gold"] == int(np.argmax(chars))
        n = len(rows)
        for m in picks:
            shortest = sum(p == int(np.argmin([len(c) for c in r["doc"]["choices"]]))
                           for p, r in zip(picks[m], rows))
            longest = sum(p == int(np.argmax([len(c) for c in r["doc"]["choices"]]))
                          for p, r in zip(picks[m], rows))
            acc = sum(p == r["doc"]["gold"] for p, r in zip(picks[m], rows)) / n
            print(f"  {m:<8} accuracy {acc:6.1%} | picks the shortest ending {shortest / n:6.1%},"
                  f" the longest {longest / n:6.1%}  (random: 25%)")
        print(f"  the correct ending is the shortest in {gold_shortest / n:.1%} of items,"
              f" the longest in {gold_longest / n:.1%}")


if __name__ == "__main__":
    main()
