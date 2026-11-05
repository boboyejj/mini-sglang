"""Reproduce lm-evaluation-harness's BLiMP score by hand, and look inside one question.

For each minimal pair, the harness asks: which sentence does the model find more probable?
  score(sentence) = sum of log P(token | all tokens before it), over the sentence's tokens
The context is empty, so the sentence is conditioned on one "start" token (the tokenizer's
BOS token, or its EOS token if it has no BOS). Correct = score(good) > score(bad).

Run:  python evaluation/blimp_by_hand.py
Reads the harness's saved answers in evaluation/results/ and checks they match.
"""

import glob
import json
import sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "tiny-qwen3": "training/tiny-qwen3",
    "qwen3-0.6b": snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True),
}
CHECK_TASKS = ["regular_plural_subject_verb_agreement_1", "principle_A_case_1", "npi_present_1"]


@torch.no_grad()
def sentence_logprob(model, tok, start_id: int, text: str, show: bool = False) -> float:
    ids = tok(text, add_special_tokens=False).input_ids
    x = torch.tensor([[start_id] + ids])
    logp = model(x).logits[0, :-1].log_softmax(-1)  # position i predicts token i+1
    per_token = logp[torch.arange(len(ids)), torch.tensor(ids)]
    if show:
        for t, lp in zip(ids, per_token):
            print(f"      {tok.decode([t])!r:>16}  log P = {lp.item():7.2f}")
    return per_token.sum().item()


def main() -> None:
    for name, path in MODELS.items():
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        start_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        print(f"\n######## {name}: start token = {tok.decode([start_id])!r}")

        for task in CHECK_TASKS:
            f = glob.glob(f"evaluation/results/{name}/*/samples_blimp_{task}_*.jsonl")[0]
            rows = [json.loads(line) for line in open(f)]
            correct = max_diff = 0.0
            for i, r in enumerate(rows):
                good, bad = (r["arguments"][k]["arg_1"] for k in ("gen_args_0", "gen_args_1"))
                show = i == 0
                if show:
                    print(f"\n  [{task}] question 0")
                    print(f"    good: {good!r}")
                lg = sentence_logprob(model, tok, start_id, good, show)
                if show:
                    print(f"    bad:  {bad!r}")
                lb = sentence_logprob(model, tok, start_id, bad, show)
                harness = [float(r["resps"][j][0][0]) for j in (0, 1)]
                max_diff = max(max_diff, abs(lg - harness[0]), abs(lb - harness[1]))
                correct += lg > lb
                if show:
                    verdict = "correct" if lg > lb else "WRONG"
                    print(f"    total: good {lg:.2f} vs bad {lb:.2f} -> {verdict}"
                          f"   (harness: {harness[0]:.2f} vs {harness[1]:.2f})")
            print(f"  [{task}] by hand: {correct / len(rows):.0%} correct of {len(rows)}"
                  f" | harness: {sum(r['acc'] for r in rows) / len(rows):.0%}"
                  f" | largest log-prob difference from harness: {max_diff:.4f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
