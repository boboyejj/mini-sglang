"""Step 3a: generate story continuations for LLM-as-judge grading, blinded.

Takes the first N_PROMPTS openings from TinyStories' "Evaluation prompts.yaml" and lets
both models continue each one (same sampling settings, fixed seed). The 2 x N_PROMPTS
continuations are shuffled and saved under anonymous IDs in stories.json; which model wrote
which story is saved separately in key.json, to be opened only after grading.

Run: python evaluation/judge/generate.py
"""

import json
import os
import random

import torch
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
N_PROMPTS = 12
MAX_NEW_TOKENS = 150
MODELS = {
    "tiny-qwen3": "training/tiny-qwen3",
    "qwen3-0.6b": snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True),
}


@torch.no_grad()
def main() -> None:
    path = hf_hub_download("roneneldan/TinyStories", "Evaluation prompts.yaml", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)[:N_PROMPTS]

    stories = []
    for name, mpath in MODELS.items():
        tok = AutoTokenizer.from_pretrained(mpath)
        model = AutoModelForCausalLM.from_pretrained(mpath, dtype=torch.float32).eval()
        for i, p in enumerate(prompts):
            torch.manual_seed(1000 + i)  # same seed per prompt for both models
            ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
            out = model.generate(
                ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.8, top_k=50,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
            )
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            stories.append({"prompt_id": i, "model": name, "prompt": p, "continuation": text})
            print(f"{name} prompt {i}: {len(out[0]) - ids.shape[1]} tokens", flush=True)

    random.Random(42).shuffle(stories)
    blinded, key = [], {}
    for n, s in enumerate(stories, 1):
        sid = f"S{n:02d}"
        blinded.append({"id": sid, "prompt": s["prompt"], "continuation": s["continuation"]})
        key[sid] = {"model": s["model"], "prompt_id": s["prompt_id"]}
    with open(os.path.join(HERE, "stories.json"), "w") as f:
        json.dump(blinded, f, indent=1, ensure_ascii=False)
    with open(os.path.join(HERE, "key.json"), "w") as f:
        json.dump(key, f, indent=1)
    print(f"saved {len(blinded)} blinded stories; key in key.json (not opened before grading)")


if __name__ == "__main__":
    main()
