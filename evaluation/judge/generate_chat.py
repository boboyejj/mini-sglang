"""Step 3d: fair rematch for Qwen3-0.6B (chat format), and the blinded round-2 set.

Qwen3-0.6B is a chat model. Here it gets each opening inside its chat template with an
instruction to continue the story (thinking mode off, same seeds and sampling as round 1).
Round 2 combines 36 stories: tiny model + Qwen raw completion (from round 1) + Qwen chat,
shuffled under IDs T01-T36. Each of three judges gets its own story order.

Run: python evaluation/judge/generate_chat.py
"""

import json
import os
import random

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "round2")
INSTRUCTION = (
    "Continue this children's story. Write only the continuation, starting exactly where "
    "the text stops, and finish the story in a few short paragraphs.\n\n"
)


@torch.no_grad()
def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    r1 = {s["id"]: s for s in json.load(open(os.path.join(HERE, "stories.json")))}
    key1 = json.load(open(os.path.join(HERE, "key.json")))
    prompts = {v["prompt_id"]: r1[sid]["prompt"] for sid, v in key1.items()}
    stories = [
        {"model": v["model"] if v["model"] == "tiny-qwen3" else "qwen3-0.6b-raw",
         "prompt_id": v["prompt_id"], "prompt": r1[sid]["prompt"],
         "continuation": r1[sid]["continuation"], "round1_id": sid}
        for sid, v in key1.items()
    ]

    path = snapshot_download("Qwen/Qwen3-0.6B", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    for i in sorted(prompts):
        msgs = [{"role": "user", "content": INSTRUCTION + prompts[i]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                      return_tensors="pt")
        torch.manual_seed(1000 + i)
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=150,
                             do_sample=True, temperature=0.8, top_k=50)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        stories.append({"model": "qwen3-0.6b-chat", "prompt_id": i, "prompt": prompts[i],
                        "continuation": text, "round1_id": None})
        print(f"chat prompt {i}: {out.shape[1] - ids.shape[1]} tokens", flush=True)

    random.Random(7).shuffle(stories)
    key = {}
    blinded = []
    for n, s in enumerate(stories, 1):
        tid = f"T{n:02d}"
        key[tid] = {k: s[k] for k in ("model", "prompt_id", "round1_id")}
        blinded.append({"id": tid, "prompt": s["prompt"], "continuation": s["continuation"]})
    json.dump(key, open(os.path.join(OUT, "key.json"), "w"), indent=1)
    orders = {"A": blinded, "B": blinded[::-1], "C": random.Random(99).sample(blinded, len(blinded))}
    for judge, order in orders.items():
        json.dump(order, open(os.path.join(OUT, f"stories_judge{judge}.json"), "w"),
                  indent=1, ensure_ascii=False)
    print(f"saved {len(blinded)} stories in 3 orders; key in round2/key.json")


if __name__ == "__main__":
    main()
