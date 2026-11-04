"""Step 2: train a tiny Qwen3 on TinyStories (CPU), or run the sanity checks first.

The model uses the same architecture as Qwen3-0.6B (grouped-query attention, RoPE,
RMSNorm, SwiGLU, QK-norm), only much smaller, so mini-sglang's models/qwen3.py could serve it.

Run:
  python training/train.py --check     # sanity checks: initial loss + overfit one batch (~2 min)
  python training/train.py             # full training (~1 h); resumes from checkpoints/latest.pt

Outputs: checkpoints/latest.pt (every --ckpt-every steps), log.jsonl, samples.md,
and the final model in tiny-qwen3/ (Hugging Face format, loadable with from_pretrained).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = ["Once upon a time", "Tom found a small dog in the park."]


def make_config(vocab_size: int) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=4,
        num_attention_heads=4,  # 4 query heads ...
        num_key_value_heads=2,  # ... share 2 KV heads (grouped-query attention)
        head_dim=64,
        max_position_embeddings=512,
        tie_word_embeddings=True,  # input and output word tables share weights
    )


def batches(data: np.ndarray, batch: int, seq: int, rng: np.random.Generator):
    """Random windows of seq+1 tokens: inputs are the first seq, labels shifted by one."""
    starts = rng.integers(0, len(data) - seq - 1, size=batch)
    return torch.from_numpy(np.stack([data[s : s + seq + 1] for s in starts]).astype(np.int64))


def lr_at(step: int, args) -> float:
    """Linear warm-up, then cosine decay to 10% of the peak."""
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


@torch.no_grad()
def eval_loss(model, data, args, n_batches: int = 20) -> float:
    model.eval()
    rng = np.random.default_rng(1234)  # same validation windows every time
    losses = []
    for _ in range(n_batches):
        x = batches(data, args.batch, args.seq, rng)
        losses.append(model(input_ids=x[:, :-1], labels=x[:, :-1]).loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample(model, tok, prompt: str, max_new_tokens: int = 120) -> str:
    model.eval()
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8, top_k=50,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id,
    )
    model.train()
    return tok.decode(out[0], skip_special_tokens=True)


def sanity_checks(model, tok, train, args) -> None:
    print(f"expected initial loss for random guessing: ln({len(tok)}) = {math.log(len(tok)):.2f}")
    x = batches(train, args.batch, args.seq, np.random.default_rng(0))
    with torch.no_grad():
        print(f"measured initial loss:                       {model(input_ids=x, labels=x).loss.item():.2f}")
    print("\noverfit one batch (the loss should fall toward 0 if the training loop is correct):")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    t0 = time.time()
    for step in range(1, 201):
        loss = model(input_ids=x, labels=x).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step in (1, 25, 50, 100, 150, 200):
            print(f"  step {step:>3}  loss {loss.item():.3f}   ({time.time() - t0:.0f} s)")
    print("PASS" if loss.item() < 0.5 else "FAIL: the model could not memorize one batch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run sanity checks and exit")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=500)
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = PreTrainedTokenizerFast.from_pretrained(os.path.join(HERE, "tokenizer"))
    train = np.fromfile(os.path.join(HERE, "data", "train.bin"), dtype=np.uint16)
    val = np.fromfile(os.path.join(HERE, "data", "val.bin"), dtype=np.uint16)
    model = Qwen3ForCausalLM(make_config(len(tok)))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"tiny Qwen3: {n_params / 1e6:.1f}M parameters | train {len(train):,} tokens | val {len(val):,} tokens")
    print(f"torch threads: {torch.get_num_threads()}\n")

    if args.check:
        sanity_checks(model, tok, train, args)
        return

    ckpt_dir = os.path.join(HERE, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    latest = os.path.join(ckpt_dir, "latest.pt")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    start, elapsed = 0, 0.0
    if os.path.exists(latest):  # resume after an interruption
        ck = torch.load(latest)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, elapsed = ck["step"], ck["elapsed"]
        print(f"resumed from step {start}")
    rng = np.random.default_rng(start)

    log = open(os.path.join(HERE, "log.jsonl"), "a")
    samples = open(os.path.join(HERE, "samples.md"), "a")
    t_last, window = time.time(), []
    for step in range(start, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)
        x = batches(train, args.batch, args.seq, rng)
        loss = model(input_ids=x[:, :-1], labels=x[:, :-1]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        window.append(loss.item())

        done = step + 1
        if done % args.log_every == 0:
            now = time.time()
            elapsed += now - t_last
            t_last = now
            rec = {"step": done, "train_loss": round(float(np.mean(window)), 4),
                   "lr": round(lr_at(step, args), 6), "minutes": round(elapsed / 60, 1),
                   "tokens_seen": done * args.batch * args.seq}
            window = []
            if done % args.eval_every == 0 or done == args.steps:
                rec["val_loss"] = round(eval_loss(model, val, args), 4)
                samples.write(f"\n## step {done} (val loss {rec['val_loss']})\n\n")
                for p in PROMPTS:
                    samples.write(f"- {sample(model, tok, p)!r}\n")
                samples.flush()
                t_last = time.time()  # don't count eval time as training time
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
        if done % args.ckpt_every == 0 or done == args.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": done, "elapsed": elapsed}, latest)

    out = os.path.join(HERE, "tiny-qwen3")
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"\nsaved final model to {out}")


if __name__ == "__main__":
    main()
