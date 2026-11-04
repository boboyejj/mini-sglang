"""Step 1: download TinyStories, train a small tokenizer, and turn the text into token IDs.

Outputs (in training/):
  tokenizer/        an 8,192-token byte-level BPE tokenizer, saved in Hugging Face format
  data/train.bin    training token IDs (uint16)
  data/val.bin      validation token IDs from the official held-out file (uint16)

Run:  python training/prepare_data.py
"""

import os
import time

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "roneneldan/TinyStories"
TRAIN_FILE, VALID_FILE = "TinyStoriesV2-GPT4-train.txt", "TinyStoriesV2-GPT4-valid.txt"
TRAIN_BYTES = 100_000_000  # use the first ~100 MB of the 2.2 GB training file
VOCAB_SIZE = 8192
EOS = "<|endoftext|>"  # TinyStories separates stories with this marker


def read_stories(path: str, max_bytes: int | None = None) -> list[str]:
    with open(path, encoding="utf-8") as f:
        text = f.read(max_bytes) if max_bytes else f.read()
    stories = [s.strip() for s in text.split(EOS)]
    if max_bytes:
        stories = stories[:-1]  # the last one was cut off mid-story
    return [s for s in stories if s]


def main() -> None:
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    t0 = time.time()
    print("downloading TinyStories (2.2 GB train + 22 MB validation)...")
    train_path = hf_hub_download(REPO, TRAIN_FILE, repo_type="dataset")
    valid_path = hf_hub_download(REPO, VALID_FILE, repo_type="dataset")
    print(f"  done in {time.time() - t0:.0f} s")

    train, valid = read_stories(train_path, TRAIN_BYTES), read_stories(valid_path)
    print(f"stories: {len(train):,} train (first {TRAIN_BYTES / 1e6:.0f} MB), {len(valid):,} validation")
    print(f"example story:\n---\n{train[0][:400]}\n---")

    print(f"training a {VOCAB_SIZE:,}-token BPE tokenizer...")
    t0 = time.time()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(train, trainer=trainer)
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token=EOS, bos_token=EOS)
    hf_tok.save_pretrained(os.path.join(HERE, "tokenizer"))
    print(f"  done in {time.time() - t0:.0f} s; example: {hf_tok.tokenize('Once upon a time, Lily saw a dog.')}")

    eos_id = hf_tok.convert_tokens_to_ids(EOS)
    for name, stories in [("train", train), ("val", valid)]:
        t0 = time.time()
        ids = []
        for enc in tok.encode_batch(stories):
            ids.extend(enc.ids)
            ids.append(eos_id)  # mark where each story ends
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(os.path.join(HERE, "data", f"{name}.bin"))
        print(f"{name}: {len(arr):,} tokens ({len(arr) / len(stories):.0f} per story) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
