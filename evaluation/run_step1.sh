#!/usr/bin/env bash
# Step 1: run standard benchmarks with EleutherAI's lm-evaluation-harness on two models.
#   tiny-qwen3  - our 5.2M-parameter TinyStories model (training/tiny-qwen3)
#   qwen3-0.6b  - Qwen/Qwen3-0.6B (local copy; no Hugging Face API calls)
# LIMIT = number of questions per benchmark (BLiMP: per sub-task, 67 sub-tasks).
# --limit takes the FIRST N questions of each benchmark, so both models see the same ones.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

TASKS="blimp,lambada_openai,hellaswag,piqa,arc_easy"
LIMIT=${LIMIT:-300}
BLIMP_LIMIT=${BLIMP_LIMIT:-50}
QWEN=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-0.6B', local_files_only=True))")

run() {  # name, model path
  echo "=== $1: $(date +%H:%M:%S)"
  # BLiMP's 67 sub-tasks get a smaller per-task limit than the other benchmarks
  lm_eval --model hf --model_args "pretrained=$2,dtype=float32" \
    --tasks blimp --limit "$BLIMP_LIMIT" --device cpu --batch_size 16 \
    --output_path "evaluation/results/$1" --log_samples 2>&1 | grep -v -i warn | grep -E "^\|" | grep -E "blimp \||Tasks" || true
  lm_eval --model hf --model_args "pretrained=$2,dtype=float32" \
    --tasks lambada_openai,hellaswag,piqa,arc_easy --limit "$LIMIT" --device cpu --batch_size 8 \
    --output_path "evaluation/results/$1" --log_samples 2>&1 | grep -v -i warn | grep -E "^\|" || true
  echo "=== $1 done: $(date +%H:%M:%S)"
}

run tiny-qwen3 training/tiny-qwen3
run qwen3-0.6b "$QWEN"
