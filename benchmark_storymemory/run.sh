#!/usr/bin/env bash
# Run StoryMemory (with story, and the no-story baseline) on both models.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
QWEN=$(python -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-0.6B', local_files_only=True))")
for pair in "tiny-qwen3=training/tiny-qwen3" "qwen3-0.6b=$QWEN"; do
  name=${pair%%=*}; path=${pair#*=}
  echo "=== $name"
  lm_eval --model hf --model_args "pretrained=$path,dtype=float32" \
    --include_path benchmark_storymemory/tasks --tasks storymemory,storymemory_nostory,storymemory_easy,storymemory_easy_nostory \
    --device cpu --batch_size 16 --output_path "benchmark_storymemory/results/$name" --log_samples \
    2>&1 | grep -v -i warn | grep -E "^\|" || true
done
