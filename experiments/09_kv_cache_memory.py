"""Experiment 9: what the KV cache costs in memory, and how mini-sglang sizes its pool.

Concept: every token stores one K and one V vector per layer per KV head. We measure the
real cache tensors after a prefill, check them against the formula mini-sglang uses in
Engine._determine_num_pages, then estimate how many tokens a GPU could hold.

Run:  python experiments/09_kv_cache_memory.py
Try:  change GPU_GB, or count what a model with 4x the layers would need.
"""

import _model as M

GPU_GB = 24  # e.g. an RTX 4090 / L4-class GPU
MEMORY_RATIO = 0.9  # mini-sglang's default --memory-ratio
SERVING_BYTES = 2  # mini-sglang serves in bf16 (2 bytes); this CPU run uses float32 (4)


def main() -> None:
    tok, model = M.load()
    cfg = model.config
    ids = M.chat_ids(tok, "Count from one to ten.")
    res = M.generate(model, ids, max_new_tokens=1)  # prefill only: cache holds the prompt
    cache = res.cache

    n_tokens = cache.get_seq_length()
    k0 = cache.layers[0].keys
    print(f"\nprompt tokens in cache: {n_tokens}")
    print(f"layers in cache:        {len(cache.layers)}")
    print(f"layer 0 K shape:        {tuple(k0.shape)}  = (batch, kv_heads, tokens, head_dim)")

    measured = sum(layer.keys.nbytes + layer.values.nbytes for layer in cache.layers)
    per_token_measured = measured / n_tokens
    # the same formula as mini-sglang's Engine._determine_num_pages (page_size = 1)
    per_token_formula = 2 * cfg.head_dim * cfg.num_key_value_heads * 4 * cfg.num_hidden_layers
    print(f"\nbytes per token, measured (float32):   {per_token_measured:,.0f}")
    print(
        f"bytes per token, formula  (float32):   {per_token_formula:,}"
        f"  = 2 (K,V) x {cfg.head_dim} head_dim x {cfg.num_key_value_heads} kv_heads"
        f" x 4 bytes x {cfg.num_hidden_layers} layers"
    )

    per_token_bf16 = per_token_formula // 4 * SERVING_BYTES
    weights = sum(p.numel() for p in model.parameters()) * SERVING_BYTES
    pool = GPU_GB * 1e9 * MEMORY_RATIO - weights
    print(f"\nserving in bf16 on a {GPU_GB} GB GPU (memory_ratio={MEMORY_RATIO}):")
    print(f"  weights:          {weights / 1e9:5.2f} GB")
    print(f"  KV per token:     {per_token_bf16 / 1024:5.0f} KiB")
    print(f"  KV pool:          {pool / 1e9:5.2f} GB  ->  ~{pool / per_token_bf16:,.0f} tokens")
    print(f"  = ~{pool / per_token_bf16 / 4096:,.0f} requests of 4,096 tokens at once")
    print(
        "\nQuestion: Qwen3 has 16 query heads but only 8 KV heads (grouped-query attention)."
        "\nHow many tokens would fit if every query head had its own K/V?"
    )


if __name__ == "__main__":
    main()
