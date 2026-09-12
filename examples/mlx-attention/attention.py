"""Scaled dot-product attention in MLX — the function under optimization.

Layout: q is [B, H_q, L, D]; k and v are [B, H_kv, S, D] with H_q % H_kv == 0
(grouped-query attention). Output is [B, H_q, L, D] in q's dtype.

This baseline is deliberately naive: it materializes the repeated K/V heads, the full
[L, S] score matrix, and the mask. Semantics match
``mx.fast.scaled_dot_product_attention`` — which is the reference in the tests and is
off-limits inside this function.
"""

from __future__ import annotations

import mlx.core as mx


def attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float | None = None,
    causal: bool = False,
) -> mx.array:
    B, H_q, L, D = q.shape
    _, H_kv, S, _ = k.shape
    assert H_q % H_kv == 0, "query heads must be a multiple of key/value heads"
    scale = D**-0.5 if scale is None else scale

    if H_kv != H_q:
        k = mx.repeat(k, H_q // H_kv, axis=1)
        v = mx.repeat(v, H_q // H_kv, axis=1)

    scores = (q * scale) @ k.transpose(0, 1, 3, 2)
    if causal:
        # Query i (aligned to the end of the key sequence) may see keys 0..S-L+i.
        mask = mx.tril(mx.ones((L, S), dtype=mx.bool_), k=S - L)
        scores = mx.where(mask, scores, mx.array(-mx.inf, dtype=scores.dtype))
    probs = mx.softmax(scores, axis=-1, precise=True)
    return probs @ v
