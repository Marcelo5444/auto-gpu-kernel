"""Naive cuTile scaled dot-product attention — the function under optimization.

Layout: q is [B, H_q, L, D]; k and v are [B, H_kv, S, D] with H_q % H_kv == 0
(grouped-query attention). Output is [B, H_q, L, D] in q's dtype.

This baseline is deliberately naive: it materializes the repeated K/V heads, the full
[L, S] score matrix in registers, and the mask. Semantics match
`torch.nn.functional.scaled_dot_product_attention` — which is the reference in the tests.
"""

import cuda.tile as ct
import torch
import math

# Tile sizes (powers of 2)
TILE_M = 64   # query sequence chunk
TILE_N = 64   # key/value sequence chunk  
TILE_D = 128  # head dimension (must match D, rounded up to power of 2)

# Pre-computed constants
INV_LOG_2 = ct.Constant[float](1.0 / math.log(2))
NEG_INF = ct.Constant[float](-math.inf)

@ct.kernel
def attention_kernel(
    q: ct.Array,      # [B, H_q, L, D]
    k: ct.Array,      # [B, H_kv, S, D]
    v: ct.Array,      # [B, H_kv, S, D]
    out: ct.Array,    # [B, H_q, L, D]
    scale: ct.Constant[float],
    causal: ct.Constant[bool],
    B: ct.Constant[int],
    H_q: ct.Constant[int],
    H_kv: ct.Constant[int],
    L: ct.Constant[int],
    S: ct.Constant[int],
    D: ct.Constant[int],
):
    # Map block IDs: bid_x -> query chunk (L // TILE_M), bid_y -> (batch * H_q + head)
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    
    batch_idx = bid_y // H_q
    head_idx = bid_y % H_q
    kv_head_idx = head_idx // (H_q // H_kv)
    
    # Query tile offset
    q_start = bid_x * TILE_M
    
    # Load query tile: [TILE_M, D]
    q_tile = ct.load(
        q,
        index=(batch_idx, head_idx, q_start, 0),
        shape=(1, 1, TILE_M, TILE_D),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((TILE_M, TILE_D))
    
    # Online softmax accumulators
    m_i = ct.full((TILE_M, 1), NEG_INF, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)
    
    # Scale for exp2 (TileGym uses exp2, so qk_scale = scale / log(2))
    qk_scale = scale * INV_LOG_2
    
    # Loop over K/V tiles
    num_kv_tiles = (S + TILE_N - 1) // TILE_N
    for kv_tile_idx in range(num_kv_tiles):
        kv_start = kv_tile_idx * TILE_N
        
        # Load K tile: [TILE_N, D] -> transpose to [D, TILE_N]
        k_tile = ct.load(
            k,
            index=(batch_idx, kv_head_idx, kv_start, 0),
            shape=(1, 1, TILE_N, TILE_D),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((TILE_N, TILE_D)).transpose()  # [D, TILE_N]
        
        # Load V tile: [TILE_N, D]
        v_tile = ct.load(
            v,
            index=(batch_idx, kv_head_idx, kv_start, 0),
            shape=(1, 1, TILE_N, TILE_D),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((TILE_N, TILE_D))
        
        # Q @ K^T: [TILE_M, D] @ [D, TILE_N] = [TILE_M, TILE_N] -- NO SCALE YET
        qk = ct.mma(
            q_tile.astype(ct.float16),
            k_tile.astype(ct.float16),
            ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32),
        )
        
        # Causal mask (before scaling) - EXACT TileGym pattern
        if causal:
            # Query i sees keys 0..S-L+i
            q_idx = ct.full((TILE_M, 1), q_start, dtype=ct.int32) + ct.arange(TILE_M, dtype=ct.int32)[:, None]
            kv_idx = ct.full((1, TILE_N), kv_start, dtype=ct.int32) + ct.arange(TILE_N, dtype=ct.int32)[None, :]
            mask = q_idx >= kv_idx  # [TILE_M, TILE_N]
            # TileGym: mask = ct.where(mask, 0.0, -math.inf); qk += mask
            mask_values = ct.where(mask, ct.full_like(qk, 0.0), NEG_INF)
            qk = qk + mask_values
        
        # Online softmax - EXACT pattern from TileGym (exp2 based)
        max_qk = ct.max(qk, axis=-1, keepdims=True)
        m_ij = ct.max(m_i, max_qk * qk_scale)  # Use ct.max for tile comparison
        qk = qk * qk_scale - m_ij
        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha
        acc = ct.mma(
            p.astype(ct.float16),
            v_tile.astype(ct.float16),
            acc,
        )
        
        m_i = m_ij
    
    # Final normalize
    out_tile = acc / l_i
    
    # Store output
    ct.store(
        out,
        index=(batch_idx, head_idx, q_start, 0),
        tile=out_tile.astype(ct.float16).reshape((1, 1, TILE_M, TILE_D)),
    )


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    B, H_q, L, D = q.shape
    _, H_kv, S, _ = k.shape
    assert H_q % H_kv == 0, "query heads must be a multiple of key/value heads"
    scale = D ** -0.5 if scale is None else scale
    
    out = torch.empty_like(q)
    
    # Grid: (L_chunks, B * H_q)
    grid_x = (L + TILE_M - 1) // TILE_M
    grid_y = B * H_q
    grid = (grid_x, grid_y, 1)
    
    stream = torch.cuda.current_stream().cuda_stream
    ct.launch(
        stream,
        grid,
        attention_kernel,
        (
            q, k, v, out,
            ct.Constant[float](scale),
            ct.Constant[bool](causal),
            ct.Constant[int](B),
            ct.Constant[int](H_q),
            ct.Constant[int](H_kv),
            ct.Constant[int](L),
            ct.Constant[int](S),
            ct.Constant[int](D),
        ),
    )
    torch.cuda.synchronize()
    return out


if __name__ == "__main__":
    # Validation test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--H", type=int, default=2)
    parser.add_argument("--L", type=int, default=32)
    parser.add_argument("--S", type=int, default=32)
    parser.add_argument("--D", type=int, default=32)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()
    
    torch.manual_seed(42)
    dtype = getattr(torch, args.dtype)
    
    q = torch.randn(args.B, args.H, args.L, args.D, dtype=dtype, device="cuda")
    k = torch.randn(args.B, args.H, args.S, args.D, dtype=dtype, device="cuda")
    v = torch.randn(args.B, args.H, args.S, args.D, dtype=dtype, device="cuda")
    
    # Reference
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=args.causal)
    
    # cuTile
    out = attention(q, k, v, causal=args.causal)
    
    # Validate
    is_close = torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3)
    if is_close:
        print("✓ Validation PASSED")
    else:
        max_diff = (out - ref_out).abs().max().item()
        print(f"✗ Validation FAILED - max diff: {max_diff}")