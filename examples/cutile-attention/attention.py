"""Naive cuTile scaled dot-product attention — the function under optimization.

Layout: q is [B, H_q, L, D]; k and v are [B, H_kv, S, D] with H_q % H_kv == 0
(grouped-query attention). Output is [B, H_q, L, D] in q's dtype.

This baseline follows the TileGym production pattern from:
/home/marcelo/TileGym/src/tilegym/ops/cutile/attention.py
"""

import cuda.tile as ct
import torch
import math
from cuda.tile import RoundingMode as RMd

# Tile sizes (powers of 2) - matching TileGym autotune for SM121
# TILE_D will be computed as next_power_of_2(D) at runtime
TILE_M = 64   # query sequence chunk
TILE_N = 64   # key/value sequence chunk  

INV_LOG_2 = 1.0 / math.log(2)  # Python float constant

def next_power_of_2(x: int) -> int:
    """Return the next power of 2 >= x."""
    return 1 << (x - 1).bit_length()

@ct.kernel
def attention_kernel(
    q: ct.Array,           # [B, H_q, L, D]
    k: ct.Array,           # [B, H_kv, S, D]
    v: ct.Array,           # [B, H_kv, S, D]
    out: ct.Array,         # [B, H_q, L, D]
    scale: ct.Constant[float],
    causal: ct.Constant[bool],
    B: ct.Constant[int],
    H_q: ct.Constant[int],
    H_kv: ct.Constant[int],
    L: ct.Constant[int],
    S: ct.Constant[int],
    D: ct.Constant[int],
    tile_m: ct.Constant[int],  # Compile-time TILE_M
    tile_n: ct.Constant[int],  # Compile-time TILE_N
    tile_d: ct.Constant[int],  # Compile-time TILE_D (next_power_of_2(D))
):
    # Map block IDs: bid_x -> query chunk (L // tile_m), bid_y -> (batch * H_q + head)
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    
    batch_idx = bid_y // H_q
    head_idx = bid_y % H_q
    kv_head_idx = head_idx // (H_q // H_kv)
    
    # Compute qk_scale for exp2
    qk_scale = scale * INV_LOG_2
    
    # Initialize offsets for current query tile (M-dimension)
    offs_m = bid_x * tile_m + ct.arange(tile_m, dtype=ct.int32)  # [tile_m]
    offs_m = offs_m[:, None]  # [tile_m, 1]
    
    # Initialize local offsets for key/value tile (N-dimension)
    offs_n_tile = ct.arange(tile_n, dtype=ct.int32)  # [tile_n]
    offs_n_tile = offs_n_tile[None, :]  # [1, tile_n]
    
    # Initialize online softmax accumulators in float32 for stability
    m_i = ct.full((tile_m, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((tile_m, 1), 0.0, dtype=ct.float32)
    acc = ct.full((tile_m, tile_d), 0.0, dtype=ct.float32)
    
    # Load query tile for this batch, head, and M-chunk
    q_tile = ct.load(
        q,
        index=(batch_idx, head_idx, bid_x, 0),
        shape=(1, 1, tile_m, tile_d),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((tile_m, tile_d))  # [tile_m, tile_d]
    
    # Loop over K, V blocks (N-dimension chunks)
    num_kv_tiles = ct.cdiv(S, tile_n)
    for j in range(0, num_kv_tiles):
        k_tile = ct.load(
            k,
            index=(batch_idx, kv_head_idx, 0, j),
            shape=(1, 1, tile_d, tile_n),
            order=(0, 1, 3, 2),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((tile_d, tile_n))  # [tile_d, tile_n]
        
        # Q @ K^T: [tile_m, tile_d] @ [tile_d, tile_n] = [tile_m, tile_n]
        qk = ct.full((tile_m, tile_n), 0.0, dtype=ct.float32)
        qk = ct.mma(q_tile, k_tile, qk)
        
        # Mask out-of-bounds N positions (S < tile_n)
        oob_n = offs_n_tile >= S
        oob_mask_n = ct.full((tile_m, tile_n), -math.inf, dtype=ct.float32)
        oob_mask_n = ct.where(oob_n, -math.inf, 0.0)
        qk = qk + oob_mask_n
        
        # Causal mask - only apply for partial tiles or when causal
        if causal:
            offs_n = j * tile_n + offs_n_tile
            mask = ct.full((tile_m, tile_n), True, dtype=ct.bool_)
            # Out of bounds mask for partial tiles
            if (j + 1) * tile_n > S:
                mask = mask & (offs_n < S)
            # Causal mask: query i sees keys 0..S-L+i
            mask = mask & (offs_m >= offs_n)
            mask = ct.where(mask, 0.0, -math.inf)
            qk = qk + mask
        
        # Online softmax - EXACT TileGym pattern
        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij  # [tile_m, tile_n]
        
        # Attention weights
        p = ct.exp2(qk, flush_to_zero=True)  # [tile_m, tile_n]
        l_ij = ct.sum(p, axis=-1, keepdims=True)  # [tile_m, 1]
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # [tile_m, 1]
        
        # Update m_i and l_i
        l_i = l_i * alpha + l_ij  # [tile_m, 1]
        
        # Scale acc
        acc = acc * alpha  # [tile_m, tile_d]
        
        # Load V tile
        v_tile = ct.load(
            v,
            index=(batch_idx, kv_head_idx, j, 0),
            shape=(1, 1, tile_n, tile_d),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((tile_n, tile_d))  # [tile_n, tile_d]
        
        # P @ V: cast p to input dtype, then MMA
        p = p.astype(q_tile.dtype)
        acc = ct.mma(p, v_tile, acc)  # [tile_m, tile_d]
        m_i = m_ij  # [tile_m, 1]
    
    # Final normalize
    out_tile = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    out_tile = out_tile.astype(q_tile.dtype)
    
    # Store output (slice to actual D)
    ct.store(
        out,
        index=(batch_idx, head_idx, bid_x, 0),
        tile=out_tile.reshape((1, 1, tile_m, tile_d)),
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
    
    # TILE_D must be power of 2 >= D (like TileGym)
    TILE_D = next_power_of_2(D)
    
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
            ct.Constant[int](TILE_M),
            ct.Constant[int](TILE_N),
            ct.Constant[int](TILE_D),
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