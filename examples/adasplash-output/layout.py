"""Host-side dtype / shape / layout plumbing shared by the forward host wrappers.

Everything here ran identically (copy-pasted) in all three ``get_*`` wrappers
before the restructure: the torch->cutlass dtype map, the BSHD view, the
``varlen``-to-``seqused`` conversion, the zero-vs-empty allocator choice, and the
two dimension formulas (``select_bins`` from kernel A, ``mask_dims`` from B).

Pure host Python -- no ``cute.jit``, nothing traced.
"""

from typing import NamedTuple, Optional

import torch

import cutlass


TORCH_TO_CUTLASS = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


def cutlass_dtype(t: torch.Tensor):
    """Map a torch dtype to its cutlass counterpart, naming the constraint on failure."""
    try:
        return TORCH_TO_CUTLASS[t.dtype]
    except KeyError:
        raise TypeError(
            f"AdaSplash forward supports bfloat16 and float16 Q/K/V; got {t.dtype}. "
            "fp32 inputs must be cast by the caller (the wgmma path has no fp32 MMA)."
        ) from None


class QkvDims(NamedTuple):
    batch: int
    num_heads: int
    n_ctx: int
    head_dim: int
    num_kv_heads: int
    qhead_per_kvhead: int


def qkv_dims(q: torch.Tensor, k: torch.Tensor) -> QkvDims:
    """Unpack ``(B, N_H, N_CTX, H_DIM)`` + K's head count into named fields."""
    batch, num_heads, n_ctx, head_dim = q.shape
    num_kv_heads = k.shape[1]
    return QkvDims(batch, num_heads, n_ctx, head_dim,
                   num_kv_heads, num_heads // num_kv_heads)


def default_sm_scale(head_dim: int) -> float:
    return 1.0 / (head_dim ** 0.5)


def bshd(*tensors: torch.Tensor):
    """``(b, h, s, d) -> (b, s, h, d)`` views for the kernels.

    The kernels (like all of ``flash_attn/cute``) index Q/K/V as (b, s, h, d).
    ``transpose(1, 2)`` is a pure view and keeps d innermost + contiguous, which
    is all TMA requires.
    """
    return tuple(t.transpose(1, 2) for t in tensors)


def seqused_from_varlen(varlen: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Dense layout + per-batch valid lengths == "seqused", not "cu_seqlens"."""
    return varlen.to(torch.int32) if varlen is not None else None


def output_allocator(varlen: Optional[torch.Tensor]):
    """``torch.empty`` when dense, ``torch.zeros`` under varlen.

    Dense: every output row is written by some tile, so empty is safe.  varlen
    skips whole OOB tiles, so those rows must be zero-filled to stay defined.
    """
    return torch.empty if varlen is None else torch.zeros


def select_bins(n_ctx: int, tile_n: int = 128) -> int:
    """Largest BINS in {8, 4, 2} whose 16-bit counter cannot overflow.

    A thread owns ``tile_n/4`` columns of a row per K-block, so its private
    count for one bin is at most ``(tile_n/4) * cdiv(n_ctx, tile_n)`` ~=
    ``n_ctx/4``.  Unlike the Triton kernel -- where the counter width was
    ``64/BINS`` bits and so traded directly against BINS, forcing h=1/4 at 16K --
    the width here is a fixed 16 bits regardless of BINS.  So this returns 8 for
    every sequence length that fits in memory; the loop is kept only so that the
    failure would be explicit rather than silent.
    """
    max_count = (n_ctx + 3) // 4
    for bins in (8, 4, 2):
        if max_count <= (1 << 16) - 1:
            return bins
    raise ValueError(f"N_CTX={n_ctx} too large for the 16-bit packed histogram")


def mask_dims(n_ctx: int):
    """(M64, W32) for a given key/query length: M64 = 2*ceil(N/128) 64-row
    blocks (two per 128-row tile, one per warpgroup -- NOT ceil(N/64), so the
    last odd block exists even when N % 128 <= 64 and is stored all-zero),
    W32 = 4*ceil(N/128) little-endian u32 words per block row."""
    n_tiles = (n_ctx + 127) // 128
    return 2 * n_tiles, 4 * n_tiles
