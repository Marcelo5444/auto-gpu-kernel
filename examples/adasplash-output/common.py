"""Shared machinery for the AdaSplash forward harnesses.

Everything here was duplicated across the old per-kernel ``compare.py`` /
``benchmark.py`` pairs: the float64 sort-solver oracle, the fp32-semantics score
reference, mask packing/unpacking, the varlen row mask, the rotated-round timing
loop, and the argument parsing + shape sweep.

House rules these encode (do not quietly relax them):

* Correctness runs on the GPU directly, never through ``TRITON_INTERPRET``.
* ``tau_ref`` is float64 and is the oracle; kernels are certified against the
  scores they actually consumed, not against each other.
* Timing is the MIN over rotated rounds of the ``do_bench`` median.  GH200
  throttles under sustained load, so ``--rep 1500`` is the default: short reps
  inflate ratios by measuring a colder GPU.
"""

import argparse
import time

import torch
import triton


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


# ///////////////////////////////////////////////////////////////////////////////
# References
# ///////////////////////////////////////////////////////////////////////////////


def tau_ref(y):
    """Exact entmax-1.5 threshold, float64 sort solver (house standard)."""
    z, _ = torch.sort(y.double(), dim=-1, descending=True)
    kk = torch.arange(1, z.shape[-1] + 1, dtype=torch.float64, device=z.device)
    S1, S2 = z.cumsum(-1), (z * z).cumsum(-1)
    tau_k = (S1 - (S1 * S1 - kk * (S2 - 1.0)).clamp_min(0).sqrt()) / kk
    supp = (z > tau_k).sum(-1).clamp_min(1)
    return tau_k.gather(-1, (supp - 1).unsqueeze(-1)).squeeze(-1)


def scores(q, k, sm_scale, varlen, n_ctx, bins=1, prescale_q=False):
    """The exact scores a kernel sees: bf16 GEMM inputs, fp32 accumulate.

    The CuteDSL and Triton kernels do NOT see the same scores, and giving them a
    shared reference is wrong.  Triton multiplies q by ``0.5*sm_scale*BINS`` in
    fp32 and rounds the product back to bf16 *before* the GEMM
    (``prescale_q=True``), so its inputs carry a ~2^-8 relative perturbation.
    The CuTe kernels cannot do that -- Q arrives at the wgmma as bf16 already in
    smem -- so they scale the fp32 accumulator afterwards and their inputs are
    the raw bf16 q.
    """
    n_h, n_kv = q.shape[1], k.shape[1]
    kk = k.repeat_interleave(n_h // n_kv, dim=1).float()
    if prescale_q:
        qq = (q.float() * (0.5 * sm_scale * bins)).to(q.dtype).float()
        y = torch.einsum("bhnd,bhmd->bhnm", qq, kk) / bins
    else:
        y = 0.5 * sm_scale * torch.einsum("bhnd,bhmd->bhnm", q.float(), kk)
    causal = torch.ones(n_ctx, n_ctx, dtype=torch.bool, device=q.device).tril()
    y = y.masked_fill(~causal, -10000.0)
    if varlen is not None:
        cols = torch.arange(n_ctx, device=q.device)
        valid = cols[None, :] < varlen[:, None]  # (B, N)
        y = y.masked_fill(~valid[:, None, None, :], -10000.0)
    return y


def err(a, ref, rowmask=None):
    """max |a - ref| over valid rows.

    Uses ``torch.where`` rather than a multiply so that a non-finite value in an
    invalid row cannot leak through as ``nan * 0``.
    """
    e = (a.double() - ref.double()).abs()
    if rowmask is not None:
        e = torch.where(rowmask[..., None].bool(), e, torch.zeros_like(e))
    return float(e.max())


def ulp_check(a, ref, rowmask, ulps, extra_rel, extra_abs):
    """Elementwise |a - ref| <= ulps * ulp(ref) + extra_rel*|ref| + extra_abs.
    Returns (n_violations, worst_excess)."""
    a64, r64 = a.double(), ref.double()
    e = (a64 - r64).abs()
    ulp = (r64.abs() * (2.0 ** -8 if ref.dtype == torch.bfloat16 else 2.0 ** -11))
    tol = ulps * ulp + extra_rel * r64.abs() + extra_abs
    bad = e > tol
    if rowmask is not None:
        bad &= rowmask[..., None].bool()
    worst = float((e - tol)[bad].max()) if bad.any() else 0.0
    return int(bad.sum()), worst


# ///////////////////////////////////////////////////////////////////////////////
# Masks and shapes
# ///////////////////////////////////////////////////////////////////////////////


def unpack_mask(mask):
    """(B,H,M64,W32) int32 -> (B,H,M64,W32*32) bool; bit b of word w = column 32w+b."""
    shifts = torch.arange(32, device=mask.device, dtype=torch.int32)
    bits = (mask.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(*mask.shape[:-1], -1).bool()


def pack_mask(bits):
    """Inverse of ``unpack_mask``: (..., M64, W32*32) bool -> (..., M64, W32) int32.

    Bit b of word w encodes column 32w+b, matching what kernel B emits and what
    kernel C decodes.  The trailing column count must be a multiple of 32.
    """
    assert bits.shape[-1] % 32 == 0, "column count must be a multiple of 32"
    grouped = bits.reshape(*bits.shape[:-1], bits.shape[-1] // 32, 32)
    shifts = torch.arange(32, device=bits.device, dtype=torch.int64)
    words = (grouped.to(torch.int64) << shifts).sum(-1)
    ## int64 -> int32 bit pattern (values < 2^32; bit 31 needs the wrap)
    return (words - ((words >> 31) & 1) * (1 << 32)).to(torch.int32).contiguous()


def pack_triton_bmask(mask, n_ctx):
    """Block-reduce kernel B's column mask to Triton ``_get_output``'s layout:
    (B,H,mblocks,N_INT32s) int32, bit i of word w of row m = 64-col block
    32w+i has a live column for 64-row block m."""
    B, H, M64, _ = mask.shape
    bits = unpack_mask(mask)[..., :n_ctx]
    nb = (n_ctx + 63) // 64
    pad = nb * 64 - n_ctx
    blk = torch.nn.functional.pad(bits, (0, pad)).reshape(B, H, M64, nb, 64).any(-1)
    mblocks = (n_ctx + 63) // 64  # triton grid rows; <= M64
    ni32 = (nb + 31) // 32
    blk = blk[:, :, :mblocks]
    padw = ni32 * 32 - nb
    blk = torch.nn.functional.pad(blk, (0, padw)).reshape(B, H, mblocks, ni32, 32)
    shifts = torch.arange(32, device=mask.device, dtype=torch.int64)
    words = (blk.to(torch.int64) << shifts).sum(-1)
    ## int64 -> int32 bit pattern (values < 2^32; bit 31 needs the wrap)
    return (words - ((words >> 31) & 1) * (1 << 32)).to(torch.int32).contiguous()


def density_metrics(mask, n_ctx):
    """Two different densities of a 64x1 column mask -- report BOTH, they differ a lot.

    Returns ``(rho_w, rho64)``:

    * ``rho_w``  -- WIDTH-WEIGHTED: total live columns / total causal columns.
      This is the one that tracks kernel C's cost, because cost follows the total
      live-column count.
    * ``rho64``  -- the unweighted mean of per-64-row-block ratios, i.e. what
      ``probe_density.py`` prints.  It gives a narrow early block the same vote as
      a full-width late one, and entmax keeps proportionally more of the early
      (narrow) blocks -- so ``rho64`` runs ~1.3-1.5x ABOVE ``rho_w`` on real masks.

    Quoting one where the other is meant is the easiest way to make two otherwise
    consistent benchmark tables look like they disagree.
    """
    M64 = mask.shape[-2]
    width = torch.clamp((torch.arange(M64, device=mask.device) + 1) * 64,
                        max=n_ctx).double()
    cnt64 = unpack_mask(mask)[..., :n_ctx].sum(-1, dtype=torch.int64).double()
    bh = mask.shape[0] * mask.shape[1]
    return float(cnt64.sum() / (bh * width.sum())), float((cnt64 / width).mean())


def row_mask(B, H, N, varlen, device="cuda"):
    """(B,H,N) bool: rows the kernels actually define.  Under varlen, rows past
    each batch's seqlen belong to skipped tiles."""
    if varlen is not None:
        rows = torch.arange(N, device=device)[None, None, :] < varlen[:, None, None]
    else:
        rows = torch.ones(B, 1, N, dtype=torch.bool, device=device)
    return rows.expand(B, H, N)


def make_qkv(B, H, KV, N, D, dtype, seed=0, device="cuda", need_v=True, varlen=False):
    """Deterministic inputs; the fixed seed is what makes harness runs comparable."""
    torch.manual_seed(seed)
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, KV, N, D, device=device, dtype=dtype)
    v = torch.randn(B, KV, N, D, device=device, dtype=dtype) if need_v else None
    vl = None
    if varlen:
        vl = torch.randint(N // 2, N + 1, (B,), device=device, dtype=torch.int32)
    return q, k, v, vl


# ///////////////////////////////////////////////////////////////////////////////
# Timing
# ///////////////////////////////////////////////////////////////////////////////


def rotated_bench(fns, warmup=25, rep=1500, rounds=4, cool=0.4):
    """min over ``rounds`` rotated rounds of the ``do_bench`` median, in ms.

    Rotating the order each round keeps any single entry from always running on
    the hottest (or coldest) GPU.  Every fn is called once up front so no JIT or
    autotune cost lands inside a timed region.
    """
    for f in fns.values():
        f()
    torch.cuda.synchronize()

    names = list(fns)
    best = {n: float("inf") for n in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[: r % len(names)]
        for n in order:
            time.sleep(cool)
            t = triton.testing.do_bench(fns[n], warmup=warmup, rep=rep)
            best[n] = min(best[n], t)
    return best


# ///////////////////////////////////////////////////////////////////////////////
# CLI
# ///////////////////////////////////////////////////////////////////////////////


def parser(desc):
    """The argparse surface every harness shares.

    ``--mode check`` (default) runs the correctness checks; ``--mode bench``
    runs the timing table.
    """
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--mode", choices=["check", "bench"], default="check")
    p.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    p.add_argument("--d", type=int, default=128)
    ## check-mode shape sweep
    p.add_argument("--n", type=int, default=None, help="check: only this N_CTX")
    p.add_argument("--gqa", action="store_true", help="check: also sweep GQA (KV = H/4)")
    p.add_argument("--varlen", action="store_true", help="check: also sweep varlen")
    p.add_argument("--n16k", action="store_true", help="check: add the N=16384 case")
    ## bench-mode shapes + timing
    p.add_argument("--b", type=int, default=4, help="bench: batch")
    ## NOT --h: the CuTe DSL re-parses sys.argv on its first JIT compile with its
    ## own argparse (-h/--help/-diagnostic), and argparse abbreviation makes --h
    ## match --help -- so a harness taking --h dies mid-run with a usage message.
    p.add_argument("--heads", type=int, default=32, help="bench: heads")
    p.add_argument("--bench-n", type=int, nargs="+",
                   default=[2048, 4096, 8192, 16384], help="bench: N_CTX sweep")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=1500)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--cool", type=float, default=0.4)
    return p


def check_sweep(args, default_ns=(256, 512, 1024, 2048)):
    """Yield ``(B, H, KV, N, use_varlen)`` for the check-mode shape sweep."""
    ns = [args.n] if args.n else list(default_ns)
    for N in ns:
        for use_varlen in ([False, True] if args.varlen else [False]):
            for kv_div in ([1, 4] if args.gqa else [1]):
                B, H = 2, 4
                yield B, H, H // kv_div, N, use_varlen
    if args.n16k:
        yield 1, 2, 2, 16384, False
