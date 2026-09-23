# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Derived from adasplash/forward/cute/fwd_tau_hist.py (kernel "A"),
# which is itself derived from flash_attn/cute/flash_fwd_sm90.py.
"""One-pass tau refinement + bitpacked column mask for entmax-1.5 (CuTe DSL, SM90).

This is kernel "B" of the pipeline A -> B -> C.  It replaces the Triton
``_get_tau_v3`` pass, but algorithmically revamped around what A guarantees:

  * A (``fwd_tau_hist``) certifies ``0 <= tau* - t < h`` with h = 1/8 at
    every length, so B does NOT need Triton-v3's NITER+1 full Q@K^T sweeps from
    the bracket midpoint.  ONE pass evaluates the Halley sums at tau0 = t, takes
    one safeguarded Halley step, and clamps back into [t, t+h].
  * The mask is COLUMN-level, not Triton's 64x64-block-level: for every 64-row
    block (one warpgroup's rows) B stores the union over rows of the per-key
    live set at threshold t, bitpacked u32 little-endian (memory-compatible with
    a u64 view), plus a popcount sidecar.  Thresholding at t makes the mask a
    certified superset of the support at ANY tau >= t -- in particular at the
    tau-hat B stores and at anything C re-derives, because C's wgmma reproduces
    B's raw scores bit-identically.

Per-element cost (the whole point -- see the design notes below):

  * ``refine_mode='fused'``  : ~6 instructions/element -- FSUB (staged), FMNMX
    (relu), FADD (S1), FFMA (S2), and setp + predicated OR-immediate for the
    mask bit.  The row count n falls out of a per-k-block POPC of the per-row
    mask word, NOT a per-element add.
  * ``refine_mode='mask_only'``: 1.5 instructions/element -- ``setp.gt.f32`` on
    row 0, ``setp.gt.or.f32`` folding row 1's compare and the two-row union
    into the SAME predicate, one predicated OR-immediate.  tau-hat is then the
    bisection-safe midpoint t + h/2 (refinement deferred to C).

Why predicated OR-immediate and not ballots: the element loop is
``range_constexpr``-unrolled, so every bit position is a compile-time
immediate; ``@p or.b32 m, m, IMM`` is ONE predicated LOP3 in SASS.  A ballot
also costs one issue, but the per-ballot fold and stride-2 bit deposit sit in
every thread's stream (~9-10 ops per 64 raw elements marginal); a 1-bit OR is
already minimal, so vote hardware has nothing to amortize.  The DSL route
(``Boolean(..).to(Uint32) << j | m``) compiles to setp+selp+shl+or; the
``inline_ptx`` chunks below pin the 2-op form.

Mask data path (element -> warp -> warpgroup -> gmem), designed so the mainloop
never synchronizes per k-block:

  * element: each thread's u32 covers its 32 accumulator columns in the natural
    MN-order bit layout: bit j = 2k+b <-> physical column 8k + 2*(lane%4) + b.
  * warp: 3 x ``shfl.bfly`` (XOR 4, 8, 16) + OR folds the 8 lanes that share a
    column set -> every lane uniformly holds its column set's 16-row union.
  * warpgroup: NO ``red.shared``, NO barrier -- each warp owns a private N-bit
    smem replica; lanes 0-3 do one STS.32 per k-block into words
    ``n_block*4 + lane``.  Different k-blocks hit different words, so the
    mainloop has no read-modify-write anywhere.
  * tile end: ONE warpgroup barrier, then a merge pass ORs the 4 replicas,
    fixes the c-interleaved bit order into absolute column order (the 2-step
    magic-mask spread -- see ``_spread_pairs``), popcounts for the sidecar, and
    stores the (m64, head) mask row with plain coalesced STG.  A permuted gmem
    format was rejected: it would poison the backward pass and host tooling,
    and the fix is ~2% of the mainloop stream once per tile.

Each warpgroup's 64 rows are exactly one m64 output block, so the two masks per
128-row tile fall out with zero cross-WG traffic and gmem needs no atomics.

Refinement correctness details (the two certificate-breaking traps):

  * ROUNDING OF THE RAW THRESHOLD.  The compare runs in raw accumulator units
    against t' = t * (2/scale).  If that product rounded UP, a truly-live key
    with 2t/scale < s < t' would be dropped and the superset guarantee would
    die silently.  Directed rounding is sign-treacherous, so the fix is a fixed
    conservative nudge: the kernel compares against
    ``t_cmp = t' - (|t'| + 1) * 2^-22`` -- a few ulps of slack, an unmeasurable
    extra band against the ~0.08 keep fraction.  The SAME t_cmp feeds the
    compare, the staged d = s - t_cmp, and the closing solve's expansion point
    tau0 = t_cmp * (scale/2), so the sums are self-consistent by construction.
  * OOB QUERY ROWS POISON THE UNION.  TMA zero-fills OOB Q rows, so their
    scores are 0 -- which can exceed a garbage-loaded t.  A never cared (its
    outputs were store-guarded per row); B's union is a cross-row OR, so the
    guard must move to the t LOAD: rows >= seqlen_q get t_cmp = +inf, every
    compare goes false, and the mask/sums stay clean.  (d = s - inf = -inf,
    relu clamps to 0, so the unconditional S1/S2 adds are also clean; the
    predicated mask form never materializes -inf * 0 = NaN.)

Causal/seqlen masking is inherited unchanged (``mask_block`` -> -inf), and
-inf falls through everything for free: setp false, relu to 0.

Schedule: A/v3's phase-split skeleton verbatim (see that file's ``mma`` notes
for why the two-fragment double buffer is a ptxas trap).  Everything that must
READ the accumulator runs post-``wait_group(0)`` / pre-issue (phase1): the
mask select and either the staged FSUBs (fused) or the whole setp chain
(mask_only).  The acc-free work -- relu/S1/S2/popc out of the staging buffer
(fused), butterfly, STS, stage release -- runs while the next block's wgmma is
in flight (phase2).  All wgmma stay straight-line, waits are ``wait_group(0)``
only, no group crosses the back-edge: no new ptxas exposure.  B carries LESS
register state than A (no rowmax/M_top/histogram/flush), so ``num_stages`` has
headroom to 4 if TMA wants it.

Outputs (host layout):
  * ``taus_out`` (B, H, N) fp32 -- refined tau-hat, clamped to [t, t+h].
  * ``mask`` (B, H, M64, W32) int32 -- M64 = 2*ceil(N/128) row blocks, W32 =
    4*ceil(N/128) little-endian u32 words; ``mask.view(torch.int64)`` is the
    u64 view; bit b of word w = absolute key column 32w + b.
  * ``cnt`` (B, H, M64) int32 -- popcount of each mask row (live keys at t).

Measured on GH200, B=4 H=32 D=128 bf16 causal, min over 4 rotated rounds of
the do_bench median (rep=1500); ``tri1``/``tri0`` = Triton ``_get_tau_v3`` at
niter=1 (the shipped default: TWO full sweeps) / niter=0 (the structural
one-sweep equivalent; ratios from the tri0 run's own columns):

    N_CTX    tri1     tri0     B_mask   B_fused  fused/tri1  fused/tri0  mask/tri0
     2048    0.416    0.214    0.136    0.182    2.29x       1.17x       1.57x
     4096    1.508    0.810    0.440    0.619    2.43x       1.18x       1.64x
     8192    6.100    3.487    1.738    2.495    2.45x       1.20x       1.73x
    16384   27.559   13.853    7.369   10.780    2.56x       1.20x       1.78x

B_gemm (GEMM+mask floor) = 0.124/0.413/1.674/7.212: fused sits at ~1.49x the
floor (exposure tracks per-element op count, as in A), mask_only at ~1.04x.
Overlap vs serial = 1.04-1.06x.  stages=4 measured ~2% slower than 3;
ptx_chunk in {8,16,32} is a measured no-op (the boundary movs coalesce).
Against tri0's SAME sweep count, B_fused is 1.17-1.20x faster while holding
~9x better max tau error (3.5e-3 vs 3.13e-2 ~= h/4, Triton's
midpoint-bisection-fallback worst case) plus the certified-superset column
mask; tri0's bmask thresholds at the PRE-update tau, so a downward final
update can silently drop live blocks -- the certificate class this kernel's
threshold-at-t design closes.
"""

from typing import Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, Boolean, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute.named_barrier import NamedBarrierFwd
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
)

from flash_attn.cute.flash_fwd import FlashAttentionForwardBase

from adasplash.forward.cute.utils import fma_rn, div_rn
from adasplash.forward.cute.named_barrier import NamedBarrierTauRefine


## Triton's halley_bisect_update acceptance slack, mirrored exactly.
HALLEY_EPS = 1e-5

# ///////////////////////////////////////////////////////////////////////////////
# Trace-time PTX generators (B-specific codegen, not shared leaves)
# ///////////////////////////////////////////////////////////////////////////////


def _mask_union_chunk_ptx(n_cols: int, bit_base: int) -> str:
    """PTX for the mask_only hot loop over ``n_cols`` columns x 2 rows.

    Per column: ``setp.gt.f32`` on row 0, ``setp.gt.or.f32`` folding row 1 AND
    the two-row union into the same predicate, one predicated OR-immediate --
    3 instructions per column = 1.5/element, the floor reachable from PTX
    (SASS P2R would be ~1/element but predicates are not addressable here).

    Operands: $r0 = t_cmp row0, $r1 = t_cmp row1, $r2.. = row-0 scores,
    $r{2+n}.. = row-1 scores, $r{2+2n} = mask word in; $w0 = mask word out.
    The in/out movs bracket the chunk and coalesce away in ptxas.
    """
    lines = ["{", ".reg .pred p;", "mov.b32 {$w0}, {$r%d};" % (2 + 2 * n_cols)]
    for j in range(n_cols):
        lines.append("setp.gt.f32 p, {$r%d}, {$r0};" % (2 + j))
        lines.append("setp.gt.or.f32 p, {$r%d}, {$r1}, p;" % (2 + n_cols + j))
        lines.append("@p or.b32 {$w0}, {$w0}, %d;" % (1 << (bit_base + j)))
    lines.append("}")
    return "\n".join(lines)


def _bits_from_d_chunk_ptx(n_cols: int, bit_base: int) -> str:
    """PTX for the fused-mode mask bits over ``n_cols`` relu'd d values of ONE
    row: ``setp.gt.f32 p, d, +0`` + predicated OR-immediate = 2/element.
    After the relu, d > 0 is EXACTLY s > t_cmp: a nonzero difference of
    finite fp32 values never rounds to zero (cancellation is exact), and
    masked -inf lanes were clamped to +0.

    Operands: $r0..$r{n-1} = d values, $r{n} = mask word in; $w0 = out.
    """
    lines = ["{", ".reg .pred p;", "mov.b32 {$w0}, {$r%d};" % n_cols]
    for j in range(n_cols):
        lines.append("setp.gt.f32 p, {$r%d}, 0f00000000;" % j)
        lines.append("@p or.b32 {$w0}, {$w0}, %d;" % (1 << (bit_base + j)))
    lines.append("}")
    return "\n".join(lines)


# ///////////////////////////////////////////////////////////////////////////////
# Kernel
# ///////////////////////////////////////////////////////////////////////////////


class AdaSplashTauRefineSm90(FlashAttentionForwardBase):
    """Q@K^T + per-key threshold mask + one Halley step, warp-specialized.

    The producer warp issues TMA loads of Q and K; the consumer warpgroups run
    wgmma, OR each score tile's live columns into per-warp smem mask replicas,
    and (fused mode) accumulate the entmax-1.5 Halley sums at tau0 = t.
    """

    def __init__(self, *args, n_ctx: int, refine_mode: str = "fused",
                 ptx_chunk: int = 16, ping_pong: bool = False, overlap: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        ## 'fused'     -- mask + Halley sums + one step (tau-hat refined)
        ## 'mask_only' -- mask only; tau-hat = t + h/2 (refinement deferred to C)
        ## 'gemm_only' -- DEBUG floor: GEMM + causal mask + a 1-op/block acc
        ##                sink; no mask, no sums (the maxonly analog)
        self.refine_mode = refine_mode
        ## columns per inline_ptx chunk in the hot loop (amortizes the 2-3
        ## boundary movs; 32 would be a single call, 8 is the paranoid floor)
        self.ptx_chunk = ptx_chunk
        self.ping_pong = ping_pong
        ## Phase-split pipelined consumer (A/v3's schedule) vs serial.  Outputs
        ## are bit-identical; only the schedule differs.
        self.overlap = overlap
        ## Mask words are sized by the static key length: ceil(n_ctx/32) u32
        ## per m64 row, rounded to a multiple of 4 so every mask row is
        ## 16B-aligned.  One replica per consumer warp lives in smem.
        self.n_ctx = n_ctx
        self.mask_words = 4 * ((n_ctx + 127) // 128)
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = (1, 1)
        assert refine_mode in ("fused", "mask_only", "gemm_only")
        assert self.tile_n == 128, "mask bitpacking assumes tile_n == 128"
        assert 32 % ptx_chunk == 0 and ptx_chunk <= 32
        assert not self.pack_gqa, "pack_gqa not supported; GQA is handled by head indexing"
        assert not self.is_local, "sliding-window attention not supported"
        assert self.arch.is_family_of(Arch.sm_90a), "Only SM 9.x is supported"

    # -- layouts / mma -------------------------------------------------------

    def _setup_attributes(self):
        self.sQ_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_hdim), None
        )
        self.sK_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdim), self.num_stages
        )

    def _get_tiled_mma(self):
        return sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            tiler_mn=(64, self.tile_n),
        )

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sQ_layout, self.sK_layout)
        ]
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        ## one N-bit replica per consumer warp (num_wg * 4), plus the per-warp
        ## sidecar partial counts
        n_warps = self.num_wg_mma * 4
        sMaskRep_struct = cute.struct.MemRange[cutlass.Uint32, n_warps * self.mask_words]
        sCnt_struct = cute.struct.MemRange[cutlass.Int32, n_warps]

        @cute.struct
        class SharedStorageQK:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            sCnt: sCnt_struct
            sMaskRep: sMaskRep_struct
            sQ: sQ_struct
            sK: sK_struct

        return SharedStorageQK

    # -- host entry ----------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if cu_seqlens_k
        mTauIn: cute.Tensor,  # (b, h, s_q) fp32 -- A's certified lower bound t
        mTauOut: cute.Tensor,  # (b, h, s_q) fp32 -- refined tau-hat
        mMask: cute.Tensor,  # (b, h, m64, w32) int32 -- bitpacked column mask
        mCnt: cute.Tensor,  # (b, h, m64) int32 -- mask popcount sidecar
        softmax_scale: Float32,
        h_bracket: Float32,  # A's bracket width (1/BINS)
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        # Always keep stream as the last parameter (EnvStream: implicit via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK = [assume_tensor_aligned(t) for t in (mQ, mK)]
        Q_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        mQ = layout_utils.select(mQ, Q_layout_transpose)
        K_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensK is None) else [0, 2, 1]
        mK = layout_utils.select(mK, K_layout_transpose)
        Tau_layout_transpose = [2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0]
        mTauIn = layout_utils.select(mTauIn, Tau_layout_transpose)
        mTauOut = layout_utils.select(mTauOut, Tau_layout_transpose)
        ## (b, h, m64, w) -> (w, m64, h, b); (b, h, m64) -> (m64, h, b)
        mMask = layout_utils.select(mMask, [3, 2, 1, 0])
        mCnt = layout_utils.select(mCnt, [2, 1, 0])

        tiled_mma_qk = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2]
        self.num_threads = self.num_threads_per_warp_group * (self.num_wg_mma + 1)
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24)}[self.num_wg_mma]
        self.use_scheduler_barrier = self.num_wg_mma == 2 and self.ping_pong
        self._setup_attributes()

        SharedStorage = self._get_shared_storage_cls()

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_K = cpasync.CopyBulkTensorTileG2SOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [("Q", mQ, self.sQ_layout), ("K", mK, self.sK_layout)]
        }
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q, mQ, self.sQ_layout, (self.tile_m, self.tile_hdim)
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_K,
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
            1,  # no mcast
        )

        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler if const_expr(not self.is_causal) else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            1,  # num_splits
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mQ.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        ## Raw-accumulator units: the score the kernel thresholds is s_raw with
        ## y = half_scale * s_raw; thresholds move the other way via inv_half.
        ## Q is NOT prescaled (it reaches the wgmma as bf16 already in smem).
        half_scale = softmax_scale * Float32(0.5)
        inv_half = Float32(2.0) / softmax_scale

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            mTauIn,
            mTauOut,
            mMask,
            mCnt,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            tma_atom_Q,
            tma_atom_K,
            half_scale,
            inv_half,
            h_bracket,
            self.sQ_layout,
            self.sK_layout,
            tiled_mma_qk,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    # -- device --------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mTauIn: cute.Tensor,
        mTauOut: cute.Tensor,
        mMask: cute.Tensor,
        mCnt: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        half_scale: Float32,
        inv_half: Float32,
        h_bracket: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K):
                cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        pipeline_q = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=1,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["Q"],
            defer_sync=True,
        )
        pipeline_k = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=tma_warp,
            consumer_group=mma_warps,
            tx_count=self.tma_copy_bytes["K"],
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        n_warps = self.num_wg_mma * 4
        sMaskRep = storage.sMaskRep.get_tensor(
            cute.make_layout(n_warps * self.mask_words)
        )
        sCnt = storage.sCnt.get_tensor(cute.make_layout(n_warps))

        block_info = BlockInfo(self.tile_m, self.tile_n, self.is_causal, False, False, None, None)
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(AttentionMask, self.tile_m, self.tile_n)
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                sQ,
                sK,
                tma_atom_Q,
                tma_atom_K,
                pipeline_q,
                pipeline_k,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
        else:  # Consumer
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                mTauIn,
                mTauOut,
                mMask,
                mCnt,
                sQ,
                sK,
                sMaskRep,
                sCnt,
                pipeline_q,
                pipeline_k,
                tidx,
                half_scale,
                inv_half,
                h_bracket,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_k: pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        if warp_idx_in_wg == 0:
            q_producer_phase = Int32(1)
            k_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()
            while work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, _ = work_tile.tile_idx
                seqlen = SeqlenInfoCls(batch_idx)
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                head_idx_kv = head_idx // self.qhead_per_kvhead
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]

                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                )
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK
                )
                tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)

                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)

                pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
                q_producer_phase ^= 1

                ## Same descending walk as A -- B has no running max, but the
                ## shared order keeps the two kernels' TMA traffic identical.
                for i in cutlass.range(n_block_max - n_block_min, unroll=1):
                    n_block = n_block_max - 1 - i
                    pipeline_k.producer_acquire(k_producer_state)
                    tma_load_K_fn(src_idx=n_block, producer_state=k_producer_state)
                    pipeline_k.producer_commit(k_producer_state)
                    k_producer_state.advance()

                tile_scheduler.prefetch_next_work()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

            pipeline_k.producer_tail(k_producer_state)

    # -- per-block phases ----------------------------------------------------

    @cute.jit
    def phase1(
        self,
        acc_S: cute.Tensor,
        dbuf: cute.Tensor,
        t_cmp: cute.Tensor,
        sink: cute.Tensor,
        m_words: cute.Tensor,
    ):
        """Everything that must READ the accumulator, at a pipe-empty point.

        fused    : stage d = s - t_cmp[row] for every element; the FSUB's
                   destination register IS the dbuf slot, so staging costs
                   zero extra instructions (A's ubuf pattern).
        mask_only: the whole setp/setp.or/@p-or chain -- there is nothing else
                   per element, and routing it through a staging buffer would
                   add a pure-overhead FSUB per element.
        gemm_only: a 1-op/block sink so the GEMM stays live through DCE.
        """
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        n_cols: cutlass.Constexpr = cute.size(acc_S_mn, mode=[1])
        CHUNK: cutlass.Constexpr = self.ptx_chunk

        if const_expr(self.refine_mode == "fused"):
            for r in cutlass.range_constexpr(2):
                for c in cutlass.range_constexpr(n_cols):
                    dbuf[r * n_cols + c] = acc_S_mn[r, c] - t_cmp[r]
        elif const_expr(self.refine_mode == "mask_only"):
            m = Uint32(0)
            for c0 in cutlass.range_constexpr(0, n_cols, CHUNK):
                m = cute.arch.inline_ptx(
                    _mask_union_chunk_ptx(CHUNK, c0),
                    write_only_types=[Uint32],
                    read_only_args=(
                        [t_cmp[0], t_cmp[1]]
                        + [acc_S_mn[0, c0 + j] for j in range(CHUNK)]
                        + [acc_S_mn[1, c0 + j] for j in range(CHUNK)]
                        + [m]
                    ),
                )
            m_words[0] = m
        else:  # gemm_only
            sink[0] = sink[0] + acc_S_mn[0, 0]

    @cute.jit
    def phase2_elem(
        self,
        dbuf: cute.Tensor,
        s1: cute.Tensor,
        s2: cute.Tensor,
        n_pop: cute.Tensor,
        m_words: cute.Tensor,
    ):
        """Fused mode's acc-free per-element work, run in the wgmma shadow.

        Out of the staging buffer: relu (masked -inf -> +0, so S1/S2 need no
        predicate), FADD into S1, FFMA into S2, and the 2-op setp/@p-or mask
        bit per row.  The per-row count n is NOT a per-element add: it is one
        POPC of the per-row mask word per k-block.
        """
        n_cols: cutlass.Constexpr = cute.size(dbuf) // 2
        CHUNK: cutlass.Constexpr = self.ptx_chunk
        for r in cutlass.range_constexpr(2):
            d_vals = [
                cute.arch.fmax(dbuf[r * n_cols + c], Float32(0.0)) for c in range(n_cols)
            ]
            for c in cutlass.range_constexpr(n_cols):
                s1[r] = s1[r] + d_vals[c]
                s2[r] = fma_rn(d_vals[c], d_vals[c], s2[r])
            m = Uint32(0)
            for c0 in cutlass.range_constexpr(0, n_cols, CHUNK):
                m = cute.arch.inline_ptx(
                    _bits_from_d_chunk_ptx(CHUNK, c0),
                    write_only_types=[Uint32],
                    read_only_args=[d_vals[c0 + j] for j in range(CHUNK)] + [m],
                )
            n_pop[r] = n_pop[r] + cute.arch.popc(m)
            m_words[r] = m

    @cute.jit
    def phase2_store(
        self,
        m_words: cute.Tensor,
        sMaskRep: cute.Tensor,
        rep_base: Int32,
        lane_idx: Int32,
        n_block: Int32,
    ):
        """Warp fold + replica store -- the ONLY collective work per k-block.

        Butterfly-OR across the 8 lanes sharing a column set (XOR 4, 8, 16)
        leaves every lane holding its column set's 16-row union; lanes 0-3
        then store one word each into this warp's private replica at words
        ``n_block*4 + lane``.  Different k-blocks hit different words: no
        accumulation, no red/atom, no barrier in the mainloop.
        """
        m = m_words[0]
        if const_expr(self.refine_mode == "fused"):
            m = m | m_words[1]
        m = m | cute.arch.shuffle_sync_bfly(m, offset=4)
        m = m | cute.arch.shuffle_sync_bfly(m, offset=8)
        m = m | cute.arch.shuffle_sync_bfly(m, offset=16)
        if lane_idx < 4:
            sMaskRep[rep_base + n_block * 4 + lane_idx] = m

    # -- tile epilogue -------------------------------------------------------

    @cute.jit
    def solve_store_tau(
        self,
        t_y: cute.Tensor,
        t_cmp: cute.Tensor,
        row_valid: cute.Tensor,
        s1: cute.Tensor,
        s2: cute.Tensor,
        n_pop: cute.Tensor,
        sink: cute.Tensor,
        taccTgTauOut: cute.Tensor,
        col0: Int32,
        half_scale: Float32,
        h_bracket: Float32,
    ):
        """Quad-reduce the sums, one safeguarded Halley step, clamp, store.

        The sums were accumulated in RAW units over exactly {s > t_cmp}; they
        convert to y units by half_scale powers (safe in fp32: liveness at t
        bounds d_y <= 1+h, so S2_raw <~ 5e6).  The expansion point is
        tau0 = t_cmp * half_scale -- the SAME threshold the sums used -- and
        the step mirrors Triton's ``halley_bisect_update`` (bounds update,
        EPS-slack acceptance, midpoint fallback) followed by a hard clamp to
        [t, t+h], which is what preserves A's certificate.  NaN steps (e.g.
        the degenerate OOB rows) fail the acceptance compares and fall to the
        midpoint; the store guard drops OOB rows regardless.
        """
        for r in cutlass.range_constexpr(2):
            tau_hat = t_y[r] + Float32(0.5) * h_bracket
            if const_expr(self.refine_mode == "fused"):
                n_f = cute.arch.warp_reduction_sum(Float32(n_pop[r]), threads_in_group=4)
                S1 = cute.arch.warp_reduction_sum(s1[r], threads_in_group=4)
                S2 = cute.arch.warp_reduction_sum(s2[r], threads_in_group=4)
                S1_y = S1 * half_scale
                S2_y = S2 * (half_scale * half_scale)
                tau0 = t_cmp[r] * half_scale
                ff = S2_y - Float32(1.0)
                df = Float32(-2.0) * S1_y
                ddf = Float32(2.0) * n_f
                new_t = tau0 - div_rn(ff * df, df * df - Float32(0.5) * ff * ddf)
                t_lo = tau0 if ff > Float32(0.0) else t_y[r]
                t_hi = tau0 if ff < Float32(0.0) else t_y[r] + h_bracket
                good = (new_t > t_lo - Float32(HALLEY_EPS)) & (
                    new_t < t_hi + Float32(HALLEY_EPS)
                )
                tau_hat = new_t if good else Float32(0.5) * (t_lo + t_hi)
                ## hard clamp: A's certificate says tau* is in [t, t+h], and C
                ## relies on tau_hat >= t for the mask superset property.  A
                ## NaN step already fell to the (finite) midpoint via ``good``.
                lo_b = t_y[r]
                hi_b = t_y[r] + h_bracket
                tau_hat = lo_b if tau_hat < lo_b else tau_hat
                tau_hat = hi_b if tau_hat > hi_b else tau_hat
            elif const_expr(self.refine_mode == "gemm_only"):
                tau_hat = tau_hat + sink[0] * Float32(0.0)
            if col0 != 0:
                if row_valid[r] != 0:
                    taccTgTauOut[r, 0] = tau_hat

    @cute.jit
    def mask_merge_store(
        self,
        sMaskRep: cute.Tensor,
        sCnt: cute.Tensor,
        mMask_row: cute.Tensor,
        mCnt_row: cute.Tensor,
        m64_glob: Int32,
        wg_idx: Int32,
        tid_in_wg: Int32,
    ):
        """Tile-end: merge the 4 warp replicas, fix the bit order, store.

        Merged word (n_block, c) holds bits j = 2k+b <-> column 8k+2c+b; the
        absolute-order output u32 for columns 32t..32t+31 is the OR over c of
        ``spread(byte t of word_c) << 2c``, where spread is the 2-step
        magic-mask expand of four 2-bit pairs to stride-8 positions (the same
        ``(x | x<<s) & M`` family as A's hist_flush).  ~120 instructions per
        thread per k-block group, once per tile: <2% of the mainloop stream.
        POPC for the sidecar rides the same pass; the store is plain
        coalesced STG (TMA S2G buys nothing at 1 KB per row).

        Caller brackets this with the two named barriers: MASK_MERGE (replica
        stores visible) before, MASK_DONE (replica reads + sCnt partials
        done -- also gates the next tile's re-zero) after.
        """
        W: cutlass.Constexpr = self.mask_words
        KB: cutlass.Constexpr = W // 4  # k-block groups (128 columns each)
        M_NIB: cutlass.Constexpr = 0x000F000F
        M_PAIR: cutlass.Constexpr = 0x03030303
        warp_in_wg = cute.arch.make_warp_uniform(tid_in_wg // 32)
        lane_idx = cute.arch.lane_idx()

        ## mutation inside a dynamic ``if`` must go through memory, not SSA
        out = cute.make_rmem_tensor(4, Uint32)
        cnt = cute.make_rmem_tensor(1, Int32)
        cnt[0] = Int32(0)
        for it in cutlass.range_constexpr((KB + 127) // 128):
            kb = tid_in_wg + it * 128
            if kb < KB:
                for c in cutlass.range_constexpr(4):
                    w = sMaskRep[(wg_idx * 4 + 0) * W + kb * 4 + c]
                    for rw in cutlass.range_constexpr(1, 4):
                        w = w | sMaskRep[(wg_idx * 4 + rw) * W + kb * 4 + c]
                    for t in cutlass.range_constexpr(4):
                        x = (w >> Uint32(8 * t)) & Uint32(0xFF)
                        x = (x | (x << Uint32(12))) & Uint32(M_NIB)
                        x = (x | (x << Uint32(6))) & Uint32(M_PAIR)
                        if const_expr(c == 0):
                            out[t] = x
                        else:
                            out[t] = out[t] | (x << Uint32(2 * c))
                for t in cutlass.range_constexpr(4):
                    cnt[0] = cnt[0] + cute.arch.popc(out[t]).to(Int32)
                    mMask_row[kb * 4 + t] = out[t].bitcast(Int32)
        ## all lanes reach this point together (the shuffle must be convergent)
        total = cute.arch.warp_reduction_sum(cnt[0], threads_in_group=32)
        if lane_idx == 0:
            sCnt[wg_idx * 4 + warp_in_wg] = total
        cute.arch.barrier(
            barrier_id=NamedBarrierTauRefine.MaskDoneBase + wg_idx,
            number_of_threads=self.num_threads_per_warp_group,
        )
        if tid_in_wg == 0:
            mCnt_row[m64_glob] = (
                sCnt[wg_idx * 4 + 0]
                + sCnt[wg_idx * 4 + 1]
                + sCnt[wg_idx * 4 + 2]
                + sCnt[wg_idx * 4 + 3]
            )

    # -- consumer mainloop ---------------------------------------------------

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        mTauIn: cute.Tensor,
        mTauOut: cute.Tensor,
        mMask: cute.Tensor,
        mCnt: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sMaskRep: cute.Tensor,
        sCnt: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_k: pipeline.PipelineAsync,
        tidx: Int32,
        half_scale: Float32,
        inv_half: Float32,
        h_bracket: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
        )
        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK
        )
        if const_expr(self.overlap):
            ## ONE persistent accumulator; see A's mma() for why the classic
            ## two-fragment double buffer is a ptxas trap on sm90.
            acc_shape = tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
            acc_S = cute.make_rmem_tensor(acc_shape, Float32)

        self.mma_init()

        tid_in_wg = tidx % self.num_threads_per_warp_group
        warp_in_wg = cute.arch.make_warp_uniform(tid_in_wg // 32)
        lane_idx = cute.arch.lane_idx()
        rep_base = (warp_group_idx * 4 + warp_in_wg) * self.mask_words

        ## per-thread state: 2 rows (the wgmma 2-rows-per-thread fact)
        num_rows: cutlass.Constexpr = 2
        n_cols: cutlass.Constexpr = self.tile_n // 4
        t_y = cute.make_rmem_tensor(num_rows, Float32)  # A's t, y units
        t_cmp = cute.make_rmem_tensor(num_rows, Float32)  # nudged raw threshold
        row_valid = cute.make_rmem_tensor(num_rows, Int32)
        s1 = cute.make_rmem_tensor(num_rows, Float32)
        s2 = cute.make_rmem_tensor(num_rows, Float32)
        n_pop = cute.make_rmem_tensor(num_rows, Uint32)
        sink = cute.make_rmem_tensor(1, Float32)
        m_words = cute.make_rmem_tensor(num_rows, Uint32)
        if const_expr(self.refine_mode == "fused"):
            dbuf = cute.make_rmem_tensor(num_rows * n_cols, Float32)
        else:
            dbuf = cute.make_rmem_tensor(1, Float32)  # unused placeholder

        ## tau row-tile partitioning pattern (store_tau's trick, used for BOTH
        ## the t load and the tau-hat store): give the rank-1 (tile_m,) row a
        ## stride-0 column mode so the MMA C layout can partition it; every
        ## lane of a quad then sees the same address for its rows.
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        taccTcT = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
        t0accTcT = layout_utils.reshape_acc_to_mn(
            tiled_mma_qk.get_slice(0).partition_C(cS)
        )
        ## only the column-0 lane of each quad stores tau-hat (all 4 hold it)
        col0 = (taccTcT[0][1] == 0).to(Int32)

        q_consumer_phase = Int32(0)
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        if const_expr(self.overlap):
            k_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            mask = AttentionMaskCls(seqlen)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=False,
            )
            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            n_block_min_causal_mask = (
                block_info.get_n_block_min_causal_local_mask(seqlen, m_block, n_block_min)
                if const_expr(self.is_causal)
                else n_block_min
            )

            ## -- prologue: per-row t load (guarded: OOB rows get +inf so they
            ## -- can never poison the union), prescale + conservative nudge,
            ## -- zero this warp's mask replica (own replica only -> intra-warp
            ## -- program order suffices; the previous tile's MASK_DONE barrier
            ## -- already proved everyone finished READING it).
            mTauIn_cur = seqlen.offset_batch_Q(mTauIn, batch_idx, dim=2)[None, head_idx]
            gTauIn = cute.local_tile(mTauIn_cur, (self.tile_m,), (m_block,))
            gTauIn_x = cute.make_tensor(
                gTauIn.iterator,
                cute.append(gTauIn.layout, cute.make_layout((self.tile_n,), stride=(0,))),
            )
            taccTgTauIn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(gTauIn_x))
            mTauOut_cur = seqlen.offset_batch_Q(mTauOut, batch_idx, dim=2)[None, head_idx]
            gTauOut = cute.local_tile(mTauOut_cur, (self.tile_m,), (m_block,))
            gTauOut_x = cute.make_tensor(
                gTauOut.iterator,
                cute.append(gTauOut.layout, cute.make_layout((self.tile_n,), stride=(0,))),
            )
            taccTgTauOut = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(gTauOut_x))

            for r in cutlass.range_constexpr(num_rows):
                valid = (
                    t0accTcT[r, 0][0]
                    < seqlen.seqlen_q - m_block * self.tile_m - taccTcT[0][0]
                )
                row_valid[r] = valid.to(Int32)
                ## a real branch, NOT a select: the OOB address must never be
                ## dereferenced, and +inf on dead rows is the union guard
                t_y[r] = Float32(0.0)
                t_cmp[r] = Float32(float("inf"))
                if valid:
                    tv = taccTgTauIn[r, 0]
                    t_y[r] = tv
                    tp = tv * inv_half
                    tp_abs = cute.arch.fmax(tp, Float32(0.0) - tp)
                    t_cmp[r] = tp - (tp_abs + Float32(1.0)) * Float32(2.0**-22)
            s1.fill(Float32(0.0))
            s2.fill(Float32(0.0))
            n_pop.fill(Uint32(0))
            sink.fill(Float32(0.0))
            m_words.fill(Uint32(0))
            if const_expr(self.refine_mode != "gemm_only"):
                W: cutlass.Constexpr = self.mask_words
                for it in cutlass.range_constexpr((W + 31) // 32):
                    widx = lane_idx + it * 32
                    if widx < W:
                        sMaskRep[rep_base + widx] = Uint32(0)

            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)

            if const_expr(self.overlap):
                ## A/v3's phase-split intra-warpgroup pipeline, verbatim shape:
                ## finish block i-1 (phase1: acc reads), issue block i, then
                ## run block i-1's acc-free work (phase2) in the GEMM shadow.
                mask_block_fn = partial(
                    self.mask_block,
                    n_block_max=n_block_max,
                    n_block_min_causal_mask=n_block_min_causal_mask,
                    mask_seqlen_fn=partial(mask_fn, mask_seqlen=True),
                    mask_inner_fn=partial(mask_fn, mask_seqlen=False),
                )
                n_blocks = n_block_max - n_block_min
                pipeline_k.consumer_wait(
                    k_consumer_state, pipeline_k.consumer_try_wait(k_consumer_state)
                )
                self.warp_scheduler_barrier_sync()
                sm90_utils.gemm_w_idx(
                    tiled_mma_qk, acc_S, tSrQ, tSrK, True,
                    B_idx=k_consumer_state.index, wg_wait=-1,
                )
                k_consumer_state.advance()
                for i in cutlass.range(1, n_blocks, 1, unroll=1):
                    warpgroup.wait_group(0)
                    self.warp_scheduler_barrier_arrive()
                    mask_block_fn(acc_S, jp=i - 1)
                    self.phase1(acc_S, dbuf, t_cmp, sink, m_words)
                    pipeline_k.consumer_wait(
                        k_consumer_state, pipeline_k.consumer_try_wait(k_consumer_state)
                    )
                    self.warp_scheduler_barrier_sync()
                    sm90_utils.gemm_w_idx(
                        tiled_mma_qk, acc_S, tSrQ, tSrK, True,
                        B_idx=k_consumer_state.index, wg_wait=-1,
                    )
                    k_consumer_state.advance()
                    pipeline_k.consumer_release(k_release_state)
                    k_release_state.advance()
                    if const_expr(self.refine_mode == "fused"):
                        self.phase2_elem(dbuf, s1, s2, n_pop, m_words)
                    if const_expr(self.refine_mode != "gemm_only"):
                        self.phase2_store(
                            m_words, sMaskRep, rep_base, lane_idx,
                            n_block=n_block_max - i,
                        )
                warpgroup.wait_group(0)
                self.warp_scheduler_barrier_arrive()
                pipeline_k.consumer_release(k_release_state)
                k_release_state.advance()
                mask_block_fn(acc_S, jp=n_blocks - 1)
                self.phase1(acc_S, dbuf, t_cmp, sink, m_words)
                if const_expr(self.refine_mode == "fused"):
                    self.phase2_elem(dbuf, s1, s2, n_pop, m_words)
                if const_expr(self.refine_mode != "gemm_only"):
                    self.phase2_store(
                        m_words, sMaskRep, rep_base, lane_idx, n_block=n_block_min
                    )
            else:
                one_n_block_fn = partial(
                    self.one_n_block,
                    mma_qk_fn=mma_qk_fn,
                    pipeline_k=pipeline_k,
                    dbuf=dbuf,
                    t_cmp=t_cmp,
                    sink=sink,
                    m_words=m_words,
                    s1=s1,
                    s2=s2,
                    n_pop=n_pop,
                    sMaskRep=sMaskRep,
                    rep_base=rep_base,
                    lane_idx=lane_idx,
                )
                k_consumer_state = one_n_block_fn(
                    k_consumer_state,
                    n_block=n_block_max - 1,
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                )
                for n_tile in cutlass.range(n_block_max - 1 - n_block_min_causal_mask, unroll=1):
                    k_consumer_state = one_n_block_fn(
                        k_consumer_state,
                        n_block=n_block_max - 2 - n_tile,
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                    )
                n_block_upper = cutlass.min(n_block_max - 1, n_block_min_causal_mask)
                for n_tile in cutlass.range(n_block_upper - n_block_min, unroll=1):
                    k_consumer_state = one_n_block_fn(
                        k_consumer_state, n_block=n_block_upper - 1 - n_tile, mask_fn=None
                    )

            pipeline_q.consumer_release_w_index(0)
            q_consumer_phase ^= 1

            ## -- tile epilogue: solve + store tau-hat (register-only), then
            ## -- barrier, merge the replicas, store mask + sidecar.
            self.solve_store_tau(
                t_y, t_cmp, row_valid, s1, s2, n_pop, sink, taccTgTauOut,
                col0, half_scale, h_bracket,
            )
            if const_expr(self.refine_mode != "gemm_only"):
                m64_glob = m_block * (self.tile_m // 64) + warp_group_idx
                mMask_row = mMask[None, m64_glob, head_idx, batch_idx]
                mCnt_row = mCnt[None, head_idx, batch_idx]
                cute.arch.barrier(
                    barrier_id=NamedBarrierTauRefine.MaskMergeBase + warp_group_idx,
                    number_of_threads=self.num_threads_per_warp_group,
                )
                self.mask_merge_store(
                    sMaskRep,
                    sCnt,
                    mMask_row,
                    mCnt_row,
                    m64_glob,
                    warp_group_idx,
                    tid_in_wg,
                )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def one_n_block(
        self,
        k_consumer_state,
        n_block: Int32,
        mma_qk_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        dbuf: cute.Tensor,
        t_cmp: cute.Tensor,
        sink: cute.Tensor,
        m_words: cute.Tensor,
        s1: cute.Tensor,
        s2: cute.Tensor,
        n_pop: cute.Tensor,
        sMaskRep: cute.Tensor,
        rep_base: Int32,
        lane_idx: Int32,
        mask_fn: Optional[Callable] = None,
    ):
        """Serial schedule: gemm -> wait -> mask -> phase1 -> phase2, one block."""
        pipeline_k.consumer_wait(k_consumer_state, pipeline_k.consumer_try_wait(k_consumer_state))
        self.warp_scheduler_barrier_sync()
        acc_S = mma_qk_fn(B_idx=k_consumer_state.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(k_consumer_state)
        k_consumer_state.advance()

        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)

        self.phase1(acc_S, dbuf, t_cmp, sink, m_words)
        if const_expr(self.refine_mode == "fused"):
            self.phase2_elem(dbuf, s1, s2, n_pop, m_words)
        if const_expr(self.refine_mode != "gemm_only"):
            self.phase2_store(m_words, sMaskRep, rep_base, lane_idx, n_block=n_block)
        return k_consumer_state

    @cute.jit
    def mask_block(
        self,
        acc_S: cute.Tensor,
        jp: Int32,
        n_block_max: Int32,
        n_block_min_causal_mask: Int32,
        mask_seqlen_fn: Callable,
        mask_inner_fn: Callable,
    ):
        """Apply the right mask variant for the block at descending position
        ``jp`` (0 == the diagonal block); see A for the trace-time rationale."""
        n_block = n_block_max - 1 - jp
        if jp == 0:
            mask_seqlen_fn(acc_S=acc_S, n_block=n_block)
        else:
            if n_block >= n_block_min_causal_mask:
                mask_inner_fn(acc_S=acc_S, n_block=n_block)

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            next_wg = 1 - cur_wg
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )


# ///////////////////////////////////////////////////////////////////////////////
# Host wrapper
# ///////////////////////////////////////////////////////////////////////////////

import torch  # noqa: E402
from flash_attn.cute.cute_dsl_utils import to_cute_tensor  # noqa: E402

from adasplash.forward.cute.cache import get_jit_cache  # noqa: E402
from adasplash.forward.cute import layout  # noqa: E402
from adasplash.forward.cute.layout import mask_dims  # noqa: E402,F401  (re-exported)


def get_tau_refine(q, k, taus_in, sm_scale=None, h=0.125, varlen=None,
                   refine_mode="fused", tile_m=128, tile_n=128, num_stages=3,
                   ptx_chunk=16, ping_pong=False, overlap=True):
    """Refine A's certified tau + build the bitpacked column mask, ONE pass.

    Args:
        q: ``(B, N_H, N_CTX, H_DIM)`` (cuda, bf16/fp16).
        k: ``(B, N_KV_H, N_CTX, H_DIM)`` (GQA; MHA when ``N_KV_H == N_H``).
        taus_in: ``(B, N_H, N_CTX)`` float32 -- A's output t, certified
            ``0 <= tau* - t < h`` per valid row.
        h: A's bracket width (1/BINS; 1/8 for the CuTe A at every length).
        sm_scale: softmax scale; defaults to ``1/sqrt(H_DIM)``.
        varlen: optional ``(B,)`` valid seqlens (dense-layout ``seqused``).
        refine_mode: 'fused' (Halley tau-hat) | 'mask_only' (tau-hat = t+h/2)
            | 'gemm_only' (DEBUG floor: no mask, no sums).
        overlap: pipelined schedule (default) vs serial; bit-identical outputs.

    Returns:
        ``(taus_out, mask, cnt)``:
          * taus_out ``(B, N_H, N_CTX)`` fp32, in [t, t+h];
          * mask ``(B, N_H, M64, W32)`` int32 bitpacked (u64-view compatible):
            bit b of word w of row i64 == key column 32w+b is live (score
            above the nudged t) for SOME query row in [64*i64, 64*i64+64);
          * cnt ``(B, N_H, M64)`` int32 == popcount of each mask row.
    """
    B, N_H, N_CTX, H_DIM, N_KV_H, qhead_per_kvhead = layout.qkv_dims(q, k)
    if sm_scale is None:
        sm_scale = layout.default_sm_scale(H_DIM)
    M64, W32 = mask_dims(N_CTX)

    ## Under varlen this kernel skips whole OOB tiles, so taus_out/mask/cnt must
    ## all be zero-filled -- NOT just mask/cnt.  taus_out used to be an
    ## unconditional torch.empty, which returned recycled memory (nan included)
    ## in OOB rows and made every consumer's OOB tau undefined.
    alloc = layout.output_allocator(varlen)
    taus_out = alloc((B, N_H, N_CTX), device=q.device, dtype=torch.float32)
    mask = alloc((B, N_H, M64, W32), device=q.device, dtype=torch.int32)
    cnt = alloc((B, N_H, M64), device=q.device, dtype=torch.int32)

    q_bshd, k_bshd = layout.bshd(q, k)
    seqused = layout.seqused_from_varlen(varlen)

    dtype = layout.cutlass_dtype(q)
    ## N_CTX is MANDATORY in this key: self.mask_words = 4*ceil(n_ctx/128) is a
    ## compile-time smem size, so every sequence length recompiles B.
    key = (dtype, H_DIM, qhead_per_kvhead, N_CTX, tile_m, tile_n, num_stages,
           refine_mode, ptx_chunk, ping_pong, overlap, seqused is not None)
    if key not in get_tau_refine.compile_cache:
        kern = AdaSplashTauRefineSm90(
            dtype,
            H_DIM,
            head_dim_v=H_DIM,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=True,
            is_local=False,
            pack_gqa=False,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=num_stages,
            n_ctx=N_CTX,
            refine_mode=refine_mode,
            ptx_chunk=ptx_chunk,
            ping_pong=ping_pong,
            overlap=overlap,
        )
        cq, ck = to_cute_tensor(q_bshd), to_cute_tensor(k_bshd)
        ctau_in = to_cute_tensor(taus_in, assumed_align=4)
        ctau_out = to_cute_tensor(taus_out, assumed_align=4)
        cmask = to_cute_tensor(mask, assumed_align=16)
        ccnt = to_cute_tensor(cnt, assumed_align=4)
        cseq = to_cute_tensor(seqused, assumed_align=4, leading_dim=0) if seqused is not None else None
        get_tau_refine.compile_cache[key] = cute.compile(
            kern, cq, ck, ctau_in, ctau_out, cmask, ccnt,
            Float32(sm_scale), Float32(h), None, None, cseq, cseq,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    get_tau_refine.compile_cache[key](
        q_bshd.detach(), k_bshd.detach(), taus_in, taus_out, mask, cnt,
        sm_scale, h, None, None, seqused, seqused
    )
    return taus_out, mask, cnt


get_tau_refine.compile_cache = get_jit_cache("adasplash_fwd_tau_refine")
