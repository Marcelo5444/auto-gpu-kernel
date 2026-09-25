# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Derived from flash_attn/cute/flash_fwd_sm90.py (SM90 forward pass), stripped to a
# Q@K^T-only kernel that computes the AdaSplash entmax-1.5 tau lower bound.
"""Single-pass certified tau estimation for entmax-1.5 attention (CuTe DSL, SM90).

V3 == the SHIPPED kernel: v2's arithmetic plus the phase-split intra-warpgroup
schedule that actually overlaps the integer insert with the wgmma (v2's
dual-buffer attempt is ptxas-serialized and net slower; see the pipelining
notes in ``mma``).  The serial-schedule predecessor it rolls back to is
recorded under "Rejected variants" below.

Same contract as ``adasplash.forward.triton.get_tau_hist``: one kernel replaces the
``get_tau`` (row max) + ``get_tau_v2`` (histogram around the seed) pair, and per
valid row the stored ``t`` satisfies ``0 <= tau* - t < 1/BINS``.

WHY THIS EXISTS -- the structural fact a tile DSL cannot express
----------------------------------------------------------------
The Triton kernel is pinned at ~64x64 tiles because its histogram is a live
``(BLOCK_M, BLOCK_N)`` uint64 tile: 32 KiB of register state.  That is not
because a histogram needs to be that big -- it is because in a tile language
``hist += ...`` is elementwise, so every (row, column) slot must own a private
counter, and there is no way to say "fold all of my columns into one register".

On SM90 the wgmma f32 accumulator hands each thread **exactly 2 rows** and
``tile_n/4`` columns of them (PTX ISA 9.7.14.5: ``row = groupID (+8)``,
``col = 2*threadID_in_group + (i&1)``; the 4 lanes of a quad share a row).  So
once we write the accumulator loop ourselves, the private histogram collapses
from ``2 rows x tile_n/4 columns`` of counters to ``2 rows x 1`` -- 4 registers
instead of ~64.  Three costs die together with that state:

  * the window slide stops touching a tile and becomes 2 shifts per row;
  * register pressure stops dictating the tile shape, so we can run 128x128 with
    two warpgroups over one shared K tile -- half the L2 traffic of 64x64;
  * BINS decouples from N_CTX (see ``layout.select_bins``), so h = 1/8 holds at every
    sequence length instead of degrading to 1/4 at 16K.

Notably this is NOT the warp-ballot scheme.  Ballots stay shelved, but for the
right reason: the per-bin LOP3+SWAR reduction after a ballot is warp-UNIFORM
work, which issues once per warp -- lane redundancy is free in SIMT issue
accounting, so that is not the cost.  The real tax is forming the per-bit vote
predicates: in the integer domain that is 4-5 int ops per element, erasing the
win; as fp32 compares against window edges (a binary search) it drops to
~2-2.5 shared int ops per element and could plausibly beat private counters by
1.5-2x on the insert alone.  Not pursued because the insert now runs in the
shadow of the GEMM (see the pipelining note below), where its absolute cost no
longer sets the wall clock.  The win here is state, not vote hardware.

Inner loop, per score element (detail in ``hist_update``)
---------------------------------------------------------
  * classification WITHOUT the conversion pipe and WITHOUT a multiply:
    ``fma.rm.f32(s_raw, cell_scale, 1.5*2**23)`` folds the cell scaling and the
    floor into ONE full-rate FP op (``cvt.rmi.s32.f32``, what Triton's
    ``float2int_rd`` lowers to, issues at a quarter of INT32 rate on Hopper).
    The low bits of the result ARE the biased cell index, exact for products
    inside +-2**22 (see MAGIC_BITS for why 1.5*2**23 and the verification).
  * the window test is the shift clamp itself: with BINS=8 and 8-bit fields an
    out-of-window cell shifts by >= 64 and ``shl.b64`` yields 0 (PTX
    semantics), so there is no compare, no index mask, and -inf masked lanes
    drop for free.
  * RELATIVE (not circular) window: field j counts cell ``A + j``, so a slide is
    a right shift by ``d*width`` bits and the solve needs no rotation.  The
    Triton kernel went circular because there a shift-based slide meant a
    data-dependent shift across the whole tile; with 2 registers per row the
    trade-off reverses.  ``shr.u64``/``shl.b64`` clamp to 0 past 64 bits (PTX),
    so one branch-free formula covers the whole 0..128 range, including the
    full-clear case.

Counters are TWO-TIER (see ``hist_update``): a narrow ``BINS x 8-bit`` uint64
takes every per-element insert, draining into wide ``BINS x 16-bit`` counters
every few K-blocks.  The narrow tier exists purely so the hot insert targets 64
bits rather than 128 -- measured at N=4096, the second shifted add costs more
than the whole rest of the insert (0.706ms vs 0.452ms).

Relative to the first CuTe port (v1, since deleted -- it certified but ran the
insert at ~7 ops/element), this kernel rebuilt the insert down to 3-4 ops
(0.487ms -> 0.319ms at N=4096) via the fused fma-floor and the clamp-as-window
test, and then PIPELINED it: the per-block work is split into a phase that
must read the accumulator (mask, max, slide, fma-floor into a staging buffer)
and a pure-integer phase that does not, and the integer phase runs while the
NEXT block's wgmma is in flight (see ``mma``).  ``overlap=False`` selects the
serial schedule -- bit-identical output, only the schedule differs.

Measured on GH200, B=4 H=32 D=128 bf16 causal, min over 4 rotated rounds of the
do_bench median (rep=1500); ``SUM`` = get_tau + get_tau_v2 (the two-pass pair
this replaces), ``tri`` = the tuned Triton one-shot, ``ser`` = this kernel on
the serial schedule (overlap=False, bit-identical output):

    N_CTX    SUM      tri      ser      v3      v3/SUM  v3/tri  v3/ser  v3/tau
     2048    0.399    0.343    0.221    0.205   1.95x   1.67x   1.08x   1.37x
     4096    1.345    1.166    0.748    0.672   2.00x   1.74x   1.11x   1.37x
     8192    4.939    4.300    2.737    2.510   1.97x   1.71x   1.09x   1.43x
    16384   18.704   16.329   10.479    9.620   1.94x   1.70x   1.09x   1.45x

At 16K this also holds 2x the bracket resolution of the Triton kernel (h=1/8
vs its forced h=1/4), now pinned by ``compare.py --n16k``'s float64
certificate rather than by a docstring.

WHERE THE FLOOR IS (N=4096 decomposition, short-rep, 2026-07-23 corrected).
``maxonly`` (GEMM + row max + slide) = 0.448ms.  ``phase1only`` = 0.536ms --
BUT only after ``hist_sink`` gave the staging buffer a live consumer: the
original "phase1 is free (0.464 == maxonly)" reading was a DEAD-CODE
ARTIFACT -- with ``ubuf`` write-only, the compiler deleted phase1's whole
per-element chain and the phase1only cubin came out byte-identical to
maxonly.  ``full`` = 0.672ms.  So the honest split of the ~0.22ms exposed
cost is ~0.09 phase1 (incl. the sink's 1 op/elt, so an upper bound) + ~0.14
phase2: exposure tracks TOTAL per-element op count, which retro-explains the
"paradoxical" nulls (slimming phase2 alone moved ~1% because phase2 was
never the whole exposed cost).  The schedule itself is exhausted -- v4's
clean two-accumulator wait_group(1) pipeline (v4; see "Rejected variants") measured
0.689 full / 0.463 maxonly: halving the intra-WG TC drains did not move the
GEMM-only floor (the other warpgroup's gemms already cover them), and every
intra-WG scheduling variant (phase placement, interleaving, cross-WG
stagger, pair schedule, pair+deferred-phase2 hybrid) lands within a few
percent of 0.67-0.71.  The kernel is issue-throughput-saturated across both
warpgroups; counter-level attribution stays blocked on this node
(ERR_NVGPUCTRPERM).  Going below this means fewer ops per element (the
fp32-compare ballot variant is the one known candidate -- reopen only if a
profile ever shows the int pipe alone exposed) or a different algorithm.

Rejected variants
-----------------
Both lived beside this file as ``cutedsl_impl_v2.py`` / ``cutedsl_impl_v4.py``
until the forward was packaged; they are deleted, not lost -- ``git show
f8014fd:adasplash/forward/get_tau_onepass/cutedsl_impl_v{2,4}.py``.

  * v2 -- the serial-schedule predecessor and rollback point for this kernel.
    Same arithmetic; ``hist_update`` monolithic instead of split into
    ``hist_phase1``/``hist_phase2``/``hist_sink``.  Superseded: the phase split
    is what the shipped pipelined schedule needs.
  * v4 -- the clean two-accumulator ``wait_group(1)`` pipeline referenced above.
    Bit-identical output, measured 0.689 full / 0.463 maxonly, i.e. 3-5% SLOWER.
    It is the runnable evidence for the refined ptxas rule (no group across a
    back-edge into ``wait(1)``; ``wait(0)`` exempt) and for the finding that the
    tensor cores are never intra-WG starved.
"""

from typing import Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, Uint64, Boolean, const_expr
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

from adasplash.forward.cute.utils import (
    MAGIC_BITS,
    fma_floor_bits,
    shl_u64,
    shr_u64,
    sqrt_rn,
    div_rn,
)
from adasplash.forward.cute.layout import select_bins


## Seeds the running row max ONLY -- masked scores themselves stay at the -inf
## that AttentionMask writes (-inf is the fmax identity, and the insert's shift
## clamp drops it; see hist_update).  The seed must be finite because the row
## max goes through the magic-constant floor, which needs |max| < 2**22.  Same
## constant the Triton kernel uses as its mask sentinel, with the same implicit
## assumption: every valid row's true max must exceed SMALL_NUMBER + BINS.
SMALL_NUMBER = -10000.0


# ///////////////////////////////////////////////////////////////////////////////
# Kernel
# ///////////////////////////////////////////////////////////////////////////////


class AdaSplashTauHistSm90(FlashAttentionForwardBase):
    """Q@K^T + sliding-window histogram + entmax-1.5 solve, warp-specialized.

    The producer warp issues TMA loads of Q and K (there is no V); the consumer
    warpgroups run wgmma and fold each score tile into their private per-row
    histograms.
    """

    def __init__(self, *args, bins: int = 8, ping_pong: bool = False,
                 hist_mode: str = 'full', bits_per_bin: int = 16, n_acc: int = 1,
                 overlap: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.bins = bins
        self.ping_pong = ping_pong
        ## Software-pipeline the consumer: issue block i's wgmma, then fold
        ## block i-1 into the histogram while it runs (two acc fragments,
        ## wait_group(1)).  False = the serial schedule (bit-identical result;
        ## GEMM and insert costs then ADD instead of overlapping).
        self.overlap = overlap
        ## 'full' | 'maxonly' (DEBUG: drop the insert to expose the
        ## GEMM+pipeline floor) | 'noslide' (insert but no window slide)
        self.hist_mode = hist_mode
        ## Number of partial pack accumulators per row.  Intended to break the
        ## insert's serial add-chain into n_acc independent chains -- but
        ## MEASURED NULL: n_acc in {1,2,4} all land at 0.77ms (N=4096).  The
        ## insert is int-pipe THROUGHPUT-bound, not latency-bound, so extra
        ## chains buy nothing and only cost registers.  Kept as a knob (default
        ## 1) so the finding is reproducible, not as an optimization.
        self.n_acc = n_acc
        self.bits_per_bin = bits_per_bin
        self.buffer_align_bytes = 1024
        self.cluster_shape_mn = (1, 1)
        assert self.bins in (2, 4, 8), "BINS must be a power of two <= 8"
        assert not self.pack_gqa, "pack_gqa not supported; GQA is handled by head indexing"
        assert not self.is_local, "sliding-window attention not supported"
        assert self.arch.is_family_of(Arch.sm_90a), "Only SM 9.x is supported"

    # -- layouts / mma -------------------------------------------------------

    def _setup_attributes(self):
        """Minimal replacement for the base version.

        The base ``_setup_attributes`` derives V and O copy layouts from an sV
        atom and computes five smem layouts that the SM90 path immediately
        overwrites.  With no V and no O tile, only Q and K remain.
        """
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

        @cute.struct
        class SharedStorageQK:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            sQ: sQ_struct
            sK: sK_struct

        return SharedStorageQK

    # -- host entry ----------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d) or (total_q, h, d) if cu_seqlens_q
        mK: cute.Tensor,  # (b_k, s_k, h_k, d) or (total_k, h_k, d) if cu_seqlens_k
        mTau: cute.Tensor,  # (b, h, s_q) fp32, or (h, total_q) if cu_seqlens_q
        softmax_scale: Float32,
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
        mTau = layout_utils.select(mTau, Tau_layout_transpose)

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
        # Cross-WG ping-pong on the shared wgmma path.  In the OVERLAP schedule
        # this is a COMPLETION stagger (arrive sits after wait_group(0)): a WG
        # issues its 8-HGMMA burst only after the other WG's burst retired, so
        # the issue never stalls on a full tensor-core queue -- the stall that
        # otherwise pins the warp at its wgmma and keeps it from reaching the
        # phase2 int work below (a warp issues in order).  In the SERIAL
        # schedule the arrive sits before wait_group(0) (mere issue), which
        # measured as a no-op -- kept only for the record.
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

        ## Fold the entmax-1.5 convention (0.5) and the cell quantization (BINS)
        ## into one scale.  BINS is a power of two, so the quantization commutes
        ## exactly with it.  Unlike Triton -- which prescales q -- we scale the
        ## accumulator, because Q reaches the wgmma as bf16 already in smem.
        cell_scale = softmax_scale * Float32(0.5 * self.bins)

        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            mTau,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            tma_atom_Q,
            tma_atom_K,
            cell_scale,
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
        mTau: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        cell_scale: Float32,
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
                mTau,
                sQ,
                sK,
                pipeline_q,
                pipeline_k,
                tidx,
                cell_scale,
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

                ## Walk K blocks from the diagonal backwards, exactly like the
                ## Triton kernel: the running max climbs fastest that way, so the
                ## window settles early and most later inserts fall out of it.
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

    @cute.jit
    def hist_update(
        self,
        acc_S: cute.Tensor,
        row_max: cute.Tensor,
        M_top: cute.Tensor,
        hist_pack: cute.Tensor,
        hist_lo: cute.Tensor,
        hist_hi: cute.Tensor,
        flush_cnt: cute.Tensor,
        cell_scale: Float32,
    ):
        """Fold one score tile into the private per-row histograms.

        TWO TIERS, because the per-element insert is the whole cost of this
        kernel and it is the only thing that scales with N^2:

          * ``hist_pack`` -- BINS fields x 8 bits in ONE uint64.  Every score
            element hits this, and a 64-bit target makes that a single
            ``shl.b64`` + single 64-bit add.  Measured: widening the target to
            128 bits (the obvious "just use 16-bit fields" layout) costs
            0.706ms vs 0.452ms at N=4096 -- the second shifted add is 56% more
            expensive than the entire rest of the insert.
          * ``hist_lo``/``hist_hi`` -- the same BINS fields at 16 bits across
            two uint64s.  ``hist_pack`` drains into this every FLUSH_EVERY
            K-blocks, before an 8-bit field could overflow.

        Field j of either tier counts absolute cell ``A + j`` with
        ``A = M_top - (BINS-1)``; field ``BINS-1`` is the max cell.  Slide
        first, then insert -- so a K-block that raises the max evicts the cells
        it pushes out before its own elements land.  Both tiers slide every
        iteration (by d*8 and d*16 bits respectively), which keeps their fields
        aligned so the drain is a plain field-wise add.

        v2 insert -- three ops per element, down from v1's ~seven:

          * ``fma.rm.f32(s_raw, cell_scale, MAGIC)`` folds the cell scaling and
            the floor into ONE full-rate FP op, so there is no per-element FMUL
            and no store-back of a rescaled tile (v1 did both).  ``M_top`` and
            the window base now live in this biased *bit* domain, never as a
            reconstructed integer.
          * ``rel = fma_floor_bits(s, cell_scale) - base`` with ``base = M_bits
            - (BINS-1)`` is a plain subtract whose result is the field index
            directly -- no ``& (BINS-1)`` mask.
          * THE keep test is the shift clamp itself.  With BINS=8 and 8-bit
            fields, ``BINS*PACK_BITS == 64``, so an out-of-window cell has
            ``rel >= BINS`` => ``rel*8 >= 64`` => ``shl.b64`` yields 0.  (No cell
            aliases back under 64: ``|c_s - A| < 2**23`` so ``rel*8`` can only be
            < 64 for ``rel in [0, BINS)``.)  Masked ``-inf`` lanes floor to a
            huge ``rel`` and clamp to 0 the same way -- which is why v2 needs no
            fmax-clamp fixup of the masked tile at all.
        """
        BINS: cutlass.Constexpr = self.bins
        PACK_BITS: cutlass.Constexpr = 8
        BPB: cutlass.Constexpr = self.bits_per_bin
        NA: cutlass.Constexpr = self.n_acc
        CLAMP_IS_KEEP: cutlass.Constexpr = BINS * PACK_BITS == 64
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        n_cols: cutlass.Constexpr = cute.size(acc_S_mn, mode=[1])
        ## a thread adds at most n_cols counts to one field per K-block (summed
        ## across all NA partials), so the flush cadence is set by n_cols, not by
        ## the per-partial rate.
        FLUSH_EVERY: cutlass.Constexpr = ((1 << PACK_BITS) - 1) // n_cols
        assert FLUSH_EVERY >= 1, f"n_cols={n_cols} overflows an 8-bit field in one block"

        for r in cutlass.range_constexpr(cute.size(row_max)):
            ## raw accumulator -- scaling is folded into every floor below
            row = acc_S_mn[r, None].load()

            ## -- running row max in RAW units: intra-thread tree, then across
            ## -- the 4 lanes of the quad that share this row.  -inf masked lanes
            ## -- are the identity for fmax, so no clamp is needed.
            m_cur = utils.fmax_reduce(row, init_val=row_max[r], arch=90)
            m_cur = cute.arch.warp_reduction_max(m_cur, threads_in_group=4)
            row_max[r] = m_cur
            ## biased bits of floor(cell_scale * m): MAGIC_BITS + max cell
            M_bits = fma_floor_bits(m_cur, cell_scale)

            ## -- slide: the window moved up by d cells (the MAGIC_BITS bias
            ## -- cancels in the difference), so fields shift down by d.  shr/shl
            ## -- clamp past 64, covering d in [0, BINS] including full-clear.
            if const_expr(self.hist_mode == "noslide"):
                M_top[r] = M_bits.bitcast(Int32)
            cells = cutlass.min(M_bits.bitcast(Int32) - M_top[r], Int32(BINS))
            d = Uint32(cells * BPB)
            dp = Uint32(cells * PACK_BITS)
            for a in cutlass.range_constexpr(NA):
                hist_pack[r * NA + a] = shr_u64(hist_pack[r * NA + a], dp)
            lo, hi = hist_lo[r], hist_hi[r]
            hist_lo[r] = (
                shr_u64(lo, d) | shl_u64(hi, Uint32(64) - d) | shr_u64(hi, d - Uint32(64))
            )
            hist_hi[r] = shr_u64(hi, d)
            M_top[r] = M_bits.bitcast(Int32)

            if const_expr(self.hist_mode == "maxonly"):
                continue
            ## window base in the biased bit domain: fma-floor bits of a cell-A value
            base = M_bits - Uint32(BINS - 1)
            ## -- THE hot loop.  The insert is a serial add chain on ONE
            ## -- accumulator; NA partials (round-robin by column) cut the chain
            ## -- to length n_cols/NA so the adds pipeline.  The round-robin index
            ## -- ``ai`` is a pure-python constant per unrolled iteration, so each
            ## -- write hits a compile-time-fixed slot of the rmem hist_pack --
            ## -- distinct slots => distinct registers => independent chains.
            ai = 0
            for c in cutlass.range_constexpr(n_cols):
                s = acc_S_mn[r, c]
                rel = fma_floor_bits(s, cell_scale) - base
                if const_expr(CLAMP_IS_KEEP):
                    inc = shl_u64(Uint64(1), rel * Uint32(PACK_BITS))
                else:
                    one = Uint64(Boolean(rel < Uint32(BINS)).to(Uint32))
                    inc = shl_u64(one, (rel & Uint32(BINS - 1)) * Uint32(PACK_BITS))
                hist_pack[r * NA + ai] = hist_pack[r * NA + ai] + inc
                ai = (ai + 1) % NA

        ## -- drain the narrow tier before any 8-bit field can overflow.  The
        ## -- counter is driven purely by the loop trip count, so this branch is
        ## -- uniform across the whole warpgroup -- no divergence.
        if const_expr(self.hist_mode != "maxonly"):
            flush_cnt[0] = flush_cnt[0] + 1
            if flush_cnt[0] >= Int32(FLUSH_EVERY):
                flush_cnt[0] = Int32(0)
                self.hist_flush(hist_pack, hist_lo, hist_hi)

    @cute.jit
    def hist_flush(self, hist_pack: cute.Tensor, hist_lo: cute.Tensor, hist_hi: cute.Tensor):
        """Drain BINS x 8-bit fields into BINS x 16-bit fields, then zero.

        First sum the NA partial accumulators of a row (fields are byte-disjoint
        counts, total <= 255, so a plain 64-bit add per partial suffices).  Then
        byte j of the total becomes 16-bit field j of the wide pair via the
        standard two-step spread: ``(x | x<<16) & 0x0000FFFF0000FFFF`` splits the
        halves, ``(t | t<<8) & 0x00FF00FF00FF00FF`` splits the bytes.  Both tiers
        were slid identically, so fields line up and the merge is a plain add.
        """
        NA: cutlass.Constexpr = self.n_acc
        num_rows: cutlass.Constexpr = cute.size(hist_pack) // NA
        M16: cutlass.Constexpr = 0x0000FFFF0000FFFF
        M8: cutlass.Constexpr = 0x00FF00FF00FF00FF
        for r in cutlass.range_constexpr(num_rows):
            p = hist_pack[r * NA]
            for a in cutlass.range_constexpr(1, NA):
                p = p + hist_pack[r * NA + a]
                hist_pack[r * NA + a] = Uint64(0)
            hist_pack[r * NA] = Uint64(0)
            x = p & Uint64(0xFFFFFFFF)
            x = (x | (x << Uint64(16))) & Uint64(M16)
            x = (x | (x << Uint64(8))) & Uint64(M8)
            hist_lo[r] = hist_lo[r] + x
            y = p >> Uint64(32)
            y = (y | (y << Uint64(16))) & Uint64(M16)
            y = (y | (y << Uint64(8))) & Uint64(M8)
            hist_hi[r] = hist_hi[r] + y

    @cute.jit
    def solve(
        self,
        row_max: cute.Tensor,
        M_top: cute.Tensor,
        hist_lo: cute.Tensor,
        hist_hi: cute.Tensor,
        cell_scale: Float32,
    ) -> cute.Tensor:
        """entmax-1.5 threshold from the per-row bin counts.

        Mirrors the Triton sweep exactly, in cell units (edges ``sj``, threshold
        ``BINS**2``), rescaling the accumulators by exact powers of two before
        the closing sqrt/div so those see bit-identical operands.

        ``M_top`` is the biased bit pattern ``MAGIC_BITS + max_cell`` and
        ``row_max`` is the RAW accumulator max, so both are converted back to
        cell units here (once per row, off the hot path).
        """
        BINS: cutlass.Constexpr = self.bins
        BPB: cutlass.Constexpr = self.bits_per_bin
        BIN_MASK: cutlass.Constexpr = (1 << self.bits_per_bin) - 1
        taus = cute.make_rmem_tensor(cute.size(row_max), Float32)

        for r in cutlass.range_constexpr(cute.size(row_max)):
            ## strip the MAGIC_BITS bias to recover the integer window base
            A_f = Float32(M_top[r] - Int32(MAGIC_BITS + (BINS - 1)))
            w_max = cell_scale * row_max[r] - A_f

            sum_zz = w_max * w_max
            sum_z = w_max
            sum_n = Float32(1.0)

            for sj in cutlass.range_constexpr(BINS - 1, -1, -1):
                ## field sj lives in hist_lo when sj*BPB < 64, else in hist_hi
                if const_expr(sj * BPB < 64):
                    packed = hist_lo[r] >> Uint64(sj * BPB)
                else:
                    packed = hist_hi[r] >> Uint64(sj * BPB - 64)
                c_bin = Float32((packed & Uint64(BIN_MASK)).to(Uint32))
                ## each row is spread over the 4 lanes of a quad
                c_bin = cute.arch.warp_reduction_sum(c_bin, threads_in_group=4)
                if const_expr(sj == BINS - 1):
                    c_bin = c_bin - 1.0  # the max itself is already seeded above

                c_tau = Float32(sj)
                new_zz = sum_zz + c_bin * (c_tau * c_tau)
                new_z = sum_z + c_bin * c_tau
                new_n = sum_n + c_bin

                good = new_n * c_tau * c_tau - 2.0 * new_z * c_tau + new_zz < Float32(BINS * BINS)
                sum_zz = new_zz if good else sum_zz
                sum_z = new_z if good else sum_z
                sum_n = new_n if good else sum_n

            ## back to score units by exact powers of two, THEN sqrt/div
            sum_z = sum_z * Float32(1.0 / BINS)
            sum_zz = sum_zz * Float32(1.0 / (BINS * BINS))
            A_val = A_f * Float32(1.0 / BINS)
            taus[r] = A_val + div_rn(
                sum_z - sqrt_rn(sum_z * sum_z - sum_n * (sum_zz - 1.0)), sum_n
            )
        return taus

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        mTau: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_k: pipeline.PipelineAsync,
        tidx: Int32,
        cell_scale: Float32,
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
            ## ONE persistent accumulator plus a u32 staging buffer of the same
            ## footprint (2 rows x tile_n/4 biased cell bits per thread).
            ##
            ## NOT the classic two-fragment double buffer -- that is a trap on
            ## sm90: ptxas statically tracks wgmma async groups, and when an
            ## accumulator register is written by wgmmas under RUNTIME branches
            ## (the natural parity-alternation loop), the tracking fails and
            ## ptxas serializes EVERY wgmma touching those registers with a
            ## full drain (SASS: WARPGROUP.DEPBAR.LE gsb0, 0x0 after each
            ## HGMMA; measured 24 drains and a net SLOWDOWN vs serial).  The
            ## phase-split below gets the same overlap with every wgmma in
            ## straight-line code -- the exact def-use shape the serial kernel
            ## already proved clean.
            acc_shape = tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
            acc_S = cute.make_rmem_tensor(acc_shape, Float32)
            ## 2 == num_rows (the wgmma 2-rows-per-thread fact, see below)
            ubuf = cute.make_rmem_tensor(2 * (self.tile_n // 4), Uint32)

        self.mma_init()

        ## num_rows == 2 on SM90: each thread owns rows r and r+8 of its warp's
        ## 16-row band.  That is the whole reason the histogram fits in registers.
        num_rows: cutlass.Constexpr = 2
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        M_top = cute.make_rmem_tensor(num_rows, Int32)
        hist_pack = cute.make_rmem_tensor(num_rows * self.n_acc, Uint64)  # narrow, hot
        hist_lo = cute.make_rmem_tensor(num_rows, Uint64)  # wide, drained into
        hist_hi = cute.make_rmem_tensor(num_rows, Uint64)
        flush_cnt = cute.make_rmem_tensor(1, Int32)

        q_consumer_phase = Int32(0)
        k_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages
        )
        if const_expr(self.overlap):
            ## Trails k_consumer_state by the one in-flight block: a K stage is
            ## released only after wait_group proves the GEMM that read it
            ## retired.  Both counters advance once per block, so they stay one
            ## block apart inside a tile and equal at tile boundaries.
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
            ## Blocks in [n_block_min_causal_mask, n_block_max) touch the causal
            ## diagonal and need masking; blocks below are clean interior.
            n_block_min_causal_mask = (
                block_info.get_n_block_min_causal_local_mask(seqlen, m_block, n_block_min)
                if const_expr(self.is_causal)
                else n_block_min
            )

            row_max.fill(SMALL_NUMBER)
            M_top.fill(Int32(0))  # biased-bit domain; first slide clears
            hist_pack.fill(Uint64(0))
            hist_lo.fill(Uint64(0))
            hist_hi.fill(Uint64(0))
            flush_cnt.fill(Int32(0))

            pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)

            if const_expr(self.overlap):
                ## ----------------------------------------------------------------
                ## Intra-warpgroup software pipeline, PHASE-SPLIT around one acc.
                ##
                ## The per-block work is split by WHAT IT TOUCHES:
                ##   phase1 -- everything that must read the accumulator: mask,
                ##     row max, window slide, and the fma-floor of every element
                ##     into ``ubuf`` (the fma's destination register IS the ubuf
                ##     slot, so staging costs zero extra instructions).  Runs
                ##     with the async pipe EMPTY: after wait_group(0), before
                ##     the next GEMM is issued.
                ##   phase2 -- the pure-integer insert out of ``ubuf``.  Touches
                ##     no accumulator register, so it runs AFTER block i's wgmma
                ##     is issued -- i.e. the int-pipe work executes entirely in
                ##     the tensor core's shadow, which is the point of the
                ##     whole exercise.
                ##
                ## Every wgmma is in straight-line code (see the note at the
                ## acc_S allocation for why that is load-bearing); the only
                ## dynamic branches are the mask selects, which run at
                ## pipe-empty points and touch memory only.  K stages release
                ## at the top of the next iteration (the GEMM that read them
                ## has provably retired), so TMA needs num_stages >= 3.
                ## ----------------------------------------------------------------
                mask_block_fn = partial(
                    self.mask_block,
                    n_block_max=n_block_max,
                    n_block_min_causal_mask=n_block_min_causal_mask,
                    mask_seqlen_fn=partial(mask_fn, mask_seqlen=True),
                    mask_inner_fn=partial(mask_fn, mask_seqlen=False),
                )
                n_blocks = n_block_max - n_block_min
                ## prologue: issue the diagonal block.  The scheduler-barrier
                ## sync/arrive calls are the cross-WG completion stagger --
                ## compiled out unless ping_pong=True, which measured SLOWER
                ## (see the steady-state note below); kept for the record.
                pipeline_k.consumer_wait(
                    k_consumer_state, pipeline_k.consumer_try_wait(k_consumer_state)
                )
                self.warp_scheduler_barrier_sync()
                sm90_utils.gemm_w_idx(
                    tiled_mma_qk, acc_S, tSrQ, tSrK, True,
                    B_idx=k_consumer_state.index, wg_wait=-1,
                )
                k_consumer_state.advance()
                ## steady state: finish block i-1 (phase1), issue block i, then
                ## run block i-1's insert (phase2) while block i's GEMM is in
                ## flight.  The stage release also sits in the shadow.
                ##
                ## Two placement variants were built and measured NULL at
                ## N=4096: interleaving phase2 chunks BETWEEN the 8 wgmma
                ## issues (survived in SASS, changed nothing), and a strict
                ## cross-WG completion stagger via the scheduler barrier
                ## (ping_pong=True: 0.776ms vs 0.685ms -- the forced TC
                ## alternation costs more than it saves).  The exposure of the
                ## int work is placement-invariant; what actually helped was
                ## making phase2 SMALLER (see hist_phase1's shift-amount
                ## staging).
                for i in cutlass.range(1, n_blocks, 1, unroll=1):
                    warpgroup.wait_group(0)
                    self.warp_scheduler_barrier_arrive()
                    mask_block_fn(acc_S, jp=i - 1)
                    self.hist_phase1(
                        acc_S, ubuf, row_max, M_top, hist_pack, hist_lo, hist_hi, cell_scale
                    )
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
                    if const_expr(self.hist_mode not in ("maxonly", "phase1only")):
                        self.hist_phase2(ubuf, hist_pack, hist_lo, hist_hi, flush_cnt)
                    if const_expr(self.hist_mode == "phase1only"):
                        self.hist_sink(ubuf, hist_pack)
                ## epilogue: the last block has no successor GEMM to hide under
                warpgroup.wait_group(0)
                self.warp_scheduler_barrier_arrive()
                pipeline_k.consumer_release(k_release_state)
                k_release_state.advance()
                mask_block_fn(acc_S, jp=n_blocks - 1)
                self.hist_phase1(
                    acc_S, ubuf, row_max, M_top, hist_pack, hist_lo, hist_hi, cell_scale
                )
                if const_expr(self.hist_mode not in ("maxonly", "phase1only")):
                    self.hist_phase2(ubuf, hist_pack, hist_lo, hist_hi, flush_cnt)
                if const_expr(self.hist_mode == "phase1only"):
                    self.hist_sink(ubuf, hist_pack)
            else:
                hist_one_n_block = partial(
                    self.one_n_block,
                    mma_qk_fn=mma_qk_fn,
                    pipeline_k=pipeline_k,
                    row_max=row_max,
                    M_top=M_top,
                    hist_pack=hist_pack,
                    hist_lo=hist_lo,
                    hist_hi=hist_hi,
                    flush_cnt=flush_cnt,
                    cell_scale=cell_scale,
                )
                ## Serial schedule: blocks needing a mask are separated from the
                ## clean interior at trace time, as in the reference kernel.
                k_consumer_state = hist_one_n_block(
                    k_consumer_state,
                    n_block=n_block_max - 1,
                    mask_fn=partial(mask_fn, mask_seqlen=True),
                )
                for n_tile in cutlass.range(n_block_max - 1 - n_block_min_causal_mask, unroll=1):
                    k_consumer_state = hist_one_n_block(
                        k_consumer_state,
                        n_block=n_block_max - 2 - n_tile,
                        mask_fn=partial(mask_fn, mask_seqlen=False),
                    )
                n_block_upper = cutlass.min(n_block_max - 1, n_block_min_causal_mask)
                for n_tile in cutlass.range(n_block_upper - n_block_min, unroll=1):
                    k_consumer_state = hist_one_n_block(
                        k_consumer_state, n_block=n_block_upper - 1 - n_tile, mask_fn=None
                    )

            pipeline_q.consumer_release_w_index(0)
            q_consumer_phase ^= 1

            ## mandatory final drain: whatever is still in the narrow tier has
            ## not reached the wide counters the solve reads.
            self.hist_flush(hist_pack, hist_lo, hist_hi)
            taus = self.solve(row_max, M_top, hist_lo, hist_hi, cell_scale)
            self.store_tau(taus, mTau, seqlen, thr_mma_qk, tidx, m_block, head_idx, batch_idx)

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def one_n_block(
        self,
        k_consumer_state,
        n_block: Int32,
        mma_qk_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        row_max: cute.Tensor,
        M_top: cute.Tensor,
        hist_pack: cute.Tensor,
        hist_lo: cute.Tensor,
        hist_hi: cute.Tensor,
        flush_cnt: cute.Tensor,
        cell_scale: Float32,
        mask_fn: Optional[Callable] = None,
    ):
        pipeline_k.consumer_wait(k_consumer_state, pipeline_k.consumer_try_wait(k_consumer_state))
        ## Ping-pong on the shared wgmma issue port: acquire BEFORE issuing,
        ## release right after.  The reference kernel pairs its arrive with the
        ## sync that guards the PV GEMM; with only one GEMM per iteration the
        ## pair has to close inside this function, or the last iteration leaves
        ## a sync with no matching arrive and the two warpgroups deadlock.
        self.warp_scheduler_barrier_sync()
        acc_S = mma_qk_fn(B_idx=k_consumer_state.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(k_consumer_state)
        k_consumer_state.advance()

        if const_expr(mask_fn is not None):
            ## AttentionMask writes -inf; v2 needs no clamp -- fma-floor sends a
            ## -inf lane to a huge rel and the shift clamp drops it (see
            ## hist_update).  -inf is also the identity for the fmax row max.
            mask_fn(acc_S=acc_S, n_block=n_block)

        self.hist_update(
            acc_S, row_max, M_top, hist_pack, hist_lo, hist_hi, flush_cnt, cell_scale
        )
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
        ``jp`` (0 == the diagonal block ``n_block_max-1``).

        In the pipelined schedule a block's segment (diagonal / causal-masked /
        interior) is not knowable at trace time, so both variants are traced
        and selected by DYNAMIC branches.  The branches only mutate the
        accumulator in place -- one uniform branch, no state merging -- and
        they run at pipe-empty points, so they never sit between a wgmma and
        its wait.  Skipping the mask for interior blocks is exact: every
        element of a full block sits on or below the causal diagonal.
        """
        n_block = n_block_max - 1 - jp
        if jp == 0:
            ## the diagonal block also carries the ragged seqlen tail
            mask_seqlen_fn(acc_S=acc_S, n_block=n_block)
        else:
            if n_block >= n_block_min_causal_mask:
                mask_inner_fn(acc_S=acc_S, n_block=n_block)

    @cute.jit
    def hist_phase1(
        self,
        acc_S: cute.Tensor,
        ubuf: cute.Tensor,
        row_max: cute.Tensor,
        M_top: cute.Tensor,
        hist_pack: cute.Tensor,
        hist_lo: cute.Tensor,
        hist_hi: cute.Tensor,
        cell_scale: Float32,
    ):
        """Everything that must READ the accumulator: row max, window slide,
        and the fma-floor of every element into ``ubuf`` (biased cell bits).

        The arithmetic is hist_update's, verbatim -- only the insert is
        deferred.  Storing the fma result costs nothing extra: the fma's
        destination register IS the ubuf slot the insert later reads.  Runs
        with the async pipe empty, so the accumulator's def-use never crosses
        an in-flight wgmma (the property ptxas needs; see mma()).
        """
        BINS: cutlass.Constexpr = self.bins
        PACK_BITS: cutlass.Constexpr = 8
        BPB: cutlass.Constexpr = self.bits_per_bin
        NA: cutlass.Constexpr = self.n_acc
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        n_cols: cutlass.Constexpr = cute.size(acc_S_mn, mode=[1])

        for r in cutlass.range_constexpr(cute.size(row_max)):
            row = acc_S_mn[r, None].load()
            m_cur = utils.fmax_reduce(row, init_val=row_max[r], arch=90)
            m_cur = cute.arch.warp_reduction_max(m_cur, threads_in_group=4)
            row_max[r] = m_cur
            M_bits = fma_floor_bits(m_cur, cell_scale)

            if const_expr(self.hist_mode == "noslide"):
                M_top[r] = M_bits.bitcast(Int32)
            cells = cutlass.min(M_bits.bitcast(Int32) - M_top[r], Int32(BINS))
            d = Uint32(cells * BPB)
            dp = Uint32(cells * PACK_BITS)
            for a in cutlass.range_constexpr(NA):
                hist_pack[r * NA + a] = shr_u64(hist_pack[r * NA + a], dp)
            lo, hi = hist_lo[r], hist_hi[r]
            hist_lo[r] = (
                shr_u64(lo, d) | shl_u64(hi, Uint32(64) - d) | shr_u64(hi, d - Uint32(64))
            )
            hist_hi[r] = shr_u64(hi, d)
            M_top[r] = M_bits.bitcast(Int32)

            if const_expr(self.hist_mode != "maxonly"):
                ## Store the SHIFT AMOUNT, not the raw bits: the subtract and
                ## the *PACK_BITS run here, where they hide in phase1's issue
                ## slack, leaving phase2 at two ops per element (64-bit shift +
                ## 64-bit add).  Computing rel*8 mod 2**32 is safe for the
                ## clamp-as-window-test: |rel*8| < 2**29, so wrapped values
                ## land in [2**32 - 2**29, 2**32) or [64, 2**29) -- never back
                ## inside [0, 64).
                base = M_bits - Uint32(BINS - 1)
                for c in cutlass.range_constexpr(n_cols):
                    rel = fma_floor_bits(acc_S_mn[r, c], cell_scale) - base
                    if const_expr(BINS * PACK_BITS == 64):
                        ubuf[r * n_cols + c] = rel * Uint32(PACK_BITS)
                    else:
                        ubuf[r * n_cols + c] = rel

    @cute.jit
    def hist_phase2(
        self,
        ubuf: cute.Tensor,
        hist_pack: cute.Tensor,
        hist_lo: cute.Tensor,
        hist_hi: cute.Tensor,
        flush_cnt: cute.Tensor,
    ):
        """The pure-integer insert out of ``ubuf`` -- hist_update's hot loop.

        Touches no accumulator register, which is what lets it run while the
        next block's wgmma is in flight.  Phase1 already subtracted the window
        base and (for the clamp-is-keep layout) pre-multiplied by PACK_BITS,
        so per element this is ONE 64-bit shift and ONE 64-bit add -- the
        irreducible histogram update.  Bit-identical to the fused path: same
        adds, and uint64 adds commute exactly.
        """
        BINS: cutlass.Constexpr = self.bins
        PACK_BITS: cutlass.Constexpr = 8
        NA: cutlass.Constexpr = self.n_acc
        CLAMP_IS_KEEP: cutlass.Constexpr = BINS * PACK_BITS == 64
        num_rows: cutlass.Constexpr = 2
        n_cols: cutlass.Constexpr = cute.size(ubuf) // num_rows
        FLUSH_EVERY: cutlass.Constexpr = ((1 << PACK_BITS) - 1) // n_cols

        for r in cutlass.range_constexpr(num_rows):
            ai = 0
            for c in cutlass.range_constexpr(n_cols):
                if const_expr(CLAMP_IS_KEEP):
                    inc = shl_u64(Uint64(1), ubuf[r * n_cols + c])
                else:
                    rel = ubuf[r * n_cols + c]
                    one = Uint64(Boolean(rel < Uint32(BINS)).to(Uint32))
                    inc = shl_u64(one, (rel & Uint32(BINS - 1)) * Uint32(PACK_BITS))
                hist_pack[r * NA + ai] = hist_pack[r * NA + ai] + inc
                ai = (ai + 1) % NA

        flush_cnt[0] = flush_cnt[0] + 1
        if flush_cnt[0] >= Int32(FLUSH_EVERY):
            flush_cnt[0] = Int32(0)
            self.hist_flush(hist_pack, hist_lo, hist_hi)

    @cute.jit
    def hist_sink(self, ubuf: cute.Tensor, hist_pack: cute.Tensor):
        """DCE guard for the ``phase1only`` decomposition mode ONLY.

        Without a consumer, ``ubuf`` is a write-only rmem tensor and the
        compiler deletes phase1's entire per-element chain (fma-floor,
        subtract, x8) -- measured: the phase1only cubin came out BYTE-IDENTICAL
        to maxonly, so the old "phase1 is free" reading was an artifact.  This
        XOR-folds every slot into ``hist_pack`` (live through flush -> solve ->
        gmem) at ONE int op per element, the cheapest real data dependence.
        phase1only therefore measures phase1 + 1 op/element -- an UPPER bound
        on phase1's true cost.
        """
        num_rows: cutlass.Constexpr = 2
        n_cols: cutlass.Constexpr = cute.size(ubuf) // num_rows
        NA: cutlass.Constexpr = self.n_acc
        for r in cutlass.range_constexpr(num_rows):
            sink = Uint32(0)
            for c in cutlass.range_constexpr(n_cols):
                sink = sink ^ ubuf[r * n_cols + c]
            hist_pack[r * NA] = hist_pack[r * NA] ^ Uint64(sink)

    @cute.jit
    def store_tau(
        self,
        taus: cute.Tensor,
        mTau: cute.Tensor,
        seqlen: SeqlenInfoQK,
        thr_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        """rmem -> gmem, no smem and no TMA.

        The same trick the reference epilogue uses for LSE: give the rank-1
        ``(tile_m,)`` row tile a stride-0 second mode so it can be partitioned by
        the MMA C layout, then let only the column-0 lane of each quad store.
        """
        mTau_cur = seqlen.offset_batch_Q(mTau, batch_idx, dim=2)[None, head_idx]
        gTau = cute.local_tile(mTau_cur, (self.tile_m,), (m_block,))
        gTau_expanded = cute.make_tensor(
            gTau.iterator, cute.append(gTau.layout, cute.make_layout((self.tile_n,), stride=(0,)))
        )
        taccTgTau = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gTau_expanded))
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        taccTcT = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS))
        t0accTcT = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cS))

        if taccTcT[0][1] == 0:
            for m in cutlass.range_constexpr(cute.size(taus)):
                if t0accTcT[m, 0][0] < seqlen.seqlen_q - m_block * self.tile_m - taccTcT[0][0]:
                    taccTgTau[m, 0] = taus[m]

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
# Host wrapper -- drop-in for adasplash.forward.triton.get_tau_hist
# ///////////////////////////////////////////////////////////////////////////////

import torch  # noqa: E402
from flash_attn.cute.cute_dsl_utils import to_cute_tensor  # noqa: E402

from adasplash.forward.cute.cache import get_jit_cache  # noqa: E402
from adasplash.forward.cute import layout  # noqa: E402


def get_tau_hist(q, k, sm_scale=None, varlen=None, tile_m=128, tile_n=128,
                 num_stages=3, ping_pong=False, hist_mode='full', bins=None,
                 bits_per_bin=16, n_acc=1, overlap=True):
    """Certified entmax-1.5 tau lower bound in ONE pass (CuTe DSL, SM90).

    Args:
        q: ``(B, N_H, N_CTX, H_DIM)`` (same layout as the Triton kernel).
        k: ``(B, N_KV_H, N_CTX, H_DIM)`` (GQA; MHA when ``N_KV_H == N_H``).
        sm_scale: softmax scale; defaults to ``1/sqrt(H_DIM)``.
        varlen: optional ``(B,)`` int32 valid seqlens.
        overlap: pipelined schedule (default) vs serial; bit-identical outputs.
        hist_mode: 'full' | decomposition knobs 'maxonly' (GEMM+max+slide
            only), 'phase1only' (adds the fma-floor staging + a 1-op/elt XOR
            sink so DCE cannot delete it -- an UPPER bound on phase1; see
            hist_sink), 'noslide' (debug).  num_stages: K smem stages; the
            pipelined schedule releases a stage one block late, so keep >= 3.

    Returns:
        ``taus`` of shape ``(B, N_H, N_CTX)`` float32 with, per valid row,
        ``0 <= tau* - taus < 1/BINS``; refine on ``[taus, taus + 1/BINS]``.
    """
    B, N_H, N_CTX, H_DIM, N_KV_H, qhead_per_kvhead = layout.qkv_dims(q, k)
    if sm_scale is None:
        sm_scale = layout.default_sm_scale(H_DIM)
    ## h = 1/bins.  The two-tier counters decouple BINS from N_CTX, so this
    ## returns 8 at every practical sequence length (see layout.select_bins).
    bins = bins if bins is not None else select_bins(N_CTX, tile_n)

    ## varlen skips whole OOB tiles, so those rows are never written -- zero-fill
    ## them rather than handing back recycled memory (see fwd_tau_refine).
    taus = layout.output_allocator(varlen)(
        (B, N_H, N_CTX), device=q.device, dtype=torch.float32
    )

    q_bshd, k_bshd = layout.bshd(q, k)
    seqused = layout.seqused_from_varlen(varlen)

    dtype = layout.cutlass_dtype(q)
    key = (dtype, H_DIM, qhead_per_kvhead, tile_m, tile_n, num_stages, bins,
           ping_pong, hist_mode, bits_per_bin, n_acc, overlap, seqused is not None)
    if key not in get_tau_hist.compile_cache:
        kern = AdaSplashTauHistSm90(
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
            bins=bins,
            ping_pong=ping_pong,
            hist_mode=hist_mode,
            bits_per_bin=bits_per_bin,
            n_acc=n_acc,
            overlap=overlap,
        )
        cq, ck = to_cute_tensor(q_bshd), to_cute_tensor(k_bshd)
        ctau = to_cute_tensor(taus, assumed_align=4)
        cseq = to_cute_tensor(seqused, assumed_align=4, leading_dim=0) if seqused is not None else None
        get_tau_hist.compile_cache[key] = cute.compile(
            kern, cq, ck, ctau, Float32(sm_scale), None, None, cseq, cseq,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    get_tau_hist.compile_cache[key](
        q_bshd.detach(), k_bshd.detach(), taus, sm_scale, None, None, seqused, seqused
    )
    return taus


get_tau_hist.compile_cache = get_jit_cache("adasplash_fwd_tau_hist")
