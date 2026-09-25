#!/usr/bin/env python3
"""
AdaSplash get_output kernel - baseline implementation for auto-gpu-kernel optimization.
This is a simplified self-contained version of the AdaSplashOutputSm90 kernel from
adasplash/forward/cute/fwd_output.py (merge-cutile branch).

Target: Kernel C - no-union 64-row tiles, 2 CTAs/SM, adaptive mask decode + gather + wgmma.
"""

from typing import Optional, Callable
from functools import partial
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute import pipeline as pipeline_custom
from flash_attn.cute.named_barrier import NamedBarrierFwd
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
)

# Import our local named barriers
from named_barrier import NamedBarrierOutput


def _frgA_pair_map(acc_S: cute.Tensor):
    """Trace-time (plain python): for each linear slot of the frgA view of
    the S accumulator, the (row-pair, col) coordinate of the SAME register in
    the mn view. Both views alias one rmem array, so composing the static
    layouts gives the correspondence the packed cvt needs."""
    mn_view = layout_utils.reshape_acc_to_mn(acc_S)
    frg_view = layout_utils.reshape_acc_to_frgA(acc_S)
    reg2rc = {}
    for r in range(cute.size(mn_view, mode=[0])):
        for c in range(cute.size(mn_view, mode=[1])):
            reg2rc[mn_view.layout((r, c))] = (r, c)
    return tuple(reg2rc[frg_view.layout(i)] for i in range(cute.size(frg_view)))


BAND_CAP = 64  # the in-span suffix is at most 64 columns (2 mask words)


class AdaSplashOutputSm90:
    """Adaptive one-pass mask decode through a ring + K/V gather + dense wgmma."""

    def __init__(self, dtype, head_dim: int, qhead_per_kvhead: int = 1,
                 tile_g: int = 64, num_stages: int = 2,
                 need_backward: bool = True, mode: str = "full",
                 overlap: bool = False, ring_log2: int = 13,
                 wide_index: bool = False,
                 mma_regs: int = 232, producer_regs: int = 24):
        self.dtype = dtype
        self.head_dim = head_dim
        self.qhead_per_kvhead = qhead_per_kvhead
        self.tile_m = 64
        self.tile_g = tile_g
        self.num_stages = num_stages
        self.need_backward = need_backward
        self.mode = mode
        self.overlap = overlap
        self.ring_capacity = 1 << ring_log2
        self.idx_dtype = cutlass.Uint32 if wide_index else cutlass.Uint16
        self.wide_index = wide_index
        self.num_mma_regs, self.num_producer_regs = (mma_regs, producer_regs)
        assert mma_regs + producer_regs <= 256
        assert mode in ("full", "no_pv", "gemm_only")
        assert tile_g in (32, 64, 128)
        assert head_dim == 128, "gather chunking assumes 256B K/V rows"
        assert self.ring_capacity >= 4096 + tile_g, "ring must hold a round + carry"

    # -- host entry ----------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s, h, d)
        mK: cute.Tensor,  # (b, s, h_k, d)
        mV: cute.Tensor,  # (b, s, h_k, d)
        mTau: cute.Tensor,  # (b, h, s) fp32 -- B's tau-hat
        mMask: cute.Tensor,  # (b, h, m64, w32) int32 -- B's column mask
        mOut: cute.Tensor,  # (b, s, h, d)
        mOut2: cute.Tensor,  # (b, s, h, d) -- UNNORMALIZED (proj @ V), raw units
        softmax_scale: Float32,
        mSupp: cutlass.Constexpr[Optional[cute.Tensor]] = None,  # (b, h, s) fp32 sidecar
        mSeqUsedQ: cutlass.Constexpr[Optional[cute.Tensor]] = None,
        stream: cuda.CUstream = None,
    ):
        mQ, mK, mV, mOut, mOut2 = [
            assume_tensor_aligned(t) for t in (mQ, mK, mV, mOut, mOut2)
        ]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, [1, 3, 2, 0]))
        mOut = cute.make_tensor(mOut.iterator, cute.select(mOut.layout, [1, 3, 2, 0]))
        mOut2 = cute.make_tensor(mOut2.iterator, cute.select(mOut2.layout, [1, 3, 2, 0]))
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, [1, 3, 2, 0]))
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, [1, 3, 2, 0]))
        mTau = cute.make_tensor(mTau.iterator, cute.select(mTau.layout, [2, 1, 0]))
        mMask = cute.make_tensor(mMask.iterator, cute.select(mMask.layout, [3, 2, 1, 0]))
        if const_expr(mSupp is not None):
            mSupp = layout_utils.select(mSupp, [2, 1, 0])

        self.sQ_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_m, self.head_dim), None
        )
        self.sK_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_g, self.head_dim), self.num_stages
        )
        tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            Float32, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.tile_g),
        )
        tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype, self.dtype,
            warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            Float32, atom_layout_mnk=(1, 1, 1), tiler_mn=(64, self.head_dim),
            a_source=warpgroup.OperandSource.RMEM,
        )
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mQ, self.sQ_layout,
            (self.tile_m, self.head_dim)
        )
        self.tma_bytes_Q = cute.size_in_bytes(
            self.dtype, cute.select(self.sQ_layout, mode=[0, 1])
        )

        if const_expr(mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileLPTScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            1,  # num_splits
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mQ.shape[1],
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_g),
            mCuSeqlensQ=None,
            mSeqUsedQ=mSeqUsedQ,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=True,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        half_scale = softmax_scale * Float32(0.5)
        inv_half = Float32(2.0) / softmax_scale

        self.kernel(
            tma_tensor_Q, mK, mV, mTau, mMask, mOut, mOut2, mSupp, mSeqUsedQ,
            tma_atom_Q, half_scale, inv_half,
            self.sQ_layout, self.sK_layout, tiled_mma_qk, tiled_mma_pv,
            tile_sched_params, TileScheduler,
        ).launch(
            grid=grid_dim, block=[256, 1, 1], stream=stream, min_blocks_per_mp=2,
        )

    # -- device --------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mTau: cute.Tensor,
        mMask: cute.Tensor,
        mOut: cute.Tensor,
        mOut2: cute.Tensor,
        mSupp: cutlass.Constexpr[Optional[cute.Tensor]],
        mSeqUsedQ: cutlass.Constexpr[Optional[cute.Tensor]],
        tma_atom_Q: cute.CopyAtom,
        half_scale: Float32,
        inv_half: Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)

        smem = cutlass.utils.SmemAllocator()
        mbar_q = smem.allocate_array(cutlass.Int64, 1 * 2)
        mbar_k = smem.allocate_array(cutlass.Int64, self.num_stages * 2)
        mbar_v = smem.allocate_array(cutlass.Int64, self.num_stages * 2)
        mbar_i = smem.allocate_array(cutlass.Int64, 2 * 2)
        sNLive = smem.allocate_tensor(Int32, cute.make_layout(2), 16)
        sWarpTot = smem.allocate_tensor(Int32, cute.make_layout(4), 16)
        sBand = smem.allocate_tensor(
            self.idx_dtype, cute.make_layout(2 * BAND_CAP), 16)
        sRing = smem.allocate_tensor(
            self.idx_dtype, cute.make_layout(self.ring_capacity), 16)
        sO = smem.allocate_tensor(
            self.dtype, sQ_layout.outer, 1024, swizzle=sQ_layout.inner)
        sQ = smem.allocate_tensor(
            self.dtype, sQ_layout.outer, 1024, swizzle=sQ_layout.inner)
        sK = smem.allocate_tensor(
            self.dtype, sK_layout.outer, 1024, swizzle=sK_layout.inner)
        sV = smem.allocate_tensor(
            self.dtype, sK_layout.outer, 1024, swizzle=sK_layout.inner)

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(128)
        mma_warps = ThreadCooperativeGroup(128 // cute.arch.WARP_SIZE)
        pipeline_q = pipeline_custom.PipelineTmaAsync.create(
            barrier_storage=mbar_q, num_stages=1, producer_group=tma_warp,
            consumer_group=mma_warps, tx_count=self.tma_bytes_Q, defer_sync=True,
        )
        pipeline_k = pipeline_custom.PipelineCpAsync.create(
            barrier_storage=mbar_k, num_stages=self.num_stages,
            producer_group=load_threads, consumer_group=mma_warps,
            defer_sync=True, elect_one_release=True, syncwarp_before_release=False,
        )
        pipeline_v = pipeline_custom.PipelineCpAsync.create(
            barrier_storage=mbar_v, num_stages=self.num_stages,
            producer_group=load_threads, consumer_group=mma_warps,
            defer_sync=True, elect_one_release=True, syncwarp_before_release=False,
        )
        pipeline_i = pipeline_custom.PipelineAsync.create(
            barrier_storage=mbar_i, num_stages=2,
            producer_group=load_threads, consumer_group=mma_warps,
            defer_sync=True, elect_one_release=True, syncwarp_before_release=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype, num_bits_per_copy=128,
        )
        thr_layout = cute.make_ordered_layout((16, 8), order=(1, 0))
        val_layout = cute.make_layout((1, 8))
        gmem_tiled_copy_KV = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)

        if warp_idx < 4:  # Producer
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.prod(mQ, mK, mV, mMask, mSeqUsedQ, sQ, sK, sV, sRing, sBand,
                      sNLive, sWarpTot, tma_atom_Q, pipeline_q,
                      pipeline_k, pipeline_v, pipeline_i, gmem_tiled_copy_KV,
                      tile_sched_params, TileScheduler)
        else:  # Consumers
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            self.cons(mQ, mTau, mMask, mOut, mOut2, mSupp, mSeqUsedQ, sQ, sK, sV,
                      sO, sBand, sNLive, pipeline_q, pipeline_k, pipeline_v,
                      pipeline_i, tiled_mma_qk, tiled_mma_pv,
                      half_scale, inv_half, tile_sched_params, TileScheduler)

    # -- producer ------------------------------------------------------------

    @cute.jit
    def decode_round(self, gMask, sRing, sWarpTot, W: Int32,
                     rnd: Int32, base: Int32, ptid: Int32, warp: Int32,
                     lane: Int32) -> Int32:
        """One v1-style decode round (128 words = 4096 columns): popc + warp
        prefix + cross-warp scan give every thread its output range; bits go
        to the ring at wrapped positions. Two producer barriers, exactly as
        v1's decode_mask. Dense words take a 32-flat-store path (no serial
        peel dependency chain). Returns the round's total."""
        rmask: cutlass.Constexpr = self.ring_capacity - 1
        w_idx = ptid + rnd * 128
        w_safe = cutlass.min(w_idx, W - 1)
        u = gMask[w_safe].to(Uint32)
        u = u if w_idx < W else Uint32(0)
        pc = Int32(cute.arch.popc(u))
        incl = layout_utils.warp_prefix_sum(pc, lane)
        if lane == 31:
            sWarpTot[warp] = incl
        cute.arch.barrier(barrier_id=NamedBarrierOutput.ProducerDecode,
                          number_of_threads=128)
        wbase = Int32(0)
        rtot = Int32(0)
        for wv in cutlass.range_constexpr(4):
            t = sWarpTot[wv]
            wbase = wbase + (t if wv < warp else Int32(0))
            rtot = rtot + t
        pos = base + wbase + (incl - pc)
        col0 = w_idx * 32
        if u == Uint32(0xFFFFFFFF):
            for kb in cutlass.range_constexpr(32):
                sRing[(pos + kb) & rmask] = (col0 + kb).to(self.idx_dtype)
        else:
            for _ in cutlass.range(pc):
                lsb = u & (Uint32(0) - u)
                bit = Int32(cute.arch.popc(lsb - Uint32(1)))
                sRing[(pos & rmask)] = (col0 + bit).to(self.idx_dtype)
                pos += 1
                u = u & (u - Uint32(1))
        cute.arch.barrier(barrier_id=NamedBarrierOutput.ProducerDecode,
                          number_of_threads=128)
        return rtot

    @cute.jit
    def decode_band(self, gMask, sBand, band_base: Int32,
                    m_block: Int32, ptid: Int32):
        """Bit-parallel band decode by warp 0: the tile's own 64-column span
        is exactly words [2m, 2m+2); lane l handles bit l of each word with
        its rank from a masked popc -- no serial peel chain, no
        communication. All lanes read the same 2 words (broadcast loads)."""
        if ptid < 32:
            base_w = 2 * m_block
            lane_mask = (Uint32(1) << Uint32(ptid)) - Uint32(1)
            pre = Int32(0)
            for w in cutlass.range_constexpr(2):
                u = gMask[base_w + w].to(Uint32)
                if ((u >> Uint32(ptid)) & Uint32(1)) != Uint32(0):
                    rank = Int32(cute.arch.popc(u & lane_mask))
                    sBand[band_base + pre + rank] = \
                        ((base_w + w) * 32 + ptid).to(self.idx_dtype)
                pre = pre + Int32(cute.arch.popc(u))

    @cute.jit
    def gather_tile(self, mX, sX_stage, tiled_copy, thr_copy, sRing,
                    s: Int32, n_live: Int32):
        """v1's gather (cp.async.cg, index-clamp padding) with the index read
        wrapped through the ring: pos & (ring_capacity - 1)."""
        rmask: cutlass.Constexpr = self.ring_capacity - 1
        cX = cute.make_identity_tensor((self.tile_g, self.head_dim))
        tXsX = thr_copy.partition_D(sX_stage)
        tXcX = thr_copy.partition_S(cX)
        for gm in cutlass.range_constexpr(cute.size(tXsX, mode=[1])):
            row = tXcX[0, gm, 0][0]
            pos = cutlass.min(s * self.tile_g + row, n_live - 1)
            idx = sRing[pos & rmask].to(Int32)
            x_ptr = layout_utils.elem_pointer(mX, (idx, 0)).toint()
            gptr = cute.make_ptr(self.dtype, x_ptr, cute.AddressSpace.gmem,
                                 assumed_align=16)
            mrow = cute.make_tensor(gptr, cute.make_layout((self.head_dim,)))
            mrow8 = cute.tiled_divide(mrow, (8,))
            for kk in cutlass.range_constexpr(cute.size(tXsX, mode=[2])):
                ki = tXcX[0, 0, kk][1] // 8
                src = cute.make_tensor(mrow8[None, ki].iterator, tXsX[None, gm, kk].layout)
                cute.copy(tiled_copy, src, tXsX[None, gm, kk])

    @cute.jit
    def emit_chunk(self, mK_cur, mV_cur, sK, sV, gmem_tiled_copy_KV, thr_copy,
                   sRing, pipeline_k, pipeline_v, kv_pi: Int32, s: Int32,
                   n_live: Int32) -> Int32:
        """Gather one tile_g chunk of K then V (v1's per-stage body).

        The K/V pipeline position is ONE packed scalar (lap << 16 | index)
        because the emits sit inside NESTED dynamic loops: a mutated
        pipeline-state OBJECT does not thread as a loop-carried value across
        an outer traced loop (region dominance error, the known rw-args
        gotcha) -- a rebound scalar does. Packing keeps the unpack to a
        shift+and (phase = lap & 1) instead of a %-and-// pair."""
        lap = kv_pi >> 16
        s_idx = kv_pi & Int32(0xFFFF)
        s_ph = lap & 1
        pipeline_k.producer_acquire_w_index_phase(s_idx, s_ph)
        self.gather_tile(mK_cur, sK[None, None, s_idx],
                         gmem_tiled_copy_KV, thr_copy, sRing, s, n_live)
        cute.arch.cp_async_commit_group()
        pipeline_k.producer_commit_w_index(s_idx)
        pipeline_v.producer_acquire_w_index_phase(s_idx, s_ph)
        self.gather_tile(mV_cur, sV[None, None, s_idx],
                         gmem_tiled_copy_KV, thr_copy, sRing, s, n_live)
        cute.arch.cp_async_commit_group()
        pipeline_v.producer_commit_w_index(s_idx)
        wrap = s_idx + 1 == self.num_stages
        return ((lap + 1) << 16) if wrap else (kv_pi + 1)

    @cute.jit
    def prod(self, mQ, mK, mV, mMask, mSeqUsedQ, sQ, sK, sV, sRing, sBand,
             sNLive, sWarpTot, tma_atom_Q, pipeline_q,
             pipeline_k, pipeline_v, pipeline_i, gmem_tiled_copy_KV,
             tile_sched_params, TileScheduler: cutlass.Constexpr):
        ptid, _, _ = cute.arch.thread_idx()
        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        thr_copy = gmem_tiled_copy_KV.get_slice(ptid)
        warp = cute.arch.make_warp_uniform(ptid // 32)
        lane = cute.arch.lane_idx()

        q_phase = Int32(1)
        kv_pi = Int32(1 << 16)
        i_state = pipeline_custom.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 2)

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        more = work_tile.is_valid_tile
        while more:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoQK.create(
                batch_idx, seqlen_q_static=mQ.shape[0], seqlen_k_static=mK.shape[0],
                mCuSeqlensQ=None, mCuSeqlensK=None,
                mSeqUsedQ=mSeqUsedQ, mSeqUsedK=mSeqUsedQ)
            head_idx_kv = head_idx // self.qhead_per_kvhead
            mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, head_idx_kv]
            mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[None, None, head_idx_kv]

            if warp_idx_in_wg == 0:
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.head_dim), (m_block, 0))
                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True)
                pipeline_q.producer_acquire_w_index_phase(0, q_phase)
                load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
            q_phase ^= 1

            gMask = mMask[None, m_block, head_idx, batch_idx]

            W = cutlass.min(Int32(cute.size(gMask.shape)), 2 * (m_block + 1))
            rounds = (W + 127) // 128

            base = Int32(0)
            for rnd in cutlass.range(rounds, unroll=1):
                base = base + self.decode_round(gMask, sRing, sWarpTot,
                                                W, rnd, base, ptid, warp, lane)
            n_live = base

            pipeline_i.producer_acquire(i_state)
            band_base = i_state.index * BAND_CAP
            self.decode_band(gMask, sBand, band_base, m_block, ptid)
            if ptid == 0:
                sNLive[i_state.index] = n_live
            pipeline_i.producer_commit(i_state)
            i_state.advance()

            n_tiles_g = (n_live + self.tile_g - 1) // self.tile_g
            if n_live <= self.ring_capacity:
                emitted = Int32(0)
                for _dc in cutlass.range(n_tiles_g, unroll=1):
                    kv_pi = self.emit_chunk(mK_cur, mV_cur, sK, sV,
                                            gmem_tiled_copy_KV, thr_copy, sRing,
                                            pipeline_k, pipeline_v, kv_pi,
                                            emitted, n_live)
                    emitted += 1
            else:
                base2 = Int32(0)
                emitted = Int32(0)
                for rnd in cutlass.range(rounds, unroll=1):
                    ready = base2 >> self.tile_g_log2
                    n_drain = ready - emitted
                    for _dc in cutlass.range(n_drain, unroll=1):
                        kv_pi = self.emit_chunk(mK_cur, mV_cur, sK, sV,
                                                gmem_tiled_copy_KV, thr_copy,
                                                sRing, pipeline_k, pipeline_v,
                                                kv_pi, emitted, n_live)
                        emitted += 1
                    base2 = base2 + self.decode_round(gMask, sRing,
                                                      sWarpTot, W, rnd, base2,
                                                      ptid, warp, lane)
                n_left = n_tiles_g - emitted
                for _dc in cutlass.range(n_left, unroll=1):
                    kv_pi = self.emit_chunk(mK_cur, mV_cur, sK, sV,
                                            gmem_tiled_copy_KV, thr_copy, sRing,
                                            pipeline_k, pipeline_v, kv_pi,
                                            emitted, n_live)
                    emitted += 1

            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            more = work_tile.is_valid_tile

        tail_state = pipeline.PipelineState(
            self.num_stages, Int32(0),
            kv_pi & Int32(0xFFFF), (kv_pi >> 16) & 1)
        pipeline_v.producer_tail(tail_state)

    @property
    def tile_g_log2(self) -> int:
        return {32: 5, 64: 6, 128: 7}[self.tile_g]

    # -- consumer ------------------------------------------------------------

    @cute.jit
    def phase1(self, acc_S, tau_raw, supp, rP_i32, rP2_i32, sink_f, sink_u,
               pairs_rc: cutlass.Constexpr,
               masked: cutlass.Constexpr = False, tScS_mn=None, sBand=None,
               band_base=None, band_start=None, band_clamp=None,
               slot0=None, n_live=None, row_base=None):
        """v1's phase1 verbatim except the masked index source: the band
        sidecar replaces the full list. For slots below band_start the v1
        row test is vacuously true (idx < row_base <= row) -> off < 0; pad
        slots die on slot < n_live; band slots read sBand. The sBand read
        is unconditional with a clamped offset -- when band_cnt == 0 it reads
        a stale slot whose value is irrelevant (off < 0 already decides)."""
        frg = layout_utils.reshape_acc_to_frgA(acc_S)
        if const_expr(self.mode == "gemm_only"):
            sink_f[0] = sink_f[0] + frg[0]
        else:
            npairs: cutlass.Constexpr = cute.size(frg) // 2
            for j in cutlass.range_constexpr(npairs):
                r0, c0 = pairs_rc[2 * j]
                r1, c1 = pairs_rc[2 * j + 1]
                d0 = cute.arch.fmax(frg[2 * j] - tau_raw[r0], Float32(0.0))
                d1 = cute.arch.fmax(frg[2 * j + 1] - tau_raw[r1], Float32(0.0))
                if const_expr(masked):
                    slot_a = slot0 + tScS_mn[r0, c0][1]
                    slot_b = slot0 + tScS_mn[r1, c1][1]
                    off_a = slot_a - band_start
                    off_b = slot_b - band_start
                    oa = off_a if off_a > 0 else Int32(0)
                    ob = off_b if off_b > 0 else Int32(0)
                    idx0 = sBand[band_base + cutlass.min(oa, band_clamp)].to(Int32)
                    idx1 = sBand[band_base + cutlass.min(ob, band_clamp)].to(Int32)
                    live0 = (slot_a < n_live) & (
                        (off_a < 0) | (row_base + tScS_mn[r0, c0][0] >= idx0))
                    live1 = (slot_b < n_live) & (
                        (off_b < 0) | (row_base + tScS_mn[r1, c1][0] >= idx1))
                    d0 = d0 if live0 else Float32(0.0)
                    d1 = d1 if live1 else Float32(0.0)
                if const_expr(self.need_backward):
                    supp[r0] = supp[r0] + d0
                    supp[r1] = supp[r1] + d1
                    rP_i32[j] = layout_utils.cvt_f16x2_f32(d0, d1, self.dtype)
                rP2_i32[j] = layout_utils.cvt_f16x2_f32(d0 * d0, d1 * d1, self.dtype)
            if const_expr(self.mode == "no_pv"):
                for j in cutlass.range_constexpr(npairs):
                    sink_u[0] = sink_u[0] ^ rP2_i32[j].to(Uint32)
                    if const_expr(self.need_backward):
                        sink_u[0] = sink_u[0] ^ rP_i32[j].to(Uint32)

    @cute.jit
    def pv_pair(self, tiled_mma_pv, acc_O, acc_O2, tOrP, tOrP2, tOrVt, v_idx: Int32):
        if const_expr(self.mode == "full"):
            sm90_utils_basic.gemm_w_idx(tiled_mma_pv, acc_O, tOrP2, tOrVt, False,
                                  B_idx=v_idx, wg_wait=-1)
            if const_expr(self.need_backward):
                sm90_utils_basic.gemm_w_idx(tiled_mma_pv, acc_O2, tOrP, tOrVt, False,
                                      B_idx=v_idx, wg_wait=-1)

    @cute.jit
    def cons(self, mQ, mTau, mMask, mOut, mOut2, mSupp, mSeqUsedQ, sQ, sK, sV,
             sO, sBand, sNLive, pipeline_q, pipeline_k, pipeline_v,
             pipeline_i, tiled_mma_qk, tiled_mma_pv,
             half_scale: Float32, inv_half: Float32,
             tile_sched_params, TileScheduler: cutlass.Constexpr):
        tidx_full, _, _ = cute.arch.thread_idx()
        tidx = tidx_full - 128
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(0)
        wg_mma_pv = tiled_mma_pv.get_slice(0)
        acc_S, tSrQ, tSrK = sm90_utils_basic.partition_fragment_ABC(
            wg_mma_qk, (self.tile_m, self.tile_g, self.head_dim), sQ, sK)
        sVt = layout_utils.transpose_view(sV)
        acc_O, tOrP2, tOrVt = sm90_utils_basic.partition_fragment_ABC(
            wg_mma_pv, (self.tile_m, self.head_dim, self.tile_g), None, sVt)
        tOrP = cute.make_fragment_like(tOrP2)
        if const_expr(self.need_backward):
            acc_O2 = cute.make_rmem_tensor(acc_O.shape, Float32)
        else:
            acc_O2 = acc_O  # placeholder, never written
        rP_i32 = cute.recast_tensor(tOrP, Int32)
        rP2_i32 = cute.recast_tensor(tOrP2, Int32)
        pairs_rc = _frgA_pair_map(acc_S)

        cS = cute.make_identity_tensor((self.tile_m, self.tile_g))
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(cS))
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            tiled_mma_qk.get_slice(0).partition_C(cS))

        tau_raw = cute.make_rmem_tensor(2, Float32)
        supp = cute.make_rmem_tensor(2, Float32)
        sink_f = cute.make_rmem_tensor(1, Float32)
        sink_u = cute.make_rmem_tensor(1, Uint32)

        q_phase = Int32(0)
        k_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages)
        v_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages)
        k_rel = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages)
        v_rel = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_stages)
        i_state = pipeline_custom.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 2)

        tile_scheduler = TileScheduler.create(tile_sched_params)
        work_tile = tile_scheduler.initial_work_tile_info()
        more = work_tile.is_valid_tile
        while more:
            m_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoQK.create(
                batch_idx, seqlen_q_static=mQ.shape[0], seqlen_k_static=mQ.shape[0],
                mCuSeqlensQ=None, mCuSeqlensK=None,
                mSeqUsedQ=mSeqUsedQ, mSeqUsedK=mSeqUsedQ)
            row_base = m_block * self.tile_m

            mTau_cur = seqlen.offset_batch_Q(mTau, batch_idx, dim=2)[None, head_idx]
            gTau = cute.local_tile(mTau_cur, (self.tile_m,), (m_block,))
            gTau_x = cute.make_tensor(
                gTau.iterator,
                cute.append(gTau.layout, cute.make_layout((self.tile_g,), stride=(0,))))
            taccTgTau = layout_utils.reshape_acc_to_mn(thr_mma_qk.partition_C(gTau_x))
            row_valid = cute.make_rmem_tensor(2, Int32)
            for r in cutlass.range_constexpr(2):
                valid = (t0ScS_mn[r, 0][0]
                         < seqlen.seqlen_q - m_block * self.tile_m - tScS_mn[0, 0][0])
                row_valid[r] = valid.to(Int32)
                tau_raw[r] = Float32(float("inf"))
                if valid:
                    tau_raw[r] = taccTgTau[r, 0] * inv_half
            supp.fill(Float32(0.0))
            sink_f.fill(Float32(0.0))
            sink_u.fill(Uint32(0))
            acc_O.fill(Float32(0.0))
            if const_expr(self.need_backward):
                acc_O2.fill(Float32(0.0))

            band_cnt = Int32(0)
            for w in cutlass.range_constexpr(2):
                u = mMask[2 * m_block + w, m_block, head_idx, batch_idx].to(Uint32)
                band_cnt = band_cnt + Int32(cute.arch.popc(u))

            pipeline_i.consumer_wait(i_state, pipeline_i.consumer_try_wait(i_state))
            band_base = i_state.index * BAND_CAP
            n_live = sNLive[i_state.index]
            band_start = n_live - band_cnt
            band_clamp = band_cnt - 1 if band_cnt > 0 else Int32(0)

            p1 = partial(self.phase1, acc_S, tau_raw, supp, rP_i32, rP2_i32,
                         sink_f, sink_u, pairs_rc)
            p1m = partial(p1, masked=True, tScS_mn=tScS_mn, sBand=sBand,
                          band_base=band_base, band_start=band_start,
                          band_clamp=band_clamp, n_live=n_live, row_base=row_base)
            pv = partial(self.pv_pair, tiled_mma_pv, acc_O, acc_O2, tOrP, tOrP2, tOrVt)

            if n_live > 0:
                pipeline_q.consumer_wait_w_index_phase(0, q_phase)
                n_tiles_g = (n_live + self.tile_g - 1) // self.tile_g

                if const_expr(self.overlap):
                    pipeline_k.consumer_wait(k_state, pipeline_k.consumer_try_wait(k_state))
                    sm90_utils_basic.gemm_w_idx(tiled_mma_qk, acc_S, tSrQ, tSrK, True,
                                              B_idx=k_state.index, wg_wait=-1)
                    k_state.advance()
                    for i in cutlass.range(n_tiles_g - 1, unroll=1):
                        warpgroup.wait_group(0)
                        pipeline_k.consumer_release(k_rel)
                        k_rel.advance()
                        if i >= 1:
                            pipeline_v.consumer_release(v_rel)
                            v_rel.advance()
                        if self.tile_g * (i + 1) > band_start:
                            p1m(slot0=i * self.tile_g)
                        else:
                            p1()
                        pipeline_k.consumer_wait(k_state, pipeline_k.consumer_try_wait(k_state))
                        sm90_utils_basic.gemm_w_idx(tiled_mma_qk, acc_S, tSrQ, tSrK, True,
                                                  B_idx=k_state.index, wg_wait=-1)
                        k_state.advance()
                        pipeline_v.consumer_wait(v_state, pipeline_v.consumer_try_wait(v_state))
                        pv(v_idx=v_state.index)
                        v_state.advance()
                    warpgroup.wait_group(0)
                    pipeline_k.consumer_release(k_rel)
                    k_rel.advance()
                    if n_tiles_g >= 2:
                        pipeline_v.consumer_release(v_rel)
                        v_rel.advance()
                    p1m(slot0=(n_tiles_g - 1) * self.tile_g)
                    pipeline_i.consumer_release(i_state)
                    pipeline_v.consumer_wait(v_state, pipeline_v.consumer_try_wait(v_state))
                    pv(v_idx=v_state.index)
                    v_state.advance()
                    warpgroup.wait_group(0)
                    pipeline_v.consumer_release(v_rel)
                    v_rel.advance()
                else:
                    for s in cutlass.range(n_tiles_g, unroll=1):
                        pipeline_k.consumer_wait(k_state, pipeline_k.consumer_try_wait(k_state))
                        sm90_utils_basic.gemm_w_idx(tiled_mma_qk, acc_S, tSrQ, tSrK, True,
                                                  B_idx=k_state.index, wg_wait=0)
                        pipeline_k.consumer_release(k_rel)
                        k_rel.advance()
                        if (self.tile_g * (s + 1) > band_start) | (s == n_tiles_g - 1):
                            p1m(slot0=s * self.tile_g)
                        else:
                            p1()
                        pipeline_v.consumer_wait(v_state, pipeline_v.consumer_try_wait(v_state))
                        pv(v_idx=v_state.index)
                        warpgroup.wait_group(0)
                        pipeline_v.consumer_release(v_rel)
                        v_rel.advance()
                        k_state.advance()
                        v_state.advance()
                    pipeline_i.consumer_release(i_state)
                pipeline_q.consumer_release_w_index(0)

                if const_expr(self.need_backward):
                    for r in cutlass.range_constexpr(cute.size(supp)):
                        supp[r] = cute.arch.warp_reduction_sum(
                            supp[r], threads_in_group=4)
                    mSupp_cur = seqlen.offset_batch_Q(mSupp, batch_idx, dim=2)[None, head_idx]
                    gSupp = cute.local_tile(mSupp_cur, (self.tile_m,), (m_block,))
                    gSupp_x = cute.make_tensor(
                        gSupp.iterator,
                        cute.append(gSupp.layout,
                                    cute.make_layout((self.tile_g,), stride=(0,))))
                    taccSgSupp = layout_utils.reshape_acc_to_mn(
                        thr_mma_qk.partition_C(gSupp_x))
                    if tScS_mn[0, 0][1] == 0:
                        for r in cutlass.range_constexpr(2):
                            if row_valid[r] != 0:
                                taccSgSupp[r, 0] = supp[r]

                self.epilogue(acc_O, acc_O2, sink_f, sink_u, mOut, mOut2,
                              sO, tiled_mma_pv, tidx, half_scale, seqlen,
                              m_block, head_idx, batch_idx)
            else:
                pipeline_q.consumer_wait_w_index_phase(0, q_phase)
                pipeline_q.consumer_release_w_index(0)
                pipeline_i.consumer_release(i_state)
            q_phase ^= 1
            i_state.advance()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
            more = work_tile.is_valid_tile

    # -- epilogue --------------------------------------------------------------

    @cute.jit
    def epilogue(self, acc_O, acc_O2, sink_f, sink_u, mOut, mOut2, sO,
                 tiled_mma_pv, tidx: Int32, half_scale: Float32,
                 seqlen, m_block: Int32, head_idx: Int32, batch_idx: Int32):
        half2 = half_scale * half_scale
        if const_expr(self.mode != "full"):
            acc_O[0] = acc_O[0] + sink_f[0] + Float32(sink_u[0] & Uint32(1))
        for i in cutlass.range_constexpr(cute.size(acc_O)):
            acc_O[i] = acc_O[i] * half2

        smem_copy_atom_O = layout_utils.get_smem_store_atom(90, self.dtype)
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma_pv).get_slice(tidx)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        async_copy_elems: cutlass.Constexpr = 128 // self.dtype.width
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), self.dtype, num_bits_per_copy=128)
        tO_dim1: cutlass.Constexpr = self.head_dim // async_copy_elems
        tO_layout = cute.make_ordered_layout((128 // tO_dim1, tO_dim1), order=(1, 0))
        vO_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        cO = cute.make_identity_tensor((self.tile_m, self.head_dim))
        tOcO = gmem_thr_copy_O.partition_S(cO)
        t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)

        mOut_cur = seqlen.offset_batch_Q(mOut, batch_idx, dim=3)[None, None, head_idx]
        gOut = cute.local_tile(mOut_cur, (self.tile_m, self.head_dim), (m_block, 0))
        tOgO = gmem_thr_copy_O.partition_D(gOut)
        if const_expr(self.need_backward):
            mOut2_cur = seqlen.offset_batch_Q(mOut2, batch_idx, dim=3)[None, None, head_idx]
            gOut2 = cute.local_tile(mOut2_cur, (self.tile_m, self.head_dim), (m_block, 0))
            tOgO2 = gmem_thr_copy_O.partition_D(gOut2)

        n_outs: cutlass.Constexpr = 2 if self.need_backward else 1
        for which in cutlass.range_constexpr(n_outs):
            acc = acc_O if const_expr(which == 0) else acc_O2
            rO = layout_utils.cvt_f16(acc, self.dtype)
            taccOrO = smem_thr_copy_O.retile(rO)
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue),
                              number_of_threads=128)
            tOrO = cute.make_fragment_like(tOsO, self.dtype)
            cute.autovec_copy(tOsO, tOrO)
            dst = tOgO if const_expr(which == 0) else tOgO2
            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if (t0OcO[0, rest_m, 0][0]
                        < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]):
                    cute.copy(gmem_tiled_copy_O, tOrO[None, rest_m, None],
                              dst[None, rest_m, None])
            cute.arch.barrier(barrier_id=int(NamedBarrierFwd.Epilogue),
                              number_of_threads=128)


# ///////////////////////////////////////////////////////////////////////////////
# Host wrapper
# ///////////////////////////////////////////////////////////////////////////////

import torch
from flash_attn.cute.cute_dsl_utils import to_cute_tensor
from adasplash.forward.cute.cache import get_jit_cache
from adasplash.forward.cute import layout


def get_output(q, k, v, taus, mask, cnt=None, sm_scale=None, need_backward=True,
               varlen=None, mode="full", tile_g=64, num_stages=2, overlap=False,
               ring_log2=13, mma_regs=232, producer_regs=24):
    """Kernel C: no-union 64-row tiles, 2 CTAs/SM. Contract identical to the
    deprecated v1/v2 wrappers (unnormalized out2 + supp sidecar; uncapped N);
    outputs agree with them to summation-order rounding (NOT bitwise:
    dropping union-only zero columns reorders wgmma reduction trees)."""
    B, N_H, N_CTX, H_DIM, N_KV_H, qhead_per_kvhead = layout.qkv_dims(q, k)
    if sm_scale is None:
        sm_scale = layout.default_sm_scale(H_DIM)
    wide_index = N_CTX > 65536

    alloc = layout.output_allocator(varlen)
    out = alloc((B, N_H, N_CTX, H_DIM), device=q.device, dtype=q.dtype)
    out2 = torch.zeros((B, N_H, N_CTX, H_DIM), device=q.device, dtype=q.dtype) \
        if need_backward or varlen is not None else \
        torch.empty((B, N_H, N_CTX, H_DIM), device=q.device, dtype=q.dtype)
    supp = torch.zeros((B, N_H, N_CTX), device=q.device, dtype=torch.float32) \
        if need_backward else None

    q_bshd, k_bshd, v_bshd = layout.bshd(q, k, v)
    out_bshd, out2_bshd = layout.bshd(out, out2)
    seqused = layout.seqused_from_varlen(varlen)

    dtype = layout.cutlass_dtype(q)
    key = (dtype, H_DIM, qhead_per_kvhead, tile_g, num_stages, need_backward,
           mode, overlap, seqused is not None, ring_log2, wide_index,
           mma_regs, producer_regs)
    if key not in get_output.compile_cache:
        kern = AdaSplashOutputSm90(
            dtype, H_DIM, qhead_per_kvhead=qhead_per_kvhead, tile_g=tile_g,
            num_stages=num_stages, need_backward=need_backward, mode=mode,
            overlap=overlap, ring_log2=ring_log2, wide_index=wide_index,
            mma_regs=mma_regs, producer_regs=producer_regs,
        )
        cq, ck, cv = to_cute_tensor(q_bshd), to_cute_tensor(k_bshd), to_cute_tensor(v_bshd)
        co, co2 = to_cute_tensor(out_bshd), to_cute_tensor(out2_bshd)
        ctau = to_cute_tensor(taus, assumed_align=4)
        cmask = to_cute_tensor(mask, assumed_align=16)
        csupp = to_cute_tensor(supp, assumed_align=4) if supp is not None else None
        cseq = to_cute_tensor(seqused, assumed_align=4, leading_dim=0) if seqused is not None else None
        get_output.compile_cache[key] = cute.compile(
            kern, cq, ck, cv, ctau, cmask, co, co2,
            Float32(sm_scale), csupp, cseq,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    get_output.compile_cache[key](
        q_bshd.detach(), k_bshd.detach(), v_bshd.detach(), taus, mask,
        out_bshd, out2_bshd, sm_scale, supp, seqused
    )
    return out, out2, supp


get_output.compile_cache = get_jit_cache("adasplash_fwd_output")