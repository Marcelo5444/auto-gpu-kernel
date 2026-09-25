import cuda.tile as ct
import torch
from cuda.tile import RoundingMode as RMd

NEG_INF = -1000.0
_CT_DTYPE = ct.float16  # Will be set by _set_input_dtype
_TORCH_DTYPE = torch.float16

def _set_input_dtype(q):
    global _CT_DTYPE, _TORCH_DTYPE
    s = str(getattr(q, 'dtype', q))
    if 'float16' in s and 'bfloat16' not in s:
        _CT_DTYPE = ct.float16
        _TORCH_DTYPE = torch.float16
    else:
        _CT_DTYPE = ct.bfloat16
        _TORCH_DTYPE = torch.bfloat16

def _acc_output_block(q, K, V, tau, out, out2, supp_acc, ones_tile,
                      off_kv_hz, c_block, mask_needed,
                      offs_m, offs_n_base,
                      BLOCK_M, BLOCK_N, H_DIM,
                      LATENCY_K, LATENCY_V, use_tma_k, use_tma_v):
    vt = ct.load(V, index=(off_kv_hz, 0, c_block, 0),
                 shape=(1, 1, BLOCK_N, H_DIM), latency=LATENCY_V, allow_tma=use_tma_v)
    k_t = ct.load(K, index=(off_kv_hz, 0, 0, c_block),
                  shape=(1, 1, H_DIM, BLOCK_N), latency=LATENCY_K, order=(0, 1, 3, 2), allow_tma=use_tma_k)
    k_t = k_t.reshape((H_DIM, BLOCK_N)).astype(_CT_DTYPE)
    qk = ct.full((BLOCK_M, BLOCK_N), 0.0, ct.float32)
    qk = ct.mma(q, k_t, qk)

    if mask_needed:
        offs_n = c_block * BLOCK_N + offs_n_base
        mask_qk = offs_m[:, None] >= offs_n[None, :]
        qk = ct.where(mask_qk, qk, ct.float32(NEG_INF))

    vt = vt.reshape((BLOCK_N, H_DIM))
    proj = ct.maximum(qk - tau[:, None], ct.float32(0.0))
    proj_f16 = proj.astype(_CT_DTYPE)
    out = ct.mma((proj * proj).astype(_CT_DTYPE), vt, out)
    out2 = ct.mma(proj_f16, vt, out2)
    supp_acc = ct.mma(proj_f16, ones_tile, supp_acc)
    return out, out2, supp_acc

@ct.kernel
def get_output(
    Q,               # [B*N_H, 1, N_CTX, H_DIM]
    K,               # [B*N_KV_H, 1, N_CTX, H_DIM]
    V,               # [B*N_KV_H, 1, N_CTX, H_DIM]
    TAUS,            # [B*N_H, 1, N_CTX, 1]
    N_ACTIVE,        # [B*N_H, N_M, 1, 1]
    ACTIVE_IDX,      # [B*N_H, N_M, MAX_BLOCKS, 1]
    OUT,             # [B*N_H, 1, N_CTX, H_DIM]
    OUT2,            # [B*N_H, 1, N_CTX, H_DIM]
    sm_scale: float,
    H_DIM: ct.Constant[int],
    N_CTX: ct.Constant[int],
    N_H: ct.Constant[int],
    N_KV_H: ct.Constant[int],
    GROUP_SIZE: ct.Constant[int],

    BLOCK_M: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    LATENCY_Q: ct.Constant[int],
    LATENCY_K: ct.Constant[int],
    LATENCY_V: ct.Constant[int],
    USE_TMA: ct.Constant[int],        # bitfield: bit0=Q, bit1=K, bit2=V load via TMA
    USE_TMA_STORE: ct.Constant[int],  # bitfield: bit0=OUT, bit1=OUT2 store via TMA
):
    n_m_tiles: ct.constexpr = (N_CTX + BLOCK_M - 1) // BLOCK_M
    start_m = n_m_tiles - 1 - ct.bid(0)
    off_h = ct.bid(1)
    off_z = ct.bid(2)
    off_hz = off_z * N_H + off_h
    off_kv_hz = off_z * N_KV_H + (off_h // GROUP_SIZE)

    scalar = 0.5 * sm_scale
    F0 = ct.float32(0.0)
    NEG_INF_F32 = ct.float32(NEG_INF)
    ONE_E_8 = ct.float32(1e-8)

    use_tma_q: ct.constexpr = (USE_TMA & 1) != 0
    use_tma_k: ct.constexpr = (USE_TMA & 2) != 0
    use_tma_v: ct.constexpr = (USE_TMA & 4) != 0

    use_tma_out: ct.constexpr = (USE_TMA_STORE & 1) != 0
    use_tma_out2: ct.constexpr = (USE_TMA_STORE & 2) != 0

    q = ct.load(Q, index=(off_hz, 0, start_m, 0), shape=(1, 1, BLOCK_M, H_DIM), latency=LATENCY_Q, padding_mode=ct.PaddingMode.ZERO, allow_tma=use_tma_q)
    q = q.reshape((BLOCK_M, H_DIM))
    q = q.astype(ct.float32) * scalar
    q = q.astype(_CT_DTYPE)

    offs_m_base = start_m * BLOCK_M

    tau = ct.load(TAUS, index=(off_hz, 0, start_m, 0), shape=(1, 1, BLOCK_M, 1))
    tau = tau.reshape((BLOCK_M,))

    n_active_tile = ct.load(N_ACTIVE, index=(off_hz, start_m, 0, 0), shape=(1, 1, 1, 1))
    n_active = n_active_tile.reshape(()).astype(ct.int32)

    out = ct.full((BLOCK_M, H_DIM), 0.0, ct.float32)
    out2 = ct.full((BLOCK_M, H_DIM), 0.0, ct.float32)
    SUPP_COLS: ct.constexpr = 8
    supp_acc = ct.full((BLOCK_M, SUPP_COLS), 0.0, ct.float32)
    ones_tile = ct.full((BLOCK_N, SUPP_COLS), 1.0, _CT_DTYPE)

    offs_m = offs_m_base + ct.arange(BLOCK_M, dtype=ct.int32)
    offs_n_base = ct.arange(BLOCK_N, dtype=ct.int32)
    mask_start = (start_m * BLOCK_M) // BLOCK_N
    total_blocks = ((start_m + 1) * BLOCK_M + BLOCK_N - 1) // BLOCK_N

    if n_active == total_blocks:
        for c_block in range(mask_start, total_blocks):
            mask_needed: ct.constexpr = True
            out, out2, supp_acc = _acc_output_block(
                q, K, V, tau, out, out2, supp_acc, ones_tile,
                off_kv_hz, c_block, mask_needed,
                offs_m, offs_n_base,
                BLOCK_M, BLOCK_N, H_DIM,
                LATENCY_K, LATENCY_V, use_tma_k, use_tma_v)
        for c_block in range(0, mask_start):
            mask_needed: ct.constexpr = False
            out, out2, supp_acc = _acc_output_block(
                q, K, V, tau, out, out2, supp_acc, ones_tile,
                off_kv_hz, c_block, mask_needed,
                offs_m, offs_n_base,
                BLOCK_M, BLOCK_N, H_DIM,
                LATENCY_K, LATENCY_V, use_tma_k, use_tma_v)
    else:
        for slot in range(0, n_active):
            idx_tile = ct.load(ACTIVE_IDX, index=(off_hz, start_m, slot, 0),
                               shape=(1, 1, 1, 1))
            c_block = idx_tile.reshape(()).astype(ct.int32)
            out, out2, supp_acc = _acc_output_block(
                q, K, V, tau, out, out2, supp_acc, ones_tile,
                off_kv_hz, c_block, c_block >= mask_start,
                offs_m, offs_n_base,
                BLOCK_M, BLOCK_N, H_DIM,
                LATENCY_K, LATENCY_V, use_tma_k, use_tma_v)

    out = out.astype(_CT_DTYPE)

    ct.store(
        OUT, index=(off_hz, 0, start_m, 0),
        tile=out.reshape((1, 1, BLOCK_M, H_DIM)),
        allow_tma=use_tma_out,
    )

    out2 = out2.astype(ct.float32)
    supp_size = ct.sum(supp_acc.astype(ct.float32), axis=1) * (1.0 / SUPP_COLS)
    supp_size = ct.maximum(supp_size, ONE_E_8)
    out2 = ct.truediv(out2, supp_size[:, None], rounding_mode=RMd.APPROX)
    out2 = out2.astype(_CT_DTYPE)

    ct.store(
        OUT2, index=(off_hz, 0, start_m, 0),
        tile=out2.reshape((1, 1, BLOCK_M, H_DIM)),
        allow_tma=use_tma_out2,
    )

# --- Host launcher ---
def launch_get_output(q, k, v, taus, mask, cnt=None, sm_scale=None, need_backward=True,
                      varlen=None, mode="full", tile_g=64, num_stages=2, overlap=False,
                      ring_log2=13, mma_regs=232, producer_regs=24):
    """cuTile get_output launcher matching the cutedsl contract."""
    _set_input_dtype(q)
    
    B, N_H, N_CTX, H_DIM = q.shape
    N_KV_H = k.shape[1]
    GROUP_SIZE = N_H // N_KV_H
    
    if sm_scale is None:
        sm_scale = 1.0 / (H_DIM ** 0.5)
    
    # Layout: [B*N_H, 1, N_CTX, H_DIM]
    q_bnh = q.reshape(B * N_H, 1, N_CTX, H_DIM).contiguous()
    k_bnh = k.reshape(B * N_KV_H, 1, N_CTX, H_DIM).contiguous()
    v_bnh = v.reshape(B * N_KV_H, 1, N_CTX, H_DIM).contiguous()
    
    # Build mask/taus in the format expected by the kernel
    # This is a simplified version - the real kernel expects pre-computed N_ACTIVE and ACTIVE_IDX
    # For now, we'll use the affine path (all blocks active)
    
    M_TILES = (N_CTX + tile_g - 1) // tile_g  # BLOCK_M = tile_g
    MAX_BLOCKS = (N_CTX + tile_g - 1) // tile_g  # BLOCK_N = tile_g
    
    # Allocate outputs
    out = torch.empty((B, N_H, N_CTX, H_DIM), device=q.device, dtype=q.dtype)
    out2 = torch.empty((B, N_H, N_CTX, H_DIM), device=q.device, dtype=q.dtype)
    out_bnh = out.reshape(B * N_H, 1, N_CTX, H_DIM).contiguous()
    out2_bnh = out2.reshape(B * N_H, 1, N_CTX, H_DIM).contiguous()
    
    # TAUS: [B*N_H, 1, N_CTX, 1]
    taus_bnh = taus.reshape(B * N_H, 1, N_CTX, 1).contiguous()
    
    # N_ACTIVE: [B*N_H, N_M, 1, 1] - all blocks active
    n_active = torch.full((B * N_H, M_TILES, 1, 1), MAX_BLOCKS, device=q.device, dtype=torch.int32)
    
    # ACTIVE_IDX: [B*N_H, N_M, MAX_BLOCKS, 1] - affine indices
    active_idx = torch.arange(MAX_BLOCKS, device=q.device, dtype=torch.int32).view(1, 1, MAX_BLOCKS, 1)
    active_idx = active_idx.expand(B * N_H, M_TILES, MAX_BLOCKS, 1).contiguous()
    
    # Launch kernel
    grid = (M_TILES, N_H, B)
    stream = torch.cuda.current_stream().cuda_stream
    ct.launch(
        stream,
        grid,
        get_output,
        (
            q_bnh, k_bnh, v_bnh, taus_bnh,
            n_active, active_idx,
            out_bnh, out2_bnh,
            sm_scale,
            H_DIM, N_CTX, N_H, N_KV_H, GROUP_SIZE,
            tile_g, tile_g,
            1, 2, 4,  # LATENCY_Q=1, LATENCY_K=2, LATENCY_V=4 (must be 1-10)
            0, 0,     # USE_TMA=0, USE_TMA_STORE=0
        )
    )
    
    torch.cuda.synchronize()
    return out, out2, None