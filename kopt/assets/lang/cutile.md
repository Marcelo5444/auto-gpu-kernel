## Language: cuTile

cuTile is NVIDIA's Python-based tile programming model (Tile IR). JIT-compiled via `tileiras`, 
targets Tensor Cores across architectures (Ampere SM80, Ada SM89, Hopper SM90, Blackwell SM120+).

- **Stay in cuTile.** No switching to Triton/CUDA mid-run.
- **Entry point:** `@ct.kernel` decorated function + host `ct.launch()` wrapper.
- **Data model:** Arrays (global memory, mutable, strided) ↔ Tiles (immutable, compile-time shape, register/SMEM).
- **Key ops:** `ct.load()`, `ct.store()`, `ct.mma()`, `ct.cp_async()`, `ct.barrier()`, `ct.bid()`, `ct.tid()`.
- **Interoperability:** Accepts PyTorch tensors, CuPy arrays as kernel arguments.

### Read this before your first change

cuTile compilation goes through `tileiras` → PTX. Failures are **compile-time** (layout mismatches, 
atom type errors, tile shape constraints). Rules:

1. **Compile before benchmark.** Clean build + `--quick` correctness first.
2. **Tile shapes = powers of 2.** `ct.Tile[(M, N), dtype]` — M,N,K must be 2^n.
3. **Layout algebra is explicit.** Thread-value layouts, swizzles, shared-memory maps — verify against 
   `tileiras` version in container.
4. **MMA atoms have fixed types.** `ct.mma(mma_atom, A, B, C)` — operand/accumulator types must match.
5. **Bitwise ops work on int32/int64** — `ct.bitwise_and/or/xor/lshift/rshift/not` (see `references/17_bitwise_ops.md`). 
   No `popc` or `fns.b32` — scan 32 positions if needed.
6. **Bank conflicts on Blackwell (SM120):** All conflicts are **write-side (ST-path)**. 
   TMA (`allow_tma=True`, default) eliminates them for tiles ≤64-wide. 
   See `references/sass_bank_conflict_swizzle.md` and `references/bank_conflict_deep_analysis.md`.

### Progression Ladder

1. **Naive tile kernel** — correct `@ct.kernel` signature, load→compute→store, `ct.launch()` works
2. **Correct tiling/layout** — tile shapes match problem, thread-value layout valid
3. **Shared-memory staging** — `ct.cp_async` + `ct.barrier` for global→SMEM→register
4. **Tiled MMA** — select correct `ct.MmaAtom` for dtype/shape, accumulate in f32
5. **Pipelining / multistage** — K-stage async copy + compute overlap
6. **Architecture atoms** — TMA, cluster launch, Blackwell TMEM (when available)

Establish correctness at each step; never carry two unverified changes.

### Tuning Knobs

- Tile shapes: `(BLOCK_M, BLOCK_N, BLOCK_K)` — powers of 2, fit in SMEM
- Thread-value layouts: `ct.Layout`, swizzle patterns
- MMA atom selection: `ct.MmaAtom(kind, m, n, k, dtype_a, dtype_b, dtype_c)`
- Pipeline stages: `NUM_STAGES` for `cp_async` depth
- Shared memory budget: `ct.shared_memory()` allocation
- Cluster/CTA shape: `ct.cluster()` for Hopper/Blackwell
- Grid/block: `ct.launch(stream, grid, block, kernel, args)`

### Numerical Hazards

- **Accumulate in f32** even for fp16/bf16/fp8 inputs — MMA atoms may require f32 acc
- **Layout errors manifest as numerical errors** — wrong swizzle = wrong results, not crash
- **Tile immutability** — every `ct.load`/`ct.mma` produces new tile; no in-place updates
- **Boundary handling** — `ct.load` with OOB returns undefined; use `ct.if_` guards or pad inputs
- **FP8 support** — requires Blackwell (SM 120+); check `ct.device_capability()`
- **Reduction rank drop** — `ct.max`/`ct.min` reduce 2D→1D; broadcast before `ct.store`: `max_val[None, :]`

### Reference Implementations (Study First)

Before optimizing, search **TileGym** (primary) then **fallback examples**:

| Operation | TileGym Location |
|-----------|------------------|
| FMHA (dense) | `tilegym/ops/cutile/attention.py::fmha_kernel_impl` |
| Flash decode | `tilegym/ops/cutile/flash_decode.py` |
| MLA decoding | `tilegym/ops/cutile/mla_decoding.py` |
| AdaSPLASH-2 sparse | `tilegym/ops/cutile/attention_sink.py` |
| MatMul | `tilegym/ops/cutile/matmul.py` |
| Softmax | `tilegym/ops/cutile/softmax.py` |
| LayerNorm | `tilegym/ops/cutile/rms_norm.py` |
| Group GEMM | `tilegym/ops/cutile/group_gemm.py` |

### Build / Runtime Notes

- Requires `cuda-tile[tileiras]` + CUDA 13.1+ (or `pip install cuda-toolkit[tileiras,nvcc,nvvm]`)
- Container base: `nvidia/cuda:13.2-devel` or `flashinfer/flashinfer-ci-cu132:latest` + `pip install cuda-tile[tileiras]`
- `tileiras` version must match CUDA toolkit minor version
- JIT compilation on first launch — warmup runs essential before timing
- Nsight Compute works: `ncu --set detailed python kernel.py`
- Benchmarking: Use TileGym's 3-tier infra — C++ `_benchmark` (L2-flushed), autotuner, or `report_benchmark` (no flush)
  See `references/benchmarking_l2_flush.md`

### Validation Pattern (Mandatory)

Every kernel file must include inline validation:

```python
if __name__ == "__main__":
    # Setup inputs
    q = torch.randn(...).cuda().half()
    k = torch.randn(...).cuda().half()
    v = torch.randn(...).cuda().half()
    out = torch.empty_like(q)
    
    # Reference
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    
    # cuTile kernel
    ct.launch(stream, grid, attention_kernel, (q, k, v, out, scale, causal))
    torch.cuda.synchronize()
    
    # Validate
    is_close = torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3)
    if is_close:
        print("✓ Validation PASSED")
    else:
        max_diff = (out - ref_out).abs().max().item()
        print(f"✗ Validation FAILED - max diff: {max_diff}")
```

### Escape Hatch

If 15-20 iterations produce no improvement, spin up a sub-agent with fresh context to rebuild the
kernel around a different decomposition (e.g., persistent scheduling vs standard, different tiling).
See `references/attention_kernels.md` for warp scheduler/occupancy analysis and persistent patterns.