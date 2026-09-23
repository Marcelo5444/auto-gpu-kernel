# Benchmark Results: cuTile Attention Kernel Optimization

## Summary
Optimized cuTile attention kernel (exp_6: LPT chunk ordering + causal early-exit + constant caching + fast stream + no sync) tested on two architectures:

| Architecture | GPU | Speedup vs Baseline |
|-------------|-----|---------------------|
| **DGX Spark (GB10)** | Blackwell SM120, 128GB unified | **~8×** (0.187 → 0.023 ms) |
| **Builder (RTX 4090)** | Ada Lovelace, 24GB | **~2×** (0.055 → 0.029 ms geomean) |

## DGX Spark (GB10) - Primary Target
- **Baseline (exp_1):** 0.187 ms geomean_median_ms
- **Optimized (exp_6):** 0.0235 ms geomean_median_ms
- **Cumulative speedup:** 8×
- **Noise floor:** ±7% (self-A/B)

## Builder (RTX 4090) - Cross-Architecture Validation
| Case | Baseline (ms) | Optimized (ms) | Speedup |
|------|--------------|----------------|---------|
| B1 H8 L64 S64 D64 fp16 NC | 0.0339 | 0.0147 | 2.3× |
| B1 H8 L128 S128 D64 fp16 NC | 0.0342 | 0.0185 | 1.8× |
| B2 H16 L128 S128 D128 fp16 C | 0.0393 | 0.0209 | 1.9× |
| B4 H32 L256 S256 D128 fp16 C | 0.0793 | 0.0532 | 1.5× |
| B1 H8 L256 S512 D128 bf16 C | 0.0527 | 0.0235 | 2.2× |
| B2 H16 L512 S1024 D128 fp16 C | 0.1379 | 0.0520 | 2.7× |
| **Geomean** | **0.055** | **0.029** | **~1.9×** |

## Correctness (Both Machines)
- 17/17 pytest tests pass
- 7/7 reference correctness checks pass (vs torch SDPA)
- Max absolute diffs identical across runs

## Key Optimizations (exp_6)
1. **LPT chunk ordering** - longest-trip query chunks first
2. **Causal early-exit** - skip fully-masked KV tiles
3. **Constant caching** - per launch signature
4. **Fast raw-stream lookup** - `torch._C._cuda_getCurrentRawStream`
5. **Removed `torch.cuda.synchronize()`** from launch path

## Notes
- The ~8× speedup on GB10 is **Blackwell-specific** (SM120 features, unified memory, cuTile JIT behavior)
- The ~2× speedup on Ada is still significant but reflects different architectural benefits
- Optimized kernel at: `work/cutile-attention/repo/attention.py`
- Clean baseline at: `examples/cutile-attention/attention.py`
