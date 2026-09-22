"""Benchmark for cuTile attention."""

import argparse
import json
import statistics
import torch
import sys
sys.path.insert(0, ".")
from attention import attention

def run_benchmark(B, H, L, S, D, dtype, causal, iterations=100, warmup=10):
    """Run benchmark and return list of latencies in ms."""
    torch.manual_seed(42)
    dt = getattr(torch, dtype)
    
    q = torch.randn(B, H, L, D, dtype=dt, device="cuda")
    k = torch.randn(B, H, S, D, dtype=dt, device="cuda")
    v = torch.randn(B, H, S, D, dtype=dt, device="cuda")
    
    # Warmup (includes JIT compilation)
    for _ in range(warmup):
        _ = attention(q, k, v, causal=causal)
    
    # Timed runs
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = attention(q, k, v, causal=causal)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    
    return latencies

def geometric_mean(vals):
    import math
    return math.exp(sum(math.log(v) for v in vals) / len(vals))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="Output JSON path")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (smaller shapes)")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    
    # Workload configurations
    if args.quick:
        workloads = [
            (1, 8, 64, 64, 64, "float16", False),   # small decode
            (2, 16, 128, 128, 128, "float16", True), # medium prefill
        ]
    else:
        workloads = [
            (1, 8, 64, 64, 64, "float16", False),
            (1, 8, 128, 128, 64, "float16", False),
            (2, 16, 128, 128, 128, "float16", True),
            (4, 32, 256, 256, 128, "float16", True),
            (1, 8, 256, 512, 128, "bfloat16", True),
            (2, 16, 512, 1024, 128, "float16", True),
        ]
    
    results = []
    for B, H, L, S, D, dtype, causal in workloads:
        print(f"Benchmarking: B={B}, H={H}, L={L}, S={S}, D={D}, dtype={dtype}, causal={causal}")
        latencies = run_benchmark(B, H, L, S, D, dtype, causal, args.iterations, args.warmup)
        
        geo_mean = geometric_mean(latencies)
        median = statistics.median(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        
        print(f"  geomean: {geo_mean:.3f} ms, median: {median:.3f} ms, min: {min_lat:.3f} ms, max: {max_lat:.3f} ms")
        
        results.append({
            "B": B, "H": H, "L": L, "S": S, "D": D,
            "dtype": dtype, "causal": causal,
            "latencies_ms": latencies,
            "geomean_ms": geo_mean,
            "median_ms": median,
            "min_ms": min_lat,
            "max_ms": max_lat,
        })
    
    # Overall geomean of geomeans
    overall_geomean = geometric_mean([r["geomean_ms"] for r in results])
    print(f"\nOverall geomean_median_ms: {overall_geomean:.3f}")
    
    output = {
        "geomean_median_ms": overall_geomean,
        "workloads": results,
    }
    
    if args.json:
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {args.json}")
    
    # Print final metric for kbench parsing
    print(f"geomean_median_ms={overall_geomean:.6f}")

if __name__ == "__main__":
    main()