#!/usr/bin/env python3
"""Benchmark for cuTile AdaSplash get_output kernel."""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repo"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "adasplash-output"))

import torch
from cutile_get_output import launch_get_output

def run_benchmark(B, H, KV, N, D, dtype=torch.bfloat16, reps=50, warmup=5):
    """Run benchmark for a single configuration."""
    device = "cuda"
    
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, KV, N, D, device=device, dtype=dtype)
    v = torch.randn(B, KV, N, D, device=device, dtype=dtype)
    taus = torch.randn(B, H, N, device=device, dtype=torch.float32).abs()
    mask = None
    cnt = None
    sm_scale = 1.0 / (D ** 0.5)
    
    # Warmup
    for _ in range(warmup):
        launch_get_output(q, k, v, taus, mask, cnt, sm_scale=sm_scale)
    
    # Benchmark
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(reps):
        launch_get_output(q, k, v, taus, mask, cnt, sm_scale=sm_scale)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / reps
    
    return {
        "B": B, "H": H, "KV": KV, "N": N, "D": D, "dtype": str(dtype).split(".")[-1],
        "elapsed_ms": elapsed_ms,
        "reps": reps
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    
    configs = []
    if args.mode == "quick":
        configs = [
            (1, 32, 32, 1024, 128, torch.bfloat16),
            (4, 32, 32, 2048, 128, torch.bfloat16),
        ]
    else:
        configs = [
            (1, 32, 32, 1024, 128, torch.bfloat16),
            (4, 32, 32, 2048, 128, torch.bfloat16),
            (2, 16, 16, 4096, 128, torch.bfloat16),
            (1, 8, 8, 8192, 128, torch.bfloat16),
        ]
    
    results = []
    for B, H, KV, N, D, dtype in configs:
        try:
            result = run_benchmark(B, H, KV, N, D, dtype)
            results.append(result)
            print(f"B={B} H={H} KV={KV} N={N} D={D} {dtype}: {result['elapsed_ms']:.3f} ms")
        except Exception as e:
            print(f"B={B} H={H} KV={KV} N={N} D={D} {dtype}: FAILED - {e}")
            results.append({"B": B, "H": H, "KV": KV, "N": N, "D": D, "dtype": str(dtype).split(".")[-1], "error": str(e)})
    
    if args.json:
        output = {
            "results": results,
            "geomean_ms": None
        }
        if all("elapsed_ms" in r for r in results):
            import math
            product = 1.0
            for r in results:
                product *= r["elapsed_ms"]
            output["geomean_ms"] = product ** (1.0 / len(results))
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()