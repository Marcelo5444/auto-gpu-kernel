#!/usr/bin/env python3
"""Correctness tests for cuTile AdaSplash get_output kernel."""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "repo"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "adasplash-output"))

import torch
from cutile_get_output import launch_get_output

def fp64_reference_get_output(q, k, v, taus, sm_scale, mask=None):
    """FP64 reference implementation matching the kernel contract."""
    B, H, N, D = q.shape
    KV = k.shape[1]
    GROUP_SIZE = H // KV
    device = q.device
    
    out = torch.zeros(B, H, N, D, dtype=torch.float64, device=device)
    out2 = torch.zeros(B, H, N, D, dtype=torch.float64, device=device)
    
    for b in range(B):
        for h in range(H):
            q_bh = q[b, h].double()
            k_bh = k[b, h // GROUP_SIZE].double()
            v_bh = v[b, h // GROUP_SIZE].double()
            tau_bh = taus[b, h].double()
            
            # Q @ K^T
            scores = q_bh @ k_bh.T  # [N, N] in fp64
            scores = scores * (0.5 * sm_scale)
            
            # Causal mask
            causal = torch.ones(N, N, dtype=torch.bool, device=device).tril()
            scores = scores.masked_fill(~causal, float('-inf'))
            
            # Entmax-1.5 (alpha=2): proj = max(0, scores - tau)^2
            proj = (scores - tau_bh[:, None]).clamp_min(0.0) ** 2
            
            # out = proj @ V
            out[b, h] = proj @ v_bh
            
            # out2 = proj @ V / sum(proj)  (normalized)
            proj_sum = proj.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            out2[b, h] = (proj @ v_bh) / proj_sum
    
    return out, out2

def test_correctness(B, H, KV, N, D, dtype=torch.bfloat16):
    """Test kernel against fp64 reference."""
    device = "cuda"
    torch.manual_seed(42)
    
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, KV, N, D, device=device, dtype=dtype)
    v = torch.randn(B, KV, N, D, device=device, dtype=dtype)
    taus = torch.randn(B, H, N, device=device, dtype=torch.float32).abs()
    mask = None
    cnt = None
    sm_scale = 1.0 / (D ** 0.5)
    
    # Run kernel
    out, out2, _ = launch_get_output(q, k, v, taus, mask, cnt, sm_scale=sm_scale)
    
    # Run reference
    ref_out, ref_out2 = fp64_reference_get_output(q, k, v, taus, sm_scale)
    
    # Compare
    max_diff_out = (out.double() - ref_out).abs().max().item()
    max_diff_out2 = (out2.double() - ref_out2).abs().max().item()
    
    # Tolerance for bfloat16 - relaxed for degenerate cases where
    # proj_sum is clamped to 1e-8 (amplifies tiny numerator errors)
    if dtype == torch.bfloat16:
        atol = 0.05
        rtol = 0.05
    else:
        atol = 1e-3
        rtol = 1e-3
            
    # Also check relative error for non-degenerate cases
    ref_out_mean = ref_out.abs().mean().item()
    ref_out2_mean = ref_out2.abs().mean().item()
    rel_out = max_diff_out / max(ref_out_mean, 1e-8)
    rel_out2 = max_diff_out2 / max(ref_out2_mean, 1e-8)
            
    passed = (max_diff_out < atol or rel_out < rtol) and (max_diff_out2 < atol or rel_out2 < rtol)
    
    return {
        "B": B, "H": H, "KV": KV, "N": N, "D": D, "dtype": str(dtype).split(".")[-1],
        "max_diff_out": max_diff_out,
        "max_diff_out2": max_diff_out2,
        "atol": atol,
        "rtol": rtol,
        "rel_out": rel_out,
        "rel_out2": rel_out2,
        "passed": passed
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    
    test_configs = [
        (1, 32, 32, 1024, 128, torch.bfloat16),
        (2, 16, 16, 512, 128, torch.bfloat16),
        (1, 8, 8, 1024, 128, torch.bfloat16),
        (4, 32, 32, 2048, 128, torch.bfloat16),
    ]
    
    results = []
    all_passed = True
    for B, H, KV, N, D, dtype in test_configs:
        try:
            result = test_correctness(B, H, KV, N, D, dtype)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"B={B} H={H} KV={KV} N={N} D={D} {dtype}: {status} (out: {result['max_diff_out']:.2e}, out2: {result['max_diff_out2']:.2e})")
            if not result["passed"]:
                all_passed = False
        except Exception as e:
            print(f"B={B} H={H} KV={KV} N={N} D={D} {dtype}: ERROR - {e}")
            results.append({"B": B, "H": H, "KV": KV, "N": N, "D": D, "dtype": str(dtype).split(".")[-1], "error": str(e), "passed": False})
            all_passed = False
    
    if args.json:
        output = {"passed": all_passed, "details": results}
        print(json.dumps(output, indent=2))
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()