"""Correctness tests for cuTile attention."""

import pytest
import torch
import sys
sys.path.insert(0, ".")
from attention import attention

def _test_attention(B, H, L, S, D, dtype, causal, atol=1e-3, rtol=1e-3):
    torch.manual_seed(42)
    dt = getattr(torch, dtype)
    
    q = torch.randn(B, H, L, D, dtype=dt, device="cuda")
    k = torch.randn(B, H, S, D, dtype=dt, device="cuda")
    v = torch.randn(B, H, S, D, dtype=dt, device="cuda")
    
    # Reference
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    
    # cuTile
    out = attention(q, k, v, causal=causal)
    
    assert torch.allclose(out, ref_out, atol=atol, rtol=rtol), \
        f"Max diff: {(out - ref_out).abs().max().item()}"

@pytest.mark.parametrize("B,H,L,S,D", [
    (1, 4, 32, 32, 32),
    (2, 8, 64, 64, 64),
    (1, 8, 128, 256, 128),
])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("causal", [False, True])
def test_attention(B, H, L, S, D, dtype, causal):
    _test_attention(B, H, L, S, D, dtype, causal)

@pytest.mark.parametrize("B,H,L,S,D", [
    (1, 8, 1, 128, 64),    # decode (L=1)
    (1, 8, 1, 2048, 128),  # long decode
])
@pytest.mark.parametrize("dtype", ["float16"])
def test_decode(B, H, L, S, D, dtype):
    _test_attention(B, H, L, S, D, dtype, causal=True)

@pytest.mark.parametrize("B,H,L,S,D", [
    (2, 16, 512, 512, 128),
])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_large_prefill(B, H, L, S, D, dtype):
    _test_attention(B, H, L, S, D, dtype, causal=True)

# GQA test
def test_gqa():
    torch.manual_seed(42)
    B, H_q, H_kv, L, S, D = 2, 16, 4, 128, 128, 64
    q = torch.randn(B, H_q, L, D, dtype=torch.float16, device="cuda")
    k = torch.randn(B, H_kv, S, D, dtype=torch.float16, device="cuda")
    v = torch.randn(B, H_kv, S, D, dtype=torch.float16, device="cuda")
    
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = attention(q, k, v, causal=True)
    
    assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])