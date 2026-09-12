"""Correctness contract: `attention` must match MLX's fused SDPA."""

from __future__ import annotations

import mlx.core as mx
import pytest
from attention import attention

# (B, H_q, H_kv, L, S, D, causal)
CASES = [
    pytest.param(1, 8, 8, 1, 512, 64, False, id="decode-mha"),
    pytest.param(2, 8, 2, 1, 1024, 128, False, id="decode-gqa"),
    pytest.param(1, 8, 8, 256, 256, 64, True, id="prefill-causal"),
    pytest.param(2, 16, 4, 128, 128, 128, True, id="prefill-gqa-causal"),
    pytest.param(1, 4, 4, 32, 512, 64, True, id="chunked-prefill-causal"),
    pytest.param(1, 4, 4, 7, 7, 96, False, id="odd-shapes"),
]
DTYPES = [pytest.param(mx.float32, id="f32"), pytest.param(mx.float16, id="f16")]
TOL = {mx.float32: 1e-5, mx.float16: 2e-3}


@pytest.mark.parametrize("B,H_q,H_kv,L,S,D,causal", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_fused_sdpa(B, H_q, H_kv, L, S, D, causal, dtype):
    keys = mx.random.split(mx.random.key(0), 3)
    q = mx.random.normal((B, H_q, L, D), key=keys[0]).astype(dtype)
    k = mx.random.normal((B, H_kv, S, D), key=keys[1]).astype(dtype)
    v = mx.random.normal((B, H_kv, S, D), key=keys[2]).astype(dtype)

    out = attention(q, k, v, causal=causal)
    ref = mx.fast.scaled_dot_product_attention(
        q, k, v, scale=D**-0.5, mask="causal" if causal else None
    )
    mx.eval(out, ref)

    assert out.shape == ref.shape
    assert out.dtype == dtype
    err = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item()
    assert err < TOL[dtype], f"max abs error {err}"


def test_explicit_scale():
    q = mx.random.normal((1, 2, 4, 16))
    k = mx.random.normal((1, 2, 8, 16))
    v = mx.random.normal((1, 2, 8, 16))
    out = attention(q, k, v, scale=0.1)
    ref = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.1)
    assert mx.allclose(out, ref, atol=1e-5).item()
