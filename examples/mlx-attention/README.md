# mlx-attention

A deliberately naive scaled dot-product attention in [MLX](https://github.com/ml-explore/mlx),
used as the development example for `kopt init-task` on Apple Silicon.

```bash
pip install mlx pytest
python -m pytest -q          # correctness vs. mx.fast.scaled_dot_product_attention
python bench.py --quick      # latency on one decode + one prefill shape
python bench.py              # all workloads; geomean median ms is the metric
```

- `attention.py` — the function to optimize (`attention(q, k, v, scale=, causal=)`).
- `test_attention.py` — the contract it must keep.
- `bench.py` — decode (L=1) and causal prefill workloads in float16.
