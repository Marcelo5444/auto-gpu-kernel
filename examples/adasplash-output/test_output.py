#!/usr/bin/env python3
"""
Validation harness for AdaSplash get_output kernel.
Runs correctness checks against the fp64 mirror reference (from adasplash bench/fwd_output.py).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from fwd_output import get_output

# Import reference functions from adasplash bench
sys.path.insert(0, "/home/marcelo/auto-gpu-kernel/examples/adasplash-output")
import common


def refs_for_case(q, k, v, tau, mask, sm_scale, varlen, need2=True):
    """Per-(b,h) streamed fp64 references - copied from adasplash bench/fwd_output.py."""
    B, H, N, D = q.shape
    KV = k.shape[1]
    g = H // KV
    dev = q.device
    half = torch.tensor(sm_scale, dtype=torch.float32) * 0.5
    half64 = half.double()
    inv_half = (torch.tensor(2.0, dtype=torch.float32) / torch.tensor(sm_scale, dtype=torch.float32))
    half2_f32 = (half * half).to(dev)
    tau_raw = (tau.float() * inv_half.to(dev))
    causal = torch.ones(N, N, dtype=torch.bool, device=dev).tril()
    rows = torch.arange(N, device=dev)

    io_dtype = q.dtype
    out_p = torch.zeros(B, H, N, D, dtype=torch.float64, device=dev)
    out2_p = torch.zeros_like(out_p)
    out_m = torch.zeros(B, H, N, D, dtype=io_dtype, device=dev)
    out2_m = torch.zeros_like(out_m)
    supp_m = torch.zeros(B, H, N, dtype=torch.float64, device=dev)
    sump_min, sump_max = float("inf"), float("-inf")
    pmax = 0.0

    bits = common.unpack_mask(mask)[..., :N]
    for b in range(B):
        for h in range(H):
            q64 = q[b, h].double()
            k64 = k[b, h // g].double()
            v64 = v[b, h // g].double()
            s_raw = q64 @ k64.T
            valid = causal.clone()
            if varlen is not None:
                valid &= (rows[None, :] < int(varlen[b])) & (rows[:, None] < int(varlen[b]))

            # platonic: y units, full causal columns
            y = half64 * s_raw
            proj = (y - tau[b, h].double()[:, None]).clamp_min(0.0) * valid
            out_p[b, h] = (proj * proj) @ v64
            if need2:
                out2_p[b, h] = (proj @ v64) / proj.sum(-1, keepdim=True).clamp_min(1e-300)

            # mirror: raw units, union columns, band rule, bf16 P/P2
            tr = tau_raw[b, h].double()[:, None]
            d = (s_raw - tr).float().double().clamp_min(0.0)
            union = (bits[b, h, 0::2] | bits[b, h, 1::2])
            uexp = union.repeat_interleave(128, dim=0)[:N]
            d = d * uexp * causal
            if varlen is not None:
                d = d * valid
            P = d.to(io_dtype).double()
            P2 = (d * d).float().to(io_dtype).double()
            acc = (P2 @ v64).float()
            out_m[b, h] = (acc * half2_f32).to(io_dtype)
            sp = (P2 * half2_f32.double()).sum(-1)
            if need2:
                acc2 = (P @ v64).float()
                out2_m[b, h] = acc2.to(io_dtype)
                supp_m[b, h] = d.sum(-1)
                pmax = max(pmax, float(P.max()))
            if varlen is not None:
                sp = sp[: int(varlen[b])]
            if sp.numel():
                sump_min = min(sump_min, float(sp.min()))
                sump_max = max(sump_max, float(sp.max()))
    return out_p, out2_p, out_m, out2_m, supp_m, (sump_min, sump_max, pmax)


def check_case(B, H, KV, N, D, dtype, use_varlen):
    """Run a single correctness check case."""
    q, k, v, varlen = common.make_qkv(B, H, KV, N, D, dtype, varlen=use_varlen)
    sm_scale = 1.0 / D ** 0.5
    h_val = 1.0 / common.select_bins(N, 128)

    t_in = common.get_tau_hist(q, k, sm_scale, varlen)
    tau, mask, cnt = common.get_tau_refine(q, k, t_in, sm_scale, h=h_val, varlen=varlen)

    kw = dict(sm_scale=sm_scale, need_backward=True, varlen=varlen)
    out, out2, supp = get_output(q, k, v, tau, mask, cnt, overlap=True, **kw)
    out_b, out2_b, supp_b = get_output(q, k, v, tau, mask, cnt, overlap=True, **kw)
    out_s, out2_s, supp_s = get_output(q, k, v, tau, mask, cnt, overlap=False, **kw)
    out_f, _, _ = get_output(q, k, v, tau, mask, cnt, sm_scale=sm_scale,
                             need_backward=False, varlen=varlen, overlap=True)
    torch.cuda.synchronize()

    rowmask = common.row_mask(B, H, N, varlen, q.device).double() if varlen is not None else None
    ok = True

    # Invariants
    det = (torch.equal(out, out_b) and torch.equal(out2, out2_b)
           and torch.equal(supp, supp_b))
    sched = (torch.equal(out, out_s) and torch.equal(out2, out2_s)
             and torch.equal(supp, supp_s))
    fwd = torch.equal(out, out_f)
    print(f"  determinism: {'PASS' if det else 'FAIL'}  overlap==serial: {'PASS' if sched else 'FAIL'}  fwd-only: {'PASS' if fwd else 'FAIL'}")
    ok &= det and sched and fwd

    if varlen is not None:
        oob = ~rowmask.bool()
        z = float(out.double().abs()[oob].max()) if oob.any() else 0.0
        z2 = float(out2.double().abs()[oob].max()) if oob.any() else 0.0
        zs = float(supp.double().abs()[oob].max()) if oob.any() else 0.0
        zt = z == 0.0 and z2 == 0.0 and zs == 0.0
        print(f"  varlen OOB zero: {'PASS' if zt else f'FAIL (|out| {z:.1e}, |out2| {z2:.1e}, |supp| {zs:.1e})'}")
        ok &= zt

    out_p, out2_p, out_m, out2_m, supp_m, (spmin, spmax, pmax) = refs_for_case(
        q, k, v, tau, mask, sm_scale, varlen)

    # Mirror check (tight)
    eps_p = 2.0 ** -8 if dtype == torch.bfloat16 else 2.0 ** -11
    def _valid_absmax(t):
        a = t.double().abs()
        if rowmask is not None:
            a = torch.where(rowmask[..., None].bool(), a, torch.zeros_like(a))
        return float(a.max())
    exc_bound = 2.5 * eps_p * _valid_absmax(out_m)
    exc2_bound = 2.5 * eps_p * _valid_absmax(out2_m)
    frac_bound = 5e-4
    nv, wx = common.ulp_check(out, out_m, rowmask, ulps=2, extra_rel=0.0, extra_abs=2e-5)
    nv2, wx2 = common.ulp_check(out2, out2_m, rowmask, ulps=2, extra_rel=0.0, extra_abs=1e-3)
    nel = out.numel() if rowmask is None else int(rowmask.sum()) * out.shape[-1]
    m_ok = (nv / max(nel, 1) < frac_bound and wx < exc_bound
            and nv2 / max(nel, 1) < frac_bound and wx2 < exc2_bound)
    print(f"  vs mirror: out {nv} viol (excess {wx:.2e}) out2 {nv2} viol (excess {wx2:.2e}) {'PASS' if m_ok else 'FAIL'} sum(p_hat) in [{spmin:.4f}, {spmax:.4f}]")
    ok &= m_ok

    # Supp sidecar
    sd = (supp.double() - supp_m).abs()
    s_tol = 1e-4 * supp_m.abs() + 1e-3
    margin = sd - s_tol
    if rowmask is not None:
        margin = torch.where(rowmask.bool(), margin, torch.full_like(margin, -float("inf")))
    s_bad = int((margin > 0).sum())
    s_worst = float(margin.max())
    s_ok = s_bad == 0
    print(f"  supp sidecar: {s_bad} viol (worst excess {s_worst:.2e}, max P {pmax:.1f}) {'PASS' if s_ok else 'FAIL'}")
    ok &= s_ok

    print(f"  {'ALL PASS' if ok else 'FAIL'}")
    return ok


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--gqa", action="store_true")
    parser.add_argument("--varlen", action="store_true")
    args = parser.parse_args()

    dtype = common.DTYPES[args.dtype]
    all_ok = True
    for B, H, KV, N, uv in common.check_sweep(args, default_ns=(192, 256, 512, 1024, 2048)):
        print(f"B={B} H={H} KV={KV} N={N} D={args.d} {args.dtype} varlen={uv}")
        all_ok &= check_case(B, H, KV, N, args.d, dtype, uv)
    print(f"\nALL {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())