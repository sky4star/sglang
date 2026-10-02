#!/usr/bin/env python3
"""Fully-fused TileLang MoE (M1/M2):

    K0 gather+quant   : X, token_ids -> A1 fp4 + E4M3 scales (blockscaled layout)
    K1 w13 gate+up dual block-scaled MMA + fused SwiGLU -> act bf16
    K2 requant        : act -> A2 fp4 + scales
    K3 w2 block-scaled GEMM + fused routing-weight + atomic scatter -> y

Target: match flashinfer.cutlass_fused_moe (~17.7 ms at M=4096).

Quant convention (K0/K2): absolute E4M3 block-16 scales (sf = E4M3(amax/6)),
dequant = e2m1 * sf (no global scale). Weights keep the fp4_quantize global
scale, so the GEMM outputs are divided by the per-expert weight global scale.
"""
import argparse
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from flashinfer import fp4_quantize

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from .grouped_nvfp4_gemm import (
        pack_scale_words, swizzle_chunk_kmajor, decode_fp4, decode_scale_words,
    )
    from .w13_swiglu_kernel import w13_swiglu
    from .w2_scatter_kernel import w2_scatter
    from .act_quant_kernel import gather_quant
except ImportError:
    from grouped_nvfp4_gemm import (
        pack_scale_words, swizzle_chunk_kmajor, decode_fp4, decode_scale_words,
    )
    from w13_swiglu_kernel import w13_swiglu
    from w2_scatter_kernel import w2_scatter
    from act_quant_kernel import gather_quant

BLOCK_K = 128
GS_CONST = 448.0 * 6.0


def quant_per_expert(w):
    E, R, K = w.shape
    q = torch.empty((E, R, K // 2), device=w.device, dtype=torch.uint8)
    sf = torch.empty((E, R, K // 16), device=w.device, dtype=torch.uint8)
    gs = torch.empty((E,), device=w.device, dtype=torch.float32)
    for e in range(E):
        g = GS_CONST / w[e].float().abs().max().clamp_min(1e-8)
        qe, sfe = fp4_quantize(w[e], g.reshape(1), sf_vec_size=16, is_sf_swizzled_layout=False)
        q[e] = qe; sf[e] = sfe; gs[e] = g
    return q, sf, gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--E", type=int, default=144)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--lim", type=float, default=10.0)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    M, E, K, N, topk = args.M, args.E, args.K, args.N, args.topk
    rows = M * topk
    g = torch.Generator(device=dev).manual_seed(0)

    X = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g) / 10
    w1 = torch.randn(E, 2 * N, K, device=dev, dtype=torch.bfloat16, generator=g) / 10
    w2 = torch.randn(E, K, N, device=dev, dtype=torch.bfloat16, generator=g) / 10
    score = torch.randn(M, E, device=dev, dtype=torch.bfloat16, generator=g)
    tw, ti = torch.topk(score, topk, dim=-1, sorted=False)
    tw = torch.softmax(tw.float(), dim=-1)

    flat = ti.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    token_ids = (order // topk).to(torch.int32)
    counts = torch.bincount(flat, minlength=E)
    sizes = counts.cpu().tolist()
    cap = math.ceil(max(sizes) / 128) * 128
    bpe = cap // 128
    offsets = torch.zeros(E + 1, device=dev, dtype=torch.int32)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    counts_dev = torch.tensor(sizes, device=dev)
    row_w = tw.reshape(-1)[order].contiguous()
    padded = torch.empty(rows, dtype=torch.int32, device=dev)
    off = 0
    for e in range(E):
        n = sizes[e]
        padded[off:off + n] = e * cap + torch.arange(n, device=dev)
        off += n
    src_identity = torch.arange(rows, device=dev, dtype=torch.int32)

    print(f"M={M} E={E} K={K} N={N} topk={topk} rows={rows} bpe={bpe} cap={cap} "
          f"sizes[min/max]={min(sizes)}/{max(sizes)}")

    # ---- weights ----
    wg_q, wg_sf, wg_gs = quant_per_expert(w1[:, :N].contiguous())
    wu_q, wu_sf, wu_gs = quant_per_expert(w1[:, N:].contiguous())
    w2_q, w2_sf, w2_gs = quant_per_expert(w2)
    words = BLOCK_K // 64
    SFBg = swizzle_chunk_kmajor(pack_scale_words(wg_sf.reshape(E * N, -1))).reshape(-1, words)
    SFBu = swizzle_chunk_kmajor(pack_scale_words(wu_sf.reshape(E * N, -1))).reshape(-1, words)
    SFB2 = swizzle_chunk_kmajor(pack_scale_words(w2_sf.reshape(E * K, -1))).reshape(-1, words)

    # ---- K0: gather+quant A1 ----
    A1 = torch.empty(rows, K // 2, device=dev, dtype=torch.uint8)
    SFA1sem = torch.empty(E * cap, K // 16, device=dev, dtype=torch.float8_e4m3fn)
    k0 = gather_quant(rows, M, K, E, cap)
    k0(X, token_ids, padded, A1, SFA1sem)
    SFA1 = swizzle_chunk_kmajor(pack_scale_words(SFA1sem.view(torch.uint8))).reshape(-1, words)

    # ---- K1 ----
    scale1 = 1.0 / wg_gs  # no activation global scale
    act = torch.empty(rows, N, device=dev, dtype=torch.bfloat16)
    k1 = w13_swiglu(E, N, K, rows, bpe, block_K=BLOCK_K, lim=args.lim)
    k1(A1, wg_q, wu_q, SFA1, SFBg, SFBu, offsets, scale1, act)
    torch.cuda.synchronize()

    # ---- K2: requant act ----
    A2 = torch.empty(rows, N // 2, device=dev, dtype=torch.uint8)
    SFA2sem = torch.empty(E * cap, N // 16, device=dev, dtype=torch.float8_e4m3fn)
    k2 = gather_quant(rows, rows, N, E, cap)
    k2(act, src_identity, padded, A2, SFA2sem)
    SFA2 = swizzle_chunk_kmajor(pack_scale_words(SFA2sem.view(torch.uint8))).reshape(-1, words)

    # ---- K3 ----
    row_scale = (1.0 / w2_gs).repeat_interleave(counts_dev) * row_w
    y = torch.zeros(M, K, device=dev, dtype=torch.float32)
    k3 = w2_scatter(E, K, N, M, rows, bpe, block_N=BLOCK_K)
    k3(A2, w2_q, SFA2, SFB2, offsets, token_ids, row_scale, y)
    torch.cuda.synchronize()

    if not args.no_verify:
        def deq_e4m3(s):
            b = s.view(torch.uint8).to(torch.int64)
            sign = torch.where((b >> 7) & 1 == 1, -1.0, 1.0)
            e = ((b >> 3) & 0xF).float()
            m = (b & 7).float() / 8.0
            return sign * torch.where(e == 0, m * 2 ** -6, (1.0 + m) * torch.pow(2.0, e - 7.0))

        def deq_a(q, sem):  # q uint8 [r,cols/2], sem e4m3 [E*cap,cols/16]
            full = decode_fp4(q, q.shape[0], q.shape[1] * 2)
            sc = deq_e4m3(sem[padded.long()]).repeat_interleave(16, dim=1)
            return full * sc

        A1d = deq_a(A1, SFA1sem)
        A2d = deq_a(A2, SFA2sem)
        wgd = (decode_fp4(wg_q.reshape(E * N, K // 2), E * N, K).reshape(E, N, K)
               * deq_e4m3(wg_sf).reshape(E, N, K // 16).repeat_interleave(16, dim=2)) / wg_gs.view(E, 1, 1)
        wud = (decode_fp4(wu_q.reshape(E * N, K // 2), E * N, K).reshape(E, N, K)
               * deq_e4m3(wu_sf).reshape(E, N, K // 16).repeat_interleave(16, dim=2)) / wu_gs.view(E, 1, 1)
        w2d = (decode_fp4(w2_q.reshape(E * K, N // 2), E * K, N).reshape(E, K, N)
               * deq_e4m3(w2_sf).reshape(E, K, N // 16).repeat_interleave(16, dim=2)) / w2_gs.view(E, 1, 1)
        y_ref = torch.zeros(M, K, device=dev, dtype=torch.float32)
        for e in range(E):
            s0, s1 = int(offsets[e]), int(offsets[e + 1])
            if s1 <= s0:
                continue
            a = A1d[s0:s1]
            g_ = a @ wgd[e].t()
            u_ = a @ wud[e].t()
            act_ref = F.silu(g_.clamp(max=args.lim)) * u_.clamp(-args.lim, args.lim)
            # A2 was quantized from act_ref, so use the dequantized A2 for the oracle
            o = A2d[s0:s1] @ w2d[e].t()
            y_ref.index_add_(0, token_ids[s0:s1].long(), o * row_w[s0:s1].unsqueeze(1))
        err = (y - y_ref).abs()
        rel = err.max().item() / (y_ref.abs().max().item() + 1e-6)
        print(f"verify (dequant oracle): max_abs={err.max().item():.5f} ref_absmax={y_ref.abs().max().item():.3f} max_rel={rel:.5f}")
        assert rel < 2e-2, f"pipeline mismatch {rel}"

    def bench(fn, n=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record(); e.synchronize()
        return s.elapsed_time(e) / n

    t0 = bench(lambda: k0(X, token_ids, padded, A1, SFA1sem))
    t1 = bench(lambda: k1(A1, wg_q, wu_q, SFA1, SFBg, SFBu, offsets, scale1, act))
    t2 = bench(lambda: k2(act, src_identity, padded, A2, SFA2sem))
    t3 = bench(lambda: (y.zero_(), k3(A2, w2_q, SFA2, SFB2, offsets, token_ids, row_scale, y)))
    print(f"K0={t0:.3f}  K1={t1:.3f}  K2={t2:.3f}  K3={t3:.3f} ms   total={t0+t1+t2+t3:.3f} ms "
          f"(baseline cutlass_fused_moe ~17.73)")


if __name__ == "__main__":
    main()
