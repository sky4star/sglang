#!/usr/bin/env python3
"""K1: grouped w13 (gate+up dual block-scaled MMA) + fused SwiGLU(limit).

One CTA = (expert e, m-tile); loops n-blocks; two `T.mma_gemm_blockscaled`
calls into Cg/Cu against wg/wu (= w1 split at N in the checkpoint).

Host prep (M1): weights quantized/laid out as in grouped_nvfp4_gemm; activations
and their scale factors are produced by the existing quant path
(flashinfer fp4_quantize + host pack/swizzle) for now.
"""
import math

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

try:
    from .grouped_nvfp4_gemm import pack_scale_words, swizzle_chunk_kmajor, decode_fp4, decode_scale_words
except ImportError:
    from grouped_nvfp4_gemm import pack_scale_words, swizzle_chunk_kmajor, decode_fp4, decode_scale_words

FLOAT8_E4M3_MAX = 448.0
FLOAT4_E2M1_MAX = 6.0


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def w13_swiglu(
    E: int,
    N: int,
    K: int,
    rows: int,
    bpe: int,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 256,
    lim: float = 10.0,
    out_dtype=T.bfloat16,
):
    assert N % 128 == 0 and K % 256 == 0 and block_K % 64 == 0
    k_blocks = K // block_K
    words = block_K // 64
    n_blocks = N // 128
    BM = 128
    BN = 128
    in_dtype = T.float4_e2m1fn

    @T.prim_func
    def main(
        A: T.Tensor((rows, K), in_dtype),
        Bg: T.Tensor((E, N, K), in_dtype),
        Bu: T.Tensor((E, N, K), in_dtype),
        SFA: T.Tensor((E * bpe * BM * k_blocks, words), T.uint32),
        SFBg: T.Tensor((E * n_blocks * k_blocks * BN, words), T.uint32),
        SFBu: T.Tensor((E * n_blocks * k_blocks * BN, words), T.uint32),
        offsets: T.Tensor((E + 1,), T.int32),
        scale: T.Tensor((E,), T.float32),
        act: T.Tensor((rows, N), out_dtype),
    ):
        with T.Kernel(n_blocks, E * bpe, threads=threads) as (bn, bm):
            A_s = T.alloc_shared((BM, block_K), in_dtype)
            Bg_s = T.alloc_shared((BN, block_K), in_dtype)
            Bu_s = T.alloc_shared((BN, block_K), in_dtype)
            SFA_s = T.alloc_shared((BM, words), T.uint32)
            SFBg_s = T.alloc_shared((BN, words), T.uint32)
            SFBu_s = T.alloc_shared((BN, words), T.uint32)
            Cg = T.alloc_fragment((BM, BN), T.float32)
            Cu = T.alloc_fragment((BM, BN), T.float32)

            e = bm // bpe
            local = bm % bpe
            m_start = offsets[e] + local * BM
            actual = T.max(0, T.min(BM, offsets[e + 1] - m_start))
            sc = scale[e]

            T.clear(Cg)
            T.clear(Cu)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[m_start:m_start + BM, ko * block_K:(ko + 1) * block_K], A_s)
                T.copy(Bg[e, bn * BN:(bn + 1) * BN, ko * block_K:(ko + 1) * block_K], Bg_s)
                T.copy(Bu[e, bn * BN:(bn + 1) * BN, ko * block_K:(ko + 1) * block_K], Bu_s)
                for r, w in T.Parallel(BM, words):
                    SFA_s[r, w] = SFA[((e * bpe + local) * k_blocks + ko) * BM + r, w]
                for r, w in T.Parallel(BN, words):
                    SFBg_s[r, w] = SFBg[((e * n_blocks + bn) * k_blocks + ko) * BN + r, w]
                for r, w in T.Parallel(BN, words):
                    SFBu_s[r, w] = SFBu[((e * n_blocks + bn) * k_blocks + ko) * BN + r, w]
                T.mma_gemm_blockscaled(A_s, Bg_s, Cg, SFA_s, SFBg_s, transpose_B=True,
                                       clear_accum=False, k_start=ko * block_K,
                                       sf_a_granularity_k=16, sf_b_granularity_k=16,
                                       sf_layout="blockscaled_chunk_kmajor")
                T.mma_gemm_blockscaled(A_s, Bu_s, Cu, SFA_s, SFBu_s, transpose_B=True,
                                       clear_accum=False, k_start=ko * block_K,
                                       sf_a_granularity_k=16, sf_b_granularity_k=16,
                                       sf_layout="blockscaled_chunk_kmajor")

            for i, j in T.Parallel(BM, BN):
                if i < actual:
                    g = Cg[i, j] * sc
                    u = Cu[i, j] * sc
                    g = T.min(g, lim)
                    u = T.max(T.min(u, lim), -lim)
                    act[m_start + i, bn * BN + j] = T.Cast(out_dtype, g / (1.0 + T.exp(-g)) * u)

    return main


def _quant_w(w):
    """w [E, R, K] bf16 -> packed fp4 [E,R,K/2] uint8, semantic sf [E,R,K/16] uint8, gs [E]."""
    from flashinfer import fp4_quantize
    E, R, K = w.shape
    q = torch.empty((E, R, K // 2), dtype=torch.uint8, device=w.device)
    sf = torch.empty((E, R, K // 16), dtype=torch.uint8, device=w.device)
    gs = torch.empty((E,), dtype=torch.float32, device=w.device)
    for e in range(E):
        g = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / w[e].float().abs().max().clamp_min(1e-8)
        qe, sfe = fp4_quantize(w[e], g.reshape(1), sf_vec_size=16, is_sf_swizzled_layout=False)
        q[e] = qe; sf[e] = sfe; gs[e] = g
    return q, sf, gs


def build_test(E, N, K, rows, bpe, seed=0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    cap = bpe * 128
    base = rows // E
    sizes = torch.full((E,), base, dtype=torch.int64)
    sizes[: rows - base * E] += 1
    assert sizes.max() <= cap
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes, 0).to(torch.int32)

    A = torch.randint(-128, 128, (rows, K // 2), device=dev, dtype=torch.int8, generator=g)
    n_blocks = N // 128
    # semantic A scales padded per expert
    sfa_sem = torch.zeros((E * cap, K // 16), device=dev, dtype=torch.uint8)
    for e in range(E):
        s0, s1 = int(offsets[e]), int(offsets[e + 1])
        sfa_sem[e * cap:e * cap + (s1 - s0)] = torch.randint(56, 64, (s1 - s0, K // 16), device=dev, dtype=torch.uint8, generator=g)
    SFA = swizzle_chunk_kmajor(pack_scale_words(sfa_sem)).reshape(-1, 128 // 64)

    Wg = torch.randint(-128, 128, (E, N, K // 2), device=dev, dtype=torch.int8, generator=g)
    Wu = torch.randint(-128, 128, (E, N, K // 2), device=dev, dtype=torch.int8, generator=g)
    sfbg_sem = torch.randint(56, 64, (E * N, K // 16), device=dev, dtype=torch.uint8, generator=g)
    sfbu_sem = torch.randint(56, 64, (E * N, K // 16), device=dev, dtype=torch.uint8, generator=g)
    words = 128 // 64
    SFBg = swizzle_chunk_kmajor(pack_scale_words(sfbg_sem)).reshape(-1, words)
    SFBu = swizzle_chunk_kmajor(pack_scale_words(sfbu_sem)).reshape(-1, words)
    scale = torch.rand(E, device=dev, dtype=torch.float32) + 0.05
    return dict(A=A, Bg=Wg, Bu=Wu, SFA=SFA, SFBg=SFBg, SFBu=SFBu, offsets=offsets,
                scale=scale, sizes=sizes, sfa_sem=sfa_sem, sfbg_sem=sfbg_sem, sfbu_sem=sfbu_sem)


def main():
    E, N, K, rows, bpe, lim = 8, 256, 512, 1024, 1, 10.0
    dev = "cuda"
    d = build_test(E, N, K, rows, bpe)
    act = torch.empty(rows, N, device=dev, dtype=torch.bfloat16)
    kern = w13_swiglu(E, N, K, rows, bpe, block_K=128, lim=lim)
    kern(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"], d["offsets"], d["scale"], act)
    torch.cuda.synchronize()

    # reference (dequant + fp32)
    K16 = K // 16
    Ad = decode_fp4(d["A"], rows, K) * decode_scale_words(pack_scale_words(d["sfa_sem"]), K).repeat_interleave(16, 1)
    Wgd = decode_fp4(d["Bg"].reshape(E * N, K // 2), E * N, K).reshape(E, N, K) * \
        decode_scale_words(pack_scale_words(d["sfbg_sem"]), K).reshape(E, N, K16).repeat_interleave(16, 2)
    Wud = decode_fp4(d["Bu"].reshape(E * N, K // 2), E * N, K).reshape(E, N, K) * \
        decode_scale_words(pack_scale_words(d["sfbu_sem"]), K).reshape(E, N, K16).repeat_interleave(16, 2)
    ref = torch.empty(rows, N, device=dev, dtype=torch.float32)
    cap = bpe * 128
    for e in range(E):
        s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
        if s1 <= s0:
            continue
        a = Ad[s0:s1]
        g = (a @ Wgd[e].t()) * d["scale"][e]
        u = (a @ Wud[e].t()) * d["scale"][e]
        g = g.clamp(max=lim)
        u = u.clamp(-lim, lim)
        ref[s0:s1] = F.silu(g) * u
    err = (act.float() - ref).abs()
    rel = err.max().item() / (ref.abs().max().item() + 1e-6)
    print(f"w13+swiglu verify: max_abs={err.max().item():.5f} ref_absmax={ref.abs().max().item():.3f} max_rel={rel:.5f}")
    assert rel < 2e-2, "mismatch"

    ms = do_bench(lambda: kern(d["A"], d["Bg"], d["Bu"], d["SFA"], d["SFBg"], d["SFBu"], d["offsets"], d["scale"], act))
    print(f"w13+swiglu: {ms:.4f} ms")


if __name__ == "__main__":
    main()
