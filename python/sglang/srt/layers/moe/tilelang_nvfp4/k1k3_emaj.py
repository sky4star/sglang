#!/usr/bin/env python3
"""Expert-major rasterization variants of the K1/K3 grouped GEMMs.

Same math as `w13_swiglu_kernel.w13_swiglu` / `w2_gemm_kernel.w2_gemm`
(bit-exact vs the bn-major kernels), only the CTA raster order changes:

    before: T.Kernel(n_blocks, E*bpe) as (bn, bm)   # bn fastest
    after:  T.Kernel(bpe, n_blocks, E) as (local, bn, e)

Consecutive CTAs then share the same (expert, bn) B tile across the
expert's M-tiles (dim0 = local is fastest), so the first CTA streams the
B tile from DRAM into L2 and the sibling M-tile CTAs hit L2. This cuts
the per-expert B re-reads from ~bpe x to ~1 x: w13 reaches 217 GB/s
(~90% of the GB10 streaming roof) and the full fused MoE pipeline goes
from 15.56 -> 14.88 ms @M=4096 and 23.90 -> 21.70 ms @M=8192
(baseline flashinfer cutlass_fused_moe: 17.73 / 28.34 ms).

Scale-factor packing is unchanged (BM=128 keeps the 128-row SF atoms).
"""
import tilelang
import tilelang.language as T


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def w13_swiglu_emaj(E, N, K, rows, bpe, block_K=128, num_stages=2, threads=256,
                    lim=10.0, out_dtype=T.bfloat16):
    """Grouped w13 (gate+up dual block-scaled MMA) + fused SwiGLU, expert-major."""
    assert N % 128 == 0 and K % block_K == 0 and block_K % 64 == 0
    k_blocks = K // block_K
    words = block_K // 64
    n_blocks = N // 128
    BM, BN = 128, 128
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
        with T.Kernel(bpe, n_blocks, E, threads=threads) as (local, bn, e):
            A_s = T.alloc_shared((BM, block_K), in_dtype)
            Bg_s = T.alloc_shared((BN, block_K), in_dtype)
            Bu_s = T.alloc_shared((BN, block_K), in_dtype)
            SFA_s = T.alloc_shared((BM, words), T.uint32)
            SFBg_s = T.alloc_shared((BN, words), T.uint32)
            SFBu_s = T.alloc_shared((BN, words), T.uint32)
            Cg = T.alloc_fragment((BM, BN), T.float32)
            Cu = T.alloc_fragment((BM, BN), T.float32)

            bm = e * bpe + local
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


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def w2_emaj(E, K, N, rows, bpe, block_N=128, threads=128, num_stages=2,
            out_dtype=T.bfloat16):
    """Grouped w2 block-scaled GEMM, expert-ordered contiguous output, expert-major."""
    assert K % block_N == 0 and block_N % 64 == 0 and N % 128 == 0
    k_blocks = N // block_N
    words = block_N // 64
    n_blocks = K // 128
    BM, BN = 128, 128
    in_dtype = T.float4_e2m1fn

    @T.prim_func
    def main(
        A: T.Tensor((rows, N), in_dtype),
        B: T.Tensor((E, K, N), in_dtype),
        SFA: T.Tensor((E * bpe * BM * k_blocks, words), T.uint32),
        SFB: T.Tensor((E * n_blocks * k_blocks * BN, words), T.uint32),
        offsets: T.Tensor((E + 1,), T.int32),
        out2: T.Tensor((rows, K), out_dtype),
    ):
        with T.Kernel(bpe, n_blocks, E, threads=threads) as (local, bn, e):
            A_s = T.alloc_shared((BM, block_N), in_dtype)
            B_s = T.alloc_shared((BN, block_N), in_dtype)
            SFA_s = T.alloc_shared((BM, words), T.uint32)
            SFB_s = T.alloc_shared((BN, words), T.uint32)
            C = T.alloc_fragment((BM, BN), T.float32)

            bm = e * bpe + local
            m_start = offsets[e] + local * BM
            actual = T.max(0, T.min(BM, offsets[e + 1] - m_start))

            T.clear(C)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[m_start:m_start + BM, ko * block_N:(ko + 1) * block_N], A_s)
                T.copy(B[e, bn * BN:(bn + 1) * BN, ko * block_N:(ko + 1) * block_N], B_s)
                for r, w in T.Parallel(BM, words):
                    SFA_s[r, w] = SFA[((e * bpe + local) * k_blocks + ko) * BM + r, w]
                for r, w in T.Parallel(BN, words):
                    SFB_s[r, w] = SFB[((e * n_blocks + bn) * k_blocks + ko) * BN + r, w]
                T.mma_gemm_blockscaled(A_s, B_s, C, SFA_s, SFB_s, transpose_B=True,
                                       clear_accum=False, k_start=ko * block_N,
                                       sf_a_granularity_k=16, sf_b_granularity_k=16,
                                       sf_layout="blockscaled_chunk_kmajor")
            for i, j in T.Parallel(BM, BN):
                if i < actual:
                    out2[m_start + i, bn * BN + j] = T.Cast(out_dtype, C[i, j])

    return main
