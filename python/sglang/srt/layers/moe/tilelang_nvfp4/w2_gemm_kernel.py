#!/usr/bin/env python3
"""K3-A: grouped w2 block-scaled GEMM with a plain (expert-ordered) output.

No scatter/atomics here: each CTA writes its tile to ``out2`` in expert-sorted
row order, contiguously.  The token combine is a separate gather-reduction
kernel (`moe_combine_kernel.py`).
"""
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

try:
    from .grouped_nvfp4_gemm import pack_scale_words, swizzle_chunk_kmajor, decode_fp4, decode_scale_words
except ImportError:
    from grouped_nvfp4_gemm import pack_scale_words, swizzle_chunk_kmajor, decode_fp4, decode_scale_words


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def w2_gemm(
    E: int,
    K: int,
    N: int,
    rows: int,
    bpe: int,
    block_N: int = 128,
    threads: int = 128,
    num_stages: int = 2,
    out_dtype=T.bfloat16,
):
    assert K % 128 == 0 and N % 256 == 0
    k_blocks = N // block_N
    words = block_N // 64
    n_blocks = K // 128
    BM = 128
    BN = 128
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
        with T.Kernel(n_blocks, E * bpe, threads=threads) as (bn, bm):
            A_s = T.alloc_shared((BM, block_N), in_dtype)
            B_s = T.alloc_shared((BN, block_N), in_dtype)
            SFA_s = T.alloc_shared((BM, words), T.uint32)
            SFB_s = T.alloc_shared((BN, words), T.uint32)
            C = T.alloc_fragment((BM, BN), T.float32)

            e = bm // bpe
            local = bm % bpe
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


def build_test(E, K, N, rows, bpe, block_N=128, seed=0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    cap = bpe * 128
    base = rows // E
    sizes = torch.full((E,), base, dtype=torch.int64); sizes[: rows - base * E] += 1
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes, 0).to(torch.int32)
    A = torch.randint(-128, 128, (rows, N // 2), device=dev, dtype=torch.int8, generator=g)
    sfa_sem = torch.zeros((E * cap, N // 16), device=dev, dtype=torch.uint8)
    for e in range(E):
        s0, s1 = int(offsets[e]), int(offsets[e + 1])
        sfa_sem[e * cap:e * cap + (s1 - s0)] = torch.randint(56, 64, (s1 - s0, N // 16), device=dev, dtype=torch.uint8, generator=g)
    words = block_N // 64
    SFA = swizzle_chunk_kmajor(pack_scale_words(sfa_sem)).reshape(-1, words)
    B = torch.randint(-128, 128, (E, K, N // 2), device=dev, dtype=torch.int8, generator=g)
    sfb_sem = torch.randint(56, 64, (E * K, N // 16), device=dev, dtype=torch.uint8, generator=g)
    SFB = swizzle_chunk_kmajor(pack_scale_words(sfb_sem)).reshape(-1, words)
    return dict(A=A, B=B, SFA=SFA, SFB=SFB, offsets=offsets, sfa_sem=sfa_sem, sfb_sem=sfb_sem)


def main():
    E, K, N, rows, bpe = 8, 256, 256, 1024, 1
    dev = "cuda"
    d = build_test(E, K, N, rows, bpe)
    out2 = torch.empty(rows, K, device=dev, dtype=torch.bfloat16)
    kern = w2_gemm(E, K, N, rows, bpe)
    kern(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], out2)
    torch.cuda.synchronize()

    K16 = N // 16
    Ad = decode_fp4(d["A"], rows, N) * decode_scale_words(pack_scale_words(d["sfa_sem"]), N).repeat_interleave(16, 1)
    Bd = decode_fp4(d["B"].reshape(E * K, N // 2), E * K, N).reshape(E, K, N) * \
        decode_scale_words(pack_scale_words(d["sfb_sem"]), N).reshape(E, K, K16).repeat_interleave(16, 2)
    ref = torch.empty(rows, K, device=dev, dtype=torch.float32)
    for e in range(E):
        s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
        if s1 > s0:
            ref[s0:s1] = Ad[s0:s1] @ Bd[e].t()
    err = (out2.float() - ref).abs()
    rel = err.max().item() / (ref.abs().max().item() + 1e-6)
    print(f"w2_gemm verify: max_abs={err.max().item():.5f} max_rel={rel:.5f}")
    assert rel < 2e-2


if __name__ == "__main__":
    main()
