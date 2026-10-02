#!/usr/bin/env python3
"""K3: grouped w2 block-scaled GEMM + fused routing-weight multiply + atomic scatter.

One CTA = (expert e, m-tile, k-tile); epilogue does
    y[token_ids[row], col] += C[i,j] * row_scale[row]
where row_scale folds 1/(gs_a2*gs_w2[e]) and the routing weight.
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
def w2_scatter(
    E: int,
    K: int,
    N: int,
    M: int,
    rows: int,
    bpe: int,
    block_N: int = 128,   # reduction tile (over N)
    threads: int = 128,
    num_stages: int = 2,
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
        token_ids: T.Tensor((rows,), T.int32),
        row_scale: T.Tensor((rows,), T.float32),
        y: T.Tensor((M, K), T.float32),
    ):
        with T.Kernel(n_blocks, E * bpe, threads=threads) as (bn, bm):
            A_s = T.alloc_shared((BM, block_N), in_dtype)
            B_s = T.alloc_shared((BN, block_N), in_dtype)
            SFA_s = T.alloc_shared((BM, words), T.uint32)
            SFB_s = T.alloc_shared((BN, words), T.uint32)
            C = T.alloc_fragment((BM, BN), T.float32)
            C_sh = T.alloc_shared((BM, BN), T.float32)
            tok_sh = T.alloc_shared((BM,), T.int32)
            rs_sh = T.alloc_shared((BM,), T.float32)

            e = bm // bpe
            local = bm % bpe
            m_start = offsets[e] + local * BM
            actual = T.max(0, T.min(BM, offsets[e + 1] - m_start))

            for i in T.Parallel(BM):
                tok_sh[i] = token_ids[m_start + i]
                rs_sh[i] = row_scale[m_start + i]

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

            # stage through shared so the global atomics are coalesced
            T.copy(C, C_sh)
            for i, j in T.Parallel(BM, BN):
                if i < actual:
                    T.atomic_add(y[tok_sh[i], bn * BN + j], C_sh[i, j] * rs_sh[i])

    return main


def build_test(E, K, N, M, rows, bpe, block_N=128, seed=0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    cap = bpe * 128
    base = rows // E
    sizes = torch.full((E,), base, dtype=torch.int64)
    sizes[: rows - base * E] += 1
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
    token_ids = torch.randint(0, M, (rows,), device=dev, dtype=torch.int32)
    row_scale = torch.rand(rows, device=dev, dtype=torch.float32)
    return dict(A=A, B=B, SFA=SFA, SFB=SFB, offsets=offsets, token_ids=token_ids,
                row_scale=row_scale, sfa_sem=sfa_sem, sfb_sem=sfb_sem)


def main():
    E, K, N, M, rows, bpe = 8, 256, 256, 64, 1024, 1
    dev = "cuda"
    d = build_test(E, K, N, M, rows, bpe)
    y = torch.zeros(M, K, device=dev, dtype=torch.float32)
    kern = w2_scatter(E, K, N, M, rows, bpe)
    kern(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], d["token_ids"], d["row_scale"], y)
    torch.cuda.synchronize()

    K16 = N // 16
    Ad = decode_fp4(d["A"], rows, N) * decode_scale_words(pack_scale_words(d["sfa_sem"]), N).repeat_interleave(16, 1)
    Bd = decode_fp4(d["B"].reshape(E * K, N // 2), E * K, N).reshape(E, K, N) * \
        decode_scale_words(pack_scale_words(d["sfb_sem"]), N).reshape(E, K, K16).repeat_interleave(16, 2)
    y_ref = torch.zeros(M, K, device=dev, dtype=torch.float32)
    for e in range(E):
        s0, s1 = int(d["offsets"][e]), int(d["offsets"][e + 1])
        if s1 <= s0:
            continue
        o = Ad[s0:s1] @ Bd[e].t()
        y_ref.index_add_(0, d["token_ids"][s0:s1].long(), o * d["row_scale"][s0:s1].unsqueeze(1))
    err = (y - y_ref).abs()
    rel = err.max().item() / (y_ref.abs().max().item() + 1e-6)
    print(f"w2+scatter verify: max_abs={err.max().item():.5f} ref_absmax={y_ref.abs().max().item():.3f} max_rel={rel:.5f}")
    assert rel < 2e-2, "mismatch"

    def run():
        y.zero_()
        kern(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], d["token_ids"], d["row_scale"], y)
    ms = do_bench(run)
    print(f"w2+scatter: {ms:.4f} ms")


if __name__ == "__main__":
    main()
