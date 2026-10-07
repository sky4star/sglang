#!/usr/bin/env python3
"""TileLang grouped NVFP4 block-scaled GEMM for GLM-5.3-Flash prefill MoE.

SM120a `mma.sync...kind::mxf4nvf4.block_scale...ue4m3` (E2M1 weights/acts,
E4M3 block-16 scales) on GB10 (sm_121a).  Non-persistent tile grid
(`blocks_per_expert * E` x `N/block_N`), unlike the 48-CTA persistent
FlashInfer baseline.

This first milestone is the grouped GEMM only (w13 / w2 shapes):
    for each expert e:  C[rows_e] = dequant(A[rows_e]) @ dequant(B[e])^T

Run (tilelang 0.1.15 shadow install + SM121A guard patch):
    PYTHONPATH=/tmp/tl15site python3 grouped_nvfp4_gemm.py --E 144 --N 4096 --K 4096 --rows 32768
"""

import argparse
import math

import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

FP4_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


# --------------------------------------------------------------------------
# host scale layout helpers (mirrors upstream sm120 example)
# --------------------------------------------------------------------------
def pack_scale_words(scale_bytes_int: torch.Tensor) -> torch.Tensor:
    """(rows, K/16) uint8 E4M3 -> (rows, K/64) uint32, 4 bytes per word (int32 path)."""
    b = scale_bytes_int.to(torch.uint8).reshape(scale_bytes_int.shape[0], -1, 4).to(torch.int32)
    w = b[:, :, 0] | (b[:, :, 1] << 8) | (b[:, :, 2] << 16) | (b[:, :, 3] << 24)
    return w.contiguous().view(torch.uint32)


def swizzle_chunk_kmajor(words: torch.Tensor, block_rows: int = 128) -> torch.Tensor:
    """(rows, words_per_k64) semantic -> SM120 BlockScaledBasicChunk K-major."""
    rows, cols = words.shape
    if rows % block_rows != 0:
        padded = torch.zeros((math.ceil(rows / block_rows) * block_rows, cols),
                             dtype=words.dtype, device=words.device)
        padded[:rows] = words
        words = padded
        rows = padded.shape[0]
    rb = rows // block_rows
    src = words.contiguous().reshape(rb, 4, 32, cols)
    return src.permute(0, 3, 2, 1).contiguous().reshape(rows, cols)


def decode_fp4(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    u = packed.contiguous().view(torch.uint8)
    lut = torch.tensor(FP4_LUT, device=packed.device, dtype=torch.float32)
    out = torch.empty((rows, cols), device=packed.device, dtype=torch.float32)
    out[:, 0::2] = lut[(u & 0x0F).long()]
    out[:, 1::2] = lut[((u >> 4) & 0x0F).long()]
    return out


def decode_scale_words(words: torch.Tensor, k: int) -> torch.Tensor:
    """(rows, K/64) uint32 -> (rows, K/16) float (E4M3 byte -> float approx via ue4m3 decode)."""
    w = words.to(torch.int64)
    b = torch.empty((words.shape[0], k // 16), device=words.device, dtype=torch.int64)
    b[:, 0::4] = w & 0xFF
    b[:, 1::4] = (w >> 8) & 0xFF
    b[:, 2::4] = (w >> 16) & 0xFF
    b[:, 3::4] = (w >> 24) & 0xFF
    # E4M3 decode
    sign = torch.where((b >> 7) & 1 == 1, -1.0, 1.0)
    exp = ((b >> 3) & 0x0F).float()
    man = (b & 0x07).float() / 8.0
    val = torch.where(exp == 0, man * (2.0 ** -6), (1.0 + man) * torch.pow(2.0, exp - 7))
    return sign * val


# --------------------------------------------------------------------------
# kernel
# --------------------------------------------------------------------------
@tilelang.jit
def grouped_nvfp4_gemm(
    E: int,
    N: int,
    K: int,
    rows: int,
    blocks_per_expert: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 256,
    num_stages: int = 2,
    threads: int = 128,
    out_dtype=T.bfloat16,
):
    assert block_M == 128 and block_N == 128
    assert block_K % 64 == 0 and K % 256 == 0 and N % block_N == 0
    k_blocks = K // block_K
    words = block_K // 64
    n_blocks = N // block_N
    in_dtype = T.float4_e2m1fn
    total_m_blocks = E * blocks_per_expert

    @T.prim_func
    def main(
        A: T.Tensor((rows, K), in_dtype),
        B: T.Tensor((E, N, K), in_dtype),
        SFA: T.Tensor((total_m_blocks * block_M * k_blocks, words), T.uint32),
        SFB: T.Tensor((E * n_blocks * k_blocks * block_N, words), T.uint32),
        expert_offsets: T.Tensor((E + 1,), T.int32),
        C: T.Tensor((rows, N), out_dtype),
    ):
        with T.Kernel(n_blocks, total_m_blocks, threads=threads) as (bn, bm):
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_s = T.alloc_shared((block_M, words), T.uint32)
            SFB_s = T.alloc_shared((block_N, words), T.uint32)
            C_l = T.alloc_fragment((block_M, block_N), T.float32)

            e = bm // blocks_per_expert
            local = bm % blocks_per_expert
            m_start = expert_offsets[e] + local * block_M
            actual = T.max(0, T.min(block_M, expert_offsets[e + 1] - m_start))

            T.clear(C_l)
            for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                T.copy(A[m_start:m_start + block_M, ko * block_K:(ko + 1) * block_K], A_s)
                T.copy(B[e, bn * block_N:(bn + 1) * block_N, ko * block_K:(ko + 1) * block_K], B_s)
                for r, w in T.Parallel(block_M, words):
                    SFA_s[r, w] = SFA[(bm * k_blocks + ko) * block_M + r, w]
                for r, w in T.Parallel(block_N, words):
                    SFB_s[r, w] = SFB[((e * n_blocks + bn) * k_blocks + ko) * block_N + r, w]
                T.mma_gemm_blockscaled(
                    A_s, B_s, C_l, SFA_s, SFB_s,
                    transpose_B=True, clear_accum=False,
                    k_start=ko * block_K,
                    sf_a_granularity_k=16, sf_b_granularity_k=16,
                    sf_layout="blockscaled_chunk_kmajor",
                )
            for i, j in T.Parallel(block_M, block_N):
                if i < actual:
                    C[m_start + i, bn * block_N + j] = C_l[i, j]

    return main


@tilelang.jit
def grouped_nvfp4_gemm_reuse(
    E: int,
    N: int,
    K: int,
    rows: int,
    blocks_per_expert: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 256,
    num_stages: int = 2,
    threads: int = 128,
    out_dtype=T.bfloat16,
):
    """Variant where one CTA owns (expert, n-block) and loops over all local
    m-blocks, so the expert weight tile stays L2-resident -> 1x DRAM weight
    traffic (the plain grid reloads B per m-block CTA)."""
    assert block_M == 128 and block_N == 128
    assert block_K % 64 == 0 and K % 256 == 0 and N % block_N == 0
    k_blocks = K // block_K
    words = block_K // 64
    n_blocks = N // block_N
    in_dtype = T.float4_e2m1fn

    @T.prim_func
    def main(
        A: T.Tensor((rows, K), in_dtype),
        B: T.Tensor((E, N, K), in_dtype),
        SFA: T.Tensor((E * blocks_per_expert * block_M * k_blocks, words), T.uint32),
        SFB: T.Tensor((E * n_blocks * k_blocks * block_N, words), T.uint32),
        expert_offsets: T.Tensor((E + 1,), T.int32),
        C: T.Tensor((rows, N), out_dtype),
    ):
        with T.Kernel(n_blocks, E, threads=threads) as (bn, e):
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_s = T.alloc_shared((block_M, words), T.uint32)
            SFB_s = T.alloc_shared((block_N, words), T.uint32)
            C_l = T.alloc_fragment((block_M, block_N), T.float32)
            for lm in T.serial(blocks_per_expert):
                m_start = expert_offsets[e] + lm * block_M
                actual = T.max(0, T.min(block_M, expert_offsets[e + 1] - m_start))
                T.clear(C_l)
                for ko in T.Pipelined(k_blocks, num_stages=num_stages):
                    T.copy(A[m_start:m_start + block_M, ko * block_K:(ko + 1) * block_K], A_s)
                    T.copy(B[e, bn * block_N:(bn + 1) * block_N, ko * block_K:(ko + 1) * block_K], B_s)
                    for r, w in T.Parallel(block_M, words):
                        SFA_s[r, w] = SFA[((e * blocks_per_expert + lm) * k_blocks + ko) * block_M + r, w]
                    for r, w in T.Parallel(block_N, words):
                        SFB_s[r, w] = SFB[((e * n_blocks + bn) * k_blocks + ko) * block_N + r, w]
                    T.mma_gemm_blockscaled(
                        A_s, B_s, C_l, SFA_s, SFB_s,
                        transpose_B=True, clear_accum=False,
                        k_start=ko * block_K,
                        sf_a_granularity_k=16, sf_b_granularity_k=16,
                        sf_layout="blockscaled_chunk_kmajor",
                    )
                for i, j in T.Parallel(block_M, block_N):
                    if i < actual:
                        C[m_start + i, bn * block_N + j] = C_l[i, j]

    return main
# --------------------------------------------------------------------------
def build_grouped(E, N, K, rows, blocks_per_expert, block_K=256, seed=0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    # per-expert row counts (roughly uniform; <= blocks_per_expert*128)
    cap = blocks_per_expert * 128
    base = rows // E
    sizes = torch.full((E,), base, dtype=torch.int64)
    sizes[: rows - base * E] += 1
    assert sizes.max() <= cap, f"sizes max {sizes.max()} > cap {cap}"
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(sizes, 0).to(torch.int32)

    A_packed = torch.randint(-128, 128, (rows, K // 2), device=dev, dtype=torch.int8, generator=g)
    B_packed = torch.randint(-128, 128, (E, N, K // 2), device=dev, dtype=torch.int8, generator=g)

    # semantic E4M3 scales (use small positive exponents to keep values sane)
    def scale_bytes(nrows):
        # random E4M3 bytes with exponent ~ 7 (2^0) so scale ~ [0.5,1.875]
        e4m3 = (torch.randint(0, 7, (nrows, K // 16), device=dev, dtype=torch.int64, generator=g) | (7 << 3))
        return e4m3

    # A semantic scales padded to blocks_per_expert*128 per expert
    sfa_sem = torch.zeros((E * blocks_per_expert * 128, K // 16), device=dev, dtype=torch.int64)
    for e in range(E):
        s0 = int(offsets[e])
        s1 = int(offsets[e + 1])
        sfa_sem[e * cap: e * cap + (s1 - s0)] = scale_bytes(s1 - s0)
    sfb_sem = scale_bytes(E * N).reshape(E, N, K // 16)

    # reshape to (total_m_blocks*128*k_blocks, words); words = block_K // 64
    words = block_K // 64
    SFA = swizzle_chunk_kmajor(pack_scale_words(sfa_sem)).reshape(-1, words)
    SFB = swizzle_chunk_kmajor(pack_scale_words(sfb_sem.reshape(E * N, -1))).reshape(-1, words)

    return dict(A=A_packed, B=B_packed, SFA=SFA, SFB=SFB, offsets=offsets,
                sizes=sizes, sfa_sem=sfa_sem, sfb_sem=sfb_sem)


def reference(d, E, N, K, offsets, blocks_per_expert):
    dev = "cuda"
    cap = blocks_per_expert * 128
    A_full = decode_fp4(d["A"], d["A"].shape[0], K)
    A_sf = decode_scale_words(pack_scale_words(d["sfa_sem"]), K)  # padded per expert
    C = torch.zeros((d["A"].shape[0], N), device=dev, dtype=torch.float32)
    kb = K // 16
    for e in range(E):
        s0 = int(offsets[e]); s1 = int(offsets[e + 1])
        if s1 <= s0:
            continue
        b = decode_fp4(d["B"][e].reshape(N, K // 2), N, K)
        bsf = decode_scale_words(pack_scale_words(d["sfb_sem"][e]), K)
        a_s = A_full[s0:s1].reshape(-1, kb, 16) * A_sf[e * cap:e * cap + (s1 - s0)].unsqueeze(-1)
        b_s = b.reshape(N, kb, 16) * bsf.unsqueeze(-1)
        C[s0:s1] = torch.einsum("mbk,nbk->mn", a_s, b_s)
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=144)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--rows", type=int, default=32768)
    ap.add_argument("--blocks-per-expert", type=int, default=2)
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--block-k", type=int, default=256)
    ap.add_argument("--stages", type=int, default=2)
    ap.add_argument("--threads", type=int, default=128)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    print(f"cap {torch.cuda.get_device_capability(0)} E={args.E} N={args.N} K={args.K} rows={args.rows} "
          f"bpe={args.blocks_per_expert} tile=128x{args.block_n}x{args.block_k}")
    d = build_grouped(args.E, args.N, args.K, args.rows, args.blocks_per_expert, block_K=args.block_k)
    C = torch.empty((args.rows, args.N), device="cuda", dtype=torch.bfloat16)
    kern_fn = grouped_nvfp4_gemm_reuse if args.reuse else grouped_nvfp4_gemm
    kern = kern_fn(args.E, args.N, args.K, args.rows, args.blocks_per_expert,
                   block_N=args.block_n, block_K=args.block_k, num_stages=args.stages,
                   threads=args.threads)
    kern(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], C)
    torch.cuda.synchronize()

    if not args.no_verify:
        ref = reference(d, args.E, args.N, args.K, d["offsets"], args.blocks_per_expert)
        err = (C.float() - ref).abs()
        rel = err.max().item() / (ref.abs().max().item() + 1e-6)
        mean_rel = (err / (ref.abs() + 1e-3)).mean().item()
        print(f"max_abs={err.max().item():.4f} ref_absmax={ref.abs().max().item():.3f} "
              f"max_rel={rel:.5f} mean_rel={mean_rel:.6f}")
        assert rel < 1e-2, f"layout likely wrong: max_rel={rel}"

    ms = do_bench(lambda: kern(d["A"], d["B"], d["SFA"], d["SFB"], d["offsets"], C))
    flops = 2.0 * args.rows * args.N * args.K
    print(f"TileLang grouped: {ms:.4f} ms  {flops/(ms*1e-3)/1e12:.1f} TFLOPS")


if __name__ == "__main__":
    main()
