#!/usr/bin/env python3
"""MoE combine (finalize): gather-reduction over the topk expert rows.

    y[t, :] = sum_s out2[slot_rows[t, s], :] * slot_scale[t, s]

No atomics: each token's ``topk`` rows are gathered and summed, writes to ``y``
are contiguous.  ``slot_rows`` / ``slot_scale`` are the per-(token, slot) row
index and (dequant * routing) scale, precomputed on the host.
"""
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def moe_combine(
    M: int,
    rows: int,
    K: int,
    topk: int,
    block_M: int = 64,
    block_K: int = 256,
    threads: int = 128,
    out_dtype=T.float32,
):
    @T.prim_func
    def main(
        out2: T.Tensor((rows, K), T.bfloat16),
        slot_rows: T.Tensor((M, topk), T.int32),
        slot_scale: T.Tensor((M, topk), T.float32),
        y: T.Tensor((M, K), out_dtype),
    ):
        with T.Kernel(T.ceildiv(K, block_K), T.ceildiv(M, block_M), threads=threads) as (bk, bm):
            acc = T.alloc_fragment((block_M, block_K), T.float32)
            T.clear(acc)
            for s in T.serial(topk):
                for i, j in T.Parallel(block_M, block_K):
                    t = bm * block_M + i
                    acc[i, j] += T.Cast("float32", out2[slot_rows[t, s], bk * block_K + j]) * slot_scale[t, s]
            for i, j in T.Parallel(block_M, block_K):
                if bm * block_M + i < M:
                    y[bm * block_M + i, bk * block_K + j] = T.Cast(out_dtype, acc[i, j])

    return main


def main():
    M, rows, K, topk = 4096, 32768, 4096, 8
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    out2 = (torch.randn(rows, K, device=dev, dtype=torch.float32, generator=g) / 100).to(torch.bfloat16)
    slot_rows = torch.stack([torch.randperm(rows, device=dev, generator=g)[:M] for _ in range(topk)], dim=1).to(torch.int32)
    slot_scale = torch.rand(M, topk, device=dev, dtype=torch.float32, generator=g)
    y = torch.empty(M, K, device=dev, dtype=torch.float32)
    kern = moe_combine(M, rows, K, topk)
    kern(out2, slot_rows, slot_scale, y)
    torch.cuda.synchronize()

    y_ref = torch.zeros(M, K, device=dev, dtype=torch.float32)
    for s in range(topk):
        y_ref += out2[slot_rows[:, s].long()].float() * slot_scale[:, s].unsqueeze(1)
    err = (y - y_ref).abs().max().item()
    print(f"combine verify: max_abs={err:.6f}")
    assert err < 1e-3

    ms = do_bench(lambda: kern(out2, slot_rows, slot_scale, y))
    print(f"moe_combine: {ms:.4f} ms  (read {rows*K*2/1e6:.0f} MB, write {M*K*4/1e6:.0f} MB)")


if __name__ == "__main__":
    main()


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def moe_combine_dyn(
    K: int,
    topk: int,
    block_M: int = 64,
    block_K: int = 256,
    threads: int = 128,
    out_dtype=T.float32,
):
    """Shape-generic variant: M / rows symbolic (bound at call time)."""
    M = T.dynamic("M")
    rows = T.dynamic("rows")

    @T.prim_func
    def main(
        out2: T.Tensor((rows, K), T.bfloat16),
        slot_rows: T.Tensor((M, topk), T.int32),
        slot_scale: T.Tensor((M, topk), T.float32),
        y: T.Tensor((M, K), out_dtype),
    ):
        with T.Kernel(T.ceildiv(K, block_K), T.ceildiv(M, block_M), threads=threads) as (bk, bm):
            acc = T.alloc_fragment((block_M, block_K), T.float32)
            T.clear(acc)
            for s in T.serial(topk):
                for i, j in T.Parallel(block_M, block_K):
                    t = bm * block_M + i
                    acc[i, j] += T.Cast("float32", out2[slot_rows[t, s], bk * block_K + j]) * slot_scale[t, s]
            for i, j in T.Parallel(block_M, block_K):
                if bm * block_M + i < M:
                    y[bm * block_M + i, bk * block_K + j] = T.Cast(out_dtype, acc[i, j])

    return main
