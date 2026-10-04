#!/usr/bin/env python3
"""K0/K2: gather + NVFP4 block-16 activation quantization (bf16 -> e2m1 + E4M3 scales).

Fuses the gather (X[token_ids]) with the quant.  One tile = [blk_m, 16] so each
row has exactly one E4M3 scale (NVFP4 block-16).  Scale is E4M3-rounded and the
quant uses the rounded value, so a dequant reference with the stored scale is
exact.
"""
import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def gather_quant(
    rows: int,
    S: int,
    Cols: int,
    E: int,
    cap: int,
    blk_m: int = 128,
    block_N: int = 128,
    threads: int = 128,
):
    assert Cols % block_N == 0 and block_N % 16 == 0
    G = block_N // 16
    in_dtype = T.bfloat16
    e4 = T.float8_e4m3
    e2 = T.float4_e2m1fn

    @T.prim_func
    def main(
        X: T.Tensor((S, Cols), in_dtype),
        src_row: T.Tensor((rows,), T.int32),
        padded_row: T.Tensor((rows,), T.int32),
        Aq: T.Tensor((rows, Cols), e2),
        SFA: T.Tensor((E * cap, Cols // 16), e4),
    ):
        with T.Kernel(T.ceildiv(rows, blk_m), Cols // block_N, threads=threads) as (bm, bk):
            x_sh = T.alloc_shared((blk_m, G, 16), in_dtype)
            x_loc = T.alloc_fragment((blk_m, G, 16), in_dtype)
            amax = T.alloc_fragment((blk_m, G), T.float32)
            s_loc = T.alloc_fragment((blk_m, G), T.float32)
            y_sh = T.alloc_shared((blk_m, block_N), e2)

            for i, g, tt in T.Parallel(blk_m, G, 16):
                x_sh[i, g, tt] = X[src_row[bm * blk_m + i], bk * block_N + g * 16 + tt]
            T.copy(x_sh, x_loc)
            T.reduce_absmax(x_loc, amax, dim=2)
            for i, g in T.Parallel(blk_m, G):
                amax[i, g] = T.max(amax[i, g], 1e-6)
                s_loc[i, g] = T.Cast("float32", T.Cast(e4, amax[i, g] / 6.0))
            for i, g, tt in T.Parallel(blk_m, G, 16):
                y_sh[i, g * 16 + tt] = T.clamp(x_loc[i, g, tt] / s_loc[i, g], -6.0, 6.0)
            for i, g in T.Parallel(blk_m, G):
                SFA[padded_row[bm * blk_m + i], bk * G + g] = T.Cast(e4, amax[i, g] / 6.0)
            T.copy(y_sh, Aq[bm * blk_m, bk * block_N])

    return main


def build(rows, M, K, E, cap, seed=0):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g) / 10
    src = torch.randint(0, M, (rows,), device=dev, dtype=torch.int32, generator=g)
    base = rows // E
    sizes = torch.full((E,), base, dtype=torch.int64); sizes[: rows - base * E] += 1
    padded = torch.empty(rows, dtype=torch.int32, device=dev)
    off = 0
    for e in range(E):
        n = int(sizes[e])
        padded[off:off + n] = e * cap + torch.arange(n, device=dev)
        off += n
    return X, src, padded


def main():
    M, K, E, rows = 4096, 4096, 144, 32768
    cap = 384
    dev = "cuda"
    X, src, padded = build(rows, M, K, E, cap)
    Aq = torch.empty(rows, K // 2, device=dev, dtype=torch.uint8)
    SFA = torch.empty(E * cap, K // 16, device=dev, dtype=torch.float8_e4m3fn)
    kern = gather_quant(rows, M, K, E, cap)
    kern(X, src, padded, Aq, SFA)
    torch.cuda.synchronize()

    # reference
    Xg = X[src.long()].float()
    xb = Xg.reshape(rows, K // 16, 16)
    amax = xb.abs().amax(-1).clamp_min(1e-6)
    s_e4 = (amax / 6.0).to(torch.float8_e4m3fn)
    s_used = s_e4.float().clamp_min(1e-12)
    q = (xb / s_used.unsqueeze(-1)).clamp(-6, 6)
    vals = torch.tensor([0.0, .5, 1., 1.5, 2., 3., 4., 6.], device=dev)
    sign = torch.sign(q); aq = q.abs()
    idx = (aq.unsqueeze(-1) - vals).abs().argmin(-1)
    qr = sign * vals[idx]
    sfa_ref = torch.zeros(E * cap, K // 16, device=dev, dtype=torch.float32)
    sfa_ref[padded.long()] = s_e4.float()
    # decode Aq
    u = Aq.view(torch.uint8)
    lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
    deq = torch.empty(rows, K, device=dev)
    deq[:, 0::2] = lut[(u & 0xF).long()]; deq[:, 1::2] = lut[((u >> 4) & 0xF).long()]
    deq = (deq.reshape(rows, K // 16, 16) * SFA.float().reshape(E * cap, K // 16)[padded.long()].unsqueeze(-1)).reshape(rows, K)
    rec = (qr * s_used.unsqueeze(-1)).reshape(rows, K)
    err = (deq - rec).abs().max().item()
    print(f"gather_quant self-consistency: max_abs={err:.5f} (fp4 rounding)")
    assert err < 0.6

    ms = do_bench(lambda: kern(X, src, padded, Aq, SFA))
    print(f"gather_quant: {ms:.4f} ms  ({rows*K*2/1e6:.0f} MB in, {rows*K/2/1e6:.0f} MB out)")


if __name__ == "__main__":
    main()


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def gather_quant_dyn(
    Cols: int,
    E: int,
    blk_m: int = 128,
    block_N: int = 128,
    threads: int = 128,
):
    """Shape-generic variant: rows / S / cap are symbolic (bound at call time
    from the tensor shapes). One compile serves every batch shape."""
    assert Cols % block_N == 0 and block_N % 16 == 0
    G = block_N // 16
    in_dtype = T.bfloat16
    e4 = T.float8_e4m3
    e2 = T.float4_e2m1fn
    rows = T.dynamic("rows")
    S = T.dynamic("S")
    cap = T.dynamic("cap")

    @T.prim_func
    def main(
        X: T.Tensor((S, Cols), in_dtype),
        src_row: T.Tensor((rows,), T.int32),
        padded_row: T.Tensor((rows,), T.int32),
        Aq: T.Tensor((rows, Cols), e2),
        SFA: T.Tensor((E * cap, Cols // 16), e4),
    ):
        with T.Kernel(T.ceildiv(rows, blk_m), Cols // block_N, threads=threads) as (bm, bk):
            x_sh = T.alloc_shared((blk_m, G, 16), in_dtype)
            x_loc = T.alloc_fragment((blk_m, G, 16), in_dtype)
            amax = T.alloc_fragment((blk_m, G), T.float32)
            s_loc = T.alloc_fragment((blk_m, G), T.float32)
            y_sh = T.alloc_shared((blk_m, block_N), e2)

            for i, g, tt in T.Parallel(blk_m, G, 16):
                x_sh[i, g, tt] = X[src_row[bm * blk_m + i], bk * block_N + g * 16 + tt]
            T.copy(x_sh, x_loc)
            T.reduce_absmax(x_loc, amax, dim=2)
            for i, g in T.Parallel(blk_m, G):
                amax[i, g] = T.max(amax[i, g], 1e-6)
                s_loc[i, g] = T.Cast("float32", T.Cast(e4, amax[i, g] / 6.0))
            for i, g, tt in T.Parallel(blk_m, G, 16):
                y_sh[i, g * 16 + tt] = T.clamp(x_loc[i, g, tt] / s_loc[i, g], -6.0, 6.0)
            for i, g in T.Parallel(blk_m, G):
                SFA[padded_row[bm * blk_m + i], bk * G + g] = T.Cast(e4, amax[i, g] / 6.0)
            T.copy(y_sh, Aq[bm * blk_m, bk * block_N])

    return main
