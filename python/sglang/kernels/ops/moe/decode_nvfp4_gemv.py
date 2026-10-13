from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Decode tiny-M NVFP4 MoE GEMV (see decode_nvfp4_gemv.cuh).
#
# K1d: gemm1  N=2048 (=2*N_rank gate|up), K=4096 fp4 -> 2048 packed bytes
# K3d: gemm2  N=4096 (=hidden_rank),      K=1024 fp4 ->  512 packed bytes
# Deterministic split-K: SLICES partial planes + fixed-order fp32 reduce.
# Contract (bitmodel-verified): fp32 accumulate SA*SB*(sum_j a4*b4) over all
# K, multiply per-row expert alpha once at the end, single bf16 cast.
#
# v7: packed u8 activation staging (chunked, SEG=4) + weights decoded once
# per chunk into a j-major smem plane (wf out of registers) -> ~60 regs,
# 4 CTA/SM target. (v2/v3 pre-decode-to-f32 experiments rejected: L2/L1TEX
# walls on GB10, see .cuh header.)

_CUH = os.path.join(os.path.dirname(__file__), "decode_nvfp4_gemv.cuh")

KCAP = 128  # padded activation row cap: 4 tokens x topk8, pad-to-4 per group
GRP_CAP = 32  # max rows per expert segment (worst case: all 32 slots on one expert)
N_TILE = 16  # output rows per CTA (2 per warp x 8 warps)


MAXA = 32  # compact active-expert list cap (<= T*topk <= KCAP)


@cache_once
def _jit_split_module(n: int, k_bytes: int, slices: int) -> Module:
    args = make_cpp_args(n, k_bytes, KCAP, GRP_CAP, N_TILE, 8, slices, MAXA)
    return load_jit(
        "decode_nvfp4_gemv_split",
        *args,
        cuda_files=[_CUH],
        cuda_wrappers=[("run", f"sglang::DecodeNvfp4GemvSplitKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_reduce_module(n: int, slices: int) -> Module:
    args = make_cpp_args(n, KCAP, slices)
    return load_jit(
        "decode_nvfp4_gemv_reduce",
        *args,
        cuda_files=[_CUH],
        cuda_wrappers=[("run", f"sglang::DecodeNvfp4GemvReduceKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def decode_nvfp4_gemm_out(
    a_q: torch.Tensor,
    a_sf: torch.Tensor,
    w_q: torch.Tensor,
    w_sf: torch.Tensor,
    m_indptr: torch.Tensor,
    active_ids: torch.Tensor,
    alphas: torch.Tensor,
    partials: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """out[KCAP, N] = alpha-epilogued fp4xfp4 GEMM with deterministic split-K.

    a_q        [KCAP, K_BYTES] u8   (rows >= Mp must stay zero)
    a_sf       [128, K_BYTES/8] u8  swizzled serving layout
    w_q        [E, N, K_BYTES] u8   checkpoint-packed fp4 weights
    w_sf       [E, N, K_BYTES/8] u8 native swizzled weight scales (e4m3 bits)
    m_indptr   [E+1] i32            full-E padded segment start offsets
    active_ids [MAXA] i32           compact active experts (-1 sentinels ok)
    alphas     [E] f32              per-expert epilogue scale
    partials   [SLICES, KCAP, N] f32 pooled scratch (zero-initialized once)
    out        [KCAP, N] bf16
    """
    e, n, k_bytes = w_q.shape
    slices = partials.shape[0]
    _jit_split_module(n, k_bytes, slices).run(a_q, a_sf, w_q, w_sf, m_indptr,
                                              active_ids, partials)
    _jit_reduce_module(n, slices).run(partials, m_indptr, alphas, out)


# ---- full decode MoE chain (HANDOFF §9.0) ----------------------------------

_POOLS: dict = {}


def _pool(dev: torch.device, e: int, n1: int, kb1: int, n2: int, kb2: int, k: int) -> dict:
    """Pooled per-shape scratch. Allocated eagerly once (pre-graph-capture so
    steady-state capture alloc_delta == 0); zeroed rows >= Mp stay zero forever
    because only rows < Mp are ever written."""
    key = (dev.index, e, n1, kb1, n2, kb2, k)
    buf = _POOLS.get(key)
    if buf is None:
        buf = {
            "x_exp": torch.zeros(KCAP, k, dtype=torch.bfloat16, device=dev),
            "partials1": torch.zeros(4, KCAP, n1, dtype=torch.float32, device=dev),
            "partials2": torch.zeros(2, KCAP, n2, dtype=torch.float32, device=dev),
            "g1": torch.empty(KCAP, n1, dtype=torch.bfloat16, device=dev),
            "g2": torch.empty(KCAP, n2, dtype=torch.bfloat16, device=dev),
        }
        _POOLS[key] = buf
    return buf


def decode_nvfp4_moe_forward(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    quant_scales: list,
    w13: torch.Tensor,
    w2: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Drop-in decode chain: gather -> fp4 quant -> K1d -> swiglu -> fp4 quant
    -> K3d -> routed combine. All GPU work; no host sync (CUDA-graph safe).

    x            [T, K] bf16           (T <= 32; caller gates)
    topk_ids     [T, topk] i32/i64
    topk_weights [T, topk] f32
    quant_scales [q0, w13_sf, g1_alphas, q3, w2_sf, g2_alphas]  (serving order)
    w13          [E, 2*Nr, K/2]  u8 packed
    w2           [E, K, Nr/2]    u8 packed
    output       [T, K] bf16 (written in place)
    """
    from flashinfer import fp4_quantize

    t, k = x.shape
    topk = topk_ids.shape[1]
    e, n1, kb1 = w13.shape
    n2, kb2 = w2.shape[1], w2.shape[2]
    assert t * topk <= KCAP, f"decode rows {t * topk} exceed KCAP {KCAP}"
    dev = x.device
    buf = _pool(dev, e, n1, kb1, n2, kb2, k)

    q0, w13_sf, g1_alphas, q3, w2_sf, g2_alphas = quant_scales

    # routing bookkeeping (all on-device, deterministic)
    flat = topk_ids.reshape(-1).to(torch.int32)
    order = torch.argsort(flat, stable=True)
    token_of = torch.div(order, topk, rounding_mode="floor")
    counts = torch.bincount(flat, minlength=e)
    padded = (counts + 3) // 4 * 4
    mip = torch.zeros(e + 1, dtype=torch.int32, device=dev)
    mip[1:] = torch.cumsum(padded, 0).to(torch.int32)
    sorted_ids = flat[order]
    idx = torch.arange(sorted_ids.numel(), device=dev)
    is_new = torch.ones_like(idx, dtype=torch.bool)
    is_new[1:] = sorted_ids[1:] != sorted_ids[:-1]
    group_start = torch.cummax(torch.where(is_new, idx, torch.zeros_like(idx)), 0).values
    rank = idx - group_start
    pad_row = mip[sorted_ids.long()].long() + rank
    minus_one = torch.full_like(flat, -1)
    active_ids = torch.full((MAXA,), -1, dtype=torch.int32, device=dev)
    active_ids[:sorted_ids.numel()] = torch.where(is_new, sorted_ids, minus_one)

    # gather + quantize gemm1 activations
    x_exp = buf["x_exp"]
    x_exp[pad_row] = x[token_of]
    a_q, a_sf = fp4_quantize(x_exp, global_scale=q0, sf_vec_size=16,
                             is_sf_swizzled_layout=True)

    # gemm1 (K1d): [KCAP, 2*Nr] = a_q @ w13^T, per-row expert alpha epilogue
    g1 = buf["g1"]
    decode_nvfp4_gemm_out(a_q, a_sf, w13, w13_sf.view(torch.uint8), mip,
                          active_ids, g1_alphas, buf["partials1"], g1)

    # swiglu in fp32 math (matches the cutlass doActivation contract)
    nr = n1 // 2
    y = torch.nn.functional.silu(g1[:, :nr].float()) * g1[:, nr:].float()
    a2_q, a2_sf = fp4_quantize(y.to(torch.bfloat16), global_scale=q3,
                               sf_vec_size=16, is_sf_swizzled_layout=True)

    # gemm2 (K3d): [KCAP, K]
    g2 = buf["g2"]
    decode_nvfp4_gemm_out(a2_q, a2_sf, w2, w2_sf.view(torch.uint8), mip,
                          active_ids, g2_alphas, buf["partials2"], g2)

    # routed combine (fp32 accumulate, single bf16 cast). Deterministic:
    # unsort the expert-major contributions back to token-slot order and
    # reduce with a fixed-order torch sum (index_add_ is atomic fp32 on CUDA
    # and its summation order is run-to-run nondeterministic).
    w_sorted = topk_weights.reshape(-1).to(torch.float32)[order]
    contrib = g2[pad_row].float() * w_sorted.unsqueeze(1)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=dev)
    out32 = contrib[inv].view(t, topk, k).sum(dim=1)
    output.copy_(out32.to(torch.bfloat16))
    return output
