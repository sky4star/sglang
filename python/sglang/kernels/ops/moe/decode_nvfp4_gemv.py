from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# Decode tiny-M NVFP4 MoE GEMV (see decode_nvfp4_gemv.cuh).
#
# K1d: gemm1  N=2048 (=2*N_rank up|gate), K=4096 fp4 -> 2048 packed bytes
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


R_CAP = 32  # fused routing sort cap (production decode rows); R > R_CAP -> torch
_COMBINE_THREADS = 512


@cache_once
def _jit_routing_module(maxa: int, r_cap: int) -> Module:
    args = make_cpp_args(r_cap, maxa, 256)
    return load_jit(
        "decode_nvfp4_moe_routing",
        *args,
        cuda_files=[_CUH],
        cuda_wrappers=[("run", f"sglang::DecodeNvfp4MoeRouting<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_combine_module(k: int) -> Module:
    args = make_cpp_args(k, _COMBINE_THREADS)
    return load_jit(
        "decode_nvfp4_moe_combine",
        *args,
        cuda_files=[_CUH],
        cuda_wrappers=[("run", f"sglang::DecodeNvfp4MoeCombine<{args}>::run")],
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
            # routing bookkeeping outputs (fused path; R <= R_CAP)
            "order": torch.empty(R_CAP, dtype=torch.int32, device=dev),
            "inv": torch.empty(R_CAP, dtype=torch.int32, device=dev),
            "token_of": torch.empty(R_CAP, dtype=torch.int64, device=dev),
            "pad_row": torch.empty(R_CAP, dtype=torch.int64, device=dev),
            "mip": torch.empty(e + 1, dtype=torch.int32, device=dev),
            "active_ids": torch.full((MAXA,), -1, dtype=torch.int32, device=dev),
        }
        _POOLS[key] = buf
    return buf


def _routing_torch(topk_ids, topk, e, dev, buf):
    """Original eager torch routing bookkeeping (fallback; R > R_CAP or when
    SGLANG_DECODE_MOE_CHAIN=torch). Returns (order, pad_row, token_of, mip,
    active_ids, inv, r)."""
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
    n_active = min(sorted_ids.numel(), MAXA)
    active_ids[:n_active] = torch.where(is_new, sorted_ids, minus_one)[:n_active]
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=dev)
    return order, pad_row, token_of, mip, active_ids, inv, sorted_ids.numel()


def _routing_fused(topk_ids, topk, e, buf):
    """Single-CTA routing kernel (R <= R_CAP). Same outputs as _routing_torch."""
    r = topk_ids.shape[0] * topk
    flat = topk_ids.reshape(-1)
    if flat.dtype != torch.int32:
        flat = flat.to(torch.int32)
    order, inv = buf["order"], buf["inv"]
    token_of, pad_row = buf["token_of"], buf["pad_row"]
    mip, active_ids = buf["mip"], buf["active_ids"]
    _jit_routing_module(MAXA, R_CAP).run(flat, r, topk, e, order[:r], inv[:r],
                                         token_of[:r], pad_row[:r], mip,
                                         active_ids)
    return order, pad_row, token_of, mip, active_ids, inv, r


def _combine_torch(g2, topk_weights, order, pad_row, inv, t, topk, k, output, dev):
    """Original eager torch routed combine (fallback)."""
    w_sorted = topk_weights.reshape(-1).to(torch.float32)[order]
    contrib = g2[pad_row].float() * w_sorted.unsqueeze(1)
    out32 = contrib[inv].view(t, topk, k).sum(dim=1)
    output.copy_(out32.to(torch.bfloat16))


def _combine_fused(g2, topk_weights, pad_row, inv, r, topk, output):
    """Single-kernel routed combine (R <= R_CAP). Same math as _combine_torch
    (j-ascending fp32 accumulate, single bf16 cast)."""
    w = topk_weights.reshape(-1)
    if w.dtype != torch.float32:
        w = w.to(torch.float32)
    _jit_combine_module(g2.shape[1]).run(g2, inv[:r], pad_row[:r], w, output, topk)


def decode_nvfp4_moe_forward(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    quant_scales: list,
    w13: torch.Tensor,
    w2: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float | None = None,
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
    swiglu_limit GPT-OSS-style activation clamp: gate and up are each clamped
                 to +-limit (fp32) before silu*mul, matching the cutlass
                 swigluoai epilogue (alpha=1/beta=0 neutral case only).

    Chain mode (SGLANG_DECODE_MOE_CHAIN, default "auto"): "auto" uses the fused
    single-pass routing + combine kernels when R = T*topk <= R_CAP (32), else
    the eager torch path; "torch" forces the torch path everywhere.
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

    # routing bookkeeping (all on-device, deterministic; fused = capture-safe)
    chain_mode = os.environ.get("SGLANG_DECODE_MOE_CHAIN", "auto")
    if chain_mode == "torch" or t * topk > R_CAP:
        order, pad_row, token_of, mip, active_ids, inv, r = _routing_torch(
            topk_ids, topk, e, dev, buf)
        fused = False
    else:
        order, pad_row, token_of, mip, active_ids, inv, r = _routing_fused(
            topk_ids, topk, e, buf)
        fused = True
    pad_v, tok_v = pad_row[:r], token_of[:r]

    # gather + quantize gemm1 activations
    x_exp = buf["x_exp"]
    x_exp[pad_v] = x[tok_v]
    a_q, a_sf = fp4_quantize(x_exp, global_scale=q0, sf_vec_size=16,
                             is_sf_swizzled_layout=True)

    # gemm1 (K1d): [KCAP, 2*Nr] = a_q @ w13^T, per-row expert alpha epilogue
    g1 = buf["g1"]
    decode_nvfp4_gemm_out(a_q, a_sf, w13, w13_sf.view(torch.uint8), mip,
                          active_ids, g1_alphas, buf["partials1"], g1)

    # swiglu in fp32 math (matches the cutlass doActivation contract; the
    # GPT-OSS-style clamp hits gate and up EACH, before silu*mul).
    # w13 row layout is the serving one: [up; gate] (gate = SECOND half) —
    # the cutlass gated-swiglu epilogue silu's the second half, and sglang
    # loads w13 up-proj-first (load_up_proj_weight_first) for the cutlass
    # backend. 2026-10-14: the [gate; up] assumption here was the serving
    # quality-collapse root cause (silu(up)*gate every block).
    nr = n1 // 2
    g1_up = g1[:, :nr].float()
    g1_gate = g1[:, nr:].float()
    if swiglu_limit is not None:
        lim = float(swiglu_limit)
        g1_gate = g1_gate.clamp(-lim, lim)
        g1_up = g1_up.clamp(-lim, lim)
    y = torch.nn.functional.silu(g1_gate) * g1_up
    a2_q, a2_sf = fp4_quantize(y.to(torch.bfloat16), global_scale=q3,
                               sf_vec_size=16, is_sf_swizzled_layout=True)

    # gemm2 (K3d): [KCAP, K]
    g2 = buf["g2"]
    decode_nvfp4_gemm_out(a2_q, a2_sf, w2, w2_sf.view(torch.uint8), mip,
                          active_ids, g2_alphas, buf["partials2"], g2)

    # routed combine (fp32 accumulate, single bf16 cast; deterministic: fixed
    # j-order sum, no atomics — index_add_ is an atomic fp32 scatter on CUDA).
    if fused:
        _combine_fused(g2, topk_weights, pad_row, inv, r, topk, output)
    else:
        _combine_torch(g2, topk_weights, order, pad_row, inv, t, topk, k, output, dev)
    return output
