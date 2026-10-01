from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# SM120/SM121 bf16 GEMV (see csrc/gemm/sm120_bf16_gemv.cuh).
#
# Consumer/workstation Blackwell (sm120) and GB10 Grace-Blackwell (sm121) have no
# cutedsl bf16 GEMM path, so decode linears fall through to cuBLAS. For M in [1, 8]
# (single-token decode, and the verify rows of speculative decoding) cuBLAS leaves
# DRAM bandwidth on the table for the wide lm_head and the tiny-N projections.
# This is a warp-per-few-rows weight-streaming GEMV: each weight element is read
# exactly once and reused across the M activation rows, the activation tile is
# staged in shared memory when it fits, and the reduction is a register +
# warp-shuffle tree (no split-K fixup kernel).
#
# Mirrors sglang.kernels.ops.gemm.hopper_bf16_gemv (SM90), extended to M > 1.

_MAX_K = 32768  # static smem budget: M * K * 2 bytes <= 48KB
_MAX_N = 131072
_SMEM_LIMIT = 48 * 1024


def _config(m: int, n: int, k: int) -> tuple[int, int, int, bool]:
    """(rows_per_warp, k_unroll, num_warps, stage_smem)."""
    # Small N is latency-bound: skip smem staging and use 4 warps for more
    # blocks (staging overhead + low occupancy loses to cuBLAS otherwise).
    if n < 1024:
        return (1, 2, 4, False)
    stage = m * k * 2 <= _SMEM_LIMIT
    if m == 1:
        return (2 if n >= 8192 else 1, 2, 8, stage)
    return (1, 2, 8, stage)


@cache_once
def _jit_sm120_bf16_gemv_module(
    n: int, k: int, m: int, k_rows: int, k_unroll: int, k_warps: int, stage: bool
) -> Module:
    args = make_cpp_args(n, k, m, k_rows, k_unroll, k_warps, stage)
    return load_jit(
        "sm120_bf16_gemv",
        *args,
        cuda_files=["gemm/sm120_bf16_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::Sm120Bf16GemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def use_sm120_bf16_gemv(m: int, n: int, k: int) -> bool:
    if not (
        1 <= m <= 8
        and k % 512 == 0
        and 512 <= k <= _MAX_K
        and n % 8 == 0
        and 8 <= n <= _MAX_N
    ):
        return False
    # Latency-bound narrow-N bucket where cuBLAS still wins at high M (measured
    # 0.83x at M=8, N=128): keep the baseline there rather than regress.
    if m >= 6 and 64 <= n <= 192:
        return False
    return True


def sm120_bf16_gemv_out(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor) -> None:
    """Write out[M, N] = x[M, K] @ w[N, K]^T into a preallocated buffer."""
    m = x.numel() // x.shape[-1]
    n, k = w.shape[0], w.shape[1]
    rows, unroll, warps, stage = _config(m, n, k)
    module = _jit_sm120_bf16_gemv_module(n, k, m, rows, unroll, warps, stage)
    module.run(x.view(m, k), w, out)


def sm120_bf16_gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """out[..., N] = x[..., K] @ w[N, K]^T, all bf16, fp32 accumulation."""
    m = x.numel() // x.shape[-1]
    n = w.shape[0]
    out = torch.empty((*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    sm120_bf16_gemv_out(x, w, out.view(m, n))
    return out
