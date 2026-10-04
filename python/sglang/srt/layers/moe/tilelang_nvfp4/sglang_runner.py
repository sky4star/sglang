#!/usr/bin/env python3
"""Wire the TileLang NVFP4 fused-MoE pipeline into sglang's MoE runner.

SGLANG_TILELANG_MOE=1 enables it for LARGE batches (prefill chunks):
  x.shape[0] >= SGLANG_TILELANG_MOE_MIN_M (default 512).
Decode batches keep the production flashinfer cutlass path (decode MoE is
already at the DRAM roof and its dynamic shapes are CUDA-graph captured).

Conversions from sglang post-load tensors:
  w13_blockscale_swizzled (cutlass 128x4 swizzle) -> semantic -> kernel-order
  chunk_kmajor SFB tiles; scale_e = g_alphas * input_scale_quant = weight_scale_2
  (activations are quantized absolutely by K0, a_gs = 1).

Topk ids are global; rows hitting non-local experts are excluded from the
gathered layout and their combine slots get scale 0 (TP expert-distributed
partial output, summed later by the TP all-reduce). Gathered rows are padded
to SGLANG_TILELANG_MOE_BUCKET (512) so TileLang compiles once per bucket.
"""
from __future__ import annotations

import os
import torch

from .k1k3_emaj import w13_swiglu_emaj, w2_emaj
from .moe_combine_kernel import moe_combine
from .act_quant_kernel import gather_quant
from .grouped_nvfp4_gemm import pack_scale_words, swizzle_chunk_kmajor

ENABLED = os.environ.get("SGLANG_TILELANG_MOE", "0") == "1"
MIN_M = int(os.environ.get("SGLANG_TILELANG_MOE_MIN_M", "512"))
BUCKET = int(os.environ.get("SGLANG_TILELANG_MOE_BUCKET", "512"))
BLOCK_K = 128
BN = 128

_states: dict[int, dict] = {}


def _unswizzle_blockscale(sw: torch.Tensor) -> torch.Tensor:
    B, Mp, Kp = sw.shape
    x = sw.reshape(B, Mp // 128, Kp // 4, 32, 4, 4).permute(0, 1, 4, 3, 2, 5)
    return x.reshape(B, Mp, Kp)


def _pack_kernel_order(sem_pad: torch.Tensor, tile_rows: int, words: int) -> torch.Tensor:
    """(tiles*tile_rows, K/16) uint8 semantic -> (tiles*k_chunks*tile_rows, words) uint32.

    Vectorized: single fused gather, no Python loop over chunks.
    """
    T_ = sem_pad.shape[0] // tile_rows
    K16 = sem_pad.shape[1]
    n_chk = (K16 * 16) // (words * 64)
    dev = sem_pad.device
    pk = pack_scale_words(sem_pad).view(torch.int32)  # (T_*tile_rows, n_chk*words)
    idx = torch.arange(tile_rows, device=dev)
    intra = (idx % 32) * (tile_rows // 32) + idx // 32
    kb = torch.arange(words, device=dev)
    flat = kb[None, :] * tile_rows + intra[:, None]          # (tile_rows, words)
    r = flat // words                                        # (tile_rows, words)
    w = flat % words
    t = torch.arange(T_, device=dev)[:, None, None, None]
    ko = torch.arange(n_chk, device=dev)[None, :, None, None]
    gl = ((t * n_chk + ko) * tile_rows + r[None, None])      # (T, n_chk, tile_rows, words)
    src_row = t * tile_rows + idx[None, None, :, None]
    src_col = ko * words + kb[None, None, None, :]
    out = torch.zeros(T_ * n_chk * tile_rows * words, dtype=torch.int32, device=dev)
    out[(gl * words + w).reshape(-1)] = pk.view(-1)[
        (src_row * (n_chk * words) + src_col).expand_as(gl).reshape(-1)]
    return out.view(T_ * n_chk * tile_rows, words).view(torch.uint32)


def _prepare_layer(quant_info) -> dict:
    E, twoN, Kh = quant_info.w13_weight.shape
    N = twoN // 2
    K = Kh * 2
    assert N % BN == 0 and K % BLOCK_K == 0
    words = BLOCK_K // 64

    w13_sf = quant_info.quant_scales[1]
    w2_sf = quant_info.quant_scales[4]
    sw13 = w13_sf.view(torch.uint8) if w13_sf.dtype == torch.float8_e4m3fn else w13_sf
    sw2 = w2_sf.view(torch.uint8) if w2_sf.dtype == torch.float8_e4m3fn else w2_sf
    if sw13.dim() == 2:
        sw13 = sw13.unsqueeze(0)
    if sw2.dim() == 2:
        sw2 = sw2.unsqueeze(0)
    sem13 = _unswizzle_blockscale(sw13)
    sem2 = _unswizzle_blockscale(sw2)
    K16 = K // 16
    sem13 = sem13[:, :twoN, :K16].contiguous()
    sem2 = sem2[:, :K, :N // 16].contiguous()

    sfbg = _pack_kernel_order(sem13[:, :N].reshape(E * N, K16), BN, words)
    sfbu = _pack_kernel_order(sem13[:, N:].reshape(E * N, K16), BN, words)
    sfb2 = _pack_kernel_order(sem2.reshape(E * K, N // 16), BN, words)

    scale1 = (quant_info.quant_scales[2] * quant_info.quant_scales[0]).float().reshape(E)
    scale2 = (quant_info.quant_scales[5] * quant_info.quant_scales[3]).float().reshape(E)

    return dict(E=E, N=N, K=K,
                Bg=quant_info.w13_weight[:, :N].contiguous(),
                Bu=quant_info.w13_weight[:, N:].contiguous(),
                W2=quant_info.w2_weight.contiguous(),
                sfbg=sfbg, sfbu=sfbu, sfb2=sfb2,
                scale1=scale1, scale2=scale2,
                ep_rank=int(quant_info.moe_ep_rank),
                ep_size=int(quant_info.moe_ep_size))


def should_run(x) -> bool:
    return ENABLED and x.shape[0] >= MIN_M


def run_tilelang_moe(*, x, topk_weights, topk_ids, quant_info, output):
    st = _states.get(id(quant_info.w13_weight))
    if st is None:
        st = _prepare_layer(quant_info)
        _states[id(quant_info.w13_weight)] = st

    E, N, K = st["E"], st["N"], st["K"]
    ep_rank = st["ep_rank"]
    M, topk = topk_ids.shape
    dev = x.device

    ids = topk_ids.to(torch.int64)
    local_mask = (ids >= ep_rank * E) & (ids < (ep_rank + 1) * E)
    local_ids = torch.where(local_mask, ids - ep_rank * E, torch.zeros_like(ids))

    keep = local_mask.reshape(-1)
    kept_pairs = keep.nonzero(as_tuple=True)[0]
    expert_of_kept = local_ids.reshape(-1)[kept_pairs]
    sort_idx = torch.argsort(expert_of_kept, stable=True)
    sorted_pairs = kept_pairs[sort_idx]
    expert_sorted = expert_of_kept[sort_idx]

    counts = torch.bincount(expert_sorted, minlength=E)
    sizes = counts.cpu().tolist()
    max_size = max(sizes) if sizes else 0
    cap = max(128, ((max_size + 127) // 128) * 128)
    bpe = cap // 128
    rows_real = int(kept_pairs.numel())
    rows = max(BUCKET, ((rows_real + BUCKET - 1) // BUCKET) * BUCKET)
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)

    token_of_pair = torch.arange(M, device=dev).unsqueeze(1).expand(M, topk).reshape(-1)
    src_rows = token_of_pair[sorted_pairs].to(torch.int32)
    within = torch.arange(rows_real, device=dev) - offsets[expert_sorted].long()
    padded = (expert_sorted * cap + within).to(torch.int32)
    # pad to the bucketed `rows` so tensor shapes match the compiled ABI; the
    # extra slots read token 0 and write an unused SF slot (harmless)
    pad_n = rows - rows_real
    if pad_n > 0:
        src_rows = torch.cat([src_rows, torch.zeros(pad_n, dtype=torch.int32, device=dev)])
        padded = torch.cat([padded, torch.full((pad_n,), E * cap - 1, dtype=torch.int32, device=dev)])
    src_rows = src_rows.contiguous()
    padded = padded.contiguous()

    # K0: gather + quant
    A1 = torch.zeros(rows, K // 2, dtype=torch.uint8, device=dev)
    SFA1sem = torch.zeros(E * cap, K // 16, dtype=torch.float8_e4m3fn, device=dev)
    gather_quant(rows, M, K, E, cap)(x, src_rows, padded, A1, SFA1sem)
    SFA1 = swizzle_chunk_kmajor(pack_scale_words(SFA1sem.view(torch.uint8))).reshape(-1, BLOCK_K // 64)

    # K1
    act = torch.zeros(rows, N, dtype=torch.bfloat16, device=dev)
    k1 = w13_swiglu_emaj(E, N, K, rows, bpe, block_K=BLOCK_K, lim=10.0)
    k1(A1, st["Bg"], st["Bu"], SFA1, st["sfbg"], st["sfbu"], offsets, st["scale1"], act)

    # K2
    A2 = torch.zeros(rows, N // 2, dtype=torch.uint8, device=dev)
    SFA2sem = torch.zeros(E * cap, N // 16, dtype=torch.float8_e4m3fn, device=dev)
    src_identity = torch.arange(rows, dtype=torch.int32, device=dev)
    gather_quant(rows, rows, N, E, cap)(act, src_identity, padded, A2, SFA2sem)
    SFA2 = swizzle_chunk_kmajor(pack_scale_words(SFA2sem.view(torch.uint8))).reshape(-1, BLOCK_K // 64)

    # K3
    out2 = torch.zeros(rows, K, dtype=torch.bfloat16, device=dev)
    k3 = w2_emaj(E, K, N, rows, bpe, block_N=BLOCK_K)
    k3(A2, st["W2"], SFA2, st["sfb2"], offsets, out2)

    # combine: slot -> DENSE gathered row (expert-ordered out2 position)
    slot_rows = torch.zeros(M, topk, dtype=torch.int32, device=dev)
    slot_scale = torch.zeros(M, topk, dtype=torch.float32, device=dev)
    dense_rows = torch.arange(rows_real, device=dev, dtype=torch.int32)
    slot_rows.reshape(-1)[sorted_pairs] = dense_rows
    slot_scale.reshape(-1)[sorted_pairs] = (
        st["scale2"][expert_sorted] * topk_weights.reshape(-1).float()[sorted_pairs])
    y = torch.empty(M, K, dtype=torch.float32, device=dev)
    moe_combine(M, rows, K, topk)(out2, slot_rows.contiguous(), slot_scale.contiguous(), y)
    output.copy_(y.to(output.dtype))
    return output
