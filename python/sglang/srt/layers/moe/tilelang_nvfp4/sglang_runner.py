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
import time
import types
import torch

from .k1k3_emaj import w13_swiglu_emaj_dyn, w13_swiglu_emaj_dyn_fused, w2_emaj_dyn
from .moe_combine_kernel import moe_combine_dyn
from .act_quant_kernel import gather_quant_dyn
from .grouped_nvfp4_gemm import pack_scale_words, swizzle_chunk_kmajor

ENABLED = os.environ.get("SGLANG_TILELANG_MOE", "0") == "1"
MIN_M = int(os.environ.get("SGLANG_TILELANG_MOE_MIN_M", "512"))
BLOCK_K = 128
BN = 128
# Bucket the per-layer routing capacity so every layer allocates IDENTICAL
# buffer shapes. Without bucketing each layer's cap differs -> the caching
# allocator maps fresh physical segments every layer; on GB10 unified memory
# (~100GB pinned by weights) that VMM churn triggers reclaim storms and can
# hang the host (2026-10-04 x2 incidents, traced to pack_scale_words region).
BUCKET = int(os.environ.get("SGLANG_TILELANG_MOE_BUCKET", "128"))
MAX_CAP = int(os.environ.get("SGLANG_TILELANG_MOE_MAX_CAP", "4096"))


class CapacityOverflow(Exception):
    """A routing batch exceeded the pinned per-expert capacity; the caller
    should fall back to the production path for this batch."""
    pass

_states: dict[int, dict] = {}

# Persistent device buffer pool: the run-time path performs ZERO torch
# allocations. Fresh VMM segment mappings under GB10 unified-memory pressure
# (weights pin ~100GB of 121GB) trigger kernel reclaim storms that hang the
# host (2026-10-04 incidents 1-3, all traced to allocation sites around
# pack_scale_words). Buffers are lazily grown and pre-warmed at MAX_CAP by
# precompile_tilelang_moe so serving never grows them.
_POOL: dict[str, torch.Tensor] = {}


def _pbuf(key: str, numel: int, dtype, dev):
    t = _POOL.get(key)
    if t is None or t.numel() < numel:
        t = torch.empty(numel, dtype=dtype, device=dev)
        _POOL[key] = t
    return t


def _warm_pool(E: int, N: int, K: int, rows_max: int, dev):
    # +BM rows on K1/K3 input/output buffers: the emaj kernels clamp their
    # tile base to rows-BM, which (a) leaves the last dense rows unwritten and
    # (b) clobbers valid rows near the tail. With rows+BM padding the clamp
    # never fires for tiles that own real rows, so writes are exact.
    maxc = MAX_CAP
    BM = 128
    for key, numel, dt in (
        ("A1", (rows_max + BM) * (K // 2), torch.uint8),
        ("act", (rows_max + BM) * N, torch.bfloat16),
        ("A2", (rows_max + BM) * (N // 2), torch.uint8),
        ("out2", (rows_max + BM) * K, torch.bfloat16),
        ("y", rows_max * K, torch.float32),
        ("SFA1sem", E * maxc * (K // 16), torch.float8_e4m3fn),
        ("SFA1sw", E * maxc * (K // 64), torch.uint32),
        ("SFA2sem", E * maxc * (N // 16), torch.float8_e4m3fn),
        ("SFA2sw", E * maxc * (N // 64), torch.uint32),
    ):
        _pbuf(key, numel, dt, dev)


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

    scale1 = (quant_info.quant_scales[2] * quant_info.quant_scales[0]).float().reshape(E)
    scale2 = (quant_info.quant_scales[5] * quant_info.quant_scales[3]).float().reshape(E)

    # NO weight copies (2026-10-06): K1 reads quant_info.w13_weight FUSED
    # (E,2N,Kh) directly (w13_swiglu_emaj_dyn_fused) and K3 reads
    # quant_info.w2_weight (E,K,N/2) directly. The old Bg/Bu/W2
    # .contiguous() duplicates retained ~1.5GB per layer -> 60 layers blew
    # the torch pool mid-request on the first tlmoe walk -> NVRM mem_desc
    # OOM storm -> host freeze (2026-10-04/05/06 incidents). Only the
    # kernel-order scale repacks (~w/16 per layer) are retained now, and
    # they are prepacked for ALL layers at init by precompile_tilelang_moe.
    return dict(E=E, N=N, K=K,
                sfbg=sfbg, sfbu=sfbu, sfb2=sfb2,
                scale1=scale1, scale2=scale2,
                ep_rank=int(quant_info.moe_ep_rank),
                ep_size=int(quant_info.moe_ep_size))


def should_run(x) -> bool:
    return ENABLED and x.shape[0] >= MIN_M


def _run_split(*, x, topk_weights, quant_info, st, output, M, topk, trace,
               sorted_pairs, expert_sorted, counts, max_size, dev):
    """Over-capacity routing (some expert > MAX_CAP rows): run the same
    K0..combine pipeline once per MAX_CAP-row slice of every expert's
    segment, accumulating combine output in fp32. Replaces the old
    CapacityOverflow -> silent flashinfer fallback. KEEP IN SYNC with the
    single-pass pipeline in run_tilelang_moe (buffers identical: cap is
    pinned to MAX_CAP, which is what the pool is warmed at)."""
    def _tick(tag, t0):
        if trace:
            torch.cuda.synchronize()
            print(f"[tlmoe-trace] {tag} +{time.perf_counter()-t0:.3f}s", flush=True)
        return time.perf_counter()

    E, N, K = st["E"], st["N"], st["K"]
    cap = MAX_CAP
    assert cap % 128 == 0
    bpe = cap // 128
    n_pass = (max_size + cap - 1) // cap
    rows_total = int(sorted_pairs.numel())
    BM = 128
    W1, W2w = K // 64, N // 64
    Rcap = E * cap

    token_of_pair = torch.arange(M, device=dev).unsqueeze(1).expand(M, topk).reshape(-1)
    offsets_full = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    offsets_full[1:] = torch.cumsum(counts, 0).to(torch.int32)
    within = torch.arange(rows_total, device=dev) - offsets_full[expert_sorted].long()

    y_acc = _pbuf("yacc", M * K, torch.float32, dev)[: M * K].view(M, K)
    y_acc.zero_()
    k0 = gather_quant_dyn(K, E)
    k1 = w13_swiglu_emaj_dyn_fused(E, N, K, block_K=BLOCK_K, lim=10.0)
    k2 = gather_quant_dyn(N, E)
    k3 = w2_emaj_dyn(E, K, N, block_N=BLOCK_K)

    t_stage = time.perf_counter()
    if trace:
        print(f"[tlmoe-trace] L{st['lid']} split begin rows={rows_total} "
              f"max_size={max_size} n_pass={n_pass} cap={cap} bpe={bpe}", flush=True)
    for p in range(n_pass):
        lo = p * cap
        sel = (within >= lo) & (within < lo + cap)
        sp = sorted_pairs[sel]
        es = expert_sorted[sel]
        rows_p = int(sp.numel())
        rows_pad = rows_p + BM
        counts_p = torch.bincount(es, minlength=E)
        offsets_p = torch.zeros(E + 1, dtype=torch.int32, device=dev)
        offsets_p[1:] = torch.cumsum(counts_p, 0).to(torch.int32)
        within_p = torch.arange(rows_p, device=dev) - offsets_p[es].long()
        # rows_pad-long index tensors (tile-tail contract, see main path)
        rows_pad = rows_p + BM
        padded = torch.empty(rows_pad, dtype=torch.int32, device=dev)
        padded[:rows_p] = (es * cap + within_p).to(torch.int32)
        padded[rows_p:] = padded[0]
        src_rows = torch.empty(rows_pad, dtype=torch.int32, device=dev)
        src_rows[:rows_p] = token_of_pair[sp].to(torch.int32)
        src_rows[rows_p:] = src_rows[0]

        # K0: gather + quant into pooled buffers (pass-local slice views)
        A1_buf = _pbuf("A1", rows_pad * (K // 2), torch.uint8, dev)
        A1 = A1_buf[: rows_pad * (K // 2)].view(rows_pad, K // 2)
        SFA1sem = _pbuf("SFA1sem", Rcap * (K // 16), torch.float8_e4m3fn,
                        dev)[: Rcap * (K // 16)].view(Rcap, K // 16)
        SFA1sem.zero_()
        k0(x, src_rows, padded, A1_buf[: rows_pad * (K // 2)].view(rows_pad, K // 2), SFA1sem)
        packed = SFA1sem.view(torch.uint8).view(Rcap, W1, 4).view(torch.uint32)
        sw1 = _pbuf("SFA1sw", Rcap * W1, torch.uint32, dev)[: Rcap * W1].view(Rcap, W1)
        sw1.view(Rcap // 128, W1, 32, 4).copy_(packed.view(Rcap // 128, 4, 32, W1).permute(0, 3, 2, 1))
        SFA1 = sw1.view(-1, BLOCK_K // 64)

        act_buf = _pbuf("act", rows_pad * N, torch.bfloat16, dev)
        act = act_buf[: rows_pad * N].view(rows_pad, N)
        k1(A1, quant_info.w13_weight, SFA1, st["sfbg"], st["sfbu"], offsets_p, st["scale1"], act)

        A2_buf = _pbuf("A2", rows_pad * (N // 2), torch.uint8, dev)
        SFA2sem = _pbuf("SFA2sem", Rcap * (N // 16), torch.float8_e4m3fn,
                        dev)[: Rcap * (N // 16)].view(Rcap, N // 16)
        SFA2sem.zero_()
        src_identity = torch.zeros(rows_pad, dtype=torch.int32, device=dev)
        src_identity[:rows_p] = torch.arange(rows_p, dtype=torch.int32, device=dev)
        k2(act_buf[: rows_pad * N].view(rows_pad, N), src_identity, padded,
           A2_buf[: rows_pad * (N // 2)].view(rows_pad, N // 2), SFA2sem)
        packed2 = SFA2sem.view(torch.uint8).view(Rcap, W2w, 4).view(torch.uint32)
        sw2 = _pbuf("SFA2sw", Rcap * W2w, torch.uint32, dev)[: Rcap * W2w].view(Rcap, W2w)
        sw2.view(Rcap // 128, W2w, 32, 4).copy_(packed2.view(Rcap // 128, 4, 32, W2w).permute(0, 3, 2, 1))
        SFA2 = sw2.view(-1, BLOCK_K // 64)

        out2_buf = _pbuf("out2", rows_pad * K, torch.bfloat16, dev)
        A2 = A2_buf[: rows_pad * (N // 2)].view(rows_pad, N // 2)
        k3(A2, quant_info.w2_weight, SFA2, st["sfb2"], offsets_p, out2_buf[: rows_pad * K].view(rows_pad, K))

        # combine: pass-local slot mapping; unfilled slots scale=0 row=0
        slot_rows = torch.zeros(M, topk, dtype=torch.int32, device=dev)
        slot_scale = torch.zeros(M, topk, dtype=torch.float32, device=dev)
        slot_rows.reshape(-1)[sp] = torch.arange(rows_p, device=dev, dtype=torch.int32)
        slot_scale.reshape(-1)[sp] = (
            st["scale2"][es] * topk_weights.reshape(-1).float()[sp])
        y = _pbuf("y", M * K, torch.float32, dev)[: M * K].view(M, K)
        moe_combine_dyn(K, topk)(out2_buf[: rows_p * K].view(rows_p, K),
                                 slot_rows.contiguous(), slot_scale.contiguous(), y)
        y_acc.add_(y)
        t_stage = _tick(f"L{st['lid']} split pass {p+1}/{n_pass} done", t_stage)

    output.copy_(y_acc)
    t_stage = _tick(f"L{st['lid']} copy out done", t_stage)
    return output


def run_tilelang_moe(*, x, topk_weights, topk_ids, quant_info, output):
    trace = os.environ.get("SGLANG_TILELANG_MOE_TRACE", "0") == "1"
    debug = os.environ.get("SGLANG_TILELANG_MOE_DEBUG", "0") == "1"

    def _tick(tag, t0):
        if trace:
            torch.cuda.synchronize()
            print(f"[tlmoe-trace] {tag} +{time.perf_counter()-t0:.3f}s", flush=True)
        return time.perf_counter()

    st = _states.get(id(quant_info.w13_weight))
    if st is None:
        if debug:
            print("[tlmoe] layer prep (kernel-order scale repack)", flush=True)
        st = _prepare_layer(quant_info)
        st["lid"] = len(_states)
        _states[id(quant_info.w13_weight)] = st

    E, N, K = st["E"], st["N"], st["K"]
    ep_rank = st["ep_rank"]
    M, topk = topk_ids.shape
    dev = x.device
    if trace:
        print(f"[tlmoe-trace] L{st['lid']} begin M={M}", flush=True)

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
    max_size = int(counts.max().item()) if E else 0
    if max_size > MAX_CAP:
        # split-loop (2026-10-06): process over-capacity routing in passes of
        # MAX_CAP rows per expert instead of raising CapacityOverflow and
        # silently falling back to flashinfer (the silent fallback both hid
        # real traffic from tlmoe and made A/B testing misleading).
        return _run_split(x=x, topk_weights=topk_weights, quant_info=quant_info,
                          st=st, output=output, M=M, topk=topk, trace=trace,
                          sorted_pairs=sorted_pairs, expert_sorted=expert_sorted,
                          counts=counts, max_size=max_size, dev=dev)
    cap = max(BUCKET, ((max_size + BUCKET - 1) // BUCKET) * BUCKET)   # bucketed (symbolic in-kernel)
    bpe = cap // 128
    rows_real = rows = int(kept_pairs.numel())
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)

    token_of_pair = torch.arange(M, device=dev).unsqueeze(1).expand(M, topk).reshape(-1)
    # Index tensors MUST be rows_pad-long: gather_quant processes 128-row
    # tiles, and a ragged tail tile otherwise reads garbage src/padded
    # indices and scatters garbage scales into REAL SFA rows (silent
    # corruption, found 2026-10-06 via the split-loop). Tail lanes duplicate
    # row 0 -> idempotent writes, no race.
    rows_pad2 = rows + 128
    src_rows = torch.empty(rows_pad2, dtype=torch.int32, device=dev)
    src_rows[:rows] = token_of_pair[sorted_pairs].to(torch.int32)
    src_rows[rows:] = src_rows[0]
    within = torch.arange(rows, device=dev) - offsets[expert_sorted].long()
    padded = torch.empty(rows_pad2, dtype=torch.int32, device=dev)
    padded[:rows] = (expert_sorted * cap + within).to(torch.int32)
    padded[rows:] = padded[0]

    t_stage = time.perf_counter()
    if trace:
        torch.cuda.synchronize()
        print(f"[tlmoe-trace] L{st['lid']} routed rows={rows} cap={cap} bpe={bpe} "
              f"+{time.perf_counter()-t_stage:.3f}s", flush=True)
        t_stage = time.perf_counter()
    if debug:
        print(f"[tlmoe] K0 rows={rows} M={M} cap={cap} bpe={bpe}", flush=True)

    # ---- pooled, zero-allocation buffers (slices only from here on) ----
    Rcap = E * cap
    BM = 128
    rows_pad = rows + BM
    A1_buf = _pbuf("A1", rows_pad * (K // 2), torch.uint8, dev)
    A1 = A1_buf[: rows_pad * (K // 2)].view(rows_pad, K // 2)
    SFA1sem = _pbuf("SFA1sem", Rcap * (K // 16), torch.float8_e4m3fn,
                    dev)[: Rcap * (K // 16)].view(Rcap, K // 16)
    SFA1sem.zero_()

    # K0: gather + quant (shape-generic kernel, compiled once)
    if trace:
        print(f"[tlmoe-trace] L{st['lid']} K0 launching", flush=True)
    gather_quant_dyn(K, E)(x, src_rows, padded, A1, SFA1sem)
    t_stage = _tick(f"L{st['lid']} K0 done", t_stage)
    # pack+swizzle, allocation-free: little-endian uint32 view == the old
    # byte-wise pack_scale_words elementwise chain; strided copy_ == swizzle
    W1 = K // 64
    packed = SFA1sem.view(torch.uint8).view(Rcap, W1, 4).view(torch.uint32)
    sw1 = _pbuf("SFA1sw", Rcap * W1, torch.uint32, dev)[: Rcap * W1].view(Rcap, W1)
    sw1.view(Rcap // 128, W1, 32, 4).copy_(packed.view(Rcap // 128, 4, 32, W1).permute(0, 3, 2, 1))
    SFA1 = sw1.view(-1, BLOCK_K // 64)
    t_stage = _tick(f"L{st['lid']} K0 pack done", t_stage)

    # K1
    act_buf = _pbuf("act", rows_pad * N, torch.bfloat16, dev)
    act = act_buf[: rows_pad * N].view(rows_pad, N)
    k1 = w13_swiglu_emaj_dyn_fused(E, N, K, block_K=BLOCK_K, lim=10.0)
    if trace:
        print(f"[tlmoe-trace] L{st['lid']} K1 launching", flush=True)
    k1(A1, quant_info.w13_weight, SFA1, st["sfbg"], st["sfbu"], offsets, st["scale1"], act)
    t_stage = _tick(f"L{st['lid']} K1 done", t_stage)

    # K2
    A2_buf = _pbuf("A2", rows_pad * (N // 2), torch.uint8, dev)
    SFA2sem = _pbuf("SFA2sem", Rcap * (N // 16), torch.float8_e4m3fn,
                    dev)[: Rcap * (N // 16)].view(Rcap, N // 16)
    SFA2sem.zero_()
    # rows_pad-long, tail duplicates row 0 (same tile-tail contract as K0)
    src_identity = torch.zeros(rows_pad, dtype=torch.int32, device=dev)
    src_identity[:rows] = torch.arange(rows, dtype=torch.int32, device=dev)
    if trace:
        print(f"[tlmoe-trace] L{st['lid']} K2 launching", flush=True)
    gather_quant_dyn(N, E)(act, src_identity, padded, A2_buf[: rows_pad * (N // 2)].view(rows_pad, N // 2), SFA2sem)
    t_stage = _tick(f"L{st['lid']} K2 done", t_stage)
    W2 = N // 64
    packed2 = SFA2sem.view(torch.uint8).view(Rcap, W2, 4).view(torch.uint32)
    sw2 = _pbuf("SFA2sw", Rcap * W2, torch.uint32, dev)[: Rcap * W2].view(Rcap, W2)
    sw2.view(Rcap // 128, W2, 32, 4).copy_(packed2.view(Rcap // 128, 4, 32, W2).permute(0, 3, 2, 1))
    SFA2 = sw2.view(-1, BLOCK_K // 64)
    t_stage = _tick(f"L{st['lid']} K2 pack done", t_stage)

    # K3
    out2_buf = _pbuf("out2", rows_pad * K, torch.bfloat16, dev)
    out2 = out2_buf[: rows_pad * K].view(rows_pad, K)
    A2 = A2_buf[: rows_pad * (N // 2)].view(rows_pad, N // 2)
    k3 = w2_emaj_dyn(E, K, N, block_N=BLOCK_K)
    if trace:
        print(f"[tlmoe-trace] L{st['lid']} K3 launching", flush=True)
    k3(A2, quant_info.w2_weight, SFA2, st["sfb2"], offsets, out2)
    t_stage = _tick(f"L{st['lid']} K3 done", t_stage)

    # combine: slot -> DENSE gathered row (expert-ordered out2 position)
    slot_rows = torch.zeros(M, topk, dtype=torch.int32, device=dev)
    slot_scale = torch.zeros(M, topk, dtype=torch.float32, device=dev)
    dense_rows = torch.arange(rows_real, device=dev, dtype=torch.int32)
    slot_rows.reshape(-1)[sorted_pairs] = dense_rows
    slot_scale.reshape(-1)[sorted_pairs] = (
        st["scale2"][expert_sorted] * topk_weights.reshape(-1).float()[sorted_pairs])
    y = _pbuf("y", M * K, torch.float32, dev)[: M * K].view(M, K)
    if trace:
        print(f"[tlmoe-trace] L{st['lid']} combine launching", flush=True)
    moe_combine_dyn(K, topk)(out2_buf[: rows * K].view(rows, K),
                             slot_rows.contiguous(), slot_scale.contiguous(), y)
    t_stage = _tick(f"L{st['lid']} combine done", t_stage)
    if debug:
        print(f"[tlmoe] combine done {time.perf_counter()-t_stage:.3f}s", flush=True)
    output.copy_(y)
    t_stage = _tick(f"L{st['lid']} copy out done", t_stage)
    return output


def precompile_tilelang_moe(model, num_tokens: int = 8192) -> bool:
    """AOT-compile the 5 TileLang MoE kernels during engine init.

    Called from the model's `precompile_kernels_after_loading` hook (runs on
    every TP rank after weight load, before serving — symmetric across ranks,
    outside the serving watchdog). Kernels are shape-generic (symbolic rows),
    so one compile per kernel covers every future batch shape.
    """
    import torch.nn as nn

    layers = []
    for m in model.modules():
        if (
            hasattr(m, "w13_weight")
            and hasattr(m, "w2_weight")
            and getattr(m, "w13_blockscale_swizzled", None) is not None
            and getattr(m, "w13_weight", None) is not None
        ):
            layers.append(m)
    if not layers:
        return False
    layer = layers[0]
    E, twoN, Kh = layer.w13_weight.shape
    N, K = twoN // 2, Kh * 2
    topk = int(getattr(layer, "topk", 8))
    quant_scales = [
        layer.w13_input_scale_quant,
        layer.w13_blockscale_swizzled,
        layer.g1_alphas,
        layer.w2_input_scale_quant,
        layer.w2_blockscale_swizzled,
        layer.g2_alphas,
    ]
    if any(q is None for q in quant_scales):
        return False
    quant_info = types.SimpleNamespace(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        quant_type="fp4",
        quant_scales=quant_scales,
        moe_ep_rank=int(layer.moe_ep_rank),
        moe_ep_size=int(layer.moe_ep_size),
        moe_tp_rank=int(layer.moe_tp_rank),
        moe_tp_size=int(layer.moe_tp_size),
    )
    dev = layer.w13_weight.device
    # Pre-warm the buffer pool at MAX_CAP so the serving path never allocates.
    _warm_pool(E, N, K, rows_max=num_tokens * topk, dev=dev)
    # Prepack kernel-order scale factors for EVERY MoE layer at init
    # (2026-10-06): the serving path must never lazily _prepare_layer —
    # per-layer first-use prep is what froze the host (NVRM OOM storm).
    n_prep = 0
    for m in layers:
        qs = [
            m.w13_input_scale_quant, m.w13_blockscale_swizzled, m.g1_alphas,
            m.w2_input_scale_quant, m.w2_blockscale_swizzled, m.g2_alphas,
        ]
        if any(q is None for q in qs):
            continue
        qi = types.SimpleNamespace(
            w13_weight=m.w13_weight, w2_weight=m.w2_weight, quant_type="fp4",
            quant_scales=qs, moe_ep_rank=int(m.moe_ep_rank),
            moe_ep_size=int(m.moe_ep_size), moe_tp_rank=int(m.moe_tp_rank),
            moe_tp_size=int(m.moe_tp_size))
        st = _prepare_layer(qi)
        st["lid"] = len(_states)
        _states[id(m.w13_weight)] = st
        n_prep += 1
    print(f"[tlmoe] prepacked kernel-order scales for {n_prep}/{len(layers)} MoE layers", flush=True)
    g = torch.Generator(device=dev).manual_seed(0)
    X = torch.randn(num_tokens, K, device=dev, dtype=torch.bfloat16, generator=g) / 10
    score = torch.randn(num_tokens, E, device=dev, dtype=torch.bfloat16, generator=g)
    tw, ti = torch.topk(score, topk, dim=-1, sorted=False)
    tw = torch.softmax(tw.float(), dim=-1)
    out = torch.empty(num_tokens, K, device=dev, dtype=torch.bfloat16)
    with torch.inference_mode():
        run_tilelang_moe(x=X, topk_weights=tw, topk_ids=ti.to(torch.int),
                         quant_info=quant_info, output=out)
    torch.cuda.synchronize()
    return True
