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
partial output, summed later by the TP all-reduce).

2026-10-06 tile-table kernels (replaces uniform-capacity + split-loop):
  gathered operands stay in DENSE gather order (K0/K2/combine plumbing
  unchanged from the proven layout); the grouped GEMMs take per-tile tables
    tile_e[t]    expert of tile t
    tile_base[t] dense row base = offsets[e] + local*128
    tile_act[t]  real rows in tile t
  and launch grid = sum_e ceil(counts[e]/128) ACTUAL tiles.
Root causes removed (HANDOFF 2026-10-06 §14):
  - dead tiles running full weight-stream K-loops under skewed routing
    (uniform cap tax, P<=1024)
  - split-loop re-streaming ALL expert weights per MAX_CAP pass (P>=2048,
    85% of the e2e regression). No capacity concept remains: any skew just
    adds tiles.

2026-10-11 K0 v2 + combine v2 (HANDOFF 2026-10-11 §6):
  - K0 = quant_nat_dyn (natural-order quant-once, dense x read) +
    fp4_gather_dyn (fp4 gather-repack into the UNCHANGED K1 contract:
    dense A1 + tile-major SFA1sem, byte-identical to gather_quant_dyn).
    Kills the routed-order x footprint tax (K0 serving 2.11 vs 0.76ms
    offline); the repack's random reads stay L2-resident in the 16.8MB
    fp4 payload.
  - combine writes bf16 directly into `output` (contiguous (M,K) bf16
    guard; pooled bf16 y + copy_ fallback otherwise) — no fp32 y, no
    copy_ tax. Zero-scale slots (TP non-local) are redirected to out2
    row 0 pre-launch (L2-hot phantom reads; bit-exact).

2026-10-11 sfb-native-read (HANDOFF_sfb_native_read_20261011):
  - SFB (weight block-scales) read DIRECTLY from the checkpoint-native
    CUTLASS 128x4 swizzled tensors via zero-copy flat u32 views + the
    closed-form inverse map inside K1/K3 — deletes the 3x repack chain in
    _prepare_layer and its ~4.8-5.6GB/process resident copies (sfbg/sfbu/
    sfb2).  Bit-exact by construction (same smem SF tiles); verified
    byte-exact offline.  SGLANG_TILELANG_MOE_SFB_REPACK=1 restores the
    old host repack path (A/B + escape hatch).
"""
from __future__ import annotations

import os
import time
import types
import torch
import tilelang.language as T

from .k1k3_emaj import w13_swiglu_emaj_dyn_fused_tbl, w2_emaj_dyn_tbl
from .moe_combine_kernel import moe_combine_dyn
from .act_quant_kernel import gather_quant_dyn, quant_nat_dyn, fp4_gather_dyn
from .grouped_nvfp4_gemm import pack_scale_words, swizzle_chunk_kmajor

ENABLED = os.environ.get("SGLANG_TILELANG_MOE", "0") == "1"
MIN_M = int(os.environ.get("SGLANG_TILELANG_MOE_MIN_M", "512"))
BLOCK_K = 128
BN = 128
BM = 128
# 2026-10-11 sfb-native-read: kernels address the checkpoint-native swizzled
# weight block-scales in place (zero-copy u32 view, closed-form inverse of
# the kernel-order pack) instead of a prepacked resident copy
# (~4.8-5.6GB/process at E=144/N=2048/K=4096). SGLANG_TILELANG_MOE_SFB_REPACK=1
# restores the old host repack path (A/B reference + escape hatch for
# non-contiguous checkpoints).
SFB_NATIVE_READ = os.environ.get("SGLANG_TILELANG_MOE_SFB_REPACK", "0") != "1"


class CapacityOverflow(Exception):
    """Legacy marker (never raised since tile-table): the hook's except path
    imports this symbol; routing can no longer overflow because tiles scale
    with actual per-expert counts."""
    pass


_states: dict[int, dict] = {}

# Persistent device buffer pool: the run-time path performs ZERO torch
# segment allocations. Fresh VMM segment mappings under GB10 unified-memory
# pressure (weights pin ~100GB of 121GB) trigger kernel reclaim storms that
# hang the host (2026-10-04 incidents 1-3). Buffers are sized for the worst
# case ONCE at init by precompile_tilelang_moe so serving never grows them.
_POOL: dict[str, torch.Tensor] = {}


def _pbuf(key: str, numel: int, dtype, dev):
    t = _POOL.get(key)
    if t is None or t.numel() < numel:
        t = torch.empty(numel, dtype=dtype, device=dev)
        _POOL[key] = t
    return t


def _warm_pool(E: int, N: int, K: int, m_max: int, rows_max: int, dev):
    """Pre-warm the pool at the worst serving shape.

    rows_pad_max = ceil128(rows_max) + BM covers the gather-order tail-tile
    contract (2026-10-11: rows_pad is 128-aligned so the gather grid never
    over-scans the index tensors);
    SFA sem/slabs round up to 128-row atoms (rows_pad is already a multiple
    of 128, else +127 slack).
    """
    rows_pad_max = (rows_max + BM - 1) // BM * BM + BM   # 128-aligned + tail tile
    sem_max = rows_max + E * BM   # tiles_max*128; tiles_max <= rows_max/BM + E
    for key, numel, dt in (
        ("A1", rows_pad_max * (K // 2), torch.uint8),
        ("act", rows_pad_max * N, torch.bfloat16),
        ("A2", rows_pad_max * (N // 2), torch.uint8),
        ("out2", rows_pad_max * K, torch.bfloat16),
        # 2026-10-11: combine writes bf16 directly into the caller's output;
        # the pooled y is now the bf16 fallback for non-contiguous outputs
        # (was fp32 — half the bytes, no fp32 y + copy_ bandwidth tax).
        ("y", m_max * K, torch.bfloat16),
        # 2026-10-11 K0 v2: natural-order quant-once payload (fp4 + scales).
        # 16.8MB + 2MB at m_max=8192, K=4096 — fits the 25MB L2 between the
        # quant pass and the repack gather.  idxL/slotL: int64 gather indices
        # for the torch index_select/index_copy_ repack; sfag: repack scratch.
        ("Aqn", m_max * (K // 2), torch.uint8),
        ("SFAn", m_max * (K // 16), torch.float8_e4m3fn),
        ("idxL", rows_pad_max, torch.int64),
        ("slotL", rows_pad_max, torch.int64),
        ("sfag", rows_pad_max * (K // 16), torch.uint8),
        ("SFA1sem", sem_max * (K // 16), torch.float8_e4m3fn),
        ("SFA1sw", sem_max * (K // 64), torch.uint32),
        ("SFA2sem", sem_max * (N // 16), torch.float8_e4m3fn),
        ("SFA2sw", sem_max * (N // 64), torch.uint32),
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

    scale1 = (quant_info.quant_scales[2] * quant_info.quant_scales[0]).float().reshape(E)
    scale2 = (quant_info.quant_scales[5] * quant_info.quant_scales[3]).float().reshape(E)
    st = dict(E=E, N=N, K=K, scale1=scale1, scale2=scale2,
              ep_rank=int(quant_info.moe_ep_rank),
              ep_size=int(quant_info.moe_ep_size))

    # NO weight copies (2026-10-06): K1 reads quant_info.w13_weight FUSED
    # (E,2N,Kh) directly and K3 reads quant_info.w2_weight (E,K,N/2) directly.
    # 2026-10-11 sfb-native-read: the scale factors stay in the checkpoint's
    # native CUTLASS 128x4 swizzled layout too — flat u32 VIEWS (zero-copy);
    # the kernels reconstruct the kernel-order SF tiles with the closed-form
    # inverse map (host-verified byte-exact vs this file's repack chain,
    # validate_sfb_map*.py; real-checkpoint EXACT on all 288 experts x
    # w13-gate/w13-up/w2).  GLM-clean pad shapes only (handoff §7-5):
    # anything else fails the asserts loudly -> run with
    # SGLANG_TILELANG_MOE_SFB_REPACK=1.
    if SFB_NATIVE_READ:
        if not (sw13.is_contiguous() and sw2.is_contiguous()):
            print("[tlmoe] sfb native-read skipped: non-contiguous checkpoint "
                  "scale (blob slice); falling back to repack for this layer",
                  flush=True)
        else:
            Mp13, Kp13 = sw13.shape[-2], sw13.shape[-1]
            Mp2, Kp2 = sw2.shape[-2], sw2.shape[-1]
            assert Mp13 == twoN and Kp13 == K // 16, \
                f"native-read shape mismatch w13 {(Mp13, Kp13)} != {(twoN, K // 16)}"
            assert Mp2 == K and Kp2 == N // 16, \
                f"native-read shape mismatch w2 {(Mp2, Kp2)} != {(K, N // 16)}"
            st["mode"] = "native"
            st["sfb_nat13"] = sw13.view(torch.uint32).reshape(-1)
            st["sfb_nat2"] = sw2.view(torch.uint32).reshape(-1)
            return st

    sem13 = _unswizzle_blockscale(sw13)
    sem2 = _unswizzle_blockscale(sw2)
    K16 = K // 16
    sem13 = sem13[:, :twoN, :K16].contiguous()
    sem2 = sem2[:, :K, :N // 16].contiguous()

    sfbg = _pack_kernel_order(sem13[:, :N].reshape(E * N, K16), BN, words)
    sfbu = _pack_kernel_order(sem13[:, N:].reshape(E * N, K16), BN, words)
    sfb2 = _pack_kernel_order(sem2.reshape(E * K, N // 16), BN, words)

    st["mode"] = "repack"
    st["sfbg"] = sfbg
    st["sfbu"] = sfbu
    st["sfb2"] = sfb2
    return st


def should_run(x) -> bool:
    return ENABLED and x.shape[0] >= MIN_M


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
            print("[tlmoe] layer prep (sfb mode: "
                  f"{'native-read' if SFB_NATIVE_READ else 'repack'})", flush=True)
        st = _prepare_layer(quant_info)
        st["lid"] = len(_states)
        _states[id(quant_info.w13_weight)] = st

    E, N, K = st["E"], st["N"], st["K"]
    ep_rank = st["ep_rank"]
    M, topk = topk_ids.shape
    dev = x.device
    if not x.is_contiguous():
        # tilelang's packed-ABI check rejects strided X (gather kernels
        # included — pre-existing since the 0.1.15 stride gate); serving
        # dispatch always hands over contiguous hidden_states, this is a
        # safety net for exotic callers.
        x = x.contiguous()
    if trace:
        print(f"[tlmoe-trace] L{st['lid']} begin M={M}", flush=True)

    # ---- routing: keep local rows, sort expert-major ----
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
    rows = int(kept_pairs.numel())
    offsets = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    token_of_pair = torch.arange(M, device=dev).unsqueeze(1).expand(M, topk).reshape(-1)

    # ---- tile tables (per-expert ACTUAL tiles; no capacity, no split) ----
    tiles_e = (counts + (BM - 1)) // BM                       # (E,) int64
    tile_start = torch.zeros(E + 1, dtype=torch.int64, device=dev)
    tile_start[1:] = torch.cumsum(tiles_e, 0)
    tiles = int(tile_start[-1].item())                        # 1 host sync
    tile_e = torch.repeat_interleave(
        torch.arange(E, device=dev), tiles_e).to(torch.int32)
    t_local = torch.arange(tiles, device=dev) - tile_start[tile_e.long()]
    tile_base = (offsets[tile_e.long()] + t_local * BM).to(torch.int32)
    tile_act = (counts[tile_e.long()] - t_local * BM).clamp_(0, BM).to(torch.int32)
    within = torch.arange(rows, device=dev) - offsets[expert_sorted].long()

    # index tensors are rows_pad-long: gather_quant works on 128-row tiles of
    # the gather order; tail lanes duplicate lane 0 (idempotent writes — the
    # 2026-10-06 tail-tile corruption contract).
    # 2026-10-11 FIX: rows_pad must be a MULTIPLE of 128.  gather_quant binds
    # `rows` from the index tensor's shape and launches ceildiv(rows,128)
    # blocks; with the old rows = rows+BM (arbitrary alignment) the grid
    # over-scanned up to 127 lanes PAST the tensors -> OOB garbage-index
    # reads/writes (nondeterministic SFA slot corruption under TP/EP; silent
    # when rows itself is 128-aligned, e.g. single-GPU offline harnesses).
    rows_pad = (rows + BM - 1) // BM * BM + BM
    src_rows = torch.empty(rows_pad, dtype=torch.int32, device=dev)
    src_rows[:rows] = token_of_pair[sorted_pairs].to(torch.int32)
    src_rows[rows:] = src_rows[0]
    ident = torch.empty(rows_pad, dtype=torch.int32, device=dev)
    ident[:rows] = torch.arange(rows, device=dev, dtype=torch.int32)
    ident[rows:] = ident[0]
    # SFA sem lives in per-GEMM-tile slot space (slab t = tile t's 128 lanes):
    # slot of gathered row i = tile_start[expert_i]*128 + within_i
    sfa_slot = torch.empty(rows_pad, dtype=torch.int32, device=dev)
    sfa_slot[:rows] = (tile_start[expert_sorted] * BM + within).to(torch.int32)
    sfa_slot[rows:] = sfa_slot[0]

    t_stage = time.perf_counter()
    if trace:
        torch.cuda.synchronize()
        print(f"[tlmoe-trace] L{st['lid']} routed rows={rows} tiles={tiles} "
              f"+{time.perf_counter()-t_stage:.3f}s", flush=True)
        t_stage = time.perf_counter()
    if debug:
        print(f"[tlmoe] K0 rows={rows} M={M} tiles={tiles}", flush=True)

    # ---- pooled, zero-segment-allocation buffers (slices only below) ----
    sem_rows = tiles * BM
    A1 = _pbuf("A1", rows_pad * (K // 2), torch.uint8, dev)[: rows_pad * (K // 2)].view(rows_pad, K // 2)
    SFA1sem = _pbuf("SFA1sem", sem_rows * (K // 16), torch.float8_e4m3fn,
                    dev)[: sem_rows * (K // 16)].view(sem_rows, K // 16)
    # NOTE: no SFA zero_() — every K0 write slot is a real-or-duplicated lane
    # (tail-tile contract above); lanes >= rows are never read for real
    # output rows (tile_act masks the epilogue).

    # K0 v2 (2026-10-11): quant-once in natural order (dense 67MB read,
    # no routing dependency, bit-identical per-row quant) then fp4
    # gather-repack into the unchanged K1 contract (A1 dense gather order,
    # SFA1sem tile-major).  The repack's 2KB random reads stay inside the
    # 16.8MB fp4 payload = L2-resident, killing the routed-order x footprint
    # tax (268MB random DRAM reads -> ~86MB dense + ~38MB L2-hot).
    Aq_nat = _pbuf("Aqn", M * (K // 2), torch.uint8, dev)[: M * (K // 2)].view(M, K // 2)
    SFA_nat = _pbuf("SFAn", M * (K // 16), torch.float8_e4m3fn,
                    dev)[: M * (K // 16)].view(M, K // 16)
    quant_nat_dyn(K)(x, Aq_nat, SFA_nat)
    t_stage = _tick(f"L{st['lid']} K0 quant-once done", t_stage)
    idxL = _pbuf("idxL", rows_pad, torch.int64, dev)[:rows_pad]
    idxL.copy_(src_rows)                       # int32 -> int64 (cast kernel)
    slotL = _pbuf("slotL", rows_pad, torch.int64, dev)[:rows_pad]
    slotL.copy_(sfa_slot)
    sfa_g = _pbuf("sfag", rows_pad * (K // 16), torch.uint8,
                  dev)[: rows_pad * (K // 16)].view(rows_pad, K // 16)
    fp4_gather_dyn(K)(Aq_nat, SFA_nat.view(torch.uint8), idxL, slotL, A1,
                      SFA1sem.view(torch.uint8), sfa_g)
    t_stage = _tick(f"L{st['lid']} K0 done", t_stage)
    # pack+swizzle, allocation-free: strided copy_ == chunk_kmajor swizzle;
    # slab s = gathered lanes [s*128,(s+1)*128) = tile s (tile-major SFA)
    W1 = K // 64
    packed = SFA1sem.view(torch.uint8).view(sem_rows, W1, 4).view(torch.uint32)
    sw1 = _pbuf("SFA1sw", sem_rows * W1, torch.uint32, dev)[: sem_rows * W1].view(sem_rows, W1)
    sw1.view(tiles, W1, 32, 4).copy_(packed.view(tiles, 4, 32, W1).permute(0, 3, 2, 1))
    SFA1 = sw1.view(-1, BLOCK_K // 64)
    t_stage = _tick(f"L{st['lid']} K0 pack done", t_stage)

    # K1: fused w13 + SwiGLU, tile-table grid
    act = _pbuf("act", rows_pad * N, torch.bfloat16, dev)[: rows_pad * N].view(rows_pad, N)
    k1 = w13_swiglu_emaj_dyn_fused_tbl(E, N, K, block_K=BLOCK_K, lim=10.0,
                                       sfb_native=(st["mode"] == "native"))
    if st["mode"] == "native":
        # gate+up share one native view; the kernel adds the +N/128 atom offset
        k1(A1, quant_info.w13_weight, SFA1, st["sfb_nat13"], st["sfb_nat13"],
           tile_e, tile_base, tile_act, st["scale1"], act)
    else:
        k1(A1, quant_info.w13_weight, SFA1, st["sfbg"].view(-1), st["sfbu"].view(-1),
           tile_e, tile_base, tile_act, st["scale1"], act)
    t_stage = _tick(f"L{st['lid']} K1 done", t_stage)

    # K2: requant activated intermediate (dense, identical to proven layout)
    A2 = _pbuf("A2", rows_pad * (N // 2), torch.uint8, dev)[: rows_pad * (N // 2)].view(rows_pad, N // 2)
    SFA2sem = _pbuf("SFA2sem", sem_rows * (N // 16), torch.float8_e4m3fn,
                    dev)[: sem_rows * (N // 16)].view(sem_rows, N // 16)
    gather_quant_dyn(N, E)(act, ident, sfa_slot, A2, SFA2sem)
    t_stage = _tick(f"L{st['lid']} K2 done", t_stage)
    W2w = N // 64
    packed2 = SFA2sem.view(torch.uint8).view(sem_rows, W2w, 4).view(torch.uint32)
    sw2 = _pbuf("SFA2sw", sem_rows * W2w, torch.uint32, dev)[: sem_rows * W2w].view(sem_rows, W2w)
    sw2.view(tiles, W2w, 32, 4).copy_(packed2.view(tiles, 4, 32, W2w).permute(0, 3, 2, 1))
    SFA2 = sw2.view(-1, BLOCK_K // 64)
    t_stage = _tick(f"L{st['lid']} K2 pack done", t_stage)

    # K3: grouped w2, tile-table grid
    out2 = _pbuf("out2", rows_pad * K, torch.bfloat16, dev)[: rows_pad * K].view(rows_pad, K)
    k3 = w2_emaj_dyn_tbl(E, K, N, block_N=BLOCK_K, sfb_native=(st["mode"] == "native"))
    if st["mode"] == "native":
        k3(A2, quant_info.w2_weight, SFA2, st["sfb_nat2"], tile_e, tile_base,
           tile_act, out2)
    else:
        k3(A2, quant_info.w2_weight, SFA2, st["sfb2"].view(-1), tile_e, tile_base,
           tile_act, out2)
    t_stage = _tick(f"L{st['lid']} K3 done", t_stage)

    # combine: slot -> dense gathered row (identical to the proven layout)
    slot_rows = torch.zeros(M, topk, dtype=torch.int32, device=dev)
    slot_scale = torch.zeros(M, topk, dtype=torch.float32, device=dev)
    dense_rows = torch.arange(rows, device=dev, dtype=torch.int32)
    slot_rows.reshape(-1)[sorted_pairs] = dense_rows
    slot_scale.reshape(-1)[sorted_pairs] = (
        st["scale2"][expert_sorted] * topk_weights.reshape(-1).float()[sorted_pairs])
    # zero-scale slots (TP non-local experts) contribute exactly +-0.0;
    # redirect their gather reads to out2 row 0 so the phantom reads hit one
    # L2-hot row instead of ~half of the 268MB out2 DRAM re-read (bit-exact:
    # the added term is still 0.0*finite = +-0.0).  NO in-kernel predicate:
    # predication serializes tilelang's FMA loop and changes rounding.
    slot_rows.masked_fill_(slot_scale == 0.0, 0)
    # combine v2 (2026-10-11): out_dtype=bfloat16, acc stays fp32 in
    # registers — one RN cast, bit-identical to the old fp32 y + copy_ cast.
    # Direct write into the caller's output when the buffer is writable
    # as-is; the pooled bf16 y + copy_ path only survives for exotic
    # outputs (non-contiguous / wrong shape).  Kills the fp32 y write+read
    # and the copy_ read+write (~2.4ms/layer of pure bandwidth tax at
    # M=8192, HANDOFF §4③).
    direct_out = (
        output.dtype == torch.bfloat16
        and output.is_contiguous()
        and output.dim() == 2
        and output.shape[0] == M
        and output.shape[1] == K
    )
    if direct_out:
        moe_combine_dyn(K, topk, out_dtype=T.bfloat16)(
            out2[:rows], slot_rows.contiguous(), slot_scale.contiguous(), output)
    else:
        y = _pbuf("y", M * K, torch.bfloat16, dev)[: M * K].view(M, K)
        moe_combine_dyn(K, topk, out_dtype=T.bfloat16)(
            out2[:rows], slot_rows.contiguous(), slot_scale.contiguous(), y)
        output.copy_(y)
    t_stage = _tick(f"L{st['lid']} combine done", t_stage)
    if debug:
        print(f"[tlmoe] combine done {time.perf_counter()-t_stage:.3}s direct={direct_out}",
              flush=True)
    return output


def precompile_tilelang_moe(model, num_tokens: int = 8192) -> bool:
    """AOT-compile the TileLang MoE kernels during engine init.

    Called from the model's `precompile_kernels_after_loading` hook (runs on
    every TP rank after weight load, before serving — symmetric across ranks,
    outside the serving watchdog). Kernels are shape-generic (symbolic dims),
    so one compile per kernel covers every future batch shape/tile count.
    (2026-10-11: 6 jit functions per layer pass — K1 tbl, K2 gather_quant,
    K3 tbl, quant_nat, fp4_gather, combine bf16.)
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
    # Pre-warm the pool at MAX serving shape so the serving path never
    # allocates a new segment.
    _warm_pool(E, N, K, m_max=num_tokens, rows_max=num_tokens * topk, dev=dev)
    # Prepack kernel-order scale factors for EVERY MoE layer at init
    # (2026-10-06): the serving path must never lazily _prepare_layer —
    # per-layer first-use prep is what froze the host (NVRM OOM storm).
    n_prep = 0
    n_nat = 0
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
        n_nat += 1 if st["mode"] == "native" else 0
    # Acceptance grep (2026-10-11): "sfb prep done" replaces the old
    # "prepacked kernel-order scales" line; repack>0 under native-read means
    # per-layer non-contiguous fallbacks (handoff §7-1 — record the share).
    print(f"[tlmoe] sfb prep done: native={n_nat}/{n_prep} repack={n_prep - n_nat} "
          f"MoE layers (SGLANG_TILELANG_MOE_SFB_REPACK="
          f"{'1' if not SFB_NATIVE_READ else '0'})", flush=True)
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
