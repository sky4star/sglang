// Decode tiny-M NVFP4 MoE GEMV with deterministic split-K (K1d/K3d) — v10.
//
// Perf history (P0 NCU evidence + v2/v3/v3.1 experiments, 2026-10-13):
//   v1: packed u8 smem staging, wf[2][32]+af[32] resident -> 115 regs -> 2
//       CTA/SM, CPI 15, decode = 1.5 inst/fp4 (F2FP+HADD2.F32). 319 us K1d.
//   v2: af pre-decoded f32 read straight from L2 -> floods GB10 L2 (~1.1 TB/s
//       ceiling), 2x SLOWER. Rejected.
//   v3: f32 staged to smem -> L1TEX wavefront wall (8x v1's smem traffic).
//       Rejected.
//   v8 (kept): packed staging(SEG=4) + SF u32 merge + wf as packed __half2 in
//       regs (115->95) + pair inner loop. K3d -23% (4 blk); K1d -4%, 95 regs
//       still 2 blk.
//   v9: v8 + 2-way m unroll in the row loop (4 independent dot chains,
//       per-row order unchanged -> bit-exact). K1d 309->285us, CPI ~flat;
//       regs hit the 128 ceiling (2 blk).
//   v10 (this): v9 + weight/SF decode hoisted ABOVE the chunk barrier
//       (read-only kernel inputs, no smem involvement -> no race): the
//       long-scoreboard wait overlaps the previous chunk's compute.
//       K1d 285->261us, K3d 155->134us (NCU, uniform_T1).
//       Rejected same day: per-warp staging planes without the barrier —
//       the 8-warp lockstep staging burst is L1-broadcast-friendly, so the
//       barrier is load-bearing; decoupled planes measured 5-7% SLOWER.
//
// Numerics are UNCHANGED vs v1 (same deq intrinsics, same per-block 16-value
// fmaf chain -> scale fma -> 5-step shuffle butterfly, fixed order) so the
// kernel stays bit-exact vs the oracle/bitmodel contract.
//
// Math contract (bitmodel.py-verified):
//   out[m,n] = bf16( alpha[expert(m)] * sum_kb SA[m,kb]*SB[n,kb] * (sum_j a4*b4) )
// with fp32 accumulation throughout, fixed iteration order (deterministic;
// no atomics — split-K partials are summed by a fixed-order reduce kernel).
//
// Addressing (byte-EXACT vs oracle):
//   * weight SF: native CUTLASS 128x4 swizzle, closed form
//       off(n,kb) = (n/128)*(128*KPB) + (kb/4)*512 + (n%32)*16 + ((n%128)/32)*4 + kb%4
//   * activation SF (decode row cap inside one 128-row tile):
//       off(m,kb) = (kb/4)*512 + (m%32)*16 + ((m%128)/32)*4 + kb%4
//
// Grid = (N/N_TILE, SLICES); each CTA walks ALL experts and skips inactive
// ones via m_indptr (no host sync, CUDA-graph safe: launch geometry depends
// only on compile-time shapes, never on routing content).
//
// Shapes: K1d = <N=2048, K_BYTES=2048, SLICES=4>; K3d = <N=4096, K_BYTES=512,
// SLICES=2>. K_BYTES is PACKED bytes per weight row (2 fp4 per byte), so one
// SF block (16 fp4) spans 8 bytes.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp4.h>

namespace sglang {

using namespace device;

// ---- fp4/e4m3 decode -------------------------------------------------------

__device__ __forceinline__ float2 deq_fp4x2(uint32_t byte) {
  const auto storage = static_cast<__nv_fp4x2_storage_t>(byte);
  const __half2 h = __half2(__nv_cvt_fp4x2_to_halfraw2(storage, __NV_E2M1));
  return __half22float2(h);
}

// Decode to a raw __half2 (kept packed in registers). e2m1 values are exactly
// representable in f16 and f16->f32 widening is exact, so unpacking at USE
// time is bit-identical to converting at decode time — the decoded weights
// just occupy half the registers until consumed.
__device__ __forceinline__ __half2 deq_fp4x2_h2(uint32_t byte) {
  const auto storage = static_cast<__nv_fp4x2_storage_t>(byte);
  return __half2(__nv_cvt_fp4x2_to_halfraw2(storage, __NV_E2M1));
}

__device__ __forceinline__ float deq_e4m3(uint8_t b) {
  return float(*reinterpret_cast<const __nv_fp8_e4m3*>(&b));
}

// Decode one 32-bit word = 8 fp4 values into out[0..7].
__device__ __forceinline__ void deq_fp4_word(uint32_t word, float* out) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = deq_fp4x2((word >> (8 * i)) & 0xFFu);
    out[2 * i] = f.x;
    out[2 * i + 1] = f.y;
  }
}

// ---- split-K main kernel (v7) ----------------------------------------------
//
//   a_q      [KCAP, K_BYTES]    u8   quantized activation rows (grouped+padded,
//                                    rows >= Mp stay zero)
//   a_sf     [128, K_BYTES/8]   u8   swizzled activation SF (serving layout)
//   w_q      [E, N, K_BYTES]    u8   packed fp4 weights (streamed, __ldcs)
//   w_sf     [E, N, K_BYTES/8]  u8   native swizzled weight SF
//   m_indptr [E+1] i32               per-expert padded row segments (full-E)
//   active_ids [MAXA] i32           expert ids, -1 = skip
//   partials [SLICES, KCAP, N]  f32  preallocated+zeroed pool
//
// One warp owns TWO output rows. Per active expert the segment's packed
// activation slice is staged into smem in SEG-row chunks (lane-local, one
// barrier per chunk); both rows' weights are decoded ONCE per chunk into the
// j-major smem plane s_wf[rr][j][lane] (each lane then reads its own values
// at bank == lane, conflict-free). Per row the lane decodes its activation
// window into af[NV] registers and runs the contract dot.

template <uint32_t N, uint32_t K_BYTES, uint32_t KCAP, uint32_t GRP_CAP, uint32_t N_TILE, uint32_t WARPS,
          uint32_t SLICES, uint32_t MAXA>
__global__ void __launch_bounds__(WARPS * 32) decode_nvfp4_gemv_split_kernel(
    const uint8_t* __restrict__ a_q,
    const uint8_t* __restrict__ a_sf,
    const uint8_t* __restrict__ w_q,
    const uint8_t* __restrict__ w_sf,
    const int32_t* __restrict__ m_indptr,
    const int32_t* __restrict__ active_ids,
    float* __restrict__ partials) {
  constexpr uint32_t KPB = K_BYTES / 8;  // SF blocks per weight row (full K)
  constexpr uint32_t SB = K_BYTES / SLICES;   // packed bytes per slice
  constexpr uint32_t SPB = SB / 8;   // SF blocks per slice
  constexpr uint32_t VEC = SB / 32;  // packed bytes per lane per row (8 or 16)
  constexpr uint32_t NV = VEC * 2;   // fp4 values per lane per row (16 or 32)
  constexpr uint32_t BPL = VEC / 8;  // SF blocks per lane (1 or 2)
  constexpr uint32_t SEG = 4;        // staged activation rows per chunk
  static_assert(VEC == 8 || VEC == 16, "lane vector must be 8 or 16 bytes");
  static_assert(KCAP <= 128, "activation SF closed form assumes a single 128-row tile");
  static_assert(N_TILE == WARPS * 2, "each warp owns 2 output rows");
  static_assert(BPL <= 2, "SF u32 merge assumes both bytes share one word");

  __shared__ uint8_t s_a[SEG][SB];
  __shared__ uint8_t s_sa[SEG][SPB];

  const uint32_t tid = threadIdx.x;
  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t slice = blockIdx.y;
  const uint32_t kb0 = slice * SPB;  // first SF block of this slice
  const uint32_t r0 = blockIdx.x * N_TILE + warp * 2;

  // walk the compact active list (interspersed -1 sentinels are ~free:
  // two scalar loads skipped, no barrier on that path — uniform branch)
  for (uint32_t gi = 0; gi < MAXA; ++gi) {
    const int32_t e = active_ids[gi];
    if (e < 0) {
      continue;
    }
    const int32_t s0 = m_indptr[e];
    const int32_t rows = m_indptr[e + 1] - s0;
    const bool active = rows > 0 && rows <= static_cast<int32_t>(GRP_CAP);

    if (active) {
      for (int32_t mc = 0; mc < rows; mc += SEG) {
        const int32_t cr = min(rows - mc, static_cast<int32_t>(SEG));

        // decode BOTH rows' weights BEFORE the chunk barrier: weights/SF are
        // read-only kernel inputs (no smem involvement -> no race), so their
        // long-scoreboard latency overlaps the PREVIOUS chunk's compute
        // instead of serializing after the barrier. Kept packed as __half2
        // (half the f32 register footprint); f16->f32 widening at use time is
        // exact, so values are bit-identical to converting at decode time.
        __half2 wfpk[2][VEC];  // NV/2 pairs per row = VEC (2 values per packed byte)
        float sbf[2][BPL];
#pragma unroll
        for (uint32_t rr = 0; rr < 2; ++rr) {
          const uint32_t n = r0 + rr;
          const uint8_t* wp = w_q + (static_cast<size_t>(e) * N + n) * K_BYTES + slice * SB + lane * VEC;
          uint32_t wv[VEC / 4];
          if constexpr (VEC == 16) {
            *reinterpret_cast<uint4*>(&wv[0]) = __ldcs(reinterpret_cast<const uint4*>(wp));
          } else {
            *reinterpret_cast<uint2*>(&wv[0]) = __ldcs(reinterpret_cast<const uint2*>(wp));
          }
#pragma unroll
          for (uint32_t i = 0; i < VEC / 4; ++i) {
#pragma unroll
            for (uint32_t k = 0; k < 4; ++k) {
              wfpk[rr][i * 4 + k] = deq_fp4x2_h2((wv[i] >> (8 * k)) & 0xFFu);
            }
          }
          const uint32_t jj = n & 31;
          const uint32_t ii = (n & 127) >> 5;
          const uint32_t tile = n >> 7;
          // weight SF: the lane's BPL bytes are consecutive inside ONE aligned
          // u32 (kbl even for BPL==2), so one 4B load replaces BPL byte loads
          const uint8_t* sbase = w_sf + static_cast<size_t>(e) * N * KPB;
          const uint32_t kbl = kb0 + lane * BPL;
          const uint32_t off0 = tile * 128 * KPB + (kbl >> 2) * 512 + jj * 16 + ii * 4 + (kbl & 3);
          const uint32_t u = *reinterpret_cast<const uint32_t*>(sbase + (off0 & ~3u));
#pragma unroll
          for (uint32_t b = 0; b < BPL; ++b) {
            sbf[rr][b] = deq_e4m3(static_cast<uint8_t>(u >> (((off0 & 3) + b) * 8)));
          }
        }

        // s_a/s_sa are shared by all 8 warps: one barrier per chunk so no
        // warp restages (next chunk or next expert) while others still read.
        // NB the lockstep burst is load-bearing: 8 warps issue the SAME a_q
        // lines together -> L1 broadcast makes the redundant staging ~free
        // (decoupled planes measured SLOWER: tmp/ncu_B_nowarpbar_uniformT1).
        __syncthreads();

        // stage this chunk's packed fp4 activation slice + SF — each lane
        // stages exactly the bytes it will later read (lane-local)
#pragma unroll
        for (uint32_t mm = 0; mm < SEG; ++mm) {
          if (static_cast<int32_t>(mm) < cr) {
            if constexpr (VEC == 16) {
              *reinterpret_cast<uint4*>(&s_a[mm][lane * VEC]) =
                  *reinterpret_cast<const uint4*>(a_q + static_cast<size_t>(s0 + mc + mm) * K_BYTES + slice * SB + lane * VEC);
            } else {
              *reinterpret_cast<uint2*>(&s_a[mm][lane * VEC]) =
                  *reinterpret_cast<const uint2*>(a_q + static_cast<size_t>(s0 + mc + mm) * K_BYTES + slice * SB + lane * VEC);
            }
#pragma unroll
            for (uint32_t b = 0; b < BPL; ++b) {
              const uint32_t kb = kb0 + lane * BPL + b;
              const uint32_t row = static_cast<uint32_t>(s0 + mc + mm);
              s_sa[mm][lane * BPL + b] =
                  a_sf[(kb >> 2) * 512 + (row & 31) * 16 + ((row & 127) >> 5) * 4 + (kb & 3)];
            }
          }
        }

        // lanes hold disjoint K windows of the same (m, n) dot products:
        // per row pair, warp-shuffle-reduce over lanes (fixed butterfly order
        // -> deterministic; no cross-CTA atomics anywhere).
        //
        // m-2way unroll (v9): process two activation rows per iteration
        // against the SAME resident wfpk/sbf weight registers -> 4 independent
        // dot chains (vs 2) cover FMA latency while the warp sits at 2 CTA/SM.
        // Per-row accumulation order is UNCHANGED (b asc, p asc, x before y,
        // scale-fma last, same butterfly), so results stay bit-identical;
        // odd tails fall back to the 1-row body below.
        int32_t mm = 0;
        for (; mm + 1 < cr; mm += 2) {
          uint32_t av0[VEC / 4], av1[VEC / 4];
          if constexpr (VEC == 16) {
            *reinterpret_cast<uint4*>(&av0[0]) = *reinterpret_cast<const uint4*>(&s_a[mm][lane * VEC]);
            *reinterpret_cast<uint4*>(&av1[0]) = *reinterpret_cast<const uint4*>(&s_a[mm + 1][lane * VEC]);
          } else {
            *reinterpret_cast<uint2*>(&av0[0]) = *reinterpret_cast<const uint2*>(&s_a[mm][lane * VEC]);
            *reinterpret_cast<uint2*>(&av1[0]) = *reinterpret_cast<const uint2*>(&s_a[mm + 1][lane * VEC]);
          }
          float saf0[BPL], saf1[BPL];
#pragma unroll
          for (uint32_t b = 0; b < BPL; ++b) {
            saf0[b] = deq_e4m3(s_sa[mm][lane * BPL + b]);
            saf1[b] = deq_e4m3(s_sa[mm + 1][lane * BPL + b]);
          }
          float acc00 = 0.0f, acc10 = 0.0f, acc01 = 0.0f, acc11 = 0.0f;
#pragma unroll
          for (uint32_t b = 0; b < BPL; ++b) {
            float dot00 = 0.0f, dot10 = 0.0f, dot01 = 0.0f, dot11 = 0.0f;
#pragma unroll
            for (uint32_t p = 0; p < 8; ++p) {
              const float2 a0 = deq_fp4x2((av0[b * 2 + (p >> 2)] >> ((p & 3) * 8)) & 0xFFu);
              const float2 a1 = deq_fp4x2((av1[b * 2 + (p >> 2)] >> ((p & 3) * 8)) & 0xFFu);
              const float2 w0 = __half22float2(wfpk[0][b * 8 + p]);
              const float2 w1 = __half22float2(wfpk[1][b * 8 + p]);
              dot00 = fmaf(a0.x, w0.x, dot00);
              dot00 = fmaf(a0.y, w0.y, dot00);
              dot10 = fmaf(a0.x, w1.x, dot10);
              dot10 = fmaf(a0.y, w1.y, dot10);
              dot01 = fmaf(a1.x, w0.x, dot01);
              dot01 = fmaf(a1.y, w0.y, dot01);
              dot11 = fmaf(a1.x, w1.x, dot11);
              dot11 = fmaf(a1.y, w1.y, dot11);
            }
            acc00 = fmaf(saf0[b] * sbf[0][b], dot00, acc00);
            acc10 = fmaf(saf0[b] * sbf[1][b], dot10, acc10);
            acc01 = fmaf(saf1[b] * sbf[0][b], dot01, acc01);
            acc11 = fmaf(saf1[b] * sbf[1][b], dot11, acc11);
          }
#pragma unroll
          for (uint32_t off = 16; off > 0; off >>= 1) {
            acc00 += __shfl_down_sync(0xffffffffu, acc00, off);
            acc10 += __shfl_down_sync(0xffffffffu, acc10, off);
            acc01 += __shfl_down_sync(0xffffffffu, acc01, off);
            acc11 += __shfl_down_sync(0xffffffffu, acc11, off);
          }
          if (lane == 0) {
            partials[(static_cast<size_t>(slice) * KCAP + s0 + mc + mm) * N + r0] = acc00;
            partials[(static_cast<size_t>(slice) * KCAP + s0 + mc + mm) * N + r0 + 1] = acc10;
            partials[(static_cast<size_t>(slice) * KCAP + s0 + mc + mm + 1) * N + r0] = acc01;
            partials[(static_cast<size_t>(slice) * KCAP + s0 + mc + mm + 1) * N + r0 + 1] = acc11;
          }
        }
        for (; mm < cr; ++mm) {
          uint32_t av[VEC / 4];
          if constexpr (VEC == 16) {
            *reinterpret_cast<uint4*>(&av[0]) = *reinterpret_cast<const uint4*>(&s_a[mm][lane * VEC]);
          } else {
            *reinterpret_cast<uint2*>(&av[0]) = *reinterpret_cast<const uint2*>(&s_a[mm][lane * VEC]);
          }
          float saf[BPL];
#pragma unroll
          for (uint32_t b = 0; b < BPL; ++b) {
            saf[b] = deq_e4m3(s_sa[mm][lane * BPL + b]);
          }
          float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
          for (uint32_t b = 0; b < BPL; ++b) {
            float dot0 = 0.0f, dot1 = 0.0f;
#pragma unroll
            for (uint32_t p = 0; p < 8; ++p) {
              const float2 a = deq_fp4x2((av[b * 2 + (p >> 2)] >> ((p & 3) * 8)) & 0xFFu);
              const float2 w0 = __half22float2(wfpk[0][b * 8 + p]);
              const float2 w1 = __half22float2(wfpk[1][b * 8 + p]);
              dot0 = fmaf(a.x, w0.x, dot0);
              dot0 = fmaf(a.y, w0.y, dot0);
              dot1 = fmaf(a.x, w1.x, dot1);
              dot1 = fmaf(a.y, w1.y, dot1);
            }
            acc0 = fmaf(saf[b] * sbf[0][b], dot0, acc0);
            acc1 = fmaf(saf[b] * sbf[1][b], dot1, acc1);
          }
#pragma unroll
          for (uint32_t off = 16; off > 0; off >>= 1) {
            acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
            acc1 += __shfl_down_sync(0xffffffffu, acc1, off);
          }
          if (lane == 0) {
            partials[(static_cast<size_t>(slice) * KCAP + s0 + mc + mm) * N + r0] = acc0;
            partials[(static_cast<size_t>(slice) * KCAP + s0 + mc + mm) * N + r0 + 1] = acc1;
          }
        }
      }
    }
  }
}

// ---- deterministic reduce --------------------------------------------------
//
//   partials [SLICES, KCAP, N] f32 -> out [KCAP, N] bf16; fp32 slice sum in
//   fixed order, then per-row expert alpha, then a single bf16 cast
//   (contract: alpha last, never intermediate bf16 rounding).
//
// Rows with no segment (m >= Mp) have an all-zero partial column -> write 0.
template <uint32_t N, uint32_t KCAP, uint32_t SLICES, uint32_t THREADS>
__global__ void decode_nvfp4_gemv_reduce_kernel(
    const float* __restrict__ partials,
    const int32_t* __restrict__ m_indptr,
    const float* __restrict__ alphas,
    bf16_t* __restrict__ out,
    int32_t num_experts) {
  __shared__ int32_t s_mip[1025];
  const uint32_t tid = threadIdx.x;
  for (int32_t i = tid; i <= num_experts; i += THREADS) {
    s_mip[i] = m_indptr[i];
  }
  __syncthreads();
  static_assert(KCAP <= 1024);
  const uint32_t n = blockIdx.x * THREADS + tid;
  for (uint32_t m = 0; m < KCAP; ++m) {
    int32_t lo = 0, hi = num_experts;
    while (lo < hi) {
      const int32_t mid = (lo + hi) >> 1;
      if (s_mip[mid + 1] <= static_cast<int32_t>(m)) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    const bool found = lo < num_experts && s_mip[lo] <= static_cast<int32_t>(m);
    float acc = 0.0f;
#pragma unroll
    for (uint32_t s = 0; s < SLICES; ++s) {
      acc += partials[(static_cast<size_t>(s) * KCAP + m) * N + n];
    }
    const float alpha = found ? alphas[lo] : 0.0f;
    out[static_cast<size_t>(m) * N + n] = cast<bf16_t>(acc * alpha);
  }
}

// ---- host wrappers ---------------------------------------------------------

template <uint32_t N, uint32_t K_BYTES, uint32_t KCAP, uint32_t GRP_CAP, uint32_t N_TILE, uint32_t WARPS,
          uint32_t SLICES, uint32_t MAXA>
struct DecodeNvfp4GemvSplitKernel {
  static void
  run(const tvm::ffi::TensorView a_q,
      const tvm::ffi::TensorView a_sf,
      const tvm::ffi::TensorView w_q,
      const tvm::ffi::TensorView w_sf,
      const tvm::ffi::TensorView m_indptr,
      const tvm::ffi::TensorView active_ids,
      const tvm::ffi::TensorView partials) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto E = SymbolicSize{"num_experts"};
    auto E1 = SymbolicSize{"num_experts_plus_one"};
    device.set_options<kDLCUDA>();
    TensorMatcher({KCAP, K_BYTES}).with_dtype<uint8_t>().with_device(device).verify(a_q);
    TensorMatcher({128, K_BYTES / 8}).with_dtype<uint8_t>().with_device(device).verify(a_sf);
    TensorMatcher({E, N, K_BYTES}).with_dtype<uint8_t>().with_device(device).verify(w_q);
    TensorMatcher({E, N, K_BYTES / 8}).with_dtype<uint8_t>().with_device(device).verify(w_sf);
    TensorMatcher({E1}).with_dtype<int32_t>().with_device(device).verify(m_indptr);
    TensorMatcher({MAXA}).with_dtype<int32_t>().with_device(device).verify(active_ids);
    TensorMatcher({SLICES, KCAP, N}).with_dtype<float>().with_device(device).verify(partials);

    const int32_t num_experts = static_cast<int32_t>(E.unwrap());
    RuntimeCheck(static_cast<int64_t>(num_experts) + 1 == E1.unwrap(), "m_indptr must be [E+1]");
    LaunchKernel(dim3(N / N_TILE, SLICES), WARPS * 32, device.unwrap())(
        decode_nvfp4_gemv_split_kernel<N, K_BYTES, KCAP, GRP_CAP, N_TILE, WARPS, SLICES, MAXA>,
        static_cast<const uint8_t*>(a_q.data_ptr()),
        static_cast<const uint8_t*>(a_sf.data_ptr()),
        static_cast<const uint8_t*>(w_q.data_ptr()),
        static_cast<const uint8_t*>(w_sf.data_ptr()),
        static_cast<const int32_t*>(m_indptr.data_ptr()),
        static_cast<const int32_t*>(active_ids.data_ptr()),
        static_cast<float*>(partials.data_ptr()));
  }
};

template <uint32_t N, uint32_t KCAP, uint32_t SLICES, uint32_t THREADS = 256>
struct DecodeNvfp4GemvReduceKernel {
  static void
  run(const tvm::ffi::TensorView partials,
      const tvm::ffi::TensorView m_indptr,
      const tvm::ffi::TensorView alphas,
      const tvm::ffi::TensorView out) {
    using namespace host;
    auto device = SymbolicDevice{};
    auto E = SymbolicSize{"num_experts"};
    auto E1 = SymbolicSize{"num_experts_plus_one"};
    device.set_options<kDLCUDA>();
    TensorMatcher({SLICES, KCAP, N}).with_dtype<float>().with_device(device).verify(partials);
    TensorMatcher({E1}).with_dtype<int32_t>().with_device(device).verify(m_indptr);
    TensorMatcher({E}).with_dtype<float>().with_device(device).verify(alphas);
    TensorMatcher({KCAP, N}).with_dtype<bf16_t>().with_device(device).verify(out);

    const int32_t num_experts = static_cast<int32_t>(E.unwrap());
    RuntimeCheck(static_cast<int64_t>(num_experts) + 1 == E1.unwrap(), "m_indptr must be [E+1]");
    RuntimeCheck(num_experts > 0 && num_experts <= 1024, "num_experts out of range: ", num_experts);
    LaunchKernel(dim3(N / THREADS), THREADS, device.unwrap())(
        decode_nvfp4_gemv_reduce_kernel<N, KCAP, SLICES, THREADS>,
        static_cast<const float*>(partials.data_ptr()),
        static_cast<const int32_t*>(m_indptr.data_ptr()),
        static_cast<const float*>(alphas.data_ptr()),
        static_cast<bf16_t*>(out.data_ptr()),
        num_experts);
  }
};

}  // namespace sglang
