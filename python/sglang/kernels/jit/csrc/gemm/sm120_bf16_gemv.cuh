// SM120/SM121 bf16 GEMV:  out[M, N] = X[M, K] @ W[N, K]^T
//
// Native-CUDA port of sglang's Hopper bf16 GEMV (csrc/gemm/hopper_bf16_gemv.cuh)
// to the SM120/SM121 (consumer/workstation + GB10 Grace-Blackwell) generation,
// extended so each warp computes kRows output rows for all M activation rows,
// streaming every weight element exactly once and reusing it across M tokens.
//
// Decode is weight-streaming bound: the weight (N*K*2 bytes) dwarfs the
// activations (M*K*2) and the output (M*N*2). For M=1 cuBLAS picks a GEMV that
// is near the DRAM ceiling on most N, but for large N (lm_head) and tiny N it
// leaves bandwidth on the table. For M>1 the only way to stay at one weight
// pass is to reuse each loaded weight across the M rows before discarding it.
//
// The activation tile is staged in shared memory when M*K*2 <= 48KB; larger
// tiles (M>=5, K>=4096) read x directly, relying on L2 for the (tiny) reuse.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

using namespace device;

constexpr uint32_t kGemvVecSize = 16 / sizeof(bf16_t);  // 8 bf16 per 16B load

__device__ __forceinline__ float dot8_bf16_f32(const float4 wv, const float4 xv) {
  const bf16x2_t* w2 = reinterpret_cast<const bf16x2_t*>(&wv);
  const bf16x2_t* x2 = reinterpret_cast<const bf16x2_t*>(&xv);
  float acc = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const auto [w0, w1] = cast<fp32x2_t>(w2[i]);
    const auto [x0, x1] = cast<fp32x2_t>(x2[i]);
    acc = fmaf(w0, x0, acc);
    acc = fmaf(w1, x1, acc);
  }
  return acc;
}

// M-row bf16 GEMV. One warp owns kRows output rows; every warp re-reads the
// activation tile (staged once in smem, or via L2 when it does not fit) and
// streams weights with evict-first loads. Reduction is a register + warp-shuffle
// tree, so there is no split-K fixup kernel.
template <uint32_t N, uint32_t K, uint32_t M, uint32_t kRows, uint32_t kUnroll, uint32_t kNumWarps,
          bool kStageSmem>
__global__ void __launch_bounds__(kNumWarps * 32) sm120_bf16_gemv_kernel(
    bf16_t* __restrict__ out,
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ w) {
  __shared__ bf16_t sx[kStageSmem ? M * K : 1];

  const uint32_t tid = threadIdx.x;
  if constexpr (kStageSmem) {
    for (uint32_t i = tid * kGemvVecSize; i < M * K; i += kNumWarps * 32 * kGemvVecSize) {
      *reinterpret_cast<float4*>(sx + i) = *reinterpret_cast<const float4*>(x + i);
    }
    __syncthreads();
  }

  const uint32_t warp = tid / 32;
  const uint32_t lane = tid % 32;
  const uint32_t r0 = (blockIdx.x * kNumWarps + warp) * kRows;

  float acc[kRows][M];
#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t m = 0; m < M; ++m) {
      acc[r][m] = 0.0f;
    }
  }
  if (r0 >= N) {
    return;
  }

  constexpr uint32_t kStep = 32 * kGemvVecSize * kUnroll;
  const bool full = (r0 + kRows <= N);
  for (uint32_t k = lane * kGemvVecSize * kUnroll; k < K; k += kStep) {
    float4 xv[M][kUnroll];
#pragma unroll
    for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
      for (uint32_t u = 0; u < kUnroll; ++u) {
        const bf16_t* xr = kStageSmem ? (sx + static_cast<size_t>(m) * K + k + u * kGemvVecSize)
                                      : (x + static_cast<size_t>(m) * K + k + u * kGemvVecSize);
        xv[m][u] = *reinterpret_cast<const float4*>(xr);
      }
    }
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (full || (r0 + r < N)) {
        const bf16_t* wr = w + static_cast<size_t>(r0 + r) * K + k;
        float4 wv[kUnroll];
#pragma unroll
        for (uint32_t u = 0; u < kUnroll; ++u) {
          wv[u] = __ldcs(reinterpret_cast<const float4*>(wr + u * kGemvVecSize));
        }
#pragma unroll
        for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
          for (uint32_t u = 0; u < kUnroll; ++u) {
            acc[r][m] += dot8_bf16_f32(wv[u], xv[m][u]);
          }
        }
      }
    }
  }

#pragma unroll
  for (uint32_t r = 0; r < kRows; ++r) {
#pragma unroll
    for (uint32_t off = 16; off > 0; off >>= 1) {
#pragma unroll
      for (uint32_t m = 0; m < M; ++m) {
        acc[r][m] += __shfl_down_sync(0xffffffff, acc[r][m], off);
      }
    }
  }
  if (lane == 0) {
#pragma unroll
    for (uint32_t r = 0; r < kRows; ++r) {
      if (r0 + r < N) {
#pragma unroll
        for (uint32_t m = 0; m < M; ++m) {
          out[static_cast<size_t>(m) * N + r0 + r] = cast<bf16_t>(acc[r][m]);
        }
      }
    }
  }
}

template <uint32_t N, uint32_t K, uint32_t M, uint32_t kRows, uint32_t kUnroll, uint32_t kNumWarps,
          bool kStageSmem>
struct Sm120Bf16GemvKernel {
  static_assert(K % (32 * kGemvVecSize * kUnroll) == 0, "K must cover full unrolled warp strides");
  static_assert(!kStageSmem || M * K * sizeof(bf16_t) <= 48 * 1024,
                "staged activation tile must fit static shared memory");

  static void run(const tvm::ffi::TensorView x, const tvm::ffi::TensorView w, const tvm::ffi::TensorView out) {
    using namespace host;

    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({N, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({M, N}).with_dtype<bf16_t>().with_device(device).verify(out);

    constexpr uint32_t kRowsPerBlock = kRows * kNumWarps;
    constexpr uint32_t kNumBlocks = (N + kRowsPerBlock - 1) / kRowsPerBlock;
    LaunchKernel(kNumBlocks, kNumWarps * 32, device.unwrap())(
        sm120_bf16_gemv_kernel<N, K, M, kRows, kUnroll, kNumWarps, kStageSmem>,
        static_cast<bf16_t*>(out.data_ptr()),
        static_cast<const bf16_t*>(x.data_ptr()),
        static_cast<const bf16_t*>(w.data_ptr()));
  }
};

}  // namespace sglang
