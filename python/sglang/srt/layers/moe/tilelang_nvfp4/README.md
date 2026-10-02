# TileLang NVFP4 block-scaled fused MoE (SM120a / GB10)

A reference routed MoE implemented entirely in TileLang's SM120 block-scaled
path, targeting NVIDIA GB10 (DGX Spark, sm_121a). It **beats the production
`flashinfer.fused_moe.cutlass_fused_moe` (TRT-LLM CUTLASS) baseline by ~1.16x**
at M=4096..16384, with exact numerics.

This is a standalone reference kernel package, not yet wired into the SGLang
`moe_runner` backends.

## Structure

Five kernels (vs ~5 in the CUDA baseline, but with different fusion):

| kernel | file | role |
|---|---|---|
| K0 | `act_quant_kernel.py` | gather tokens by expert + NVFP4 block-16 quant |
| K1 | `w13_swiglu_kernel.py` | grouped w13 with **gate+up dual block-scaled MMA** + fused SwiGLU(limit) |
| K2 | `act_quant_kernel.py` | requant activated intermediate to NVFP4 |
| K3 | `w2_gemm_kernel.py` | grouped w2 block-scaled GEMM, expert-ordered output |
| C | `moe_combine_kernel.py` | gather-reduction finalize: `y[t] = sum_s out2[slot_rows[t,s]] * scale` |

`fused_moe.py` wires the pipeline and validates against a dequant-operand oracle.
`grouped_nvfp4_gemm.py` holds the base grouped GEMM and the scale-layout helpers.

The combine is a **gather-reduction, not an atomic scatter**: each token's
`topk` rows are gathered and summed (no contention), which hits the memory
floor. A fused atomic-scatter variant measured ~2 ms slower and is not included.

## Requirements

- **tilelang >= 0.1.15** (adds the SM120a NVF4 block-scaled MMA:
  `mma.sync.aligned.m16n8k64...kind::mxf4nvf4.block_scale.scale_vec::4X...ue4m3`,
  `T.mma_gemm_blockscaled`, `sf_layout="blockscaled_chunk_kmajor"`).
- CUDA >= 12.9, target `sm_121a`.
- A one-line tilelang fix, since the guard only allows `SM120A` while GB10 is
  `SM121A`: apply `patches/mma_block_scale_sm121a.patch` to
  `src/tl_templates/cuda/instruction/mma_block_scale.h`.
- `patches/sm120_nvfp4_blockscaled_gemm.py` is the upstream dense SM120 NVFP4
  example (used to derive the scale-word packing).

## Run

```bash
# apply the tilelang guard patch, then:
cd python/sglang/srt/layers/moe/tilelang_nvfp4
python fused_moe.py --M 4096 --E 144 --K 4096 --N 2048
```

## Result (GB10, E=144/rank, topk=8, K=4096, N=2048)

| kernel | M=4096 (ms) |
|---|---:|
| K0 gather+quant | 0.76 |
| K1 w13+SwiGLU | 7.49 |
| K2 requant | 0.90 |
| K3 w2 GEMM | 4.70 |
| C combine | 1.46 |
| **total** | **15.31** |
| `cutlass_fused_moe` | 17.73 |
| **speedup** | **1.16x** |

Correctness: exact against a dequant-operand oracle (`max_rel = 0`).

| M | TileLang | baseline | speedup |
|---:|---:|---:|---:|
| 4096 | 15.31 | 17.73 | 1.16x |
| 8192 | 24.04 | 28.20 | 1.17x |
| 16384 | 44.18 | 51.76 | 1.17x |

## Notes / caveats

- The grouped NVFP4 GEMMs are weight/latency bound; the win comes from fusion
  (gather+quant, w13+SwiGLU, gather-reduction combine), not from the GEMMs.
- Disabling TileLang warp specialization is required for the dual-accumulator
  kernel (otherwise it spills `~1.2e8` times).
- Quant convention here is absolute E4M3 block-16 scales (`sf = E4M3(amax/6)`);
  parity with the CUDA baseline's activation-quant convention is not done here.
- Cold-L2 / CUDA-graph A/B and `moe_runner` integration are out of scope for
  this reference package.
