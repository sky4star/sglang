"""
Tests the SM120/SM121 bf16 GEMV JIT kernel against torch (cuBLAS + fp32
reference) on the dispatch domains where the backend enables it.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-large")


def _is_sm12x() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12


@unittest.skipIf(not _is_sm12x(), "SM120/SM121 bf16 GEMV requires an SM12x GPU")
class TestSm120Bf16Gemv(unittest.TestCase):
    def _run_case(self, m, n, k, seed=0):
        from sglang.kernels.ops.gemm.sm120_bf16_gemv import sm120_bf16_gemv

        torch.manual_seed(seed)
        x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") * 0.05
        out = sm120_bf16_gemv(x, w)
        self.assertEqual(tuple(out.shape), (m, n))
        ref = x.float() @ w.float().t()
        cub = (x @ w.t()).float()
        err = (out.float() - ref).abs().max().item()
        err_cub = (cub - ref).abs().max().item()
        # fp32 accumulation + single warp-tree reduction: at least as tight as
        # cuBLAS against the fp32 reference.
        self.assertLessEqual(err, max(err_cub * 2.0, 1e-2), (m, n, k, err, err_cub))
        self.assertFalse(torch.isnan(out).any().item(), (m, n, k))

    def test_dispatch_domain_shapes(self):
        # GLM-5.3-Flash-NVFP4 bf16 decode shapes (attention proj, dense MLP,
        # lm_head) plus DSpark verify/draft rows.
        for m, n, k in [
            (1, 77440, 4096),  # lm_head
            (5, 12576, 4096),  # dense MLP gate_up
            (5, 2048, 4096),  # down proj / shared expert
            (5, 4096, 4096),  # o_proj
            (5, 8192, 1536),  # q_b_proj
            (8, 4096, 8192),
            (8, 20480, 4096),  # draft projection
            (8, 4096, 24576),  # draft fc
            (5, 128, 4096),  # narrow attention proj
            (5, 288, 4096),  # router
            (5, 32, 4096),  # indexer
        ]:
            self._run_case(m, n, k)

    def test_tail_rows(self):
        # N not divisible by rows_per_block exercises the guarded tail path.
        for n in [104, 2056, 288]:
            self._run_case(5, n, 4096)

    def test_3d_input(self):
        from sglang.kernels.ops.gemm.sm120_bf16_gemv import sm120_bf16_gemv

        torch.manual_seed(0)
        x = torch.randn(2, 5, 4096, dtype=torch.bfloat16, device="cuda") * 0.1
        w = torch.randn(4096, 4096, dtype=torch.bfloat16, device="cuda") * 0.05
        out = sm120_bf16_gemv(x, w)
        self.assertEqual(tuple(out.shape), (2, 5, 4096))
        ref = x.reshape(-1, 4096).float() @ w.float().t()
        self.assertLessEqual((out.reshape(-1, 4096).float() - ref).abs().max().item(), 1e-1)

    def test_predicate(self):
        from sglang.kernels.ops.gemm.sm120_bf16_gemv import use_sm120_bf16_gemv

        # production decode shapes
        self.assertTrue(use_sm120_bf16_gemv(1, 77440, 4096))
        self.assertTrue(use_sm120_bf16_gemv(5, 12576, 4096))
        self.assertTrue(use_sm120_bf16_gemv(8, 4096, 8192))
        self.assertTrue(use_sm120_bf16_gemv(5, 32, 4096))
        # fall back: too many rows, odd K, huge N, and the cuBLAS-win narrow-N
        # high-M bucket.
        self.assertFalse(use_sm120_bf16_gemv(9, 4096, 4096))
        self.assertFalse(use_sm120_bf16_gemv(1, 4096, 6000))
        self.assertFalse(use_sm120_bf16_gemv(1, 200000, 4096))
        self.assertFalse(use_sm120_bf16_gemv(8, 128, 4096))


if __name__ == "__main__":
    unittest.main()
