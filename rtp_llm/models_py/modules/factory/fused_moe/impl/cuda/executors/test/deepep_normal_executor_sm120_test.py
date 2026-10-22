import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_mk_alignment,
    get_theoretical_mk_alignment_for_contiguous_layout,
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    quant_weight_ue8m0_packed,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.deepep_normal_executor_test import (
    DeepGemmHybridExecutorQwen35ShapeTestBase,
    DeepGemmHybridExecutorTestBase,
)
from rtp_llm.models_py.utils.arch import is_sm12x


class DeepGemmHybridExecutorSM120Test(
    DeepGemmHybridExecutorTestBase, unittest.TestCase
):
    def test_sm120(self):
        self.assertTrue(is_sm12x())
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())

    def test_deepep_normal_executor_cuda_graph(self):
        self._run_deepep_normal_executor(enable_cuda_graph=True)

    def test_empty_local_experts_cuda_graph(self):
        self._run_deepep_normal_executor(
            empty_local_experts=True, enable_cuda_graph=True
        )

    def test_eager_releases_scattered_fp8_input(self):
        original_execute = self._execute

        def execute_and_check(executor, payload, enable_cuda_graph):
            self.assertGreater(payload.expert_x.numel(), 0)
            result = original_execute(executor, payload, enable_cuda_graph)
            self.assertEqual(payload.expert_x.untyped_storage().nbytes(), 0)
            return result

        with patch.object(self, "_execute", side_effect=execute_and_check):
            self._run_deepep_normal_executor()

    def test_deepgemm_negative_padding_eager_and_graph(self):
        # Exercise the actual vendor boundary: partial expert tiles, an empty
        # expert, and whole unused tiles in the graph workspace keep -1 IDs.
        groups, n, k = 3, 256, 512
        alignment = min(
            128, get_theoretical_mk_alignment_for_contiguous_layout(19, groups)
        )
        m = 5 * alignment
        inputs = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        weights = torch.randn((groups, n, k), device="cuda", dtype=torch.bfloat16)
        a = sgl_per_token_group_quant_fp8(
            inputs,
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        b_parts = [quant_weight_ue8m0_packed(w) for w in weights]
        b = (
            torch.stack([part[0] for part in b_parts]),
            torch.stack([part[1].T for part in b_parts]).transpose(-1, -2),
        )
        indices = torch.full((m,), -1, device="cuda", dtype=torch.int32)
        indices[:7] = 0
        indices[alignment : alignment + 12] = 2
        original_indices = indices.clone()
        valid = indices >= 0
        reference = torch.cat(
            [
                inputs[:7].float() @ weights[0].float().T,
                inputs[alignment : alignment + 12].float() @ weights[2].float().T,
            ]
        )
        output = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)

        def run():
            with configure_deep_gemm_mk_alignment(alignment):
                m_grouped_fp8_gemm_nt_contiguous(
                    a, b, output, indices, disable_ue8m0_cast=False
                )

        run()
        for capture in (False, True):
            with self.subTest(cuda_graph=capture):
                if capture:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    graph.replay()
                else:
                    run()
                torch.cuda.synchronize()
                actual = output[valid].float()
                self.assertTrue(torch.isfinite(actual).all())
                self.assertLess((actual - reference).norm() / reference.norm(), 0.06)
                self.assertTrue(torch.equal(indices, original_indices))


class DeepGemmHybridExecutorQwen35ShapeSM120Test(
    DeepGemmHybridExecutorQwen35ShapeTestBase, unittest.TestCase
):
    pass


if __name__ == "__main__":
    unittest.main()
