import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
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


class DeepGemmHybridExecutorQwen35ShapeSM120Test(
    DeepGemmHybridExecutorQwen35ShapeTestBase, unittest.TestCase
):
    pass


class DeepGemmHybridMaskedQwenSM120Test(
    DeepGemmHybridExecutorTestBase, unittest.TestCase
):
    # Keep routed tokens below the masked threshold; exercise the actual hybrid
    # dispatch with the six-scale Qwen shape, including gather/combine.
    MAX_GENERATE_BATCH_SIZE = 1
    M = 4
    # The safe ordinary FP8 pipeline has ~0.00322 error against unquantized
    # weights for this small shape. Check fused-vs-ordinary EXACTLY below too.
    DIFF_THRESHOLD = 0.004

    def _execute(self, executor, payload, enable_cuda_graph):
        self.assertFalse(enable_cuda_graph)
        self.assertLessEqual(payload.expert_x.shape[0], executor.masked_max_token_num)
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            pack_ue8m0_kernel_launcher,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
            deepgemm_hybrid_executor,
        )
        from rtp_llm.models_py.triton_kernels.common.activation import (
            silu_mul_masked_fp8_post_quant_fwd,
        )

        def ordinary_packed(x, out, scales, group_size, counts):
            # A real, safe reference pipeline injected only at the producer
            # boundary. Routing, both GEMMs and gather still execute normally.
            sf = torch.zeros((*x.shape[:2], x.shape[-1] // 256), device=x.device)
            silu_mul_masked_fp8_post_quant_fwd(
                x, out, sf, group_size, counts, 1, scale_ue8m0=True
            )
            scales.copy_(pack_ue8m0_kernel_launcher(sf, 1))

        with patch.object(
            deepgemm_hybrid_executor,
            "silu_mul_masked_packed_sm120",
            side_effect=ordinary_packed,
        ) as reference_producer:
            reference = (
                super()
                ._execute(executor, payload, enable_cuda_graph)
                .fused_expert_output.clone()
            )
            reference_producer.assert_called_once()
        with patch.object(
            deepgemm_hybrid_executor,
            "silu_mul_masked_packed_sm120",
            wraps=deepgemm_hybrid_executor.silu_mul_masked_packed_sm120,
        ) as fused_producer:
            actual = super()._execute(executor, payload, enable_cuda_graph)
            fused_producer.assert_called_once()
        torch.testing.assert_close(
            actual.fused_expert_output, reference, rtol=0, atol=0
        )
        return actual

    def test_empty_local_experts_eager(self):
        # The inherited test explicitly selects the contiguous empty fast path.
        self.skipTest("Covered by the contiguous executor tests")


if __name__ == "__main__":
    unittest.main()
