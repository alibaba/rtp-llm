import unittest

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


if __name__ == "__main__":
    unittest.main()
