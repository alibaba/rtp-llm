import platform
import unittest

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.deepep_normal_executor_test import (
    DeepGemmContinousExecutorTestBase,
)


class DeepGemmContinousExecutorSM100Test(
    DeepGemmContinousExecutorTestBase, unittest.TestCase
):
    def test_sm100_arm(self):
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())
        self.assertTrue("aarch64" in platform.machine())


if __name__ == "__main__":
    unittest.main()
