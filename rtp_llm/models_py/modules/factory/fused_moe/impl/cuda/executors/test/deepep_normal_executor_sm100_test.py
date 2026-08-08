import platform
import unittest

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.deepep_normal_executor_test import (
    DeepGemmHybridExecutorTestBase,
)


class DeepGemmHybridExecutorSM100Test(
    DeepGemmHybridExecutorTestBase, unittest.TestCase
):
    # SM100 uses FP4 for w13 (Gate/Up GEMM), which has lower precision than FP8
    DIFF_THRESHOLD = 0.01

    def test_sm100_arm(self):
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())
        self.assertTrue("aarch64" in platform.machine())


if __name__ == "__main__":
    unittest.main()
