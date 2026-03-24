import platform
import unittest

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.deepgemm_masked_executor_test import (
    DeepGemmMaskedExecutorTestBase,
)


class DeepGemmMaskedExecutorSM100Test(
    DeepGemmMaskedExecutorTestBase, unittest.TestCase
):
    def test_sm100_arm(self):
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())
        self.assertTrue("aarch64" in platform.machine())

    def test_no_fp8(self):
        # sm100 not support bf16
        pass

    def test_fp8(self):
        self._test_deepgemm_masked_executor(True)


if __name__ == "__main__":
    unittest.main()
