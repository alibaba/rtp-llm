import unittest

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.test.deepgemm_masked_executor_test import (
    DeepGemmMaskedExecutorTestBase,
)
from rtp_llm.models_py.utils.arch import is_sm12x


class DeepGemmMaskedExecutorSM120Test(
    DeepGemmMaskedExecutorTestBase, unittest.TestCase
):
    def test_sm120(self):
        self.assertTrue(is_sm12x())
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())

    def test_no_fp8(self):
        self.skipTest("SM120 DeepGEMM only supports the FP8/UE8M0 test path")

    def test_fp8(self):
        self._test_deepgemm_masked_executor(True)


if __name__ == "__main__":
    unittest.main()
