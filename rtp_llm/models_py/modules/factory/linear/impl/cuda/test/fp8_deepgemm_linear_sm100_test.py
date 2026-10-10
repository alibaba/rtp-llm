import unittest

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.test import (
    fp8_deepgemm_linear_sm120_test as online_tests,
)


class CudaFp8DeepGEMMLinearSM100Test(
    online_tests.OnlineFp8LoaderTestBase, unittest.TestCase
):
    def test_sm100(self):
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())


class OnlineLinearAttentionTPTest(online_tests.OnlineLinearAttentionTPTest):
    pass


if __name__ == "__main__":
    unittest.main()
