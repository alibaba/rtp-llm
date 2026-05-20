import platform
import unittest

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.test.fp8_linear_test import (
    CudaFp8GEMMLinearTestBase,
)


class CudaFp8DeepGEMMLinearSM100Test(CudaFp8GEMMLinearTestBase, unittest.TestCase):
    def test_sm100_arm(self):
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())
        self.assertTrue("aarch64" in platform.machine())


if __name__ == "__main__":
    unittest.main()
