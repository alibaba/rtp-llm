import unittest

import pytest

try:
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
        has_deep_gemm,
        is_deep_gemm_e8m0_used,
    )
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.test.fp8_linear_test import (
        CudaFp8GEMMLinearTestBase,
    )
except (ImportError, RuntimeError) as e:
    pytest.skip(f"deepgemm unavailable: {e}", allow_module_level=True)

pytestmark = [pytest.mark.gpu(type="SM100")]


class CudaFp8DeepGEMMLinearSM100Test(CudaFp8GEMMLinearTestBase, unittest.TestCase):
    def test_sm100(self):
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())


if __name__ == "__main__":
    unittest.main()
