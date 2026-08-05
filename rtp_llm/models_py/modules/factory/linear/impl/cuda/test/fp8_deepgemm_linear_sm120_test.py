import unittest

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.test.fp8_linear_test import (
    CudaFp8GEMMLinearTestBase,
    init_quant_config,
)
from rtp_llm.models_py.utils.arch import is_sm12x


class CudaFp8DeepGEMMLinearSM120Test(CudaFp8GEMMLinearTestBase, unittest.TestCase):
    def test_sm120(self):
        self.assertTrue(is_sm12x())
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())

    def test_factory_dispatch_matches_other_cuda_arches(self):
        self.assertTrue(
            CudaFp8GEMMLinear.can_handle(
                init_quant_config("FP8_PER_BLOCK"),
                self.weight,
                self.weight_scales,
            )
        )


if __name__ == "__main__":
    unittest.main()
