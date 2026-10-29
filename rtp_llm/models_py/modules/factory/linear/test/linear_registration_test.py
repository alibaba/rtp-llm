import unittest

import torch

from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig, ModelOptFp4Config
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
)


class LinearRegistrationTest(unittest.TestCase):
    def test_hybrid_fp4_attention_accepts_fp8_backend(self):
        config = ModelOptFp4Config(
            bits=4,
            group_size=16,
            is_quanted=True,
            hybrid_attn_quant_method="FP8_PER_BLOCK",
        )
        weight = torch.zeros(128, 128).to(torch.float8_e4m3fn)
        scales = torch.ones(1, 1)
        self.assertTrue(CudaFp8GEMMLinear.can_handle(config, weight, scales))
        self.assertFalse(CudaF16Linear.can_handle(config, weight, scales))
        for weight_scale_2, input_scale in (
            (torch.ones(1), None),
            (None, torch.ones(1)),
        ):
            self.assertFalse(
                CudaFp8GEMMLinear.can_handle(
                    config,
                    weight,
                    scales,
                    weight_scale_2=weight_scale_2,
                    input_scale=input_scale,
                )
            )
        plain_fp4 = ModelOptFp4Config(bits=4, group_size=16, is_quanted=True)
        self.assertFalse(CudaFp8GEMMLinear.can_handle(plain_fp4, weight, scales))

    def test_unscaled_fp32_projection_keeps_nonquantized_backend(self):
        self.assertIn(CudaF16Linear, LinearFactory._strategies)
        weight = torch.arange(12, dtype=torch.float32).view(4, 3)
        layer = LinearFactory.create_linear(
            weight, None, None, Fp8BlockWiseQuantConfig()
        )
        self.assertIsInstance(layer, CudaF16Linear)
        actual = layer(torch.ones(2, 4))
        torch.testing.assert_close(actual, weight.sum(0).expand(2, 3), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
