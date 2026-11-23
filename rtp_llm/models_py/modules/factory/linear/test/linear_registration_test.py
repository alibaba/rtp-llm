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

    def test_bias_gelu_falls_back_for_supported_linear_inputs(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for dtype, shape, bias_stride in (
                (torch.float32, (5, 4), 1),
                (torch.float16, (2, 5, 4), 1),
                (torch.float16, (5, 4), 2),
            ):
                with self.subTest(
                    device=device, dtype=dtype, shape=shape, stride=bias_stride
                ):
                    value = torch.randn(shape, dtype=dtype, device=device)
                    weight = torch.randn(4, 8, dtype=dtype, device=device)
                    bias = torch.randn(8 * bias_stride, dtype=dtype, device=device)[
                        ::bias_stride
                    ]
                    layer = CudaF16Linear(weight, bias=bias)
                    expected = torch.nn.functional.gelu(layer(value))
                    torch.testing.assert_close(
                        layer.forward_with_bias_gelu(value), expected, rtol=0, atol=0
                    )

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
