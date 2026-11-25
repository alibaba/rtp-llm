import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
        for hw in (None, MagicMock()):
            with self.subTest(hw=hw):
                layer = LinearFactory.create_linear(
                    weight, None, None, Fp8BlockWiseQuantConfig(), hw
                )
                self.assertIs(type(layer), CudaF16Linear)
                actual = layer(torch.ones(2, 4))
                torch.testing.assert_close(
                    actual, weight.sum(0).expand(2, 3), rtol=0, atol=0
                )

    def test_batch_invariant_layer_uses_fixed_order_operator(self):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda import batch_invariant

        weight = torch.arange(12, dtype=torch.bfloat16).view(4, 3)
        bias = torch.ones(3, dtype=weight.dtype)
        x = torch.ones(2, 4, dtype=weight.dtype)
        layer = LinearFactory.create_linear(
            weight, bias, None, None, SimpleNamespace(batch_invariant=True)
        )
        self.assertIsInstance(layer, batch_invariant.BatchInvariantLinear)
        layer.maybe_cache_quant_scale(1024)
        with patch.object(
            batch_invariant, "batch_invariant_linear", wraps=torch.nn.functional.linear
        ) as op:
            actual = layer(x)
        op.assert_called_once_with(x, layer.weight, bias)
        torch.testing.assert_close(actual, (weight.sum(0) + bias).expand(2, 3))

    def test_batch_invariant_rejects_bias_without_a_shared_output_row(self):
        from rtp_llm.models_py.determinism.batch_invariant import batch_invariant_linear

        x = torch.empty(0, 4, dtype=torch.bfloat16)
        weight = torch.empty(3, 4, dtype=x.dtype)
        for shape in ((), (1,), (3,), (1, 3)):
            with self.subTest(shape=shape):
                bias = torch.ones(shape, dtype=x.dtype)
                result = batch_invariant_linear(x, weight, bias)
                self.assertEqual(result.shape, (0, 3))
        for shape in ((2,), (2, 3)):
            with self.subTest(shape=shape), self.assertRaises(RuntimeError):
                batch_invariant_linear(x, weight, torch.ones(shape, dtype=x.dtype))


if __name__ == "__main__":
    unittest.main()
