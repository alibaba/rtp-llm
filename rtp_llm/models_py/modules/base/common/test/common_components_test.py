"""Public CPU contracts independent of model registration."""

import sys
import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
    ZeroCenteredFusedQKRMSNorm,
    exact_head_rms_norm,
)


class CommonComponentsTest(unittest.TestCase):
    def test_fp8_fallback_supports_non_model_shapes_bias_and_batches(self):
        torch.manual_seed(42)
        for out_width, in_width in ((7, 13), (11, 17)):
            weight = torch.randn(out_width, in_width).to(torch.float8_e4m3fn)
            scale = torch.tensor([0.125])
            bias = torch.randn(out_width, dtype=torch.bfloat16)
            linear = Fp8Linear(weight, scale, bias)
            for rows in (0, 3, 5):
                x = torch.randn(2, rows, in_width, dtype=torch.bfloat16)
                expected = F.linear(x, weight.bfloat16() * scale.bfloat16(), bias)
                torch.testing.assert_close(linear(x), expected, rtol=0, atol=0)

    def test_zero_centered_gain_and_fused_value_preservation(self):
        for heads, kv_heads, dim in ((3, 1, 7), (4, 2, 11)):
            source = torch.randn(5, (heads + 2 * kv_heads) * dim)
            q_weight = torch.full((dim,), -1.0)
            k_weight = torch.zeros(dim)
            module = ZeroCenteredFusedQKRMSNorm(
                q_weight, k_weight, heads, kv_heads, dim
            )
            actual = source.clone()
            self.assertEqual(module(actual).data_ptr(), actual.data_ptr())
            torch.testing.assert_close(
                actual[:, : heads * dim], torch.zeros_like(actual[:, : heads * dim])
            )
            torch.testing.assert_close(
                actual[:, (heads + kv_heads) * dim :],
                source[:, (heads + kv_heads) * dim :],
                rtol=0,
                atol=0,
            )
            self.assertTrue(torch.isfinite(actual).all())

    def test_zero_input_remains_finite(self):
        source = torch.zeros(2, 3, 17, dtype=torch.bfloat16)
        result = exact_head_rms_norm(source, torch.zeros(17), 1e-6)
        torch.testing.assert_close(result, source, rtol=0, atol=0)

    def test_public_imports_do_not_load_qwen4_modules(self):
        # Run this target in a fresh process; the wrappers must not need model setup.
        loaded = [name for name in sys.modules if ".qwen4_exp" in name]
        self.assertEqual(loaded, [])


if __name__ == "__main__":
    unittest.main()
