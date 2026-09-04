import unittest
from unittest import mock

import torch

from rtp_llm.models_py.layers.norm import RMSNorm, RMSResNorm


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is not None,
    "requires ROCm",
)
class RocmOddRmsNormTest(unittest.TestCase):
    def test_rmsnorm_odd_hidden_sizes_use_opus_and_match_reference(self):
        import aiter

        for hidden_size in (769, 771):
            with self.subTest(hidden_size=hidden_size):
                layer = RMSNorm(hidden_size, params_dtype=torch.bfloat16).cuda()
                inputs = torch.randn(
                    7, hidden_size, dtype=torch.bfloat16, device="cuda"
                )
                inputs_fp32 = inputs.float()
                expected = (
                    layer.weight.float()
                    * inputs_fp32
                    * torch.rsqrt(inputs_fp32.pow(2).mean(-1, keepdim=True) + layer.eps)
                ).to(inputs.dtype)

                with mock.patch.object(
                    aiter,
                    "rmsnorm2d_fwd_opus",
                    wraps=aiter.rmsnorm2d_fwd_opus,
                ) as opus, mock.patch.object(
                    aiter,
                    "rms_norm",
                    side_effect=AssertionError("unsafe RMSNorm path selected"),
                ):
                    output = layer(inputs)

                opus.assert_called_once()
                torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)

    def test_rmsresnorm_odd_hidden_sizes_use_opus_and_match_reference(self):
        import aiter

        for hidden_size in (769, 771):
            with self.subTest(hidden_size=hidden_size):
                layer = RMSResNorm(hidden_size, params_dtype=torch.bfloat16).cuda()
                hidden_states = torch.randn(
                    7, hidden_size, dtype=torch.bfloat16, device="cuda"
                )
                residual = torch.randn_like(hidden_states)
                expected_residual = hidden_states + residual
                residual_fp32 = expected_residual.float()
                expected_output = (
                    layer.weight.float()
                    * residual_fp32
                    * torch.rsqrt(
                        residual_fp32.pow(2).mean(-1, keepdim=True) + layer.eps
                    )
                ).to(hidden_states.dtype)

                with mock.patch.object(
                    aiter,
                    "rmsnorm2d_fwd_with_add_opus",
                    wraps=aiter.rmsnorm2d_fwd_with_add_opus,
                ) as opus, mock.patch.object(
                    aiter,
                    "rmsnorm2d_fwd_with_add",
                    side_effect=AssertionError("unsafe RMSResNorm path selected"),
                ):
                    output, residual_out = layer(hidden_states, residual)

                opus.assert_called_once()
                torch.testing.assert_close(
                    output, expected_output, rtol=2e-2, atol=2e-2
                )
                torch.testing.assert_close(
                    residual_out, expected_residual, rtol=2e-2, atol=2e-2
                )


if __name__ == "__main__":
    unittest.main()
