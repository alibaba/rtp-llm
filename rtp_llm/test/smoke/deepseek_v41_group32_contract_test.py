"""Native group-32 quantization/GEMM plus the existing V4.1 HTTP contract.

V4.1 is not registered in NewLoader: this guards the shared FP8 helper used
by its existing runtime, without claiming NewLoader model support.
"""

import unittest

import torch

from rtp_llm.test.smoke import deepseek_v41_contract_test as contract


class DeepSeekV41Group32Smoke(contract.DeepSeekV41ContractTest):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertIsNone(torch.version.hip)
        self.assertEqual(torch.cuda.get_device_capability()[0], 10)
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        # Use the actual V4.1 consumer, including packed UE8M0 allocation and
        # native GEMM. Distinct per-group amplitudes expose group-size/stride
        # mistakes that an all-ones input would conceal.
        generator = torch.Generator(device="cuda").manual_seed(37)
        weight = torch.randn(128, 128, generator=generator, device="cuda").to(
            torch.float8_e4m3fn
        )
        scales = torch.ones(4, 4, device="cuda").to(torch.float8_e8m0fnu)
        linear = V41MXFP8Linear(weight, scales)
        for tokens in (1, 5, 8):
            with self.subTest(tokens=tokens):
                x = torch.randn(
                    tokens,
                    128,
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                x *= torch.tensor([0.25, 1, 4, 16], device="cuda").repeat_interleave(32)
                quantized, packed = linear._quantize_input(x)
                self.assertEqual(quantized.shape, x.shape)
                self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
                self.assertEqual(packed.dtype, torch.int32)
                self.assertEqual(tuple(packed.shape), (tokens, 1))
                # Independently derive per-group power-of-two scales and FP8
                # round-trip values; do not reuse production packing helpers.
                blocks = x.float().reshape(tokens, 4, 32)
                expected_scale = torch.pow(
                    2.0, torch.ceil(torch.log2(blocks.abs().amax(-1) / 448.0))
                )
                shifts = torch.arange(4, device="cuda", dtype=torch.int64) * 8
                exponent = (packed.to(torch.int64) >> shifts) & 255
                actual_scale = torch.pow(2.0, exponent.float() - 127)
                torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
                reference_q = (
                    (blocks / expected_scale.unsqueeze(-1))
                    .to(torch.float8_e4m3fn)
                    .float()
                )
                torch.testing.assert_close(
                    quantized.float().reshape_as(blocks), reference_q, rtol=0, atol=0
                )
                reference = (reference_q * expected_scale.unsqueeze(-1)).reshape(
                    tokens, 128
                ) @ weight.float().T
                actual = linear.forward_quantized(quantized, packed)
                torch.testing.assert_close(
                    actual.float(), reference, rtol=2e-2, atol=0.5
                )
                self.assertTrue(torch.isfinite(actual).all().item())
        # Real checkpoint, HTTP generation, graph replay, and prefix-reuse
        # assertions from the established V4.1 service smoke.
        super().setUp()


if __name__ == "__main__":
    unittest.main()
