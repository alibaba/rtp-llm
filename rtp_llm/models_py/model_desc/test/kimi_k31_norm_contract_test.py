"""Exercise the shared RMSNorm contract consumed by natural K3.1 DSA."""

import unittest

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_fp8_quant_triton import (
    rmsnorm_fp8_quant_ue8m0,
)


class KimiK31NormContractTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "requires the CUDA CI worker")
        torch.manual_seed(311)
        self.input = torch.randn((7, 512), dtype=torch.bfloat16, device="cuda")
        self.weight = torch.linspace(0.5, 1.5, 512, dtype=torch.bfloat16, device="cuda")

    def test_fp8_only_projection_can_omit_normalized_output(self):
        _, reference_values, reference_scales = rmsnorm_fp8_quant_ue8m0(
            self.input, self.weight, eps=1e-6
        )
        normalized, values, scales = rmsnorm_fp8_quant_ue8m0(
            self.input, self.weight, eps=1e-6, store_norm=False
        )
        self.assertIsNone(normalized)
        torch.testing.assert_close(
            values.view(torch.uint8), reference_values.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(scales, reference_scales, rtol=0, atol=0)

    def test_joint_fp8_nvfp4_projection_publishes_normalized_amax(self):
        partials = torch.full((2,), float("nan"), dtype=torch.float32, device="cuda")
        normalized, _, _ = rmsnorm_fp8_quant_ue8m0(
            self.input, self.weight, eps=1e-6, amax_partials=partials
        )
        reference = F.rms_norm(
            self.input.float(), (512,), self.weight.float(), eps=1e-6
        ).bfloat16()
        torch.testing.assert_close(normalized, reference, rtol=0.01, atol=0.02)
        self.assertTrue(torch.isfinite(partials).all().item())
        expected_partials = torch.stack(
            [normalized[start : start + 4].abs().amax().float() for start in (0, 4)]
        )
        torch.testing.assert_close(partials, expected_partials, rtol=0, atol=0)

    def test_partial_contract_rejects_invalid_storage(self):
        for invalid in (
            torch.empty(2, dtype=torch.bfloat16, device="cuda"),
            torch.empty(2, dtype=torch.float32),
            torch.empty(1, dtype=torch.float32, device="cuda"),
            torch.empty(4, dtype=torch.float32, device="cuda")[::2],
        ):
            with self.subTest(
                dtype=invalid.dtype, device=invalid.device, shape=invalid.shape
            ):
                with self.assertRaisesRegex(ValueError, "amax_partials"):
                    rmsnorm_fp8_quant_ue8m0(
                        self.input, self.weight, eps=1e-6, amax_partials=invalid
                    )

    def test_empty_input_needs_no_partial_initialization(self):
        partials = torch.empty(0, dtype=torch.float32, device="cuda")
        normalized, values, scales = rmsnorm_fp8_quant_ue8m0(
            self.input[:0],
            self.weight,
            eps=1e-6,
            store_norm=False,
            amax_partials=partials,
        )
        self.assertIsNone(normalized)
        self.assertEqual(values.shape, (0, 512))
        self.assertEqual(scales.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
