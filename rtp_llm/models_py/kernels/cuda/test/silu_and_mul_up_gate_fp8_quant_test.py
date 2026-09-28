import os
from unittest import SkipTest, TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    sgl_per_token_group_quant_fp8,
    silu_and_mul_up_gate_fp8_quant,
)
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul


class SiluAndMulUpGateFp8QuantTest(TestCase):
    """Focused SM120 contract tests for contiguous routed MoE's [up | gate] ABI."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is required")
        if torch.cuda.get_device_capability() != (12, 0):
            raise SkipTest("this test is dedicated to SM120")

    @staticmethod
    def _old_chain(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        down_input = torch.empty(
            (x.size(0), x.size(1) // 2), device=x.device, dtype=torch.bfloat16
        )
        silu_and_mul(down_input, x)
        return sgl_per_token_group_quant_fp8(
            down_input,
            group_size=128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )

    def test_i256_matches_the_existing_v2_chain_and_packs_two_scales(self) -> None:
        # M=5 exercises the TMA-aligned row stride while I=256 has exactly two
        # 128-element groups packed into the low two bytes of one int32 scale.
        torch.manual_seed(20260928)
        x = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)
        with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "v2"}):
            old_q, old_s = self._old_chain(x)
            fused_q, fused_s = silu_and_mul_up_gate_fp8_quant(x)

        self.assertEqual(fused_q.shape, (5, 256))
        self.assertEqual(fused_s.shape, (5, 1))
        self.assertTrue(torch.equal(fused_q, old_q))
        self.assertTrue(torch.equal(fused_s, old_s))

        packed = fused_s.to(torch.int64)
        self.assertTrue(torch.all((packed >> 16) == 0).item())
        self.assertTrue(torch.all((packed & 0xFFFF) != 0).item())

    def test_prefill_sizes_preserve_quantized_values_and_scales(self) -> None:
        torch.manual_seed(19)
        for rows in (131, 4093, 32768):
            for magnitude in (1.0, 8.0):
                with self.subTest(rows=rows, magnitude=magnitude), patch.dict(
                    os.environ, {"DSV4_FP8_QUANT_KERNEL": "v2"}
                ):
                    x = (
                        torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
                        * magnitude
                    )
                    old_q, old_s = self._old_chain(x)
                    fused_q, fused_s = silu_and_mul_up_gate_fp8_quant(x)
                    self.assertTrue(torch.equal(fused_q, old_q))
                    self.assertTrue(torch.equal(fused_s, old_s))

    def test_auto_uses_the_same_v2_fused_route(self) -> None:
        torch.manual_seed(9)
        x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16)
        with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "v2"}):
            v2_q, v2_s = silu_and_mul_up_gate_fp8_quant(x)
        with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "auto"}):
            auto_q, auto_s = silu_and_mul_up_gate_fp8_quant(x)

        self.assertTrue(torch.equal(auto_q, v2_q))
        self.assertTrue(torch.equal(auto_s, v2_s))

    def test_legacy_backend_is_rejected_before_allocating_a_narrow_output(self) -> None:
        x = torch.ones(1, 512, device="cuda", dtype=torch.bfloat16)
        with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": "legacy"}):
            with self.assertRaisesRegex(ValueError, "does not support"):
                silu_and_mul_up_gate_fp8_quant(x)

    def test_helper_rejects_noncontiguous_and_wrong_width(self) -> None:
        x = torch.empty(512, 5, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            silu_and_mul_up_gate_fp8_quant(x)
        with self.assertRaisesRegex(ValueError, r"2 \* group_size"):
            silu_and_mul_up_gate_fp8_quant(
                torch.empty(1, 384, device="cuda", dtype=torch.bfloat16)
            )


if __name__ == "__main__":
    main()
