"""Tests for the PPU V4.1 dense-linear module (CPU-runnable).

Covers:
  * ``dequant_fp8_block32_ue8m0`` vs the ``test_mxfp8_dense.py`` oracle
    formula (exact, including e8m0fnu / uint8 / float scale inputs).
  * ``PPUResidentBF16Linear`` forward vs manual ``F.linear`` and the
    ``from_quantized`` descriptor path.
"""

import os
import unittest

import torch

from rtp_llm.models_py.modules.dsv41.ppu.linear import (
    PPUResidentBF16Linear,
    dequant_fp8_block32_ue8m0,
)

# Reference FP8 32x32 UE8M0 dequant oracle, copied from
# rtp_llm/models_py/modules/dsv41/test/test_mxfp8_dense.py:104-106.
# w_bf16 = (w.float() * scales.float()
#           .repeat_interleave(32, 0).repeat_interleave(32, 1)).to(bf16)


def _oracle(w, scale_values):
    return (
        w.float() * scale_values.repeat_interleave(32, 0).repeat_interleave(32, 1)
    ).to(torch.bfloat16)


def _random_e4m3(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=g) * 0.5).to(torch.float8_e4m3fn)


def _random_e8m0(rows, cols, seed, lo=118, hi=136):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(lo, hi, (rows, cols), generator=g)
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )


class DequantFP8Block32Test(unittest.TestCase):
    def test_matches_mxfp8_dense_oracle_exact(self):
        torch.manual_seed(101)
        for n, k in ((64, 96), (256, 512), (33 * 32, 31 * 32)):
            with self.subTest(shape=(n, k)):
                w = _random_e4m3(n, k, seed=7)
                scales = _random_e8m0(n // 32, k // 32, seed=8)
                scale_values = torch.exp2(scales.view(torch.uint8).float() - 127.0)
                actual = dequant_fp8_block32_ue8m0(w, scales)
                expected = _oracle(w, scale_values)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_accepts_uint8_scale_bytes(self):
        w = _random_e4m3(64, 64, seed=11)
        bytes_ = torch.randint(118, 136, (2, 2), dtype=torch.int64).to(torch.uint8)
        actual = dequant_fp8_block32_ue8m0(w, bytes_)
        scale_values = torch.exp2(bytes_.to(torch.float32) - 127.0)
        torch.testing.assert_close(actual, _oracle(w, scale_values), rtol=0, atol=0)

    def test_accepts_float_scales(self):
        w = _random_e4m3(64, 64, seed=12)
        bytes_ = torch.randint(118, 136, (2, 2), dtype=torch.int64).to(torch.uint8)
        scale_values = torch.exp2(bytes_.to(torch.float32) - 127.0)
        actual = dequant_fp8_block32_ue8m0(w, scale_values)
        torch.testing.assert_close(actual, _oracle(w, scale_values), rtol=0, atol=0)

    def test_scale_value_is_power_of_two_of_byte_minus_127(self):
        w = torch.ones(32, 32).to(torch.float8_e4m3fn)
        for byte, expected_value in (
            (127, 1.0),
            (130, 8.0),
            (120, 2.0**-7),
            (100, 2.0**-27),
        ):
            with self.subTest(byte=byte):
                scales = torch.full((1, 1), byte, dtype=torch.int64).to(torch.uint8)
                out = dequant_fp8_block32_ue8m0(w, scales).float()
                torch.testing.assert_close(
                    out, torch.full_like(out, expected_value), rtol=0, atol=0
                )

    def test_rejects_mismatched_scale_shape(self):
        w = _random_e4m3(64, 64, seed=13)
        bad = _random_e8m0(3, 2, seed=14)  # expected (2, 2)
        with self.assertRaises(ValueError):
            dequant_fp8_block32_ue8m0(w, bad)
        with self.assertRaises(ValueError):
            dequant_fp8_block32_ue8m0(w.unsqueeze(0), _random_e8m0(2, 2, seed=15))


class PPUResidentBF16LinearTest(unittest.TestCase):
    def test_fp32_output_preserves_projection_before_bf16_rounding(self):
        x = torch.tensor([[[1.0, 1.0]]], dtype=torch.bfloat16)
        weight = torch.tensor([[1.0, 1.0 / 512]], dtype=torch.bfloat16)
        linear = PPUResidentBF16Linear(weight, output_dtype=torch.float32)
        output = linear(x)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output.item(), 1.0 + 1.0 / 512)
        self.assertEqual(PPUResidentBF16Linear(weight)(x).item(), 1.0)
        empty = linear(x[:, :0])
        self.assertEqual(empty.shape, (1, 0, 1))
        self.assertEqual(empty.dtype, torch.float32)

    def test_forward_matches_manual(self):
        torch.manual_seed(201)
        n, k = 96, 64
        w = torch.randn(n, k, dtype=torch.bfloat16)
        b = torch.randn(n, dtype=torch.bfloat16)
        x = torch.randn(17, k, dtype=torch.bfloat16)
        linear = PPUResidentBF16Linear(w, b)
        torch.testing.assert_close(
            linear(x), torch.nn.functional.linear(x, w, b), rtol=0, atol=0
        )
        linear_nb = PPUResidentBF16Linear(w)
        torch.testing.assert_close(
            linear_nb(x), torch.nn.functional.linear(x, w), rtol=0, atol=0
        )

    def test_forward_handles_leading_dims(self):
        torch.manual_seed(202)
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
        linear = PPUResidentBF16Linear(w)
        torch.testing.assert_close(
            linear(x), torch.nn.functional.linear(x, w), rtol=0, atol=0
        )

    def test_from_quantized_fp8_matches_dequant(self):
        torch.manual_seed(203)
        w = _random_e4m3(64, 96, seed=21)
        scales = _random_e8m0(2, 3, seed=22)
        linear = PPUResidentBF16Linear.from_quantized(
            w, scales, format="fp8_32x32_ue8m0"
        )
        expected_weight = dequant_fp8_block32_ue8m0(w, scales)
        torch.testing.assert_close(linear.weight, expected_weight, rtol=0, atol=0)
        x = torch.randn(5, 96, dtype=torch.bfloat16)
        torch.testing.assert_close(
            linear(x),
            torch.nn.functional.linear(x, expected_weight),
            rtol=0,
            atol=0,
        )

    def test_from_quantized_bf16_passthrough(self):
        w = torch.randn(8, 16, dtype=torch.bfloat16)
        linear = PPUResidentBF16Linear.from_quantized(w, None, format="bf16")
        self.assertIs(linear.weight, w)

    def test_from_quantized_rejects_fp4(self):
        w = torch.zeros(8, 8, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            PPUResidentBF16Linear.from_quantized(w, None, format="fp4_ue8m0")

    def test_rejects_non_bf16_weight(self):
        with self.assertRaises(TypeError):
            PPUResidentBF16Linear(torch.randn(8, 8, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
