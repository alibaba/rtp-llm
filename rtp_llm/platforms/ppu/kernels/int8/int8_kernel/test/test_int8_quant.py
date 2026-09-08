import unittest

import torch
from rtp_llm.platforms.ppu.kernels.int8.int8_kernel.int8_quant import (
    per_token_quant_int8,
)


class PerTokenQuantInt8Test(unittest.TestCase):
    def test_rejects_rank_one_input(self):
        with self.assertRaisesRegex(ValueError, "ndim >= 2"):
            per_token_quant_int8(torch.ones(8))

    def test_empty_2d_preserves_contract(self):
        quantized, scales = per_token_quant_int8(torch.empty((0, 128)))
        self.assertEqual(quantized.shape, (0, 128))
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(scales.shape, (0, 1))
        self.assertEqual(scales.dtype, torch.float32)

    def test_empty_3d_preserves_contract(self):
        quantized, scales = per_token_quant_int8(torch.empty((2, 0, 64)))
        self.assertEqual(quantized.shape, (2, 0, 64))
        self.assertEqual(scales.shape, (2, 0, 1))

    @unittest.skipUnless(torch.cuda.is_available(), "requires PPU/CUDA")
    def test_constant_rows(self):
        self._assert_quantization(torch.full((2, 128), 2.0, device="cuda"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires PPU/CUDA")
    def test_irregular_width(self):
        self._assert_quantization(self._input((3, 257)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires PPU/CUDA")
    def test_three_dimensional_input(self):
        self._assert_quantization(self._input((2, 3, 512)))

    @unittest.skipUnless(torch.cuda.is_available(), "requires PPU/CUDA")
    def test_noncontiguous_input(self):
        value = self._input((4, 256)).transpose(0, 1)
        self.assertFalse(value.is_contiguous())
        self._assert_quantization(value)

    @unittest.skipUnless(torch.cuda.is_available(), "requires PPU/CUDA")
    def test_large_block_uses_bounded_warp_count(self):
        self._assert_quantization(self._input((2, 4096)))

    @staticmethod
    def _input(shape):
        values = torch.arange(
            1,
            torch.tensor(shape).prod().item() + 1,
            device="cuda",
            dtype=torch.float32,
        ).reshape(shape)
        return ((values % 251) - 125).to(torch.bfloat16)

    def _assert_quantization(self, value):
        quantized, scales = per_token_quant_int8(value)
        value_fp32 = value.float()
        absmax = value_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
        expected_scales = absmax / 127.0
        expected_quantized = torch.round(
            torch.clamp(value_fp32 / expected_scales, -128.0, 127.0)
        ).to(torch.int8)

        self.assertEqual(quantized.shape, value.shape)
        self.assertEqual(quantized.dtype, torch.int8)
        self.assertEqual(scales.shape, (*value.shape[:-1], 1))
        self.assertEqual(scales.dtype, torch.float32)
        torch.testing.assert_close(scales, expected_scales)
        torch.testing.assert_close(quantized, expected_quantized)


if __name__ == "__main__":
    unittest.main()
