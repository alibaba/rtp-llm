"""Exercise the native packed-row boundary and exact E2M1/E8M0 outputs."""

import unittest

import torch

from rtp_llm.ops import rtp_llm_ops


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class SiluMulMxfp4Test(unittest.TestCase):
    def test_rejects_misaligned_packed_rows(self):
        for width in (0, 2, 34, 36, 60):
            with self.subTest(width=width):
                x = torch.zeros((2, width), device="cuda", dtype=torch.bfloat16)
                with self.assertRaisesRegex(RuntimeError, "H must be a multiple of 16"):
                    rtp_llm_ops.ppu_silu_and_mul_post_quant_mxfp4(x)

    def test_aligned_rows_and_partial_blocks_have_exact_quantized_values(self):
        # SiLU(16) rounds to 16 in BF16: scale 4, E2M1 value 4 (nibble 6).
        # Clamping both inputs to 1 gives SiLU(1): scale 1/8, value 6 (nibble 7).
        for hidden in (16, 80, 256, 512):
            for limit, packed_byte, exponent in ((None, 0x66, 129), (1.0, 0x77, 124)):
                with self.subTest(hidden=hidden, limit=limit):
                    x = torch.ones((2, 2 * hidden), device="cuda", dtype=torch.bfloat16)
                    x[:, :hidden] = 16
                    packed, scales = rtp_llm_ops.ppu_silu_and_mul_post_quant_mxfp4(
                        x, limit
                    )
                    self.assertEqual(packed.shape, (2, hidden // 2))
                    self.assertTrue(bool((packed == packed_byte).all()))
                    self.assertEqual(scales.shape, (2, (hidden + 63) // 64))
                    # Each uint16 contains two consecutive 32-element scales.
                    scale_words = scales.cpu().to(torch.int32)
                    scale_bytes = torch.stack(
                        (scale_words & 255, scale_words >> 8), dim=-1
                    ).reshape(2, -1)
                    valid_groups = (hidden + 31) // 32
                    self.assertTrue(
                        bool((scale_bytes[:, :valid_groups] == exponent).all())
                    )

    def test_storage_offset_alignment(self):
        for offset in (1, 8):
            with self.subTest(offset=offset):
                raw = torch.ones(1024 + offset, device="cuda", dtype=torch.bfloat16)
                x = raw[offset:].view(2, 512)
                x[:, :256] = 16
                self.assertTrue(x.is_contiguous())
                self.assertEqual(x.storage_offset(), offset)
                if offset == 1:
                    with self.assertRaisesRegex(RuntimeError, "aligned to 16 bytes"):
                        rtp_llm_ops.ppu_silu_and_mul_post_quant_mxfp4(x)
                else:
                    packed, scales = rtp_llm_ops.ppu_silu_and_mul_post_quant_mxfp4(x)
                    self.assertTrue(bool((packed == 0x66).all()))
                    self.assertTrue(bool((scales.to(torch.int32) == 0x8181).all()))

    def test_empty_batch_retains_layout(self):
        x = torch.empty((0, 160), device="cuda", dtype=torch.bfloat16)
        packed, scales = rtp_llm_ops.ppu_silu_and_mul_post_quant_mxfp4(x)
        self.assertEqual(packed.shape, (0, 40))
        self.assertEqual(scales.shape, (0, 2))

    def test_padded_launcher_dimension_rejected_without_large_allocation(self):
        x = torch.empty((0, 2 * 2147483632), device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "padded hidden size exceeds"):
            rtp_llm_ops.ppu_silu_and_mul_post_quant_mxfp4(x)


if __name__ == "__main__":
    unittest.main()
