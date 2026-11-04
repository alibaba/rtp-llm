"""Tests for the f4f4 (FP4xFP4) MXFP4 scale preprocessing in the PPU MoE.

The DeepEP device path quantizes activations with the sglang-for-sail
``downcast_to_mxfp4`` triton kernel and packs weight scales with
``preprocess_mxfp4_f4f4_scales`` (the torch mirror of the PPU deep_gemm wheel's
``preprocess_mxfp4_scales``). The triton kernel / GEMM themselves are
device-only; these tests lock the CPU-testable layout and packing math.

Byte-pairing convention (little-endian ``view(torch.uint16)``): the EVEN
group-32 scale byte becomes the low byte of the uint16 pair, the ODD group
the high byte — the same packing the sglang triton kernel writes
(``lo | (hi << 8)``) and the wheel reads.
"""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.ppu.moe import (
    preprocess_mxfp4_f4f4_scales,
)


def _pair_bytes(out: torch.Tensor) -> torch.Tensor:
    """``[E, N, S/2]`` uint16 N-major pairs -> ``[E, N, S]`` exponent bytes."""
    e, n, s2 = out.shape
    b = (
        out.permute(0, 2, 1)
        .contiguous()
        .view(torch.uint8)
        .reshape(e, s2, n, 2)
        .permute(0, 2, 1, 3)
        .reshape(e, n, s2 * 2)
    )
    return b


class PreprocessF4F4ScalesTest(unittest.TestCase):
    def test_layout_shape_stride_dtype(self):
        # V4.1 Flash w13 scales: [E, 2I, K/32] = [8, 4608, 160]
        torch.manual_seed(601)
        scales = torch.randint(118, 138, (8, 4608, 160), dtype=torch.uint8)
        out = preprocess_mxfp4_f4f4_scales(scales)
        self.assertEqual(tuple(out.shape), (8, 4608, 80))
        self.assertEqual(out.dtype, torch.uint16)
        # N-major layout asserted by m_grouped_gemm_fp4_fp4_bf16_nt_nopad.
        self.assertEqual(out.stride(1), 1)
        self.assertEqual(out.stride(2), 4608)

    def test_byte_pairing_even_group_is_low_byte(self):
        torch.manual_seed(602)
        scales = torch.randint(0, 256, (2, 6, 4), dtype=torch.uint8)
        out = preprocess_mxfp4_f4f4_scales(scales)
        self.assertEqual(tuple(out.shape), (2, 6, 2))
        pairs = out.permute(0, 2, 1).contiguous()  # [E, S/2, N] row-major
        for e in range(2):
            for s in range(2):
                for n in range(6):
                    lo = int(scales[e, n, 2 * s])
                    hi = int(scales[e, n, 2 * s + 1])
                    self.assertEqual(
                        int(pairs[e, s, n]),
                        lo | (hi << 8),
                        f"pair (e={e}, s={s}, n={n}) must pack even group low",
                    )

    def test_is_a_byte_permutation(self):
        torch.manual_seed(603)
        scales = torch.randint(0, 256, (3, 64, 6), dtype=torch.uint8)
        out = preprocess_mxfp4_f4f4_scales(scales)
        self.assertEqual(
            sorted(_pair_bytes(out).flatten().tolist()),
            sorted(scales.flatten().tolist()),
        )

    def test_odd_group_count_zero_pads(self):
        scales = torch.randint(0, 256, (1, 8, 3), dtype=torch.uint8)
        out = preprocess_mxfp4_f4f4_scales(scales)
        self.assertEqual(tuple(out.shape), (1, 8, 2))
        b = _pair_bytes(out)
        self.assertTrue(torch.equal(b[..., :3], scales))
        self.assertTrue((b[..., 3] == 0).all())

    def test_e8m0fnu_input_viewed_as_bytes(self):
        torch.manual_seed(604)
        raw = torch.randint(118, 138, (2, 16, 4), dtype=torch.uint8)
        out_u8 = preprocess_mxfp4_f4f4_scales(raw)
        out_f8 = preprocess_mxfp4_f4f4_scales(raw.view(torch.float8_e8m0fnu))
        self.assertTrue(torch.equal(out_u8, out_f8))

    def test_non_contiguous_input(self):
        torch.manual_seed(605)
        base = torch.randint(0, 256, (2, 16, 8), dtype=torch.uint8)
        scales = base[:, :, ::2]  # non-contiguous [2, 16, 4]
        self.assertFalse(scales.is_contiguous())
        out = preprocess_mxfp4_f4f4_scales(scales)
        self.assertTrue(torch.equal(_pair_bytes(out), scales.contiguous()))

    def test_rejects_bad_rank_and_dtype(self):
        with self.assertRaises(ValueError):
            preprocess_mxfp4_f4f4_scales(torch.zeros(4, 4, dtype=torch.uint8))
        with self.assertRaises(TypeError):
            preprocess_mxfp4_f4f4_scales(torch.zeros(2, 4, 4, dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
