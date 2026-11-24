import os
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._wo_a_layout import prepare_wo_a_stacked


class V4WoALayoutTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(os.environ.get("V4_LAYOUT_TEST_DEVICE", "cpu"))

    def test_all_exponent_bytes_row_grouping_and_k_tail(self):
        # 33 block columns exercise the K tail; vary both group and scale
        # row, including the 0 (subnormal) and 255 (NaN) UE8M0 encodings.
        g, r, blocks = 4, 256, 33
        weight = torch.empty(
            (g * r, blocks * 128), device=self.device, dtype=torch.float8_e4m3fn
        )
        data = torch.arange(g * r // 128 * blocks).to(torch.uint8).reshape(-1, blocks)
        # A source view with odd storage offset and non-unit last stride must
        # work; the original checkpoint bytes need not be int32-viewable.
        owner = torch.empty(
            (data.shape[0], blocks * 2), device=self.device, dtype=torch.uint8
        )
        raw = owner[:, 1::2]
        raw.copy_(data)
        w, packed = prepare_wo_a_stacked(
            weight, raw.view(torch.float8_e8m0fnu), g, r, blocks * 128
        )
        self.assertEqual(w.data_ptr(), weight.data_ptr())
        self.assertEqual(tuple(packed.shape), (g, r, 9))
        self.assertEqual(packed.stride(), (9 * r, 1, r))
        self.assertEqual(packed.dtype, torch.int32)
        got = packed.cpu().contiguous().view(torch.uint8)
        expected = data.repeat_interleave(128, dim=0).reshape(g, r, blocks)
        torch.testing.assert_close(got[..., :blocks], expected, rtol=0, atol=0)
        self.assertEqual(got[..., blocks:].count_nonzero().item(), 0)

    def test_already_packed_row_scales_are_not_reinterpreted_as_floats(self):
        g, r, k = 2, 128, 1024
        weight = torch.empty((g * r, k), device=self.device, dtype=torch.float8_e4m3fn)
        raw = (
            torch.arange(g * r * 2, dtype=torch.int32, device=self.device).view(2, -1).T
        )
        _, packed = prepare_wo_a_stacked(weight, raw, g, r, k)
        self.assertEqual(packed.stride(), (2 * r, 1, r))
        torch.testing.assert_close(packed, raw.reshape(g, r, 2), rtol=0, atol=0)
        self.assertNotEqual(packed.data_ptr(), raw.data_ptr())

    def test_reject_misshaped_blocks(self):
        weight = torch.empty((128, 512), device=self.device, dtype=torch.float8_e4m3fn)
        for shape, dtype in [((2, 2), torch.float8_e8m0fnu), ((1, 4), torch.int32)]:
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "shape"):
                prepare_wo_a_stacked(
                    weight,
                    torch.empty(shape, device=self.device, dtype=dtype),
                    1,
                    128,
                    512,
                )


if __name__ == "__main__":
    unittest.main()
