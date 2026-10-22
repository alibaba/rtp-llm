import unittest

import torch

from rtp_llm.platforms.ppu.kernels.cuda.ppu_sglang_permute import fused_permute


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class FusedPermuteTest(unittest.TestCase):
    def test_non_fast_path_copies_short_scale_rows_completely(self):
        torch.manual_seed(890619)
        payload = torch.randn((65, 64, 128), device="cuda").to(
            torch.float8_e4m3fn
        )

        for scale_width in (2, 3):
            with self.subTest(scale_width=scale_width):
                scales = torch.randn(
                    (65, 64, scale_width), device="cuda", dtype=torch.float32
                )
                actual_payload, actual_scales = fused_permute(payload, scales)
                self.assertTrue(
                    torch.equal(actual_payload, payload.permute(1, 0, 2).contiguous())
                )
                self.assertTrue(
                    torch.equal(actual_scales, scales.permute(1, 0, 2).contiguous())
                )


if __name__ == "__main__":
    unittest.main()
