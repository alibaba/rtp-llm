"""Compare the fused MTP producer with the actual FlashInfer/PPU operators."""

import unittest

import torch

from rtp_llm.platforms.ppu.kernels.ppu_mtp_input import (
    mtp_input_quant,
    supports_mtp_input,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class MtpInputTest(unittest.TestCase):
    def test_mask_norm_quant_graph_keeps_both_byte_boundaries(self):
        import flashinfer

        from rtp_llm.platforms.ppu.modules.linear.fp8_linear import (
            quantize_ppu_fp8_activation,
        )

        torch.manual_seed(47)
        for rows in (1, 8, 32):
            for dtype in (torch.int32, torch.int64):
                with self.subTest(rows=rows, dtype=dtype):
                    e = torch.randn((rows, 4096), device="cuda", dtype=torch.bfloat16)
                    h = torch.randn(
                        (rows, 4, 4096), device="cuda", dtype=torch.bfloat16
                    )
                    ew = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
                    hw = torch.randn_like(ew)
                    positions = torch.arange(rows, device="cuda", dtype=dtype)
                    args = (e, h, positions, ew, hw, 1e-6, 1e-5)
                    mtp_input_quant(*args)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = mtp_input_quant(*args)
                    for magnitude in (0.0, 1e-30, 1.0, 1e8):
                        e.normal_().mul_(magnitude)
                        h.normal_().mul_(magnitude)
                        positions.add_(1)
                        positions[::3] = 0
                        masked = torch.where(positions[:, None] == 0, 0, e)
                        expected = tuple(
                            quantize_ppu_fp8_activation(
                                flashinfer.norm.rmsnorm(x, w, eps=eps),
                                quantization="v2_column",
                            )
                            for x, w, eps in (
                                (masked, ew, 1e-6),
                                (h.view(-1, 4096), hw, 1e-5),
                            )
                        )
                        eager = mtp_input_quant(*args)
                        graph.replay()
                        for result in (eager, actual):
                            for pair, reference in zip(result, expected):
                                self.assertEqual(
                                    pair[1].stride(), (1, pair[0].shape[0])
                                )
                                for got, want in zip(pair, reference):
                                    self.assertTrue(
                                        torch.equal(
                                            got.contiguous().view(torch.uint8),
                                            want.contiguous().view(torch.uint8),
                                        )
                                    )

    def test_unsupported_layouts_decline_before_launch(self):
        e = torch.zeros((8, 4096), device="cuda", dtype=torch.bfloat16)
        h = torch.zeros((8, 4, 4096), device="cuda", dtype=torch.bfloat16)
        p = torch.zeros(8, device="cuda", dtype=torch.int64)
        w = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(supports_mtp_input(e, h, p, w, w, 1e-6, 1e-6))
        for args in (
            (e[:0], h[:0], p[:0], w, w, 1e-6, 1e-6),
            (e, h, p.float(), w, w, 1e-6, 1e-6),
            (e, h.transpose(1, 2), p, w, w, 1e-6, 1e-6),
            (e, h, p, w, w, 0.0, 1e-6),
        ):
            self.assertFalse(supports_mtp_input(*args))
            with self.assertRaises(ValueError):
                mtp_input_quant(*args)


if __name__ == "__main__":
    unittest.main()
