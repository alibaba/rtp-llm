"""Native Prefill HC numerical acceptance on PPU 1.0 and PPU 1.5."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.platforms.ppu.models.dsv4.ppu_hc import PpuHCUnit
from rtp_llm.platforms.ppu.modules.linear.fp8_linear import quantize_ppu_fp8_activation


class PrefillOperatorsTest(unittest.TestCase):
    @torch.inference_mode()
    def test_hc_prenorm_against_fp32(self):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_hc_prenorm import (
            tf32_hc_prenorm_gemm,
        )

        torch.manual_seed(890431)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            weight = torch.randn((24, 16384), device="cuda") / 128
            # Include the former slow-conversion regime and partial M tiles.
            for rows in (1, 65, 257, 512, 513, 1024, 2049, 8192):
                with self.subTest(rows=rows):
                    x = torch.randn((rows, 16384), device="cuda", dtype=torch.bfloat16)
                    out = torch.empty((1, rows, 24), device="cuda")
                    squares = torch.empty((1, rows), device="cuda")
                    tf32_hc_prenorm_gemm(x, weight, out, squares)
                    torch.cuda.synchronize()
                    reference = x.float() @ weight.t()
                    square_reference = x.float().square().sum(dim=1)
                    self.assertTrue(bool(out.isfinite().all()))
                    self.assertLess(
                        float((out[0] - reference).norm() / reference.norm()), 0.002
                    )
                    self.assertLess(
                        float((squares[0] - square_reference).norm()
                              / square_reference.norm()), 1e-5
                    )
                    previous = out.clone(), squares.clone()
                    tf32_hc_prenorm_gemm(x, weight, out, squares)
                    torch.testing.assert_close(out, previous[0], rtol=0, atol=0)
                    torch.testing.assert_close(squares, previous[1], rtol=0, atol=0)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    @torch.inference_mode()
    def test_hc_quantization_and_post_match_fallback(self):
        torch.manual_seed(890429)
        args = (
            torch.randn((24, 16384), device="cuda") / 128,
            torch.zeros(24, device="cuda"),
            torch.ones(3, device="cuda"),
        )
        kwargs = dict(
            dim=4096,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
            tp_size=4,
            tp_rank=0,
            fuse_norm=True,
        )
        optimized = PpuHCUnit(*args, options={}, **kwargs)
        fallback = PpuHCUnit(
            *args,
            options={"DSV4_PPU_PREFILL_HC_QUANT": "0", "DSV4_PPU_PREFILL_HC_POST": "0"},
            **kwargs
        )
        norm = SimpleNamespace(
            weight=torch.linspace(
                0.25, 1.25, 4096, device="cuda", dtype=torch.bfloat16
            ),
            variance_epsilon=1e-6,
        )
        for rows in (1024, 2049):
            with self.subTest(rows=rows):
                x = torch.randn((rows, 4, 4096), device="cuda", dtype=torch.bfloat16)
                y, post, comb = fallback.pre_norm(x, norm, tp_size=4, tp_rank=0)
                actual = optimized.prefill_fast_pre_norm_quantized(
                    x, norm, tp_size=4, tp_rank=0
                )
                for value, reference in zip(actual[:3], (y, post, comb)):
                    torch.testing.assert_close(value, reference, rtol=0, atol=0)
                q, scale = quantize_ppu_fp8_activation(y, quantization="v2_row")
                self.assertTrue(
                    torch.equal(actual[3][0].view(torch.uint8), q.view(torch.uint8))
                )
                torch.testing.assert_close(actual[3][1], scale, rtol=0, atol=0)
                self.assertTrue(actual[3][1].is_contiguous())
                self.assertIsNone(
                    fallback.prefill_fast_pre_norm_quantized(
                        x, norm, tp_size=4, tp_rank=0
                    )
                )
                inputs = (
                    y.unsqueeze(0),
                    x.unsqueeze(0),
                    post.unsqueeze(0),
                    comb.unsqueeze(0),
                )
                a = optimized._post_operator(*inputs, hc_mult=4)
                b = fallback._post_operator(*inputs, hc_mult=4)
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
