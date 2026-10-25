"""Check ordered POST arithmetic and the model's in-place Graph contract."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.platforms.ppu.kernels.ppu_hc_sum import hc_post, supports_hc_post


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class HcPostTest(unittest.TestCase):
    def unit(self):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_hc import PpuHCUnit

        return PpuHCUnit(
            torch.zeros((24, 16384), device="cuda"),
            torch.zeros(24, device="cuda"),
            torch.ones(3, device="cuda"),
            options={"DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm_deterministic"},
            dim=4096,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
            allow_graph=True,
            fuse_post=True,
        )

    def same(self, got, want):
        self.assertTrue(torch.equal(got.view(torch.uint8), want.view(torch.uint8)))

    def test_provider_post_requires_snapshot_opt_in(self):
        from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS
        from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import (
            PpuDecodeProvider,
        )

        for fusions, tiling, expected in (
            ("1", None, False),
            ("1", "0", False),
            ("1", "1", True),
            ("0", "1", False),
        ):
            options = {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_FUSIONS": fusions}
            if tiling is not None:
                options["DSV4_PPU_HC_POST_TILING"] = tiling
            provider = PpuDecodeProvider(options)
            # Construction freezes the options; neither a caller's mapping
            # mutation nor a later environment change may switch the kernel.
            options["DSV4_PPU_HC_POST_TILING"] = "0" if expected else "1"
            with patch.dict(os.environ, {"DSV4_PPU_HC_POST_TILING": "1"}), patch(
                "rtp_llm.platforms.ppu.models.dsv4.ppu_hc.PpuHCUnit"
            ) as factory:
                provider.build_hc_unit()
                self.assertIs(factory.call_args.kwargs["fuse_post"], expected)

    @torch.inference_mode()
    def test_model_post_inplace_and_graph_match_tilelang_bytes(self):
        torch.manual_seed(518)
        unit = self.unit()
        original = unit._post_kernel
        for rows in (1, 3, 8, 16, 24, 32):
            for leading in ((rows,), (rows, 1)):
                x = torch.randn((*leading, 4096), device="cuda", dtype=torch.bfloat16)
                source = torch.randn(
                    (*leading, 4, 4096), device="cuda", dtype=torch.bfloat16
                )
                res = torch.empty_like(source)
                post = torch.randn((*leading, 4, 1), device="cuda")
                comb = torch.randn((*leading, 4, 4), device="cuda")

                def run():
                    res.copy_(source)
                    result = unit.post(x, res, post, comb)
                    self.assertEqual(result.data_ptr(), res.data_ptr())
                    return result

                with patch.object(
                    unit,
                    "_post_kernel",
                    side_effect=AssertionError("unexpected fallback"),
                ):
                    for _ in range(3):
                        run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        actual = run()
                    for magnitude in (0.0, 1e-30, 0.001, 1.0, 10.0, 1e8):
                        x.normal_().mul_(magnitude)
                        source.normal_().mul_(magnitude)
                        post.normal_()
                        comb.normal_()
                        # Leave signed zeros in the input; adding a dummy shared
                        # zero would change that arithmetic boundary.
                        x.reshape(-1)[::19] = -0.0
                        # Call the saved TileLang implementation independently
                        # because the model selection above must not use it.
                        reference = original(
                            x.reshape(rows, 1, 4096),
                            source.reshape(rows, 1, 4, 4096),
                            post.reshape(rows, 1, 4, 1),
                            comb.reshape(rows, 1, 4, 4),
                            enable_pdl=False,
                        ).view_as(source)
                        self.same(run(), reference)
                        actual.fill_(float("nan"))
                        graph.replay()
                        self.same(actual, reference)

    @torch.inference_mode()
    def test_out_and_fallback_contracts(self):
        unit = self.unit()
        for rows in (0, 8, 33, 64):
            x = torch.randn((rows, 1, 4096), device="cuda", dtype=torch.bfloat16)
            res = torch.randn((rows, 1, 4, 4096), device="cuda", dtype=torch.bfloat16)
            post = torch.randn((rows, 1, 4, 1), device="cuda")
            comb = torch.randn((rows, 1, 4, 4), device="cuda")
            self.assertEqual(supports_hc_post(x, res, post, comb), rows == 8)
            if rows == 8:
                expected = unit._post_kernel(x, res, post, comb, enable_pdl=False)
                original_res = res.clone()
                self.same(hc_post(x, res, post, comb), expected)
                self.same(res, original_res)
                out = torch.empty_like(res)
                self.assertEqual(
                    hc_post(x, res, post, comb, out=out).data_ptr(), out.data_ptr()
                )
                self.same(out, expected)
                self.assertFalse(supports_hc_post(x, res, post, comb, out.float()))
                self.assertFalse(supports_hc_post(x, res.transpose(-1, -2), post, comb))
                self.assertFalse(supports_hc_post(x.float(), res, post, comb))
            else:
                with patch(
                    "rtp_llm.platforms.ppu.kernels.ppu_hc_sum.hc_post",
                    side_effect=AssertionError("unexpected optimized POST"),
                ):
                    unit.post(x, res, post, comb)


if __name__ == "__main__":
    unittest.main()
