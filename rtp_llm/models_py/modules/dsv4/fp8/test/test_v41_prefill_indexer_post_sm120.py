"""Exact FP32 indexer postprocessing and eager fallback on SM120."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_indexer_post
from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import fp4_roundtrip


class V41PrefillIndexerPostTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "SM120 hardware is required")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41986)

    def check_recipe(self, dots, weights, visible):
        before = [value.clone() for value in (dots, weights, visible)]
        expected = (dots.relu() * weights[:, :, None]).sum(1)
        columns = torch.arange(dots.shape[2], device=dots.device)
        expected.masked_fill_(columns[None] >= visible[:, None], -torch.inf)
        actual = _v41_prefill_indexer_post.try_postprocess(dots, weights, visible)
        self.assertIsNotNone(actual)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(
            torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        )
        for value, snapshot in zip((dots, weights, visible), before):
            self.assertTrue(torch.equal(value, snapshot))
        return actual

    @torch.inference_mode()
    def test_production_score_shapes(self):
        for keys in (16384, 32768):
            with self.subTest(keys=keys):
                q = fp4_roundtrip(
                    torch.randn(64, 32, 128, device="cuda").bfloat16()
                ).float()
                k = fp4_roundtrip(
                    torch.randn(keys, 128, device="cuda").bfloat16()
                ).float()
                weights = torch.randn(64, 32, device="cuda") / 64
                dots = torch.einsum("thd,kd->thk", q, k)
                visible = torch.linspace(0, keys, 64, device="cuda").long()
                self.check_recipe(dots, weights, visible)

    @torch.inference_mode()
    def test_ragged_tiles_and_relative_group_scales(self):
        for exponents in ((0, 0, 0, 0), (-16, 0, 16, 8), (-24, 24, 16, -16)):
            with self.subTest(exponents=exponents):
                factors = torch.tensor(
                    [2.0**e for e in exponents], device="cuda"
                ).repeat_interleave(32)
                q = fp4_roundtrip(
                    (torch.randn(17, 32, 128, device="cuda") * factors).bfloat16()
                ).float()
                k = fp4_roundtrip(
                    (torch.randn(513, 128, device="cuda") * factors).bfloat16()
                ).float()
                dots = torch.einsum("thd,kd->thk", q, k)
                weights = torch.randn(17, 32, device="cuda") / 64
                visible = torch.linspace(0, 513, 17, device="cuda").int()
                self.check_recipe(dots, weights, visible)

    @torch.inference_mode()
    def test_cancellation_and_tied_scores(self):
        dots = torch.randn(3, 32, 513, device="cuda")
        dots[0].zero_()
        dots[1, :, ::2] = 0
        weights = torch.tensor(
            [2.0**12, -(2.0**12), 2.0**-12, -(2.0**-12)], device="cuda"
        )
        weights = weights.repeat(3, 8)
        visible = torch.tensor([0, 513, 257], device="cuda", dtype=torch.int64)
        actual = self.check_recipe(dots, weights, visible)
        self.assertTrue(torch.isneginf(actual[0]).all())

    @torch.inference_mode()
    def test_strided_head_weights(self):
        dots = torch.randn(5, 32, 257, device="cuda")
        weights = torch.randn(5, 64, device="cuda")[:, ::2]
        visible = torch.tensor([0, 1, 64, 256, 257], device="cuda", dtype=torch.int32)
        self.check_recipe(dots, weights, visible)

    def test_eager_fallback_layouts(self):
        for dots in (
            torch.zeros(2, 32, 4, device="cpu"),
            torch.zeros(2, 4, 4, device="cuda"),
            torch.zeros(2, 32, 1, device="cuda"),
            torch.zeros(2, 32, 4, device="cuda", dtype=torch.bfloat16),
            torch.zeros(2, 4, 32, device="cuda").transpose(1, 2),
        ):
            with self.subTest(shape=dots.shape, dtype=dots.dtype, device=dots.device):
                weights = torch.ones(
                    dots.shape[:2], device=dots.device, dtype=torch.float32
                )
                visible = torch.ones(2, device=dots.device, dtype=torch.int64)
                self.assertIsNone(
                    _v41_prefill_indexer_post.try_postprocess(dots, weights, visible)
                )

    def test_gradient_fallback(self):
        dots = torch.randn(2, 32, 257, device="cuda", requires_grad=True)
        weights = torch.randn(2, 32, device="cuda")
        visible = torch.ones(2, device="cuda", dtype=torch.int64)
        self.assertIsNone(
            _v41_prefill_indexer_post.try_postprocess(dots, weights, visible)
        )
        dots = dots.detach()
        weights.requires_grad_(True)
        self.assertIsNone(
            _v41_prefill_indexer_post.try_postprocess(dots, weights, visible)
        )

    @torch.inference_mode()
    def test_empty_rows(self):
        dots = torch.empty(0, 32, 257, device="cuda")
        weights = torch.empty(0, 32, device="cuda")
        visible = torch.empty(0, device="cuda", dtype=torch.int64)
        result = _v41_prefill_indexer_post.try_postprocess(dots, weights, visible)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (0, 257))


if __name__ == "__main__":
    unittest.main()
