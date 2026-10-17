"""SM120 mHC column tiling preserves the full wrapper's output bits."""

import importlib
import os
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires SM120",
)
class MhcColumnTilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initializes the vendored TileLang environment before importing it.
        importlib.import_module("rtp_llm.models_py.modules.dsv4.tilelang_kernels")
        cls.wrapper = importlib.import_module(
            "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
        )

    def inputs(self, tokens, dim=4096):
        generator = torch.Generator(device="cuda").manual_seed(1710 + tokens)
        return (
            torch.randn(
                (1, tokens, 4, dim),
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            ),
            torch.randn((24, 4 * dim), generator=generator, device="cuda") / 128,
            torch.randn((3,), generator=generator, device="cuda"),
            torch.randn((24,), generator=generator, device="cuda"),
        )

    def call(self, inputs, enabled):
        with torch.inference_mode(), patch.dict(
            os.environ,
            DSV4_SM120_MHC_COLUMN_TILES=str(int(enabled)),
            DSV4_MHC_PRE_GEMM_BACKEND="tilelang_single",
            DSV4_MHC_PRENORM_DEEPGEMM="0",
        ):
            return self.wrapper.mhc_pre_big_fuse(*inputs, 1e-6, 1e-6, 1e-6, 2.0, 20)

    def assertExact(self, actual, expected):
        for got, want in zip(actual, expected):
            self.assertEqual(got.shape, want.shape)
            self.assertEqual(got.dtype, want.dtype)
            self.assertEqual(got.stride(), want.stride())
            self.assertTrue(torch.isfinite(got).all())
            self.assertTrue(
                torch.equal(
                    got.contiguous().view(torch.uint8),
                    want.contiguous().view(torch.uint8),
                )
            )

    def test_full_wrapper_exact_across_tail_shapes_and_scales(self):
        for tokens in (1, 2, 3, 5, 6, 7, 8, 11, 16, 17, 31, 32, 33):
            for scale in (0.0, 0.001, 1.0, 1000.0):
                with self.subTest(tokens=tokens, scale=scale):
                    inputs = self.inputs(tokens)
                    inputs[0].mul_(scale)
                    before = tuple(t.clone() for t in inputs)
                    expected = self.call(inputs, False)
                    self.assertExact(self.call(inputs, True), expected)
                    self.assertExact(inputs, before)

    def test_disabled_and_unsupported_shapes_keep_original_kernel(self):
        with patch.object(
            self.wrapper,
            "_mhc_pre_norm_fn_sm120_columns",
            side_effect=AssertionError("unsupported column kernel dispatch"),
        ):
            for tokens, dim, enabled in (
                (6, 4096, False),
                (33, 4096, True),
                (6, 2048, True),
            ):
                with self.subTest(tokens=tokens, dim=dim, enabled=enabled):
                    inputs = self.inputs(tokens, dim)
                    expected = self.call(inputs, False)
                    self.assertExact(self.call(inputs, enabled), expected)

    def test_cuda_graph_uses_updated_inputs(self):
        inputs = self.inputs(6)
        for _ in range(3):
            self.call(inputs, True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.call(inputs, True)
        for scale in (0.0, 0.001, 1.0, 1000.0):
            replacement = self.inputs(6)
            replacement[0].mul_(scale)
            for target, value in zip(inputs, replacement):
                target.copy_(value)
            graph.replay()
            self.assertExact(actual, self.call(inputs, False))


if __name__ == "__main__":
    unittest.main()
