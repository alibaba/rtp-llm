import os
import unittest
from unittest import mock

import torch

from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    per_output_channel_block_cast_to_fp8,
)
from rtp_llm.utils.sm120_fp8_backend import (
    SM120_FP8_BACKEND_ENV,
    get_sm120_fp8_backend,
    resolve_sm120_fp8_backend,
)


class SM120Fp8BackendConfigTest(unittest.TestCase):
    def test_auto_is_default_and_preserves_deepgemm(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_sm120_fp8_backend(), "auto")
            self.assertEqual(resolve_sm120_fp8_backend(), "deepgemm")

    def test_explicit_backends(self):
        for backend in ("cutlass", "deepgemm"):
            with self.subTest(backend=backend), mock.patch.dict(
                os.environ, {SM120_FP8_BACKEND_ENV: backend}
            ):
                self.assertEqual(resolve_sm120_fp8_backend(), backend)

    def test_value_is_case_and_whitespace_insensitive(self):
        with mock.patch.dict(os.environ, {SM120_FP8_BACKEND_ENV: " CUTLASS "}):
            self.assertEqual(resolve_sm120_fp8_backend(), "cutlass")

    def test_invalid_backend_fails_early(self):
        with mock.patch.dict(os.environ, {SM120_FP8_BACKEND_ENV: "cublas"}):
            with self.assertRaisesRegex(ValueError, SM120_FP8_BACKEND_ENV):
                resolve_sm120_fp8_backend()


class PerOutputChannelBlockQuantTest(unittest.TestCase):
    def test_scale_shape_and_reconstruction(self):
        # Loader weights use logical (K, N) layout.  Give each output channel a
        # different dynamic range so a mistaken 128x128 reduction is visible.
        weight = torch.arange(256 * 3, dtype=torch.float32).reshape(256, 3)
        weight = (weight.remainder(31) - 15) * torch.tensor([0.01, 0.1, 1.0])

        quantized, scales = per_output_channel_block_cast_to_fp8(weight, 128)

        self.assertEqual(quantized.shape, weight.shape)
        self.assertEqual(scales.shape, (2, 3))
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float32)
        reconstructed = (
            quantized.float().reshape(2, 128, 3) * scales[:, None, :]
        ).reshape_as(weight)
        relative_l2 = (reconstructed - weight).norm() / weight.norm()
        self.assertLess(relative_l2.item(), 0.03)
        self.assertTrue(torch.all(scales[:, 0] < scales[:, 1]))
        self.assertTrue(torch.all(scales[:, 1] < scales[:, 2]))


if __name__ == "__main__":
    unittest.main()
