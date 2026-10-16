import os
import unittest
from unittest import mock

import torch

from rtp_llm.model_loader.per_block_fp8_quant_weight import (
    LoadQuantPerBlockFp8Weight,
    per_block_cast_to_fp8,
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

    def test_blank_value_uses_auto(self):
        for value in ("", "  "):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {SM120_FP8_BACKEND_ENV: value}
            ):
                self.assertEqual(get_sm120_fp8_backend(), "auto")

    def test_invalid_backend_fails_early(self):
        with mock.patch.dict(os.environ, {SM120_FP8_BACKEND_ENV: "cublas"}):
            with self.assertRaisesRegex(ValueError, SM120_FP8_BACKEND_ENV):
                resolve_sm120_fp8_backend()

    def test_loader_uses_target_device_for_sm120_layout(self):
        loader = object.__new__(LoadQuantPerBlockFp8Weight)
        loader.group_size = 64
        loader.kernel = mock.Mock()
        loader.kernel.name = "test_dense_weight"
        loader.kernel._load_raw_tensor.return_value = {
            loader.kernel.name: torch.empty((128, 128), dtype=torch.float32)
        }
        loader.scale = None

        for convert_device, expected_target in (
            ("cuda:1", torch.device("cuda:1")),
            ("cpu", None),
        ):
            with self.subTest(convert_device=convert_device), mock.patch.dict(
                os.environ, {SM120_FP8_BACKEND_ENV: "cutlass"}
            ), mock.patch(
                "rtp_llm.models_py.utils.arch.is_sm120", return_value=True
            ) as is_sm120, mock.patch(
                "rtp_llm.models_py.kernels.cuda.deepgemm_wrapper.is_deep_gemm_e8m0_used",
                return_value=False,
            ):
                with self.assertRaisesRegex(ValueError, "SM120 CUTLASS dense weights"):
                    loader._load_raw_tensor(mock.Mock(), 0, convert_device, mock.Mock())
                is_sm120.assert_called_with(expected_target)


class SquareBlockQuantTest(unittest.TestCase):
    def test_scale_shape_and_reconstruction(self):
        weight = torch.arange(256 * 384, dtype=torch.float32).reshape(256, 384)
        weight = (weight.remainder(31) - 15) * torch.tensor(
            [0.01, 0.1, 1.0]
        ).repeat_interleave(128)
        quantized, scales = per_block_cast_to_fp8(weight, 128)
        self.assertEqual(quantized.shape, weight.shape)
        self.assertEqual(scales.shape, (2, 3))
        self.assertEqual(quantized.dtype, torch.float8_e4m3fn)
        self.assertEqual(scales.dtype, torch.float32)
        reconstructed = quantized.float() * scales.repeat_interleave(
            128, 0
        ).repeat_interleave(128, 1)
        self.assertLess(((reconstructed - weight).norm() / weight.norm()).item(), 0.03)
        self.assertTrue(torch.all(scales[:, 0] < scales[:, 1]))
        self.assertTrue(torch.all(scales[:, 1] < scales[:, 2]))


if __name__ == "__main__":
    unittest.main()
