import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
    CudaFp8GEMMLinear,
    _fp8_flashinfer_gemm_enabled,
)


class Fp8GemmABSwitchTest(unittest.TestCase):
    ENV_NAME = "RTP_LLM_ENABLE_FP8_FLASHINFER_GEMM"

    def test_switch_parsing(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(self.ENV_NAME, None)
            self.assertTrue(_fp8_flashinfer_gemm_enabled())

        for value in ("1", "true", "YES", "on"):
            with self.subTest(value=value), patch.dict(
                os.environ, {self.ENV_NAME: value}, clear=False
            ):
                self.assertTrue(_fp8_flashinfer_gemm_enabled())

        for value in ("0", "false", "NO", "off"):
            with self.subTest(value=value), patch.dict(
                os.environ, {self.ENV_NAME: value}, clear=False
            ):
                self.assertFalse(_fp8_flashinfer_gemm_enabled())

        with patch.dict(
            os.environ, {self.ENV_NAME: "automatic"}, clear=False
        ), self.assertRaisesRegex(ValueError, self.ENV_NAME):
            _fp8_flashinfer_gemm_enabled()

    def test_disabled_switch_blocks_flashinfer_dispatch(self) -> None:
        linear = object.__new__(CudaFp8GEMMLinear)
        linear._enable_flashinfer_gemm = False
        linear._flashinfer_linear = object()
        linear.input_scales = None
        linear.K = 16
        linear.FLASHINFER_M_THRESHOLD = 4

        input_tensor = torch.empty((1, 16), dtype=torch.bfloat16)
        self.assertFalse(linear._should_use_flashinfer(input_tensor))

        linear._enable_flashinfer_gemm = True
        self.assertTrue(linear._should_use_flashinfer(input_tensor))


if __name__ == "__main__":
    unittest.main()
