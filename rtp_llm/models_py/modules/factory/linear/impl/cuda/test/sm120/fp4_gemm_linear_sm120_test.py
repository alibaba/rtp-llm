"""SM120 cases split out of fp4_gemm_linear_test.py.
These cases live in this sibling file, leaving the original test file
untouched."""

import logging
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from flashinfer import fp4_quantize

from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.device.device_impl import CudaImpl
from rtp_llm.models_py.modules.factory.linear import LinearFactory
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
    CudaFp4GEMMLinear,
    fp4_backend_available,
)
from rtp_llm.test.utils.numeric_util import calc_diff


class CudaFp4GEMMLinearTestSm120(unittest.TestCase):
    """Sibling of ``CudaFp4GEMMLinearTest`` carrying the SM120 cases."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.skipTest("FP4 tests require CUDA")
        logging.getLogger(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear"
        ).setLevel(logging.WARNING)
        self.hidden_size = 1024
        self.output_size = 512
        torch.manual_seed(42)
        weight_fp16 = (
            torch.randn(
                self.output_size,
                self.hidden_size,
                dtype=torch.float16,
                device=self.device,
            )
            * 0.1
        )
        global_sf_weight = (448 * 6) / weight_fp16.float().abs().nan_to_num().max()
        self.weight_scale_2 = 1.0 / global_sf_weight.to(torch.float32)
        self.weight, self.weight_scales = fp4_quantize(
            weight_fp16, global_scale=global_sf_weight, is_sf_swizzled_layout=False
        )
        self.bias = torch.randn(
            self.output_size, dtype=torch.bfloat16, device=self.device
        )
        self.input_scale = self.weight_scale_2

    def _create_fp4_linear(self, with_bias: bool = True):
        processed_weight, processed_scale = CudaImpl.convert_fp4_gemm_weight_params(
            self.weight, self.weight_scales
        )
        return CudaFp4GEMMLinear(
            weight=processed_weight,
            weight_scales=processed_scale,
            input_scales=self.input_scale,
            bias=self.bias if with_bias else None,
            quant_config=init_quant_config("modelopt_fp4"),
            weight_scale_2=self.weight_scale_2,
        )

    def test_explicit_legacy_backend_fails_during_construction_when_unavailable(self):
        with patch.dict(
            os.environ,
            {"RTP_LLM_FP4_GEMM_BACKEND": "sgl_cutlass"},
        ), patch(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear."
            "is_legacy_cutlass_fp4_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "rtp_kernel.nvfp4 CUDA provider"):
                self._create_fp4_linear(with_bias=False)

    def test_modelopt_fp4_rejects_missing_flashinfer_api(self):
        with patch(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear.mm_fp4",
            None,
        ), patch(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear."
            "is_blackwell",
            return_value=True,
        ):
            self.assertFalse(fp4_backend_available(self.weight.device))

    def test_modelopt_fp4_rejects_non_blackwell_device(self):
        with patch(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear."
            "is_blackwell",
            return_value=False,
        ):
            self.assertFalse(fp4_backend_available(self.weight.device))

    def test_modelopt_fp4_strategy_is_registered_on_cuda(self):
        self.assertIn(CudaFp4GEMMLinear, LinearFactory._strategies)


if __name__ == "__main__":
    import unittest

    unittest.main()
