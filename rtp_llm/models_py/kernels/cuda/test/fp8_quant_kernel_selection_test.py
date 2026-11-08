"""CPU tests for sgl_per_token_group_quant_fp8 kernel selection.

Records which quantization kernel the wrapper dispatches to for each
combination of ``enable_v2``, dtype, size, group size, masking and fusion.
The kernel symbols are replaced with mocks, so no CUDA device is needed.
"""

import importlib
import unittest
from unittest.mock import Mock, patch

import torch

fp8_kernel = importlib.import_module(
    "rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel"
)

LEGACY = "per_token_group_quant_fp8"
V2 = "per_token_group_quant_fp8_v2"


class Fp8QuantKernelSelectionTest(unittest.TestCase):
    def dispatch(self, x, expected, **kwargs):
        legacy = Mock()
        v2 = Mock()
        with patch.object(fp8_kernel, LEGACY, legacy, create=True), patch.object(
            fp8_kernel, V2, v2, create=True
        ):
            fp8_kernel.sgl_per_token_group_quant_fp8(x, 128, **kwargs)
        called = {LEGACY: legacy, V2: v2}
        for name, recorder in called.items():
            if name == expected:
                recorder.assert_called_once()
            else:
                recorder.assert_not_called()

    def test_default_dispatch_matches_main(self):
        bf16_under_4m = torch.zeros(4095, 1024, dtype=torch.bfloat16)
        bf16_4m = torch.zeros(4096, 1024, dtype=torch.bfloat16)
        fp32_4m = torch.zeros(4096, 1024, dtype=torch.float32)
        small_bf16 = torch.zeros(4, 1024, dtype=torch.bfloat16)
        small_fp32 = torch.zeros(4, 1024, dtype=torch.float32)
        for label, x, enable_v2, expected in (
            ("none_bf16_under_4m", bf16_under_4m, None, LEGACY),
            ("none_bf16_4m", bf16_4m, None, V2),
            ("none_fp32_4m", fp32_4m, None, V2),
            ("true_bf16_small", small_bf16, True, V2),
            ("true_fp32_small", small_fp32, True, LEGACY),
            ("true_fp32_4m", fp32_4m, True, LEGACY),
            ("false_bf16_4m", bf16_4m, False, LEGACY),
        ):
            with self.subTest(case=label):
                self.dispatch(x, expected, enable_v2=enable_v2)

    def test_masked_and_fused_always_pick_v2(self):
        x = torch.zeros(4, 1024, dtype=torch.bfloat16)
        masked_m = torch.ones(4, dtype=torch.int32)
        for enable_v2 in (None, True, False):
            for label, kwargs in (
                ("fuse_only", dict(fuse_silu_and_mul=True)),
                ("mask_only", dict(masked_m=masked_m)),
                (
                    "mask_and_fuse",
                    dict(masked_m=masked_m, fuse_silu_and_mul=True),
                ),
            ):
                with self.subTest(enable_v2=enable_v2, case=label):
                    self.dispatch(x, V2, enable_v2=enable_v2, **kwargs)

    def test_unsupported_group_size_stays_on_legacy(self):
        x = torch.zeros(4096, 1024, dtype=torch.bfloat16)
        for enable_v2 in (None, True, False):
            with self.subTest(enable_v2=enable_v2):
                legacy = Mock()
                v2 = Mock()
                with patch.object(
                    fp8_kernel, LEGACY, legacy, create=True
                ), patch.object(fp8_kernel, V2, v2, create=True):
                    fp8_kernel.sgl_per_token_group_quant_fp8(x, 8, enable_v2=enable_v2)
                legacy.assert_called_once()
                v2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
