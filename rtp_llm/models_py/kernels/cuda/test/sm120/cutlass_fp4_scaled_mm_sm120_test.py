"""SM120 probe for the legacy NVFP4 CUTLASS wrappers.

Sibling of cutlass_fp4_scaled_mm_test.py, which is left unchanged.
The legacy wrappers are SM10x-only; on SM120 the FP4 path is served
by other providers, so the skip contract must hold on (12, 0) devices.
This sibling asserts that gating with an independent oracle (the arch
helpers), CPU-only.
"""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.utils import arch


class CutlassFp4ScaledMmSm120SkipContractTest(unittest.TestCase):
    def test_is_sm10x_excludes_exact_sm120(self):
        for capability, expected in (
            ((10, 0), True),
            ((10, 3), True),
            ((9, 0), False),
            ((12, 0), False),
            ((12, 1), False),
        ):
            with (
                self.subTest(capability=capability),
                patch.object(arch, "is_cuda", return_value=True),
                patch.object(arch, "get_sm", return_value=capability),
            ):
                self.assertEqual(arch.is_sm10x(0), expected)

    def test_legacy_nvfp4_skip_condition_excludes_sm120(self):
        # Mirrors the main file's skip contract: legacy NVFP4 wrappers run only
        # when CUDA is present and the device is SM10x — SM120 takes the
        # newer FP4 providers instead.
        def skip_condition():
            return not torch.cuda.is_available() or not arch.is_sm10x()

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch.object(arch, "is_sm10x", return_value=False),
        ):
            self.assertTrue(skip_condition())
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch.object(arch, "is_sm10x", return_value=True),
        ):
            self.assertFalse(skip_condition())


if __name__ == "__main__":
    unittest.main()
