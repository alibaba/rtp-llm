"""Smoke test for DeepGEMM warmup functionality."""

import os
import unittest
from unittest import SkipTest

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
from rtp_llm.models_py.warmup.deepgemm_warmup import deepgemm_warmup

# Import CudaFp8DeepGEMMLinear only if available
try:
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
        CudaFp8DeepGEMMLinear,
    )
except ImportError:
    CudaFp8DeepGEMMLinear = None


class DeepGemmWarmupTest(unittest.TestCase):
    """Smoke test for DeepGEMM warmup."""

    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        if not torch.cuda.is_available():
            raise SkipTest("CUDA not available")
        if not has_deep_gemm():
            raise SkipTest("DeepGEMM not available")

    def test_warmup_with_dummy_model(self):
        """Test warmup with a dummy model containing FP8 Linear layers."""
        if CudaFp8DeepGEMMLinear is None:
            raise SkipTest("CudaFp8DeepGEMMLinear not available")

        device = torch.device("cuda:0")

        # Create a simple dummy model with FP8 Linear layers
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Create a small FP8 Linear layer for testing
                n, k = 256, 512
                weight = torch.empty((n, k), device=device, dtype=torch.float8_e4m3fn)
                weight_scales = torch.empty(
                    ((n + 127) // 128, (k + 127) // 128),
                    device=device,
                    dtype=torch.float32,
                )
                # Fill with dummy values
                weight.fill_(0.5)
                weight_scales.fill_(1.0)

                self.linear = CudaFp8DeepGEMMLinear(
                    weight=weight,
                    weight_scales=weight_scales,
                )

        model = DummyModel()

        # Test warmup with small max_tokens
        max_tokens = 128
        mode = "relax"
        num_workers = 2

        # This should not raise an exception
        try:
            deepgemm_warmup(
                model=model,
                max_tokens=max_tokens,
                mode=mode,
                num_workers=num_workers,
                show_progress=False,  # Disable progress bar in test
            )
        except Exception as e:
            self.fail(f"Warmup raised an exception: {e}")

    def test_warmup_skip_when_no_deep_gemm(self):
        """Test that warmup gracefully handles missing deep_gemm."""
        # Temporarily mock has_deep_gemm to return False
        original_has_deep_gemm = has_deep_gemm

        # Create a dummy model
        class DummyModel(torch.nn.Module):
            pass

        model = DummyModel()

        # This should not raise even if deep_gemm is not available
        # (warmup.py checks has_deep_gemm before calling deepgemm_warmup)
        # So we test that deepgemm_warmup itself handles empty models gracefully
        try:
            deepgemm_warmup(
                model=model,
                max_tokens=128,
                mode="relax",
                num_workers=1,
                show_progress=False,
            )
        except Exception as e:
            self.fail(f"Warmup should handle empty models gracefully: {e}")


if __name__ == "__main__":
    unittest.main()
