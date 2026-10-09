"""Test-only released-reference compressor rounding for SM120 diagnosis."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_global, attention_v41


class _RoundedCompressorKernel:
    """Select the diagnostic JIT specialization without disabling fusion."""

    def __init__(self, original):
        self.original = original

    def __getitem__(self, grid):
        launch = self.original[grid]

        def rounded(*args, **kwargs):
            kwargs["ROUND_INPUT_BF16"] = True
            return launch(*args, **kwargs)

        return rounded


@contextmanager
def compressor_bf16(enabled, *, fused=False):
    if not enabled:
        yield
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Compressor rounding diagnostic is qualified only on SM120")
    original_norm = attention_v41.rms_norm

    def rounded_norm(values, weight, eps):
        # Only compressor latents reach this helper in FP32. Index projections
        # already return BF16 and retain their existing normalization.
        if values.dtype == torch.float32:
            if values.shape[-1] != 512:
                raise ValueError("Unexpected FP32 compressor latent width")
            values = values.bfloat16()
        return original_norm(values, weight, eps)

    # Preserve the original eager diagnostic by default. The opt-in fused
    # profile uses the same input boundary inside stage A and retains the
    # original raw FP32 state publication in stage B.
    replacement = (
        patch.object(
            _v41_decode_global,
            "_compress_norm_main_store_kernel",
            _RoundedCompressorKernel(
                _v41_decode_global._compress_norm_main_store_kernel
            ),
        )
        if fused
        else patch.object(_v41_decode_global, "try_produce_global", return_value=False)
    )
    with patch.object(attention_v41, "rms_norm", rounded_norm), replacement:
        yield
