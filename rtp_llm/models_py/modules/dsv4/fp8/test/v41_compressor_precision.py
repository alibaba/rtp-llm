"""Test-only released-reference compressor rounding for SM120 diagnosis."""

from contextlib import contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import _v41_decode_global, attention_v41


@contextmanager
def compressor_bf16(enabled):
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

    # SM120 prefill already uses the eager producer. Force the same boundary
    # for decode so the fused kernel cannot bypass this scoped replacement.
    with patch.object(attention_v41, "rms_norm", rounded_norm), patch.object(
        _v41_decode_global, "try_produce_global", return_value=False
    ):
        yield
