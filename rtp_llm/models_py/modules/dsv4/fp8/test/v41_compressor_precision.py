"""Test-only released-reference compressor rounding for SM120 diagnosis."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8 import (
    _v41_decode_global,
    attention_v41,
    compressor,
)


def _bf16_hca_projection(values, weight):
    if values.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("HCA projection diagnostic requires BF16 inputs and weights")
    return torch.nn.functional.linear(values, weight).float()


def _hca_entry(original):
    def with_projection(owner, *args, **kwargs):
        if owner.compress_ratio != 1:
            return original(owner, *args, **kwargs)
        # The entry's lazy projection import observes this scoped replacement.
        # CSA enters without a patch and retains raw FP32 values/scores.
        with patch.object(compressor, "_linear_bf16_bf16_fp32", _bf16_hca_projection):
            return original(owner, *args, **kwargs)

    return with_projection


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
def compressor_bf16(enabled, *, fused=False, hca_projection=False):
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
    with ExitStack() as stack:
        stack.enter_context(patch.object(attention_v41, "rms_norm", rounded_norm))
        stack.enter_context(replacement)
        if hca_projection:
            for target, name in (
                (attention_v41.AttentionV41FP8, "_produce_global"),
                (attention_v41.AttentionV41FP8, "_produce_global_decode"),
                (_v41_decode_global, "try_produce_global"),
            ):
                stack.enter_context(
                    patch.object(target, name, _hca_entry(getattr(target, name)))
                )
        yield
