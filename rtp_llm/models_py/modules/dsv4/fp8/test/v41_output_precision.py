"""Test-only released output GEMMs through the maintained attention owner."""

from contextlib import contextmanager
from contextvars import ContextVar
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import AttentionV41FP8
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_expert_precision import (
    _dense_down,
    load_reference_kernel,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear


@contextmanager
def output_projection_reference(enabled, *, kernel=None):
    if not enabled:
        yield
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Output GEMM diagnostic is qualified only on SM120")
    kernel = load_reference_kernel() if kernel is None else kernel
    original_project = AttentionV41FP8._project_output
    original_dense = V41MXFP8Linear.forward_quantized
    selected = ContextVar("v41_diagnostic_output_projection", default=False)

    def project(owner, value, freqs, out=None):
        if owner._wo_a_bf16_v41 is None:
            raise ValueError("Output GEMM diagnostic requires native BF16 wo_a")
        token = selected.set(True)
        try:
            return original_project(owner, value, freqs, out=out)
        finally:
            selected.reset(token)

    def dense(owner, values, scales, *, out=None):
        if selected.get():
            return _dense_down(owner, values, scales, kernel, out=out)
        return original_dense(owner, values, scales, out=out)

    with patch.object(AttentionV41FP8, "_project_output", project), patch.object(
        V41MXFP8Linear, "forward_quantized", dense
    ):
        yield
