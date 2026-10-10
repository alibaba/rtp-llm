"""Test-only released query output GEMMs through maintained SM120 owners."""

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
def query_projection_reference(enabled, *, first=True, second=True, kernel=None):
    if not enabled:
        yield
        return
    if not (first or second):
        raise ValueError("Query GEMM diagnostic requires a selected projection")
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Query GEMM diagnostic is qualified only on SM120")
    kernel = load_reference_kernel() if kernel is None else kernel
    original_linear = AttentionV41FP8._lin
    original_dense = V41MXFP8Linear.forward_quantized
    selected = ContextVar("v41_diagnostic_query_projection", default=None)

    def linear(owner, layer, value, out=None):
        target = (first and layer is owner.wq_a) or (second and layer is owner.wq_b)
        if not target:
            return original_linear(owner, layer, value, out=out)
        provider = getattr(layer, "_sm120_linear", None)
        if provider is None:
            raise ValueError("Query diagnostic requires the native SM120 provider")
        token = selected.set(provider)
        try:
            return original_linear(owner, layer, value, out=out)
        finally:
            selected.reset(token)

    def dense(owner, values, scales, *, out=None):
        if owner is selected.get():
            return _dense_down(owner, values, scales, kernel, out=out)
        return original_dense(owner, values, scales, out=out)

    with patch.object(AttentionV41FP8, "_lin", linear), patch.object(
        V41MXFP8Linear, "forward_quantized", dense
    ):
        yield
