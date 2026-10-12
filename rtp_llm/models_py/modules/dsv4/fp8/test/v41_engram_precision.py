"""Test-only released Engram projection/gate for held-input accuracy diagnosis."""

from contextlib import contextmanager
from contextvars import ContextVar
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4 import engram
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_expert_precision import (
    _dense_down,
    load_reference_kernel,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear


def reference_engram_gate(
    hidden, kv, q_weight, k_weight, eps, token_mask=None, *, out=None
):
    """Evaluate the released FP32 gate, preserving native output writeback."""
    if hidden.ndim != 3:
        raise ValueError("Engram diagnostic requires [tokens, copies, dim] hidden")
    tokens, copies, dim = hidden.shape
    if kv.shape != (tokens, (copies + 1) * dim) or (
        q_weight.shape != k_weight.shape or q_weight.shape != (copies, dim)
    ):
        raise ValueError("Engram diagnostic projection/weight shapes differ")
    sources = (kv, q_weight, k_weight, token_mask)
    if any(value is not None and value.device != hidden.device for value in sources):
        raise ValueError("Engram diagnostic inputs must share a device")
    if token_mask is not None and (
        token_mask.shape != (tokens,) or token_mask.dtype != torch.bool
    ):
        raise ValueError("Engram diagnostic mask must describe every token")
    if out is not None:
        if (
            out.shape != hidden.shape
            or out.dtype != hidden.dtype
            or out.device != hidden.device
            or not out.is_contiguous()
        ):
            raise ValueError("Engram diagnostic output layout differs")
        storage = out.untyped_storage().data_ptr()
        if any(
            value is not None and storage == value.untyped_storage().data_ptr()
            for value in sources
        ):
            raise ValueError("Engram diagnostic output aliases another input")
        if storage == hidden.untyped_storage().data_ptr() and (
            torch.is_grad_enabled()
            or out.storage_offset() != hidden.storage_offset()
            or out.stride() != hidden.stride()
        ):
            raise ValueError("Engram diagnostic hidden alias must be exact inference")
    key = kv[:, : copies * dim].float().reshape(tokens, copies, dim)
    value = kv[:, copies * dim :].float().reshape(tokens, 1, dim)
    weight = q_weight.float() * k_weight.float()
    h = hidden.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    if token_mask is not None:
        gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    result = (h + gate.unsqueeze(-1) * value).to(hidden.dtype)
    return result if out is None else out.copy_(result)


@contextmanager
def engram_gate_reference(enabled):
    if not enabled:
        yield
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Engram gate diagnostic is qualified only on SM120")
    with patch.object(engram, "gated_engram_residual", reference_engram_gate):
        yield


@contextmanager
def engram_projection_reference(enabled, *, kernel=None):
    """Replace only Engram WKV's output GEMM; retain its native quantizer."""
    if not enabled:
        yield
        return
    if torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("Engram projection diagnostic is qualified only on SM120")
    kernel = load_reference_kernel() if kernel is None else kernel
    original_rows = engram.Engram._forward_rows
    original_dense = V41MXFP8Linear.forward_quantized
    selected = ContextVar("v41_diagnostic_engram_projection", default=None)

    def forward_rows(owner, *args, **kwargs):
        provider = getattr(owner.wkv, "_sm120_linear", None)
        if provider is None:
            raise ValueError("Engram projection diagnostic requires an SM120 provider")
        token = selected.set(provider)
        try:
            return original_rows(owner, *args, **kwargs)
        finally:
            selected.reset(token)

    def dense_forward(owner, values, scales, *, out=None):
        if owner is selected.get():
            return _dense_down(owner, values, scales, kernel, out=out)
        return original_dense(owner, values, scales, out=out)

    with patch.object(engram.Engram, "_forward_rows", forward_rows), patch.object(
        V41MXFP8Linear, "forward_quantized", dense_forward
    ):
        yield
