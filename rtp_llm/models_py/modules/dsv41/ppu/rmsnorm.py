"""Standalone PPU RMSNorm dispatch, independent of model phase.

Existing CSA/Mega-mHC fused operators keep their own normalization. Ordinary
RMSNorm uses FlashInfer, regular row-strided inputs use the existing strided
kernel, and explicit PTPC consumers may request the fused norm/quant producer.
There is no Torch-composed RMSNorm fallback in this module.
"""

import math

import torch


def _validate(hidden, weight, eps):
    if (
        not isinstance(hidden, torch.Tensor)
        or hidden.layout != torch.strided
        or hidden.ndim < 1
        or hidden.shape[-1] <= 0
        or hidden.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError("PPU RMSNorm requires strided BF16 or FP16 rows")
    if weight is not None and (
        not isinstance(weight, torch.Tensor)
        or weight.layout != torch.strided
        or weight.ndim != 1
        or weight.numel() != hidden.shape[-1]
        or weight.device != hidden.device
        or weight.dtype != hidden.dtype
    ):
        raise ValueError(
            "RMSNorm weight must match the feature width, dtype and device"
        )
    if not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps < 0:
        raise ValueError("RMSNorm epsilon must be finite and nonnegative")
    if torch.is_grad_enabled() and (
        hidden.requires_grad or (weight is not None and weight.requires_grad)
    ):
        raise RuntimeError("PPU RMSNorm dispatch supports inference only")


def _flashinfer_rmsnorm():
    from flashinfer.norm import rmsnorm

    return rmsnorm


def _strided_rmsnorm():
    from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import strided_q_rmsnorm

    return strided_q_rmsnorm


def _row_strided_view(hidden):
    """Return a view only; arbitrary column strides use the library path."""
    if (
        hidden.is_contiguous()
        or hidden.stride(-1) != 1
        or hidden.shape[-1] > 8192
        or hidden.is_conj()
        or hidden.is_neg()
    ):
        return None
    try:
        rows = hidden.view(-1, hidden.shape[-1])
    except RuntimeError:
        return None
    if rows.stride(0) < rows.shape[1]:
        return None
    return rows


def rms_norm(hidden, weight, eps):
    """Normalize using one complete provider, without phase-specific flags.

    A regular gapped row view is consumed without packing. Other unsupported
    layouts are packed for FlashInfer; packing never changes the arithmetic
    into a sequence of Torch normalization operations. Provider failures
    propagate instead of retrying through another implementation.
    """
    _validate(hidden, weight, eps)
    if hidden.numel() == 0:
        return torch.empty(hidden.shape, dtype=hidden.dtype, device=hidden.device)
    rows = _row_strided_view(hidden)
    if (
        rows is not None
        and weight is not None
        and weight.is_contiguous()
        and not weight.is_conj()
        and not weight.is_neg()
    ):
        return _strided_rmsnorm()(rows, weight, eps).view(hidden.shape)
    dense = hidden.resolve_conj().resolve_neg().contiguous()
    gamma = None if weight is None else weight.resolve_conj().resolve_neg().contiguous()
    output = _flashinfer_rmsnorm()(dense.view(-1, dense.shape[-1]), gamma, eps=eps)
    return output.view(hidden.shape)


def try_rms_norm_quant(hidden, weight, eps):
    """Return a fused PTPC producer result, or None before any kernel launch.

    Only explicit quant consumers call this entry. If fusion is unsupported,
    the caller performs ordinary RMSNorm and keeps its existing quant consumer.
    Regular row-strided inputs are eligible for the same producer kernel.
    """
    _validate(hidden, weight, eps)
    from rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant import ptpc_fusion_enabled

    if weight is None or not ptpc_fusion_enabled():
        return None
    from rtp_llm.models_py.modules.dsv41.ppu.ptpc_norm_quant import (
        ptpc_rms_norm_quant_candidate,
    )

    return ptpc_rms_norm_quant_candidate(hidden, weight, eps, allow_strided=True)
