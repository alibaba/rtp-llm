"""AOT CUDA dispatch for hpc-ops-derived HY4 iHC kernels."""

from __future__ import annotations

import os
from typing import Tuple

import torch

_HPC_ENV = "RTP_LLM_HY4_IHC_HPC"
_HC_MULT = 4
_SUPPORTED_HIDDEN = (4096, 6144)


def _enabled() -> bool:
    return os.environ.get(_HPC_ENV, "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _is_sm103(tensor: torch.Tensor) -> bool:
    if tensor.device.type != "cuda":
        return False
    index = tensor.device.index
    if index is None:
        index = torch.cuda.current_device()
    return torch.cuda.get_device_capability(index) == (10, 3)


def _same_cuda_device(*tensors: torch.Tensor) -> bool:
    if not tensors:
        return False
    device = tensors[0].device
    return device.type == "cuda" and all(tensor.device == device for tensor in tensors)


def _has_enabled_grad(*tensors: torch.Tensor) -> bool:
    return torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors)


def _runtime_has_ops() -> bool:
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        return hasattr(rtp_llm_ops, "fuse_hy4_ihc_head") and hasattr(
            rtp_llm_ops, "fuse_hy4_ihc_post_pre"
        )
    except (ImportError, OSError):
        return False


def can_fuse_head(
    channels: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
) -> bool:
    return (
        _enabled()
        and not _has_enabled_grad(channels, fn_weight, scale, base)
        and channels.dim() == 3
        and channels.shape[1] == _HC_MULT
        and channels.shape[2] in _SUPPORTED_HIDDEN
        and channels.shape[0] > 0
        and channels.dtype == torch.bfloat16
        and channels.is_contiguous()
        and fn_weight.dtype == torch.float32
        and fn_weight.is_contiguous()
        and tuple(fn_weight.shape) == (_HC_MULT, _HC_MULT * channels.shape[2])
        and scale.dtype == torch.float32
        and scale.is_contiguous()
        and scale.numel() == 1
        and base.dtype == torch.float32
        and base.is_contiguous()
        and base.numel() == _HC_MULT
        and _same_cuda_device(channels, fn_weight, scale, base)
        and _is_sm103(channels)
        and _runtime_has_ops()
    )


def maybe_fuse_head(
    channels: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_eps: float,
    hc_eps: float,
) -> torch.Tensor | None:
    if not can_fuse_head(channels, fn_weight, scale, base):
        return None
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return rtp_llm_ops.fuse_hy4_ihc_head(
        channels, fn_weight, scale, base, norm_eps, hc_eps
    )


def can_fuse_post_pre(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    post_gate: torch.Tensor,
    next_fn_weight: torch.Tensor,
    next_scale: torch.Tensor,
    next_base: torch.Tensor,
    rms_weight: torch.Tensor,
) -> bool:
    hidden = residual.shape[2] if residual.dim() == 3 else -1
    return (
        _enabled()
        and not _has_enabled_grad(
            block_output,
            residual,
            post_gate,
            next_fn_weight,
            next_scale,
            next_base,
            rms_weight,
        )
        and block_output.dim() == 2
        and residual.dim() == 3
        and residual.shape[1] == _HC_MULT
        and hidden in _SUPPORTED_HIDDEN
        and residual.shape[0] > 0
        and tuple(block_output.shape) == (residual.shape[0], hidden)
        and block_output.dtype == torch.bfloat16
        and block_output.is_contiguous()
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
        and tuple(post_gate.shape) == (residual.shape[0], _HC_MULT)
        and post_gate.dtype == torch.float32
        and post_gate.is_contiguous()
        and tuple(next_fn_weight.shape) == (2 * _HC_MULT, _HC_MULT * hidden)
        and next_fn_weight.dtype == torch.float32
        and next_fn_weight.is_contiguous()
        and next_scale.dtype == torch.float32
        and next_scale.is_contiguous()
        and next_scale.numel() == 2
        and next_base.dtype == torch.float32
        and next_base.is_contiguous()
        and next_base.numel() == 2 * _HC_MULT
        and tuple(rms_weight.shape) == (hidden,)
        and rms_weight.dtype == torch.bfloat16
        and rms_weight.is_contiguous()
        and _same_cuda_device(
            block_output,
            residual,
            post_gate,
            next_fn_weight,
            next_scale,
            next_base,
            rms_weight,
        )
        and _is_sm103(residual)
        and _runtime_has_ops()
    )


def maybe_fuse_post_pre(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    post_gate: torch.Tensor,
    next_fn_weight: torch.Tensor,
    next_scale: torch.Tensor,
    next_base: torch.Tensor,
    *,
    ihc_norm_eps: float,
    hc_eps: float,
    magnitude: float,
    rms_weight: torch.Tensor,
    rms_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not can_fuse_post_pre(
        block_output,
        residual,
        post_gate,
        next_fn_weight,
        next_scale,
        next_base,
        rms_weight,
    ):
        return None
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    return rtp_llm_ops.fuse_hy4_ihc_post_pre(
        block_output,
        residual,
        post_gate,
        next_fn_weight,
        next_scale,
        next_base,
        ihc_norm_eps,
        hc_eps,
        magnitude,
        rms_weight,
        rms_eps,
        True,
    )
