"""Fuse a V4.1 attention post-mix and the following FFN pre-mix/RMSNorm.

This seam is internal to one block, so Engram injection, auxiliary hidden
capture and the final delayed readout keep their existing boundaries. Both
flat prefill and batched decode layouts use the same BF16-only operation.

DeepGEMM initializes persistent split barriers per CUDA stream. Warm up on
the capture stream before recording a graph; an unwarmed capture stream
retains the unfused path instead of initializing barriers during capture.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Callable

import torch

from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit
from rtp_llm.models_py.modules.dsv4.hc.v41_prenorm import prepare_tf32_weight

_WARMED_STREAMS: set[tuple[int, int]] = set()


@lru_cache(maxsize=1)
def _get_mega_mhc() -> Callable | None:
    try:
        module = importlib.import_module("deep_gemm")
    except ModuleNotFoundError as error:
        if error.name == "deep_gemm":
            return None
        raise
    operation = getattr(module, "mega_mhc", None)
    return operation if callable(operation) else None


def _is_dense_tensor(value, shape, dtype, device) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and tuple(value.shape) == shape
        and value.dtype == dtype
        and value.device == device
        and value.is_contiguous()
    )


def _stream_key(device: torch.device) -> tuple[int, int]:
    stream = torch.cuda.current_stream(device)
    return stream.device.index, stream.cuda_stream


def is_supported(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    previous: DelayedHCUnit,
    next_hc: DelayedHCUnit,
    norm: torch.nn.Module,
) -> bool:
    """Check the exact delayed-HC/RMSNorm contract without reading GPU data."""
    if (
        not isinstance(attn_out, torch.Tensor)
        or attn_out.device.type != "cuda"
        or attn_out.dtype != torch.bfloat16
        or attn_out.ndim not in (2, 3)
        or attn_out.shape[-1] != 5120
        or not attn_out.is_contiguous()
        or not 0 < attn_out.numel() // 5120 <= 1 << 20
    ):
        return False
    # Subclasses may change either mix equations or norm rounding, so only
    # bypass the implementations whose contracts this fusion reproduces.
    if type(previous) is not DelayedHCUnit or type(next_hc) is not DelayedHCUnit:
        return False
    if (
        previous.dim != 5120
        or next_hc.dim != 5120
        or previous.hc_mult != 4
        or next_hc.hc_mult != 4
        or next_hc._previous_ref is None
        or next_hc._previous_ref() is not previous
        or next_hc.hc_sinkhorn_iters < 1
        or type(norm).__module__ != "rtp_llm.models_py.modules.base.cuda.norm"
        or type(norm).__name__ != "RMSNorm"
    ):
        return False
    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

    if type(norm) is not RMSNorm:
        return False
    device = attn_out.device
    leading = tuple(attn_out.shape[:-1])
    contracts = (
        (residual, leading + (4, 5120), torch.bfloat16),
        (post, leading + (4, 1), torch.float32),
        (comb, leading + (4, 4), torch.float32),
        (previous.pre_mix_out, leading + (4,), torch.float32),
        (next_hc.fn, (24, 20480), torch.float32),
        (next_hc.scale, (3,), torch.float32),
        (next_hc.base, (24,), torch.float32),
        (norm.weight, (5120,), torch.bfloat16),
    )
    if any(
        not _is_dense_tensor(tensor, shape, dtype, device)
        for tensor, shape, dtype in contracts
    ):
        return False
    if torch.is_grad_enabled() and (
        attn_out.requires_grad
        or any(tensor.requires_grad for tensor, _, _ in contracts)
    ):
        return False
    if any(
        not isinstance(eps, (int, float)) or eps < 0
        for eps in (next_hc.norm_eps, next_hc.hc_eps, norm.variance_epsilon)
    ):
        return False
    if torch.cuda.get_device_capability(device)[0] != 10 or _get_mega_mhc() is None:
        return False
    with torch.cuda.device(device):
        return (
            not torch.cuda.is_current_stream_capturing()
            or _stream_key(device) in _WARMED_STREAMS
        )


def try_fused_post_pre(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    previous: DelayedHCUnit,
    next_hc: DelayedHCUnit,
    norm: torch.nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return ``(residual, normalized_input, post, comb)`` or use the old path.

    A successful call also publishes the FFN pre-mix for the next sublayer's
    delayed readout. Unsupported inputs return ``None``; backend execution
    failures propagate to the caller. Inputs are never overwritten.
    """
    if not is_supported(attn_out, residual, post, comb, previous, next_hc, norm):
        return None
    operation = _get_mega_mhc()
    leading = tuple(attn_out.shape[:-1])
    tokens = attn_out.numel() // 5120
    new_residual = torch.empty_like(residual)
    new_pre = torch.empty_like(previous.pre_mix_out)
    new_post = torch.empty_like(post)
    new_comb = torch.empty_like(comb)
    normalized = torch.empty_like(attn_out)
    # Both the small-token DeepGEMM prenorm and the large-prefill TileLang
    # fallback use the same biased TF32 weight rounding. Cache that boundary
    # instead of feeding raw FP32 weights to TMA's TF32 conversion.
    weight = prepare_tf32_weight(next_hc.fn)
    with torch.cuda.device(attn_out.device):
        capturing = torch.cuda.is_current_stream_capturing()
        operation(
            x=attn_out.view(tokens, 5120),
            residual=residual.view(tokens, 4, 5120),
            shifted_prev_mix=previous.pre_mix_out.view(tokens, 4, 1),
            post_mix=post.view(tokens, 4, 1),
            comb_res_mix=comb.view(tokens, 4, 4),
            fn=weight,
            mix_scales=next_hc.scale,
            mix_bases=next_hc.base,
            hc_mult=4,
            hc_norm_eps=next_hc.norm_eps,
            hc_pre_eps=next_hc.hc_eps,
            hc_post_scale=2.0,
            sinkhorn_eps=next_hc.hc_eps,
            num_sinkhorn_iters=next_hc.hc_sinkhorn_iters,
            rmsnorm_weight=norm.weight,
            rmsnorm_eps=norm.variance_epsilon,
            rmsnorm_scale=1.0,
            new_residual=new_residual.view(tokens, 4, 5120),
            new_prev_mix=new_pre.view(tokens, 4, 1),
            new_post_mix=new_post.view(tokens, 4, 1),
            new_comb_res_mix=new_comb.view(tokens, 4, 4),
            y_bf16=normalized.view(tokens, 5120),
        )
        if not capturing:
            _WARMED_STREAMS.add(_stream_key(attn_out.device))
    next_hc.pre_mix_out = new_pre.view(*leading, 4)
    return new_residual, normalized, new_post, new_comb


__all__ = ["is_supported", "try_fused_post_pre"]
