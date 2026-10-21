"""Native BF16 K512 selection for candidate-only V4.1 prefill scores.

This helper returns candidate *column* indices; the sparse remap must check
``0 <= col < min(max(end, 0), width)`` and filter selected nonfinite scores.
There is no sorting guarantee, and exact ties may choose any legal members.
Long rows containing NaN produce a fully initialized all--1 row. Short rows
return their prefix plus -1 padding, including nonfinite prefix entries for
the remap to filter. The NaN exception contract differs from torch.topk.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from rtp_llm.ops.compute_ops import rtp_llm_ops


@lru_cache(maxsize=None)
def _device_supported(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))


def is_available(device: torch.device) -> bool:
    """Gate chunk planning before allocating sparse scores; no GPU data reads."""
    if torch.version.hip is not None or device.type != "cuda":
        return False
    available = getattr(rtp_llm_ops, "deepselect_bf16_available", None)
    return available is not None and available() and _device_supported(device)


def is_supported(
    logits: torch.Tensor, end: torch.Tensor, out: torch.Tensor | None = None
) -> bool:
    """Metadata-only gate; it does not read GPU bounds or scores on the host."""
    if not (
        torch.version.hip is None
        and logits.is_cuda
        and logits.dtype == torch.bfloat16
        and logits.ndim == 2
        and 0 < logits.shape[0] <= 2**31 - 1
        and 512 <= logits.shape[1] < 2**23
        and logits.shape[1] % 512 == 0
        and logits.stride(1) == 1
        and logits.stride(0) >= logits.shape[1]
        and logits.stride(0) % 512 == 0
        and logits.data_ptr() % 16 == 0
        and end.device == logits.device
        and end.dtype == torch.int32
        and end.ndim == 1
        and end.shape[0] == logits.shape[0]
        and end.is_contiguous()
    ):
        return False
    if out is not None and not (
        out.device == logits.device
        and out.dtype == torch.int32
        and out.shape == (logits.shape[0], 512)
        and out.stride(1) == 1
        and out.stride(0) >= 512
        and out.stride(0) % 8 == 0
        and out.data_ptr() % 32 == 0
    ):
        return False
    return is_available(logits.device)


def try_select_sparse_tokens(
    logits: torch.Tensor, end: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor | None:
    """Select BF16 scores directly, or return None before any unsupported launch.

    Once selected, execution errors propagate. All bounds are clamped inside
    the native selecting kernel; there is no bounds-copy kernel or host sync.
    ``out`` permits stable allocation for CUDA graph replay.
    """
    if not is_supported(logits, end, out):
        return None
    if out is None:
        out = torch.empty(
            (logits.shape[0], 512), dtype=torch.int32, device=logits.device
        )
    rtp_llm_ops.deepselect_bf16(logits, end, out)
    return out
