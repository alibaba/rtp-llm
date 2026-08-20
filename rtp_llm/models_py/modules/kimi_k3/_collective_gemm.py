"""Shared policy helpers for Kimi K3 collective GEMM operators."""

from __future__ import annotations

from typing import Any

import torch

DEFAULT_COLLECTIVE_GEMM_MIN_M = 32 * 1024


def collective_gemm_state_key(
    group: Any,
    device: torch.device,
) -> tuple[Any, int]:
    """Return the process-local key shared by collective GEMM schedulers."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"collective GEMM requires a CUDA device, got {device}")
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return group, int(device_index)


def should_use_collective_gemm(
    physical_m: int,
    *,
    min_m: int = DEFAULT_COLLECTIVE_GEMM_MIN_M,
) -> bool:
    """Return whether physical M amortizes a fused collective GEMM launch."""

    if physical_m < 0:
        raise ValueError(f"physical_m must be non-negative, got {physical_m}")
    if min_m < 0:
        raise ValueError(f"min_m must be non-negative, got {min_m}")
    return physical_m >= min_m


__all__ = [
    "DEFAULT_COLLECTIVE_GEMM_MIN_M",
    "collective_gemm_state_key",
    "should_use_collective_gemm",
]
