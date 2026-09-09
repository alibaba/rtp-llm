"""Shared workspace helpers for Kimi K3 collective GEMM operators."""

from __future__ import annotations

from typing import Any

import torch


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


__all__ = ["collective_gemm_state_key"]
