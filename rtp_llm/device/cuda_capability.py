"""Arch-neutral CUDA compute-capability probes."""

from __future__ import annotations

from typing import Optional

import torch

from rtp_llm.device.device_type import is_cuda

SM120_CAPABILITY: tuple[int, int] = (12, 0)


def device_cuda_capability(
    device: Optional[object] = None,
) -> Optional[tuple[int, int]]:
    """Return ``(major, minor)`` for a CUDA device, or ``None`` off CUDA."""
    if not is_cuda() or not torch.cuda.is_available():
        return None
    if device is None:
        target: object = torch.cuda.current_device()
    elif isinstance(device, str):
        stripped = device.strip()
        if stripped.isdigit():
            target = int(stripped)
        else:
            parsed = torch.device(stripped)
            if parsed.type != "cuda":
                return None
            target = parsed
    elif isinstance(device, torch.device):
        if device.type != "cuda":
            return None
        target = (
            torch.cuda.current_device() if device.index is None else int(device.index)
        )
    else:
        target = device
    cap = torch.cuda.get_device_capability(target)
    return (int(cap[0]), int(cap[1]))


def current_device_is_sm120() -> bool:
    """Return True when the active worker's current CUDA device is SM120."""
    return device_cuda_capability() == SM120_CAPABILITY


def participating_cuda_capabilities(
    local_world_size: int = 0,
) -> list[tuple[int, int]]:
    """Return capabilities for participating local CUDA devices."""
    if not is_cuda() or not torch.cuda.is_available():
        return []
    device_count = torch.cuda.device_count()
    count = min(
        device_count,
        local_world_size if local_world_size > 0 else device_count,
    )
    return [
        (int(cap[0]), int(cap[1]))
        for cap in (torch.cuda.get_device_capability(idx) for idx in range(count))
    ]


def participating_devices_include_sm120(local_world_size: int = 0) -> bool:
    """Return True if any participating local CUDA device is SM120 (12, 0)."""
    return any(
        cap == SM120_CAPABILITY
        for cap in participating_cuda_capabilities(local_world_size)
    )


def participating_sm120_state(local_world_size: int = 0) -> str:
    """Return ``"none"``, ``"all"``, or ``"mixed"`` for participating local GPUs."""
    caps = participating_cuda_capabilities(local_world_size)
    if not caps:
        return "none"
    sm120_flags = [cap == SM120_CAPABILITY for cap in caps]
    if all(sm120_flags):
        return "all"
    if any(sm120_flags):
        return "mixed"
    return "none"
