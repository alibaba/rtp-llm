"""ROCm architecture queries independent of weight loaders and DeviceImpl."""

import os
from typing import Optional

import torch


def is_gfx950(
    arch_fallback: Optional[str] = None,
    *,
    device: Optional[torch.device] = None,
) -> bool:
    """Query the target device; allow the legacy fallback only without a target.

    An explicit device is used when materializing runtime weights, where an
    unavailable device must not silently choose another device's FP8 format.
    The no-target form also works during CPU-only configuration discovery.
    """
    target = None if device is None else torch.device(device)
    if target is not None and target.type != "cuda":
        raise ValueError(
            f"ROCm architecture query requires a CUDA device, got {target}"
        )
    try:
        index = (
            torch.cuda.current_device()
            if target is None or target.index is None
            else target.index
        )
        properties = torch.cuda.get_device_properties(index)
        return "gfx950" in getattr(properties, "gcnArchName", "")
    except Exception as error:
        if target is not None:
            raise RuntimeError(
                f"Unable to query ROCm architecture for {target}: {error}"
            ) from error
        architecture = (
            arch_fallback
            if arch_fallback is not None
            else os.environ.get("ROCM_GFX_ARCH", "")
        )
        return architecture == "950"
