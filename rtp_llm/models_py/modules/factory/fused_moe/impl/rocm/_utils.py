"""Shared helpers for ROCm fused MoE implementations."""

import torch

from rtp_llm.device.device_impl import is_gfx950


def get_rocm_fp8_dtype() -> torch.dtype:
    """Pick the FP8 dtype the current ROCm device supports.

    gfx950 (MI355X) supports the OCP e4m3 format (`float8_e4m3fn`); earlier
    archs use the FNUZ variant (`float8_e4m3fnuz`). Detection is shared with
    :func:`rtp_llm.device.device_impl.is_gfx950` so the two code paths cannot
    drift — falls back to the ``ROCM_GFX_ARCH`` env var when CUDA/ROCm is not
    available (e.g. CPU-only build environments).
    """
    return torch.float8_e4m3fn if is_gfx950() else torch.float8_e4m3fnuz
