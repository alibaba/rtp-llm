"""Compatibility setup for ROCm AITer MoE kernels."""

import os
from importlib.metadata import PackageNotFoundError, version
from typing import Optional


_OPUS_PYBIND_BROKEN_VERSION_PREFIX = "0.1.17.dev79+g2570b35f9"


def _current_rocm_arch() -> str:
    try:
        import torch

        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        return str(getattr(properties, "gcnArchName", ""))
    except Exception:
        return os.environ.get("ROCM_GFX_ARCH", "")


def _installed_aiter_version() -> str:
    try:
        return version("aiter")
    except PackageNotFoundError:
        return ""


def configure_aiter_moe_sorting(
    *, arch: Optional[str] = None, aiter_version: Optional[str] = None
) -> bool:
    """Use CK sorting for the affected AITer build on gfx942.

    AITer 0.1.17.dev79+g2570b35f9 defaults to its new Opus sorter.  On gfx942,
    that build's JIT pybind module rejects the converted ``aiter_tensor_t``
    arguments before launching the kernel.  CK sorting remains bundled and is
    selected by AITer's public environment switch.

    This must run before importing :mod:`aiter.fused_moe`.  Explicit user
    configuration always wins, and newer AITer builds keep their own default.
    """
    if "AITER_USE_CK_MOE_SORTING" in os.environ:
        return False

    resolved_arch = (arch if arch is not None else _current_rocm_arch()).lower()
    is_gfx942 = resolved_arch == "942" or "gfx942" in resolved_arch
    resolved_version = (
        aiter_version if aiter_version is not None else _installed_aiter_version()
    )
    if not is_gfx942 or not resolved_version.startswith(
        _OPUS_PYBIND_BROKEN_VERSION_PREFIX
    ):
        return False

    os.environ["AITER_USE_CK_MOE_SORTING"] = "1"
    return True
