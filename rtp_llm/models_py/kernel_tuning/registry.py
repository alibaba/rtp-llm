import logging
import os
from typing import Optional

import torch

from rtp_llm.models_py.kernel_tuning.aiter.fmoe import configure_aiter_fmoe_overlays
from rtp_llm.models_py.kernel_tuning.types import KernelTuningStatus

_LOGGER = logging.getLogger(__name__)

ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV = "RTP_LLM_ROCM_FP8_MOE_DETERMINISTIC_REDUCE"

_PROVIDERS_BY_ARCH = {
    "gfx942": (configure_aiter_fmoe_overlays,),
}


def is_rocm_fp8_moe_deterministic_reduce_enabled() -> bool:
    return os.environ.get(ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV, "0") == "1"


def _current_rocm_arch() -> Optional[str]:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return None
    try:
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0] or None
    except Exception as error:
        _LOGGER.warning("Failed to resolve the current ROCm architecture: %s", error)
        return None


def configure_kernel_tuning(
    arch: Optional[str] = None,
) -> tuple[KernelTuningStatus, ...]:
    """Configure registered kernel-tuning providers for the current device."""

    if not is_rocm_fp8_moe_deterministic_reduce_enabled():
        return ()

    resolved_arch = arch if arch is not None else _current_rocm_arch()
    statuses = tuple(
        provider() for provider in _PROVIDERS_BY_ARCH.get(resolved_arch, ())
    )
    for status in statuses:
        if not status.applied:
            _LOGGER.warning(
                "Kernel tuning overlay %s is inactive: %s",
                status.overlay,
                status.reason,
            )
    return statuses
