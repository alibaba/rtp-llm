import logging
from typing import Optional

import torch

from rtp_llm.config.py_config_modules import KernelTuningConfig
from rtp_llm.models_py.kernel_tuning.aiter.fmoe import (
    configure_aiter_fmoe_overlays,
    set_aiter_fmoe_explicit_config_paths,
)
from rtp_llm.models_py.kernel_tuning.types import KernelTuningStatus

_LOGGER = logging.getLogger(__name__)

ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV = "RTP_LLM_ROCM_FP8_MOE_DETERMINISTIC_REDUCE"

_DEFAULT_MAX_TOKENS = 128

_PROVIDERS_BY_ARCH = {
    "gfx942": (configure_aiter_fmoe_overlays,),
}

# Server-startup kernel-tuning configuration, injected exactly once by
# configure_kernel_tuning() from the parsed PyEnvConfigs (paired CLI/env
# registration in server_args/kernel_tuning_group_args.py). ``None`` means
# the server has not started yet; the deterministic helpers below then keep
# their disabled defaults instead of reaching for the raw process environment.
_kernel_tuning_config: Optional[KernelTuningConfig] = None


def is_rocm_fp8_moe_deterministic_reduce_enabled() -> bool:
    return (
        _kernel_tuning_config is not None
        and _kernel_tuning_config.rocm_fp8_moe_deterministic_reduce
    )


def rocm_fp8_moe_deterministic_max_tokens() -> int:
    if _kernel_tuning_config is None:
        return _DEFAULT_MAX_TOKENS
    return max(1, _kernel_tuning_config.rocm_fp8_moe_deterministic_max_tokens)


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
    config: KernelTuningConfig,
    arch: Optional[str] = None,
) -> tuple[KernelTuningStatus, ...]:
    """Configure registered kernel-tuning providers for the current device.

    ``config`` is the parsed server configuration
    (``PyEnvConfigs.kernel_tuning_config``). The injection also publishes the
    user-provided AITER FMoE config paths to the aiter overlay module, so the
    whole kernel-tuning stack consumes the parsed config instead of the raw
    process environment for the rest of the process lifetime.
    """

    global _kernel_tuning_config
    _kernel_tuning_config = config
    set_aiter_fmoe_explicit_config_paths(config.aiter_fmoe_config_paths)

    if not config.rocm_fp8_moe_deterministic_reduce:
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
