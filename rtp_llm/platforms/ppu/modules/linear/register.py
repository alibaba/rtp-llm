"""Register the PPU Linear implementations with the open-source factory.

The public linear package drains the ``linear`` backend slot right after it
builds LinearFactory, so record a hook instead of wrapping the module loader.
"""

import logging

from rtp_llm.utils.backend_registry import register_backend_hook


def _register_impls(factory, **_ignored) -> None:
    from rtp_llm.device.device_type import DeviceType, get_device_type

    if get_device_type() != DeviceType.Ppu:
        return
    from rtp_llm.platforms.ppu.modules.linear.int8_deepgemm_linear import (
        PpuInt8DeepGemmLinear,
    )

    factory.register(PpuInt8DeepGemmLinear)
    logging.info("[ppu linear register] Registered W8A8 INT8 DeepGEMM Linear")


def install() -> None:
    """Record the registration hook. Must run before the factory is imported."""
    register_backend_hook("linear", _register_impls)
    logging.info("[ppu linear register] Linear backend hook recorded")
