"""Register the M890P provider when a DSV4 model consumes its backend slot."""

from rtp_llm.utils.backend_registry import register_backend_hook


def _register() -> None:
    from rtp_llm.device.device_type import DeviceType, get_device_type

    if get_device_type() != DeviceType.Ppu:
        return
    import torch
    from rtp_llm.platforms.ppu.models.dsv4.ppu_provider import (
        register_m890p_dsv4_provider,
    )

    register_m890p_dsv4_provider(device_name=torch.cuda.get_device_name(), ep_size=1)


def install() -> None:
    register_backend_hook("dsv4", _register)
