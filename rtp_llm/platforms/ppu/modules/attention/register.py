"""Register the PPU FA3 attention backends with the open-source factory.

The open-source attention package picks its implementations in a
``get_device_type()`` branch, which must not grow a PPU arm. Instead it drains
the ``attention`` backend slot after building its lists, so register a hook
here and let it hand the lists over.

Ordering in those lists is priority: earlier wins. FA3 inserts at the front to
outrank the FlashInfer impls the open-source CUDA branch appends (PPU is not
ROCm, so it takes that branch). Each impl still self-filters via
``support_parallelism_config`` / its own availability check, so FA3 losing
flash_attn_3 falls back to FlashInfer rather than failing.

Every backend is env-gated: set PPU_USE_FA3_DECODE / PPU_USE_FA3_PREFILL /
PPU_USE_FA3_VERIFY to 0 to disable it.
"""

import logging

from rtp_llm.utils.backend_registry import register_backend_hook


def _register_impls(prefill_mha_imps, decode_mha_imps, **_ignored) -> None:
    from rtp_llm.device.device_type import DeviceType, get_device_type

    if get_device_type() != DeviceType.Ppu:
        return
    from rtp_llm.platforms.ppu.modules.attention.fa3_mha import (
        FA3DecodeImpl,
        FA3PrefillPagedImpl,
        _check_fa3_available,
    )

    decode_mha_imps.insert(0, FA3DecodeImpl)
    prefill_mha_imps.insert(0, FA3PrefillPagedImpl)

    logging.info(
        "[ppu attn register] FA3 decode/prefill registered at highest priority "
        "(flash_attn_3 available: %s)",
        _check_fa3_available(),
    )


def install() -> None:
    """Record the registration hook. Must run before the factory is imported."""
    register_backend_hook("attention", _register_impls)
    logging.info("[ppu attn register] Attention backend hook recorded")
