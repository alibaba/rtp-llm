"""Register the PPU Linear implementations with the open-source factory.

The public linear package drains the ``linear`` backend slot right after it
builds LinearFactory, so record a hook instead of wrapping the module loader.
"""

import logging

from rtp_llm.utils.backend_registry import register_backend_hook

_registered = False


def _register_impls(factory, **_ignored) -> None:
    from rtp_llm.models_py.modules.factory.linear.ppu_impl.int8_deepgemm_linear import (
        PpuInt8DeepGemmLinear,
    )

    factory.register(PpuInt8DeepGemmLinear)
    logging.info("[ppu linear register] Registered W8A8 INT8 DeepGEMM Linear")

    # The unquantized implementation uses torch F.linear on PPU as well.
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import (
        CudaF16Linear,
    )

    factory.register(CudaF16Linear)
    logging.info("[ppu linear register] Registered unquantized torch Linear")


def install() -> None:
    """Record the registration hook. Must run before the factory is imported."""
    global _registered
    if _registered:
        return
    _registered = True
    register_backend_hook("linear", _register_impls)
    logging.info("[ppu linear register] Linear backend hook recorded")
