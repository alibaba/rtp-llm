"""Ascend MoE strategies"""

from .cann import AscendCannStrategy
from .pytorch_fallback import AscendBf16FallbackStrategy
from .w8a8_mxfp8_strategy import AscendW8A8MXFP8MoeStrategy

__all__ = [
    "AscendCannStrategy",
    "AscendBf16FallbackStrategy",
    "AscendW8A8MXFP8MoeStrategy",
]
