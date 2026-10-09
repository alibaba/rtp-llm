"""Ascend MoE strategies"""

from .pytorch_fallback import AscendBf16FallbackStrategy

__all__ = [
    "AscendBf16FallbackStrategy",
]
