"""DeepSeek-VL2 newloader language and vision implementations."""

from typing import Any

__all__ = ["DeepSeekVLV2ForCausalLM"]


def __getattr__(name: str) -> Any:
    if name == "DeepSeekVLV2ForCausalLM":
        from .language import DeepSeekVLV2ForCausalLM

        return DeepSeekVLV2ForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
