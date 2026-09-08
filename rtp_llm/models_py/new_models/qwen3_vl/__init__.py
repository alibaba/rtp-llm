"""Qwen3-VL newloader language and vision implementations."""

from typing import Any

__all__ = ["Qwen3VLForCausalLM"]


def __getattr__(name: str) -> Any:
    if name == "Qwen3VLForCausalLM":
        from rtp_llm.models_py.new_models.qwen3_vl.model import Qwen3VLForCausalLM

        return Qwen3VLForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
