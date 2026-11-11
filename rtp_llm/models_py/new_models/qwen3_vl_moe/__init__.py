"""Qwen3-VL MoE newloader language implementation."""

from typing import Any

__all__ = ["Qwen3VLMoeForCausalLM"]


def __getattr__(name: str) -> Any:
    if name == "Qwen3VLMoeForCausalLM":
        from rtp_llm.models_py.new_models.qwen3_vl_moe.model import (
            Qwen3VLMoeForCausalLM,
        )

        return Qwen3VLMoeForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
