"""Qwen2 model export; format helpers do not require the native scheduler."""

__all__ = ["Qwen2ForCausalLM"]


def __getattr__(name):
    if name == "Qwen2ForCausalLM":
        from .language import Qwen2ForCausalLM

        return Qwen2ForCausalLM
    raise AttributeError(name)
