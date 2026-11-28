"""Config helpers for direct parameter owners; runtime helpers stay in main."""

from typing import Any

from rtp_llm.models_py.model_desc.block_map import select_fmha_impl_for_layer


def required_config_value(config: Any, *names: str) -> Any:
    for name in names:
        value = (
            config.get(name)
            if isinstance(config, dict)
            else getattr(config, name, None)
        )
        if value is not None:
            return value
    raise ValueError(f"Model config requires one of {names}")


def positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def resolve_head_dim(hidden_size: Any, num_heads: Any, configured: Any) -> int:
    hidden_size = positive_int(hidden_size, "hidden_size")
    num_heads = positive_int(num_heads, "num_heads")
    if configured is not None:
        return positive_int(configured, "head_dim")
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size={hidden_size} must be divisible by num_heads={num_heads} "
            "when head_dim is not configured"
        )
    return hidden_size // num_heads


__all__ = ["positive_int", "required_config_value", "resolve_head_dim", "select_fmha_impl_for_layer"]
