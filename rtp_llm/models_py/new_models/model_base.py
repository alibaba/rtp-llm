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


__all__ = ["required_config_value", "select_fmha_impl_for_layer"]
