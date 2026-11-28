"""Shared checkpoint YaRN schema for DeepSeek MLA descriptors and parameter owners."""

import math
from typing import Optional


def checkpoint_rope_parameters(config: dict) -> dict:
    value = config.get("rope_parameters")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("DeepSeek rope_parameters must be a dictionary")
    return value


_YARN_REQUIRED_KEYS = (
    "factor",
    "original_max_position_embeddings",
    "mscale",
    "mscale_all_dim",
)
_YARN_TYPES = {"yarn", "deepseek_yarn"}


def _rope_type(config: dict) -> Optional[str]:
    rope_type = config.get("rope_type", config.get("type"))
    if rope_type is None:
        return None
    if not isinstance(rope_type, str):
        raise TypeError(f"DeepSeek rope_type must be a string, got {rope_type!r}")
    return rope_type.strip().lower()


def resolve_yarn_parameters(config_json: dict) -> Optional[dict]:
    """Resolve legacy rope_scaling and Transformers rope_parameters safely."""
    rope_scaling = config_json.get("rope_scaling")
    rope_parameters = checkpoint_rope_parameters(config_json)
    if rope_scaling is not None and not isinstance(rope_scaling, dict):
        raise TypeError("DeepSeek rope_scaling must be a dictionary")

    parameter_type = _rope_type(rope_parameters)
    parameter_has_scaling = parameter_type in _YARN_TYPES or any(
        key in rope_parameters for key in _YARN_REQUIRED_KEYS
    )
    if (
        parameter_has_scaling
        and parameter_type is not None
        and parameter_type not in _YARN_TYPES
    ):
        raise ValueError(
            f"unsupported DeepSeek rope_parameters rope_type={parameter_type!r}"
        )

    if rope_scaling is None:
        yarn = rope_parameters if parameter_has_scaling else None
    else:
        scaling_type = _rope_type(rope_scaling)
        if scaling_type is not None and scaling_type not in _YARN_TYPES:
            raise ValueError(
                f"unsupported DeepSeek rope_scaling rope_type={scaling_type!r}"
            )
        yarn = rope_scaling
        if parameter_has_scaling:
            for key in _YARN_REQUIRED_KEYS:
                if (
                    key in rope_scaling
                    and key in rope_parameters
                    and rope_scaling[key] != rope_parameters[key]
                ):
                    raise ValueError(
                        "conflicting DeepSeek RoPE scaling values for "
                        f"{key}: rope_scaling={rope_scaling[key]!r}, "
                        f"rope_parameters={rope_parameters[key]!r}"
                    )

    if yarn is None:
        return None
    return validate_yarn_parameters(yarn)


def validate_yarn_parameters(yarn: dict) -> dict:
    missing = [key for key in _YARN_REQUIRED_KEYS if key not in yarn]
    if missing:
        raise ValueError(f"DeepSeek YaRN configuration is missing keys: {missing}")
    factor = yarn["factor"]
    original_max = yarn["original_max_position_embeddings"]
    if (
        isinstance(factor, bool)
        or not isinstance(factor, (int, float))
        or not math.isfinite(float(factor))
        or factor <= 0
    ):
        raise ValueError(f"DeepSeek YaRN factor must be positive, got {factor!r}")
    for name in ("mscale", "mscale_all_dim"):
        value = yarn[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(
                f"DeepSeek YaRN {name} must be non-negative, got {value!r}"
            )
    if (
        isinstance(original_max, bool)
        or not isinstance(original_max, int)
        or original_max <= 0
    ):
        raise ValueError(
            "DeepSeek YaRN original_max_position_embeddings must be a "
            f"positive integer, got {original_max!r}"
        )
    return yarn
