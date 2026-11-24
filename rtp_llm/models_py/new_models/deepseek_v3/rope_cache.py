"""Build the shared MLA runtime cache from checkpoint and serving settings."""

import math
from typing import Optional

import torch

from .rotary_embedding import DeepseekV3RotaryEmbedding, DeepseekV3YarnRotaryEmbedding

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


def _resolve_yarn_parameters(config_json: dict) -> Optional[dict]:
    """Resolve legacy rope_scaling and Transformers rope_parameters safely."""
    rope_scaling = config_json.get("rope_scaling")
    rope_parameters = config_json.get("rope_parameters", {})
    if rope_scaling is not None and not isinstance(rope_scaling, dict):
        raise TypeError("DeepSeek rope_scaling must be a dictionary")
    if not isinstance(rope_parameters, dict):
        raise TypeError("DeepSeek rope_parameters must be a dictionary")

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
    return _validate_yarn_parameters(yarn)


def _validate_yarn_parameters(yarn: dict) -> dict:
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


def build_rope_cache(
    config_json: dict,
    max_seq_len: int,
    device: torch.device,
    *,
    model_config=None,
) -> torch.Tensor:
    """Build RoPE cos/sin cache with any finalized serving YaRN override.

    Returns [cos|sin] concatenated, shape [max_seq_len, rope_head_dim], float32.
    """
    rope_parameters = config_json.get("rope_parameters", {})
    if not isinstance(rope_parameters, dict):
        raise TypeError("DeepSeek rope_parameters must be a dictionary")
    rope_theta = rope_parameters.get(
        "rope_theta", config_json.get("rope_theta", 10000.0)
    )
    if (
        isinstance(rope_theta, bool)
        or not isinstance(rope_theta, (int, float))
        or rope_theta <= 0
    ):
        raise ValueError(f"DeepSeek rope_theta must be positive, got {rope_theta!r}")
    rope_head_dim = config_json.get(
        "qk_rope_head_dim", config_json.get("rope_head_dim")
    )
    if (
        isinstance(rope_head_dim, bool)
        or not isinstance(rope_head_dim, int)
        or rope_head_dim <= 0
        or rope_head_dim % 2
    ):
        raise ValueError(
            "DeepSeek RoPE head dimension must be a positive even integer, "
            f"got {rope_head_dim!r}"
        )

    yarn_parameters = _resolve_yarn_parameters(config_json)
    rope = getattr(getattr(model_config, "attn_config", None), "rope_config", None)
    if rope is not None:
        from rtp_llm.ops import RopeStyle

        # DeepSeek normally leaves the native style as No and supplies the
        # cache explicitly. A serving YaRN override sets the native style and
        # these four fields *after* checkpoint config parsing. Keep the
        # checkpoint's DeepSeek-specific magnitude coefficients (also for
        # modern rope_parameters), just as the original weight owner did.
        if rope.style == RopeStyle.Yarn:
            checkpoint_yarn = yarn_parameters or {
                "mscale": model_config.deepseek_rope_mscale,
                "mscale_all_dim": model_config.deepseek_mscale_all_dim,
            }
            yarn_parameters = _validate_yarn_parameters(
                {
                    **checkpoint_yarn,
                    "factor": rope.scale,
                    "original_max_position_embeddings": rope.max_pos,
                    "beta_slow": rope.factor1,
                    "beta_fast": rope.factor2,
                }
            )

    if yarn_parameters is None:
        rotary_emb = DeepseekV3RotaryEmbedding(
            dim=rope_head_dim,
            max_position_embeddings=max_seq_len,
            base=rope_theta,
            device=device,
        )
    else:
        # Match the legacy loader's CPU sin/cos evaluation exactly, then move
        # the finished cache to CUDA before the indexer captures its reference.
        rotary_emb = DeepseekV3YarnRotaryEmbedding(
            rope_head_dim,
            max_seq_len,
            rope_theta,
            scaling_factor=yarn_parameters["factor"],
            original_max_position_embeddings=yarn_parameters[
                "original_max_position_embeddings"
            ],
            beta_fast=float(yarn_parameters.get("beta_fast", 32)),
            beta_slow=float(yarn_parameters.get("beta_slow", 1)),
            mscale=yarn_parameters["mscale"],
            mscale_all_dim=yarn_parameters["mscale_all_dim"],
        )

    half_rope_dim = rope_head_dim // 2
    cos_cache = rotary_emb.cos_cached[:, :half_rope_dim]
    sin_cache = rotary_emb.sin_cached[:, :half_rope_dim]
    return (
        torch.cat([cos_cache, sin_cache], dim=-1)
        .contiguous()
        .to(device=device, dtype=torch.float32)
    )
