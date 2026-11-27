"""Build the shared MLA runtime cache from checkpoint and serving settings."""

import torch

from rtp_llm.config.deepseek_yarn import (
    checkpoint_rope_parameters,
    resolve_yarn_parameters,
    validate_yarn_parameters,
)

from .rotary_embedding import DeepseekV3RotaryEmbedding, DeepseekV3YarnRotaryEmbedding


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
    rope_parameters = checkpoint_rope_parameters(config_json)
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

    yarn_parameters = resolve_yarn_parameters(config_json)
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
            yarn_parameters = validate_yarn_parameters(
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
