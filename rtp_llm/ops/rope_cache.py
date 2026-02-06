import math
from functools import lru_cache
from typing import Optional

import torch
from libth_transformer_config import RopeConfig, RopeStyle


@lru_cache(maxsize=1)
def get_base_rope_cache(
    rope_dim: int, rope_base: int, rope_scale: float, max_position_embeddings: int
) -> torch.Tensor:
    inv_freq = 1.0 / (
        rope_base ** (torch.arange(0, rope_dim, 2, dtype=torch.float) / rope_dim)
    )
    t = torch.arange(max_position_embeddings * rope_scale, dtype=torch.float).div_(
        rope_scale
    )
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    rope_cache = (
        torch.stack([cos, sin], dim=0)
        .permute([1, 2, 0])
        .reshape([cos.size(0), -1])
        .contiguous()
    )
    return rope_cache.cuda()


def yarn_find_correction_dim(
    num_rotations: float, rope_dim: int, rope_theta: int, max_position_embeddings: int
) -> float:
    return (
        rope_dim
        * math.log(max_position_embeddings / (num_rotations * 2.0 * math.pi))
        / (2.0 * math.log(rope_theta))
    )


@lru_cache(maxsize=1)
def get_yarn_rope_cache(
    rope_dim: int,
    rope_theta: int,
    rope_scale: float,
    max_position_embeddings: int,
    beta_slow: float,
    beta_fast: float,
    extrapolation_factor: float,
    mscale: float,
) -> torch.Tensor:
    pos_freqs = torch.pow(
        rope_theta,
        torch.arange(0, rope_dim, 2, dtype=torch.int64).to(torch.float32) / rope_dim,
    )
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (rope_scale * pos_freqs)

    low = max(
        0,
        math.floor(
            yarn_find_correction_dim(
                beta_slow, rope_dim, rope_theta, max_position_embeddings
            )
        ),
    )
    high = min(
        rope_dim - 1,
        math.ceil(
            yarn_find_correction_dim(
                beta_fast, rope_dim, rope_theta, max_position_embeddings
            )
        ),
    )
    low = float(low)
    high = float(high)
    if abs(low - high) < 1e-6:
        high += 0.001

    linear = (
        torch.arange(rope_dim // 2, dtype=torch.int64).to(torch.float32) - low
    ) / (high - low)
    ramp = torch.clamp(linear, 0, 1)
    inv_freq_mask = (1.0 - ramp) * extrapolation_factor
    inv_freq = (
        inv_freq_interpolation * (1.0 - inv_freq_mask)
        + inv_freq_extrapolation * inv_freq_mask
    )
    t = torch.arange(max_position_embeddings * rope_scale, dtype=torch.int64).to(
        torch.float32
    )
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(torch.float32) * mscale
    sin = freqs.sin().to(torch.float32) * mscale
    rope_cache = (
        torch.stack([cos, sin], dim=0)
        .permute([1, 2, 0])
        .reshape([cos.size(0), -1])
        .contiguous()
    )
    return rope_cache.cuda()


def use_rope_cache(rope_config: RopeConfig, is_cuda: bool):
    if is_cuda:
        return rope_config.style in [RopeStyle.Base, RopeStyle.Yarn]
    else:
        return rope_config.style == RopeStyle.Base


def get_rope_cache(
    rope_config: RopeConfig, max_position_embeddings: int, is_cuda: bool = True
) -> Optional[torch.Tensor]:
    if not use_rope_cache(rope_config, is_cuda):
        return None

    if rope_config.style == RopeStyle.Base:
        return get_base_rope_cache(
            rope_config.dim,
            rope_config.base,
            rope_config.scale,
            max_position_embeddings,
        )
    elif rope_config.style == RopeStyle.Yarn:
        return get_yarn_rope_cache(
            rope_config.dim,
            rope_config.base,
            rope_config.scale,
            rope_config.max_pos,
            rope_config.factor1,
            rope_config.factor2,
            rope_config.extrapolation_factor,
            rope_config.mscale,
        )
    else:
        raise ValueError(f"unsupported rope_style = {rope_config.style}")
