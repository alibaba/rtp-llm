"""JIT warmup helpers dedicated to FP8xFP4 MegaMoE shared-expert kernels."""

from __future__ import annotations

import logging
import os

from .jit_warmup import (
    clamp_token_counts,
    format_token_counts,
    generate_mega_moe_jit_token_counts,
)


def mega_moe_se_jit_warmup_enabled() -> bool:
    """Enable SE warmup, falling back to the ordinary MegaMoE switch."""
    default = os.environ.get("GLM5_MEGA_MOE_JIT_WARMUP", "1")
    return os.environ.get("GLM5_MEGA_MOE_SE_JIT_WARMUP", default) != "0"


def parse_mega_moe_se_jit_warmup_tokens_override() -> list[int] | None:
    """Read the SE token buckets without sharing ordinary-path state."""
    raw_value = os.environ.get("GLM5_MEGA_MOE_SE_JIT_WARMUP_TOKENS")
    if not raw_value:
        raw_value = os.environ.get("GLM5_MEGA_MOE_JIT_WARMUP_TOKENS")
    if not raw_value:
        return None
    try:
        tokens = [int(item) for item in raw_value.replace(" ", "").split(",") if item]
    except ValueError:
        logging.warning(
            "[GLM5 MegaMoE SE] invalid JIT warmup token override %r; "
            "falling back to automatic buckets",
            raw_value,
        )
        return None
    tokens = sorted({token for token in tokens if token > 0})
    return tokens or None


def generate_mega_moe_se_jit_token_counts(**kwargs) -> list[int]:
    """Expose bucket generation through an SE-specific module."""
    return generate_mega_moe_jit_token_counts(**kwargs)


__all__ = [
    "clamp_token_counts",
    "format_token_counts",
    "generate_mega_moe_se_jit_token_counts",
    "mega_moe_se_jit_warmup_enabled",
    "parse_mega_moe_se_jit_warmup_tokens_override",
]
