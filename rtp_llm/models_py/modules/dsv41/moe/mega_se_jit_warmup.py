"""Mega MoE SE JIT-warmup helpers.

The token-to-kernel heuristic is shared with routed-only Mega MoE, while the
SE strategy owns a distinct warmed-key cache because shared templates compile
to different DeepGEMM code.
"""

from __future__ import annotations

from .mega_jit_warmup import (
    clamp_token_counts,
    format_token_counts,
    generate_mega_moe_jit_token_counts,
    mega_moe_jit_warmup_enabled,
    parse_mega_moe_jit_warmup_tokens_override,
)


def mega_moe_se_jit_warmup_enabled() -> bool:
    return mega_moe_jit_warmup_enabled()


def generate_mega_moe_se_jit_token_counts(**kwargs) -> list[int]:
    return generate_mega_moe_jit_token_counts(**kwargs)


def parse_mega_moe_se_jit_warmup_tokens_override() -> list[int] | None:
    return parse_mega_moe_jit_warmup_tokens_override()


__all__ = [
    "clamp_token_counts",
    "format_token_counts",
    "generate_mega_moe_se_jit_token_counts",
    "mega_moe_se_jit_warmup_enabled",
    "parse_mega_moe_se_jit_warmup_tokens_override",
]
