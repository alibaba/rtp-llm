"""DeepGEMM MegaMoE JIT warmup helpers for GLM-5.

The MegaMoE kernel's heuristic maps num_tokens to template parameters.
Warmup compiles one representative per bucket to avoid JIT latency at serving time.
Ported from dsv4/moe/mega_jit_warmup.py.
"""

from __future__ import annotations

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    clamp_token_counts as clamp_token_counts,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    format_token_counts as format_token_counts,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.jit_warmup import (
    generate_mega_moe_jit_token_counts as _generate_token_counts,
)


def generate_mega_moe_jit_token_counts(
    *,
    num_ranks: int,
    num_experts: int,
    num_experts_per_rank: int,
    num_topk: int,
    intermediate_hidden: int,
    num_sms: int,
    max_tokens_per_rank: int,
) -> list[int]:
    """Reuse common heuristic buckets, preserving GLM5's full-cap warmup."""
    return _generate_token_counts(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_experts_per_rank=num_experts_per_rank,
        num_topk=num_topk,
        intermediate_hidden=intermediate_hidden,
        num_sms=num_sms,
        max_tokens_per_rank=max_tokens_per_rank,
        include_cap=True,
    )


__all__ = [
    "clamp_token_counts",
    "format_token_counts",
    "generate_mega_moe_jit_token_counts",
]
