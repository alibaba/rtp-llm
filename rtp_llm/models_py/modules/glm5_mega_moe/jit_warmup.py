"""DeepGEMM MegaMoE JIT warmup helpers for GLM-5.

The MegaMoE kernel's heuristic maps num_tokens to template parameters.
Warmup compiles one representative per bucket to avoid JIT latency at serving time.
Ported from dsv4/moe/mega_jit_warmup.py.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Iterable, Sequence


def mega_moe_jit_warmup_enabled() -> bool:
    return os.environ.get("GLM5_MEGA_MOE_JIT_WARMUP", "1") != "0"


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _block_config_signature(
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    num_tokens: int,
) -> tuple[int, int]:
    """Map token count to block_m/store_block_m heuristic bucket."""
    expected_tokens_per_expert = (
        float(num_tokens) * float(num_ranks) * float(num_topk) / float(num_experts)
    )
    if expected_tokens_per_expert <= 8.5:
        return 16, 8
    if expected_tokens_per_expert <= 16.5:
        return 32, 16
    if expected_tokens_per_expert <= 32.5:
        return 64, 32
    if expected_tokens_per_expert <= 64.5:
        return 96, 16
    if expected_tokens_per_expert <= 96.5:
        return 128, 32
    return 192, 32


def _num_experts_per_wave(
    num_experts_per_rank: int,
    num_tokens: int,
    num_topk: int,
    intermediate_hidden: int,
    block_m: int,
    block_n: int,
    num_sms: int,
) -> int:
    expected_tokens_per_expert = (
        float(num_tokens) * float(num_topk) / float(num_experts_per_rank)
    )
    if expected_tokens_per_expert < 1:
        return int(num_experts_per_rank)

    num_m_blocks = _ceil_div(int(math.ceil(expected_tokens_per_expert)), block_m)
    num_n_blocks = int(2 * intermediate_hidden) // int(block_n)
    num_l1_blocks_per_expert = num_m_blocks * num_n_blocks
    num_experts_per_wave = _ceil_div(2 * int(num_sms), num_l1_blocks_per_expert)
    num_experts_per_wave = min(num_experts_per_wave, int(num_experts_per_rank))

    while (
        num_experts_per_wave < int(num_experts_per_rank)
        and int(num_experts_per_rank) % num_experts_per_wave != 0
    ):
        num_experts_per_wave += 1
    return num_experts_per_wave


def mega_moe_config_signature(
    *,
    num_ranks: int,
    num_experts: int,
    num_experts_per_rank: int,
    num_tokens: int,
    num_topk: int,
    intermediate_hidden: int,
    num_sms: int,
) -> tuple[int, int, int]:
    block_m, store_block_m = _block_config_signature(
        num_ranks=num_ranks,
        num_experts=num_experts,
        num_topk=num_topk,
        num_tokens=num_tokens,
    )
    epw = _num_experts_per_wave(
        num_experts_per_rank=num_experts_per_rank,
        num_tokens=num_tokens,
        num_topk=num_topk,
        intermediate_hidden=intermediate_hidden,
        block_m=block_m,
        block_n=128,
        num_sms=num_sms,
    )
    return block_m, store_block_m, epw


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
    """Return one token count per reachable MegaMoE heuristic bucket."""
    max_tokens = max(int(max_tokens_per_rank), 0)
    if max_tokens == 0:
        return []

    reps: list[int] = []
    last_signature: tuple[int, int, int] | None = None
    for num_tokens in range(1, max_tokens + 1):
        signature = mega_moe_config_signature(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_experts_per_rank=num_experts_per_rank,
            num_tokens=num_tokens,
            num_topk=num_topk,
            intermediate_hidden=intermediate_hidden,
            num_sms=num_sms,
        )
        if signature != last_signature:
            reps.append(num_tokens)
            last_signature = signature

    # For chunked prefill, prefer cap token as last representative
    if reps:
        cap_sig = mega_moe_config_signature(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_experts_per_rank=num_experts_per_rank,
            num_tokens=max_tokens,
            num_topk=num_topk,
            intermediate_hidden=intermediate_hidden,
            num_sms=num_sms,
        )
        last_rep_sig = mega_moe_config_signature(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_experts_per_rank=num_experts_per_rank,
            num_tokens=reps[-1],
            num_topk=num_topk,
            intermediate_hidden=intermediate_hidden,
            num_sms=num_sms,
        )
        if cap_sig == last_rep_sig:
            reps[-1] = max_tokens
    return reps


def parse_jit_warmup_tokens_override() -> list[int] | None:
    raw_value = os.environ.get("GLM5_MEGA_MOE_JIT_WARMUP_TOKENS")
    if not raw_value:
        return None
    try:
        tokens = [int(item) for item in raw_value.replace(" ", "").split(",") if item]
    except ValueError:
        logging.warning(
            "[GLM5 MegaMoE] invalid GLM5_MEGA_MOE_JIT_WARMUP_TOKENS=%r; "
            "falling back to automatic",
            raw_value,
        )
        return None
    tokens = sorted({t for t in tokens if t > 0})
    if not tokens:
        return None
    return tokens


def clamp_token_counts(
    token_counts: Iterable[int],
    max_tokens_per_rank: int,
) -> list[int]:
    max_tokens = max(int(max_tokens_per_rank), 1)
    return sorted({min(int(t), max_tokens) for t in token_counts if int(t) > 0})


def format_token_counts(token_counts: Sequence[int]) -> str:
    return ",".join(str(t) for t in token_counts)
