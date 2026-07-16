"""JIT warmup bucket generation for DeepGEMM NVFP4 Mega MoE."""

from __future__ import annotations

import logging
import math
import os
from typing import Iterable, Sequence

from rtp_llm.models_py.modules.dsv4.chunk_env import (
    dsv4_chunk_tokens_from_env,
    dsv4_global_chunk_tokens_configured,
)


def mega_moe_nvfp4_jit_warmup_enabled() -> bool:
    return os.environ.get("DSV4_MEGA_MOE_NVFP4_JIT_WARMUP", "1") != "0"


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _block_config(num_ranks, num_experts, num_topk, num_tokens):
    expected = float(num_tokens) * num_ranks * num_topk / num_experts
    if expected <= 8.5:
        return 16, 8
    if expected <= 16.5:
        return 32, 16
    if expected <= 32.5:
        return 64, 16
    if expected <= 64.5:
        return 96, 24
    if expected <= 96.5:
        return 128, 32
    return 192, 48


def mega_moe_nvfp4_config_signature(
    *,
    num_ranks: int,
    num_experts: int,
    num_experts_per_rank: int,
    num_tokens: int,
    num_topk: int,
    intermediate_hidden: int,
    num_sms: int,
) -> tuple[int, int, int]:
    block_m, store_block_m = _block_config(num_ranks, num_experts, num_topk, num_tokens)
    expected = float(num_tokens) * num_topk / num_experts_per_rank
    if expected < 1:
        experts_per_wave = num_experts_per_rank
    else:
        num_m_blocks = _ceil_div(int(math.ceil(expected)), block_m)
        num_n_blocks = (2 * intermediate_hidden) // 128
        imbalance_factor = 3 if block_m == 192 else 6
        experts_per_wave = _ceil_div(
            imbalance_factor * num_sms, num_m_blocks * num_n_blocks
        )
        experts_per_wave = min(experts_per_wave, num_experts_per_rank)
        if block_m == 192:
            experts_per_wave = max(experts_per_wave, min(4, num_experts_per_rank))
        while (
            experts_per_wave < num_experts_per_rank
            and num_experts_per_rank % experts_per_wave != 0
        ):
            experts_per_wave += 1
    return block_m, store_block_m, experts_per_wave


def generate_mega_moe_nvfp4_jit_token_counts(
    *,
    num_ranks: int,
    num_experts: int,
    num_experts_per_rank: int,
    num_topk: int,
    intermediate_hidden: int,
    num_sms: int,
    max_tokens_per_rank: int,
) -> list[int]:
    max_tokens = max(int(max_tokens_per_rank), 0)
    if max_tokens == 0:
        return []
    representatives: list[int] = []
    previous = None
    for tokens in range(1, max_tokens + 1):
        signature = mega_moe_nvfp4_config_signature(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_experts_per_rank=num_experts_per_rank,
            num_tokens=tokens,
            num_topk=num_topk,
            intermediate_hidden=intermediate_hidden,
            num_sms=num_sms,
        )
        if signature != previous:
            representatives.append(tokens)
            previous = signature

    if dsv4_global_chunk_tokens_configured():
        prefer_cap = dsv4_chunk_tokens_from_env("DSV4_MOE_CHUNK_TOKENS") > 0
    else:
        prefer_cap = os.environ.get("DSV4_MOE_CHUNK_PREFILL", "1") != "0"
    if representatives and prefer_cap:
        cap_signature = mega_moe_nvfp4_config_signature(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_experts_per_rank=num_experts_per_rank,
            num_tokens=max_tokens,
            num_topk=num_topk,
            intermediate_hidden=intermediate_hidden,
            num_sms=num_sms,
        )
        last_signature = mega_moe_nvfp4_config_signature(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_experts_per_rank=num_experts_per_rank,
            num_tokens=representatives[-1],
            num_topk=num_topk,
            intermediate_hidden=intermediate_hidden,
            num_sms=num_sms,
        )
        if cap_signature == last_signature:
            representatives[-1] = max_tokens
    return representatives


def parse_mega_moe_nvfp4_jit_warmup_tokens_override() -> list[int] | None:
    raw = os.environ.get("DSV4_MEGA_MOE_NVFP4_JIT_WARMUP_TOKENS")
    if not raw:
        return None
    try:
        tokens = sorted({int(item) for item in raw.replace(" ", "").split(",") if item})
    except ValueError:
        logging.warning(
            "[DSV4 MegaMoE NVFP4] invalid warmup token override %r; using auto",
            raw,
        )
        return None
    tokens = [token for token in tokens if token > 0]
    return tokens or None


def clamp_token_counts(
    token_counts: Iterable[int], max_tokens_per_rank: int
) -> list[int]:
    maximum = max(int(max_tokens_per_rank), 1)
    return sorted(
        {min(int(token), maximum) for token in token_counts if int(token) > 0}
    )


def format_token_counts(token_counts: Sequence[int]) -> str:
    return ",".join(str(token) for token in token_counts)
