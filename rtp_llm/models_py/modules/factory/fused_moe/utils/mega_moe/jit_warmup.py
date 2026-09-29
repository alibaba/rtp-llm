"""MegaMoE warmup for the fixed DeepGEMM version shipped by RTP.

Use the backend BLOCK_M query and an audited token-branch contract instead of
reimplementing its non-monotonic tile-selection heuristic.
"""

from __future__ import annotations

import logging
import os
from ctypes import c_float
from typing import Callable, Hashable, Iterable, Sequence

from rtp_llm.utils.warmup import model_warm_up_enabled


def mega_moe_jit_warmup_enabled() -> bool:
    return model_warm_up_enabled()


# Audited against DeepGEMM 83961ec (the version pinned by RTP's CUDA 13 wheels).
# Re-audit this contract when updating the dependency: get_block_m alone does
# not describe FP8's store/epilogue and single-pass dispatch specializations.
DEEP_GEMM_WARMUP_REVISION = "83961ec"


def mega_moe_config_signature(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    num_tokens: int,
    block_m: int,
    fp8_weights: bool = False,
) -> tuple[int, int, int, bool]:
    """Token-varying JIT determinants for the pinned FP4/FP8 C++ launchers.

    BLOCK_M comes from the installed backend. At fixed model/buffer/SM settings,
    these fields also determine SF tiles, dispatch threads, pipeline stages and
    shared memory. FP4 has no token-dependent single-pass template argument.
    Retain FP8's latency/throughput store and epilogue branches explicitly;
    single-pass dispatch can change while BLOCK_M stays the same.
    """
    store_block_m = (
        8 if block_m <= 16 else 16 if block_m <= 64 else 32 if block_m <= 192 else 40
    )
    epilogue_threads = 256
    single_pass = False
    if fp8_weights:
        # Match the float arithmetic in get_block_config_for_mega_moe_fp8.
        expected = c_float(num_tokens).value
        expected = c_float(expected * num_ranks).value
        expected = c_float(expected * num_topk).value
        expected = c_float(expected / num_experts).value
        if expected <= 32.5:
            store_block_m = block_m // 2
            epilogue_threads = 128
        elif expected <= 64.5:
            store_block_m = 16
        single_pass = num_tokens * num_topk <= 32768
    return block_m, store_block_m, epilogue_threads, single_pass


def generate_mega_moe_jit_token_counts(
    *,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    get_block_m: Callable[[int], int],
    max_tokens_per_rank: int,
    fp8_weights: bool = False,
    include_cap: bool = False,
) -> list[int]:
    """Cover the pinned backend's token branches and both packer launch paths."""
    max_tokens = max(int(max_tokens_per_rank), 0)
    if max_tokens == 0:
        return []

    def signature(tokens: int):
        return mega_moe_config_signature(
            num_ranks=num_ranks,
            num_experts=num_experts,
            num_topk=num_topk,
            num_tokens=tokens,
            block_m=int(get_block_m(tokens)),
            fp8_weights=fp8_weights,
        )

    reps = generate_jit_token_counts_from_signature(
        signature, max_tokens, include_cap=include_cap
    )
    from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
        mega_moe_input_pack_warmup_token_counts,
    )

    return sorted(set(reps) | set(mega_moe_input_pack_warmup_token_counts(max_tokens)))


def parse_mega_moe_jit_warmup_tokens_override() -> list[int] | None:
    raw_value = os.environ.get("MEGA_MOE_JIT_WARMUP_TOKENS")
    if not raw_value:
        return None
    try:
        tokens = [int(item) for item in raw_value.replace(" ", "").split(",") if item]
    except ValueError:
        logging.warning(
            "[MegaMoE] invalid MEGA_MOE_JIT_WARMUP_TOKENS=%r; "
            "falling back to automatic token-count generation",
            raw_value,
        )
        return None
    tokens = sorted({token for token in tokens if token > 0})
    if not tokens:
        logging.warning(
            "[MegaMoE] MEGA_MOE_JIT_WARMUP_TOKENS=%r contains no "
            "positive token counts; falling back to automatic token-count generation",
            raw_value,
        )
        return None
    return tokens


def clamp_token_counts(
    token_counts: Iterable[int],
    max_tokens_per_rank: int,
) -> list[int]:
    max_tokens = max(int(max_tokens_per_rank), 1)
    return sorted(
        {min(int(token), max_tokens) for token in token_counts if int(token) > 0}
    )


def format_token_counts(token_counts: Sequence[int]) -> str:
    return ",".join(str(token) for token in token_counts)


def generate_jit_token_counts_from_signature(
    signature: Callable[[int], Hashable],
    max_tokens_per_rank: int,
    *,
    include_cap: bool = False,
) -> list[int]:
    """Deduplicate actual backend signatures, including non-monotonic buckets.

    Empty ranks also execute MegaMoE. Include T=0 in the signature scan; packers
    can independently add the non-empty shapes they need.
    """
    if max_tokens_per_rank <= 0:
        return []
    representatives = {}
    for tokens in range(max_tokens_per_rank + 1):
        representatives.setdefault(signature(tokens), tokens)
    if include_cap:
        representatives[signature(max_tokens_per_rank)] = max_tokens_per_rank
    return sorted(representatives.values())
