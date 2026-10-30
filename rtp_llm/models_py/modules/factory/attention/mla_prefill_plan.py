"""Shared page-aligned prefix planning for BF16 and FP8 MLA Prefill."""

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence


@dataclass(frozen=True)
class PrefixSlice:
    owner: int
    start: int
    length: int


@dataclass(frozen=True)
class PrefixChunkPlan:
    chunked: bool
    capacity_tokens: int
    bytes_per_token: int
    slices: tuple[PrefixSlice, ...]
    max_expanded_kv_tokens: int = 0
    requires_fp32_accumulator: bool = False


def plan_expanded_kv_capacity(
    q_lens: Sequence[int], prefix_lens: Sequence[int], *,
    page_tokens: int, bytes_per_token: int, budget_gib: float,
) -> tuple[bool, int]:
    """Return the chunked flag and capacity, with shared page alignment."""
    if page_tokens <= 0 or bytes_per_token <= 0:
        raise ValueError("MLA page alignment and expanded KV size must be positive")
    if not math.isfinite(budget_gib) or budget_gib < 0:
        raise ValueError("MLA expanded KV budget must be finite and non-negative")
    capacity = int(budget_gib * 1024**3) // bytes_per_token
    full = (budget_gib == 0 or not any(prefix_lens)
            or sum(q_lens) + sum(prefix_lens) <= capacity)
    if full:
        return False, capacity
    capacity = capacity // page_tokens * page_tokens
    if capacity == 0:
        raise ValueError("MLA expanded KV budget must fit at least one cache page")
    return True, capacity


def plan_prefix_chunks(
    q_lens: Sequence[int],
    prefix_lens: Sequence[int],
    *,
    page_tokens: int,
    bytes_per_token: int,
    budget_gib: float = 6.0,
    pack_requests: bool = False,
) -> PrefixChunkPlan:
    """Bound historical KV expansion; current-Q tokens are not split.

    BF16 packs request slices into each capacity window. FP8 keeps one
    request per window and counts its live BF16 producer plus FP8 operands.
    The caller supplies that storage cost explicitly in bytes_per_token.
    """
    q_lens = tuple(int(n) for n in q_lens)
    prefix_lens = tuple(int(n) for n in prefix_lens)
    if len(q_lens) != len(prefix_lens) or not q_lens:
        raise ValueError("MLA prefix and query length vectors must match")
    if any(n < 0 for n in q_lens + prefix_lens):
        raise ValueError("MLA token lengths must be non-negative")
    return _plan_prefix_chunks_cached(
        q_lens, prefix_lens, page_tokens, bytes_per_token, budget_gib, pack_requests
    )


@lru_cache(maxsize=16)
def _plan_prefix_chunks_cached(
    q_lens: tuple[int, ...],
    prefix_lens: tuple[int, ...],
    page_tokens: int,
    bytes_per_token: int,
    budget_gib: float,
    pack_requests: bool,
) -> PrefixChunkPlan:
    chunked, capacity = plan_expanded_kv_capacity(
        q_lens, prefix_lens, page_tokens=page_tokens,
        bytes_per_token=bytes_per_token, budget_gib=budget_gib,
    )
    if not chunked:
        return PrefixChunkPlan(False, capacity, bytes_per_token, ())

    slices = []
    max_expanded = 0
    counts = [0] * len(prefix_lens)
    remaining = list(prefix_lens)
    cursors = [0] * len(prefix_lens)
    owner = 0
    while owner < len(remaining):
        free = capacity
        while owner < len(remaining):
            tokens = remaining[owner]
            if not tokens:
                owner += 1
                continue
            take = tokens if tokens <= free else free // page_tokens * page_tokens
            if take:
                slices.append(PrefixSlice(owner, cursors[owner], take))
                cursors[owner] += take
                remaining[owner] -= take
                counts[owner] += 1
                free -= take
            if take < tokens:
                break
            owner += 1
            if not pack_requests or not free:
                break
        max_expanded = max(max_expanded, capacity - free)
    return PrefixChunkPlan(
        True, capacity, bytes_per_token, tuple(slices), max_expanded,
        any(count > 1 for count in counts),
    )
