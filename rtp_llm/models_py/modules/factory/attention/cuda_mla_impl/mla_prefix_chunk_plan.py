"""Bound the overlapping KV-up and attention operands for cached MLA Prefill."""

import math
from dataclasses import dataclass
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


def plan_prefix_chunks(
    q_lens: Sequence[int],
    prefix_lens: Sequence[int],
    *,
    page_tokens: int,
    heads: int,
    qk_dim: int,
    v_dim: int,
    operand_bytes: int,
    budget_gib: float = 6.0,
) -> PrefixChunkPlan:
    """Plan page-aligned historical slices; current Q is outside the budget.

    KV-up keeps a BF16 producer while TokenSpeed consumes FP8 or BF16 K/V,
    so both live copies count toward the expanded KV budget.
    """
    q_lens = tuple(int(n) for n in q_lens)
    prefix_lens = tuple(int(n) for n in prefix_lens)
    if len(q_lens) != len(prefix_lens) or not q_lens:
        raise ValueError("MLA prefix and query length vectors must match")
    if any(n < 0 for n in q_lens + prefix_lens):
        raise ValueError("MLA token lengths must be non-negative")
    if min(page_tokens, heads, qk_dim, v_dim, operand_bytes) <= 0:
        raise ValueError("MLA page and feature geometry must be positive")
    if not math.isfinite(budget_gib) or budget_gib < 0:
        raise ValueError("MLA expanded KV budget must be finite and non-negative")

    bytes_per_token = heads * (qk_dim + v_dim) * (2 + operand_bytes)
    capacity = int(budget_gib * 1024**3) // bytes_per_token
    full = (budget_gib == 0 or not any(prefix_lens)
            or (sum(q_lens) + sum(prefix_lens)) <= capacity)
    if full:
        return PrefixChunkPlan(False, capacity, bytes_per_token, ())

    capacity = capacity // page_tokens * page_tokens
    if capacity == 0:
        raise ValueError("MLA expanded KV budget must fit at least one cache page")
    slices = []
    for owner, prefix in enumerate(prefix_lens):
        for start in range(0, prefix, capacity):
            slices.append(PrefixSlice(owner, start, min(capacity, prefix - start)))
    return PrefixChunkPlan(True, capacity, bytes_per_token, tuple(slices))
