"""Shared page-aligned prefix planning for BF16 and FP8 MLA Prefill."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence


_PREFILL_KV_BUDGET_ENV = "RTP_PREFILL_KV_WORKSPACE_BUDGET_GIB"


def prefill_kv_workspace_budget_bytes() -> int:
    """Read the per-rank budget for temporary prefill KV buffers."""
    value = os.environ.get(_PREFILL_KV_BUDGET_ENV, "8")
    error = f"{_PREFILL_KV_BUDGET_ENV} must specify a finite positive GiB amount"
    try:
        size = float(value) * 1024**3
    except ValueError as exc:
        raise ValueError(error) from exc
    if not math.isfinite(size) or size < 1:
        raise ValueError(error)
    return int(size)


@dataclass(frozen=True)
class Bf16MlaWorkspaceBudget:
    budget_bytes: int
    shard_size: int
    latent_dim: int
    rope_dim: int
    query_rows: int = 0

    def full_projection_capacity(self, expanded_bytes: int) -> int:
        # Full KV-up retains the contiguous latent and RoPE inputs.
        return self.budget_bytes // (expanded_bytes + 2 * (self.latent_dim + self.rope_dim))

    def plan_capacity(
        self, q_lens: Sequence[int], prefix_lens: Sequence[int],
        page_tokens: int, expanded_bytes: int,
    ) -> tuple[bool, int]:
        if page_tokens <= 0 or expanded_bytes <= 0 or self.shard_size <= 0:
            raise ValueError("BF16 MLA requires positive KV page, shard and projection sizes")
        q_rows = max(sum(q_lens), self.query_rows)
        prefix_rows = sum(prefix_lens)
        full_capacity = self.full_projection_capacity(expanded_bytes)
        if not prefix_rows:
            if q_rows > full_capacity:
                raise ValueError(
                    f"{_PREFILL_KV_BUDGET_ENV} cannot fit current-query BF16 MLA KV; "
                    "reduce the query chunk size"
                )
            return False, full_capacity

        compressed_bytes = 2 * (self.latent_dim + self.rope_dim)

        def owner_pages(tokens):
            stripe_tokens = page_tokens * self.shard_size
            return (tokens + stripe_tokens - 1) // stripe_tokens

        def reader_bytes(tokens, pages_per_rank):
            return compressed_bytes * (
                tokens + (self.shard_size + 1) * pages_per_rank * page_tokens
            )

        full_reader = reader_bytes(prefix_rows, sum(map(owner_pages, prefix_lens)))
        # The full path keeps the restored prefix during per-request and
        # combined latent/RoPE concatenation.
        full_concat = compressed_bytes * (prefix_rows + 2 * (prefix_rows + q_rows))
        if (
            q_rows + prefix_rows <= full_capacity
            and max(full_reader, full_concat) <= self.budget_bytes
        ):
            return False, full_capacity

        # Each prefix slice reads one request. The packed projection and
        # contiguous latent buffers coexist with pack, gather and restore.
        lo, hi = 0, self.budget_bytes // (expanded_bytes * page_tokens)
        while lo < hi:
            pages = (lo + hi + 1) // 2
            rows = pages * page_tokens
            live = (
                max(q_rows, rows) * expanded_bytes
                + rows * self.latent_dim * 2
                + reader_bytes(rows, owner_pages(rows))
            )
            if live <= self.budget_bytes:
                lo = pages
            else:
                hi = pages - 1
        if lo == 0:
            raise ValueError(f"{_PREFILL_KV_BUDGET_ENV} cannot fit one BF16 MLA KV page")
        return True, lo * page_tokens


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
    workspace_budget: Bf16MlaWorkspaceBudget | None = None,
) -> PrefixChunkPlan:
    """Bound historical KV expansion; current-Q tokens are not split.

    BF16 packs request slices into each capacity window. FP8 keeps one
    request per window and counts its live BF16 producer plus FP8 operands.
    The caller supplies that storage cost explicitly in bytes_per_token.
    The optional BF16 workspace budget accounts for the Page-RR reader too.
    """
    q_lens = tuple(int(n) for n in q_lens)
    prefix_lens = tuple(int(n) for n in prefix_lens)
    if len(q_lens) != len(prefix_lens) or not q_lens:
        raise ValueError("MLA prefix and query length vectors must match")
    if any(n < 0 for n in q_lens + prefix_lens):
        raise ValueError("MLA token lengths must be non-negative")
    if workspace_budget is None:
        chunked, capacity = plan_expanded_kv_capacity(
            q_lens, prefix_lens, page_tokens=page_tokens,
            bytes_per_token=bytes_per_token, budget_gib=budget_gib,
        )
    else:
        chunked, capacity = workspace_budget.plan_capacity(
            q_lens, prefix_lens, page_tokens, bytes_per_token,
        )
    return _plan_prefix_chunks_cached(
        prefix_lens, page_tokens, bytes_per_token, chunked, capacity, pack_requests
    )


@lru_cache(maxsize=16)
def _plan_prefix_chunks_cached(
    prefix_lens: tuple[int, ...],
    page_tokens: int,
    bytes_per_token: int,
    chunked: bool,
    capacity: int,
    pack_requests: bool,
) -> PrefixChunkPlan:
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
