"""Page-aligned KV plans under independent, complete phase memory budgets."""

from dataclasses import dataclass
from typing import Callable, Sequence


def prefill_indexer_score_bytes(rows: int, columns: int) -> int:
    """FP32 dense MQA storage, including SM100's extra K tile and row stride."""
    # Retain the planner's conservative row alignment across MQA backends.
    aligned_rows = (rows + 63) // 64 * 64
    aligned_columns = (columns + 256 + 255) // 256 * 256
    return aligned_rows * aligned_columns * 4


def indexer_selection_scratch_bytes(rows: int, columns: int, topk: int) -> int:
    """Conservative CUDA radix-select and bounded selected-K sort scratch.

    Audited against PyTorch 70d99e998b4955e0049d13a98d77ae1b14db1f45:
    TensorTopK.cu uses at least 1024 items/block, 256 radix digits, per-block
    uint16 counts plus two uint32 counts, per-row digit totals/thresholds.
    Include bounded K sort temporaries, scan scratch and allocator alignment.
    This is in addition to caller-owned keys, values and INT64 slot outputs.
    """
    blocks = (columns + 1023) // 1024
    counters = rows * (1056 + blocks * (512 + 8 + 16))
    selected_sort = rows * min(columns, topk) * 16
    return counters + selected_sort + 2 * 1024 * 1024


@dataclass(frozen=True)
class DsaKVBlock:
    requests: tuple[int, ...]
    starts: tuple[int, ...]
    lengths: tuple[int, ...]

    @property
    def tokens(self) -> int:
        return sum(self.lengths)

    def owner_pages(self, page: int, shards: int) -> int:
        # Each request is independently stripe padded by the shared reader.
        return sum(
            ((n + page - 1) // page + shards - 1) // shards for n in self.lengths
        )


@dataclass(frozen=True)
class DsaKVChunkPlan:
    blocks: tuple[DsaKVBlock, ...]
    budget_bytes: int
    required_bytes: int
    fixed_bytes: int
    page_capacity: int

    @property
    def chunked(self) -> bool:
        return len(self.blocks) > 1


def _blocks(lengths: tuple[int, ...], pages: int, page: int) -> tuple[DsaKVBlock, ...]:
    """Partition request-major physical pages without partitioning Q."""
    result = []
    ids, starts, sizes = [], [], []
    remaining = pages
    for request, length in enumerate(lengths):
        start = 0
        while start < length:
            take = min(length - start, remaining * page)
            ids.append(request)
            starts.append(start)
            sizes.append(take)
            remaining -= (take + page - 1) // page
            start += take
            if remaining == 0:
                result.append(DsaKVBlock(tuple(ids), tuple(starts), tuple(sizes)))
                ids, starts, sizes = [], [], []
                remaining = pages
    if ids:
        result.append(DsaKVBlock(tuple(ids), tuple(starts), tuple(sizes)))
    return tuple(result)


def plan_dsa_kv_chunks(
    lengths: Sequence[int],
    *,
    page_tokens: int,
    budget_bytes: int,
    fixed_bytes: int,
    temporary_bytes: Callable[[DsaKVBlock], int],
    max_page_capacity: int | None = None,
) -> DsaKVChunkPlan:
    """Choose the largest page capacity that fits every block's full workspace.

    ``fixed_bytes`` includes live projections/layouts, bounded candidates,
    partial/running output and any resident buffers from other phases.
    ``temporary_bytes`` includes padded owner pack/gather/restore, scores,
    selector scratch and merge scratch. No free-memory query occurs in replay.
    """
    values = tuple(lengths)
    if (
        type(page_tokens) is not int
        or page_tokens <= 0
        or any(type(n) is not int or n < 0 for n in values)
        or budget_bytes <= 0
        or fixed_bytes < 0
        or (max_page_capacity is not None and
            (type(max_page_capacity) is not int or max_page_capacity <= 0))
    ):
        raise ValueError("invalid DSA KV geometry or memory budget")
    pages = sum((n + page_tokens - 1) // page_tokens for n in values)
    if fixed_bytes > budget_bytes:
        raise ValueError("DSA phase fixed live data exceeds its memory budget")
    if not pages:
        return DsaKVChunkPlan((), budget_bytes, fixed_bytes, fixed_bytes, 0)

    # Stripe padding can make neighboring capacities non-monotonic for a
    # ragged batch. Scan from the largest page-aligned capacity; do not assume
    # a binary-search monotonicity that fails across request boundaries.
    for capacity in range(min(pages, max_page_capacity or pages), 0, -1):
        blocks = _blocks(values, capacity, page_tokens)
        required = fixed_bytes + max(temporary_bytes(b) for b in blocks)
        if required <= budget_bytes:
            return DsaKVChunkPlan(blocks, budget_bytes, required, fixed_bytes, capacity)
    raise ValueError("DSA phase memory budget cannot fit one physical KV page")
