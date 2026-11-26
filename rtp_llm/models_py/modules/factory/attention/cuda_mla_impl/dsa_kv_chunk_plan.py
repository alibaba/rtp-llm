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


class DsaKVCapacityError(ValueError):
    """Capacity rejection under the planner's accounting, not a physical OOM."""

    def __init__(
        self,
        *,
        budget_bytes,
        fixed_bytes,
        page_tokens,
        min_required_bytes=None,
        min_page_capacity=None,
        one_page_required_bytes=None,
        budget_scope="phase",
    ):
        self.budget_scope = budget_scope
        self.budget_bytes = budget_bytes
        self.fixed_bytes = fixed_bytes
        self.min_required_bytes = min_required_bytes
        self.min_page_capacity = min_page_capacity
        self.one_page_required_bytes = one_page_required_bytes
        if fixed_bytes > budget_bytes:
            reason = "fixed live data exceeds its memory budget"
        elif one_page_required_bytes is None:
            reason = "memory budget cannot fit resident data with empty KV history"
        else:
            reason = "memory budget cannot fit one physical KV page or any larger chunk"
        self.lower_bound_bytes = (
            fixed_bytes if min_required_bytes is None else min_required_bytes
        )
        if min_required_bytes is None:
            detail = (
                "diagnostic_reason=fixed_exceeds_budget, "
                f"lower_bound_bytes={fixed_bytes}, "
                f"shortfall_at_least_bytes={fixed_bytes - budget_bytes}, "
                "min_required_bytes=unknown, min_page_capacity=unknown, "
                "additional_workspace_bytes=unknown, kv_costs=not_computed, "
                "one_page_temporary_bytes=unknown, one_page_required_bytes=unknown"
            )
        else:
            detail = (
                f"min_required_bytes={min_required_bytes}, "
                f"shortfall_bytes={min_required_bytes - budget_bytes}, "
                f"min_page_capacity={min_page_capacity}, "
                f"one_page_temporary_bytes={None if one_page_required_bytes is None else one_page_required_bytes - fixed_bytes}, "
                f"one_page_required_bytes={one_page_required_bytes}"
            )
        super().__init__(
            f"DSA phase {reason}; budget_bytes={budget_bytes}, fixed_bytes={fixed_bytes}, "
            f"remaining_after_fixed_bytes={budget_bytes - fixed_bytes}, "
            f"page_tokens={page_tokens}, {detail}"
        )


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
    plan_temporary_bytes: Callable[[tuple[DsaKVBlock, ...]], int] | None = None,
    kv_budget_bytes: int | None = None,
    kv_temporary_bytes: Callable[[tuple[DsaKVBlock, ...]], int] | None = None,
) -> DsaKVChunkPlan:
    """Choose the largest page capacity that fits every block's full workspace.

    ``fixed_bytes`` includes live projections/layouts, bounded candidates,
    partial/running output and any resident buffers from other phases.
    ``temporary_bytes`` includes padded owner pack/gather/restore, scores,
    selector scratch and merge scratch. No free-memory query occurs in replay.
    When supplied, ``plan_temporary_bytes`` accounts for the union of buffer
    capacities, resident phase buffers and metadata across all blocks instead
    of just the maximum per-block temporary. Fixed overflow rejects without
    evaluating callbacks or capacities and reports only a lower bound. Other
    rejections report the exact minimum over all allowed capacities.
    An optional independent KV-only cap must fit at the same capacity as the
    full phase budget. A phase diagnostic then minimizes only over KV-fitting
    capacities; if none fit KV, report that separate budget instead.
    """
    values = tuple(lengths)
    if (
        type(page_tokens) is not int
        or page_tokens <= 0
        or any(type(n) is not int or n < 0 for n in values)
        or budget_bytes <= 0
        or fixed_bytes < 0
        or (kv_budget_bytes is None) != (kv_temporary_bytes is None)
        or (kv_budget_bytes is not None and kv_budget_bytes <= 0)
        or (
            max_page_capacity is not None
            and (type(max_page_capacity) is not int or max_page_capacity <= 0)
        )
    ):
        raise ValueError("invalid DSA KV geometry or memory budget")
    if fixed_bytes > budget_bytes:
        raise DsaKVCapacityError(
            budget_bytes=budget_bytes,
            fixed_bytes=fixed_bytes,
            page_tokens=page_tokens,
        )
    pages = sum((n + page_tokens - 1) // page_tokens for n in values)
    # Stripe padding and chunk-only running buffers can make capacities
    # non-monotonic. Include whole-plan accounting in this same descending
    # scan; subtracting an overhead from the budget can skip a fitting plan.
    capacities = range(min(pages, max_page_capacity or pages), 0, -1) if pages else (0,)
    min_required, min_capacity = None, 0
    one_page_required = None
    min_kv_required, min_kv_capacity, one_page_kv_required = None, 0, None
    for capacity in capacities:
        blocks = _blocks(values, capacity, page_tokens) if capacity else ()
        required = fixed_bytes + (
            plan_temporary_bytes(blocks)
            if plan_temporary_bytes is not None
            else max((temporary_bytes(b) for b in blocks), default=0)
        )
        kv_required = kv_temporary_bytes(blocks) if kv_temporary_bytes is not None else 0
        if min_kv_required is None or kv_required < min_kv_required:
            min_kv_required, min_kv_capacity = kv_required, capacity
        if capacity == 1:
            one_page_kv_required = kv_required
            one_page_required = required
        if kv_budget_bytes is not None and kv_required > kv_budget_bytes:
            continue
        if required <= budget_bytes:
            return DsaKVChunkPlan(blocks, budget_bytes, required, fixed_bytes, capacity)
        if min_required is None or required < min_required:
            min_required, min_capacity = required, capacity
    if min_required is None:
        raise DsaKVCapacityError(
            budget_bytes=kv_budget_bytes,
            fixed_bytes=0,
            page_tokens=page_tokens,
            min_required_bytes=min_kv_required,
            min_page_capacity=min_kv_capacity,
            one_page_required_bytes=one_page_kv_required,
            budget_scope="prefill_kv",
        )
    raise DsaKVCapacityError(
        budget_bytes=budget_bytes,
        fixed_bytes=fixed_bytes,
        page_tokens=page_tokens,
        min_required_bytes=min_required,
        min_page_capacity=min_capacity,
        one_page_required_bytes=one_page_required,
    )
