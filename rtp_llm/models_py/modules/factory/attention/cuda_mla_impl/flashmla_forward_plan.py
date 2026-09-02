from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Sequence


class FlashMLAForwardRoute(Enum):
    FULL = "full"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class FlashMLAPrefixSlice:
    request_idx: int
    prefix_start: int
    prefix_len: int


@dataclass(frozen=True, slots=True)
class FlashMLAPrefixLaunch:
    slices: tuple[FlashMLAPrefixSlice, ...]
    expanded_kv_tokens: int
    packed_q_tokens: int


@dataclass(frozen=True, slots=True)
class FlashMLAForwardPlan:
    route: FlashMLAForwardRoute
    capacity_tokens: int
    prefix_launches: tuple[FlashMLAPrefixLaunch, ...]
    max_expanded_kv_tokens: int
    max_packed_q_tokens: int
    max_partial_state_tokens: int
    requires_fp32_accumulator: bool


def _full_plan(capacity_tokens: int) -> FlashMLAForwardPlan:
    return FlashMLAForwardPlan(
        route=FlashMLAForwardRoute.FULL,
        capacity_tokens=capacity_tokens,
        prefix_launches=(),
        max_expanded_kv_tokens=0,
        max_packed_q_tokens=0,
        max_partial_state_tokens=0,
        requires_fp32_accumulator=False,
    )


def _take_prefix_tokens(tokens: int, free: int, page_size: int) -> int:
    if tokens <= free:
        return tokens
    return free // page_size * page_size


def _build_prefix_launch(
    remaining: Sequence[int],
    *,
    capacity_tokens: int,
    page_size: int,
) -> dict[int, int]:
    """Pack whole requests in order and split only the boundary request."""
    takes = {}
    free = capacity_tokens
    for owner, tokens in enumerate(remaining):
        if not tokens:
            continue
        take = _take_prefix_tokens(tokens, free, page_size)
        if take:
            takes[owner] = take
            free -= take
        if take < tokens or not free:
            break
    return takes


def _build_prefix_launches(
    q_lens: tuple[int, ...],
    prefix_lens: tuple[int, ...],
    *,
    page_size: int,
    capacity_tokens: int,
) -> tuple[FlashMLAPrefixLaunch, ...]:
    remaining = list(prefix_lens)
    cursors = [0] * len(prefix_lens)
    launches = []

    while any(remaining):
        takes = _build_prefix_launch(
            remaining,
            capacity_tokens=capacity_tokens,
            page_size=page_size,
        )
        selected = sorted(takes)
        launches.append(
            FlashMLAPrefixLaunch(
                slices=tuple(
                    FlashMLAPrefixSlice(
                        request_idx=owner,
                        prefix_start=cursors[owner],
                        prefix_len=takes[owner],
                    )
                    for owner in selected
                ),
                expanded_kv_tokens=sum(takes.values()),
                packed_q_tokens=sum(q_lens[owner] for owner in selected),
            )
        )
        for owner in selected:
            cursors[owner] += takes[owner]
            remaining[owner] -= takes[owner]
    return tuple(launches)


def plan_flashmla_forward(
    q_lens: Sequence[int],
    prefix_lens: Sequence[int],
    *,
    page_size: int,
    expanded_kv_budget_bytes: int,
    expanded_kv_bytes_per_token: int,
) -> FlashMLAForwardPlan:
    """Plan one causal current-Q call and page-aligned historical-prefix calls.

    The budget limits prefix expansion. Current-Q tokens are always projected
    together, so their expanded K/V may exceed it even without a prefix.
    """
    q_lens = tuple(q_lens)
    prefix_lens = tuple(prefix_lens)

    return _plan_flashmla_forward_cached(
        q_lens,
        prefix_lens,
        page_size,
        expanded_kv_budget_bytes,
        expanded_kv_bytes_per_token,
    )


@lru_cache(maxsize=16)
def _plan_flashmla_forward_cached(
    q_lens: tuple[int, ...],
    prefix_lens: tuple[int, ...],
    page_size: int,
    expanded_kv_budget_bytes: int,
    expanded_kv_bytes_per_token: int,
) -> FlashMLAForwardPlan:
    # Identical request shapes recur in every MLA layer of one model invocation.

    q_tokens = sum(q_lens)
    total_tokens = q_tokens + sum(prefix_lens)
    if expanded_kv_budget_bytes == 0:
        return _full_plan(0)

    raw_capacity_tokens = expanded_kv_budget_bytes // expanded_kv_bytes_per_token
    if total_tokens * expanded_kv_bytes_per_token <= expanded_kv_budget_bytes:
        return _full_plan(raw_capacity_tokens)
    if not any(prefix_lens):
        return _full_plan(raw_capacity_tokens)

    capacity_tokens = raw_capacity_tokens // page_size * page_size
    if capacity_tokens == 0:
        raise ValueError(
            "FlashMLA expanded KV budget must fit at least one prefix page "
            f"({page_size * expanded_kv_bytes_per_token} bytes)"
        )
    prefix_launches = _build_prefix_launches(
        q_lens,
        prefix_lens,
        page_size=page_size,
        capacity_tokens=capacity_tokens,
    )
    owner_launch_counts = [0] * len(q_lens)
    for launch in prefix_launches:
        for item in launch.slices:
            owner_launch_counts[item.request_idx] += 1

    max_expanded_kv_tokens = max(
        launch.expanded_kv_tokens for launch in prefix_launches
    )
    max_packed_q_tokens = max(launch.packed_q_tokens for launch in prefix_launches)
    return FlashMLAForwardPlan(
        route=FlashMLAForwardRoute.HYBRID,
        capacity_tokens=capacity_tokens,
        prefix_launches=prefix_launches,
        max_expanded_kv_tokens=max_expanded_kv_tokens,
        max_packed_q_tokens=max_packed_q_tokens,
        max_partial_state_tokens=max_packed_q_tokens,
        requires_fp32_accumulator=any(count > 1 for count in owner_launch_counts),
    )
