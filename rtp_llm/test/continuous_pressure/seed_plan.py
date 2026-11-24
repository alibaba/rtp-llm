"""Immutable live FULL-capacity seed planning and formal configuration screening."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .diagnostics import select_targets

MIN_REUSE_TOKENS = 32 * 1024
FORMAL_FILL_TARGET_TOKENS = 100 * 1024
MIN_FORMAL_CLIENT_INFLIGHT = 16


def formal_config_reasons(
    *,
    manual_seed_count: int | None,
    client_inflight: int,
    test_block_num: int,
    fill_target_tokens: int,
    explicit_smoke: bool = False,
    diagnostic_only: bool = False,
) -> tuple[str, ...]:
    """Return ordered static exclusions; live capacity planning is still required."""
    for name, value, minimum in (
        ("manual_seed_count", manual_seed_count, 1),
        ("client_inflight", client_inflight, 1),
        ("test_block_num", test_block_num, 0),
        ("fill_target_tokens", fill_target_tokens, 1),
    ):
        if name == "manual_seed_count" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if type(explicit_smoke) is not bool:
        raise ValueError("explicit_smoke must be a boolean")
    if type(diagnostic_only) is not bool:
        raise ValueError("diagnostic_only must be a boolean")
    reasons = []
    if manual_seed_count is not None:
        reasons.append("manual_seed_count")
    if client_inflight < MIN_FORMAL_CLIENT_INFLIGHT:
        reasons.append("client_inflight_below_16")
    if test_block_num > 0:
        reasons.append("synthetic_test_block_capacity")
    if fill_target_tokens != FORMAL_FILL_TARGET_TOKENS:
        reasons.append("nonstandard_fill_target")
    if explicit_smoke:
        reasons.append("explicit_smoke")
    if diagnostic_only:
        reasons.append("diagnostic_only")
    return tuple(reasons)


@dataclass(frozen=True)
class SeedPoolPlan:
    pool_id: str
    capacity_blocks: int
    available_blocks: int
    low_blocks: int
    seed_blocks: int
    block_tokens: int = 0
    low_ratio: float = 0.0
    high_ratio: float = 0.0
    geometry_source: str = "owner_group"


@dataclass(frozen=True)
class SeedGroupPlan:
    owner_rank: int
    owner_group_id: int
    block_tokens: int
    device_pools: tuple[SeedPoolPlan, ...]
    host_pool: SeedPoolPlan
    gpu_family_capacity: int
    gpu_low_families: int
    host_families: int
    minimum_families: int
    maximum_families: int
    participants: tuple[tuple[int, str], ...] = ()
    owner_instance: str = ""

    @property
    def seed_blocks(self) -> int:
        return max(pool.seed_blocks for pool in self.device_pools)


@dataclass(frozen=True)
class SeedPlan:
    source: str
    seed_count: int
    grid_reuse_tokens: int
    reusable_tokens: int
    commit_tail_tokens: int
    seed_input_tokens: int
    fill_target_tokens: int
    fill_input_tokens: int
    client_inflight: int
    formal_config_valid: bool
    nonformal_reasons: tuple[str, ...]
    groups: tuple[SeedGroupPlan, ...]
    cache_alignment: int = 0
    test_block_num: int = 0
    global_minimum_families: int = 0
    global_maximum_families: int = 0
    explicit_smoke: bool = False
    diagnostic_only: bool = False


def align_up(value: int, alignment: int) -> int:
    if (
        type(value) is not int
        or type(alignment) is not int
        or value <= 0
        or alignment <= 0
    ):
        raise ValueError("positive integer value/alignment required")
    return ((value + alignment - 1) // alignment) * alignment


def build_seed_plan(
    samples: tuple[Any, ...],
    targets: tuple[Any, ...],
    *,
    cases: tuple[dict, ...],
    cache_alignment: int,
    commit_tail_tokens: int,
    client_inflight: int,
    manual_seed_count: int | None,
    fill_target_tokens: int,
    test_block_num: int,
    explicit_smoke: bool = False,
    diagnostic_only: bool = False,
) -> SeedPlan:
    """Plan a uniform reusable bank from owner capacity and live host availability.

    Schema 1 reports token geometry on the logical group. Both derived device
    and host entries retain that source explicitly; byte sizes never stand in
    for token spans. Capacities are already allocatable (including reserves).
    """
    reasons = formal_config_reasons(
        manual_seed_count=manual_seed_count,
        client_inflight=client_inflight,
        test_block_num=test_block_num,
        fill_target_tokens=fill_target_tokens,
        explicit_smoke=explicit_smoke,
        diagnostic_only=diagnostic_only,
    )
    if type(commit_tail_tokens) is not int or commit_tail_tokens < 0:
        raise ValueError("commit_tail_tokens must be a nonnegative integer")
    if not cases:
        raise ValueError("grid cases must be nonempty")
    cache_lengths = []
    for case in cases:
        length = case.get("cache_len")
        if type(length) is not int or length < 0:
            raise ValueError("grid cache_len must be a nonnegative integer")
        cache_lengths.append(length)
    grid_reuse_tokens = max(cache_lengths)
    reusable_tokens = align_up(
        max(MIN_REUSE_TOKENS, grid_reuse_tokens), cache_alignment
    )
    fill_input_tokens = align_up(fill_target_tokens, cache_alignment)
    if grid_reuse_tokens < MIN_REUSE_TOKENS:
        if not reasons:
            raise ValueError("formal grid max(cache_len) must be >= 32768")
        reasons += ("grid_reuse_below_32768",)

    if not samples or not targets:
        raise ValueError("live FULL samples and targets must be nonempty")
    # Revalidate frozen targets against the supplied evidence. Dynamic counters
    # are deliberately excluded by select_targets, so stale topology fails while
    # duplicate participant views cannot multiply owner capacity.
    current = select_targets(samples, tuple(sample.rank for sample in samples))
    if set(current) != set(targets):
        raise ValueError("pressure topology or service instance discontinuity")
    by_rank = {sample.rank: sample for sample in samples}
    owners = {}
    for target in targets:
        key = (target.owner_rank, target.owner_group_id)
        owners.setdefault(key, target)
    groups = []
    for (owner_rank, owner_group_id), target in sorted(owners.items()):
        sample = by_rank[owner_rank]
        group = next(
            group for group in sample.groups if group.group_id == owner_group_id
        )
        block_tokens = group.geometry[0]
        if type(block_tokens) is not int or block_tokens <= 0:
            raise ValueError(
                f"owner_rank={owner_rank} owner_group_id={owner_group_id}: "
                "live block_tokens must be a positive integer"
            )
        pools = {pool.pool_id: pool for pool in sample.pools}

        def pool_plan(pool):
            if pool.capacity <= 0:
                raise ValueError(
                    f"owner_rank={owner_rank} owner_group_id={owner_group_id}: "
                    f"pool_id={pool.pool_id} requires positive live capacity"
                )
            return SeedPoolPlan(
                pool_id=pool.pool_id,
                capacity_blocks=pool.capacity,
                available_blocks=pool.available,
                low_blocks=math.floor(pool.capacity * pool.low_ratio),
                seed_blocks=(reusable_tokens + block_tokens - 1) // block_tokens,
                block_tokens=block_tokens,
                low_ratio=pool.low_ratio,
                high_ratio=pool.high_ratio,
            )

        devices = tuple(
            pool_plan(pools[key]) for key in sorted(set(group.device_pool_ids))
        )
        host = pool_plan(pools[group.host_pool_id])
        gpu_capacity = min(pool.capacity_blocks // pool.seed_blocks for pool in devices)
        gpu_low = min(pool.low_blocks // pool.seed_blocks for pool in devices)
        host_families = host.available_blocks // host.seed_blocks
        groups.append(
            SeedGroupPlan(
                owner_rank=owner_rank,
                owner_group_id=owner_group_id,
                block_tokens=block_tokens,
                device_pools=devices,
                host_pool=host,
                gpu_family_capacity=gpu_capacity,
                gpu_low_families=gpu_low,
                host_families=host_families,
                minimum_families=gpu_capacity + 1,
                maximum_families=gpu_low + host_families,
                participants=tuple(sorted(target.participants)),
                owner_instance=sample.instance,
            )
        )
    lower = max(group.minimum_families for group in groups)
    upper = min(group.maximum_families for group in groups)
    count = lower if manual_seed_count is None else manual_seed_count
    if (manual_seed_count is None and lower > upper) or count > upper:
        bounds = "; ".join(
            f"owner_rank={group.owner_rank} owner_group_id={group.owner_group_id} "
            f"minimum={group.minimum_families} maximum={group.maximum_families} "
            f"gpu_family_capacity={group.gpu_family_capacity} "
            f"gpu_low_families={group.gpu_low_families} host_families={group.host_families} "
            f"device_pools={group.device_pools!r} host_pool={group.host_pool!r}"
            for group in groups
        )
        source = "automatic" if manual_seed_count is None else "manual"
        raise ValueError(
            f"infeasible {source} seed_count={count}: lower={lower} upper={upper}; "
            f"{bounds}; increase live host availability or revise the smoke workload"
        )
    return SeedPlan(
        source="automatic" if manual_seed_count is None else "manual",
        seed_count=count,
        grid_reuse_tokens=grid_reuse_tokens,
        reusable_tokens=reusable_tokens,
        commit_tail_tokens=commit_tail_tokens,
        seed_input_tokens=reusable_tokens + commit_tail_tokens,
        fill_target_tokens=fill_target_tokens,
        fill_input_tokens=fill_input_tokens,
        client_inflight=client_inflight,
        formal_config_valid=not reasons,
        nonformal_reasons=reasons,
        groups=tuple(groups),
        cache_alignment=cache_alignment,
        test_block_num=test_block_num,
        global_minimum_families=lower,
        global_maximum_families=upper,
        explicit_smoke=explicit_smoke,
        diagnostic_only=diagnostic_only,
    )
