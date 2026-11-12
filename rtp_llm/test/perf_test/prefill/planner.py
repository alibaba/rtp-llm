"""Deterministic cache-aware prefill grid planning."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import asdict, replace
from typing import Dict, List, Tuple

from rtp_llm.test.perf_test.config_schema import EngineConfig, PrefillWorkloadConfig
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import ExecutionPlan
from rtp_llm.test.perf_test.prefill.block_size import resolve_cache_block_size
from rtp_llm.test.perf_test.prefill.measurement import PREFILL_MEASUREMENT_CONTRACT


def _stable_rng(seed: int, namespace: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{namespace}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _validate_contract(workload: PrefillWorkloadConfig, engine: EngineConfig) -> int:
    if workload.kind != "cache_grid":
        raise PlanningError("prefill planner requires workload.kind=cache_grid")
    if workload.max_input_len + 1 > engine.max_seq_len:
        raise PlanningError(
            "workload.max_input_len plus one output token exceeds engine.max_seq_len"
        )
    if workload.commit_tail_tokens % workload.cache_alignment:
        raise PlanningError(
            "workload.commit_tail_tokens must be aligned to cache_alignment"
        )
    alignment = math.lcm(
        workload.cache_alignment, expected_probe_block_size(workload, engine)
    )
    # Only the reusable prefix must end on a physical block boundary. A shorter
    # tail still lets the seed commit all full blocks preceding that boundary.
    if workload.min_compute_len > workload.max_input_len:
        raise PlanningError(
            "workload.min_compute_len cannot exceed workload.max_input_len"
        )
    if len(set(workload.compute_steps)) != len(workload.compute_steps) or any(
        left <= right
        for left, right in zip(workload.compute_steps, workload.compute_steps[1:])
    ):
        raise PlanningError(
            "workload.compute_steps must be unique and ordered coarse-to-fine"
        )
    if workload.max_cache_len is not None and (
        workload.max_cache_len < workload.min_cache_len
    ):
        raise PlanningError("workload cache minimum cannot exceed cache maximum")
    if any(batch > engine.concurrency_limit for batch in workload.batch_sizes):
        raise PlanningError(
            "workload.batch_sizes cannot exceed engine.concurrency_limit"
        )
    if any(
        batch * workload.max_input_len > engine.max_batch_tokens
        for batch in workload.batch_sizes
    ):
        raise PlanningError(
            "batch_size times max_input_len exceeds engine.max_batch_tokens"
        )
    return alignment


def _cache_lengths(workload: PrefillWorkloadConfig, cache_step: int) -> Tuple[int, ...]:
    compute_floor = max(workload.min_compute_len, workload.commit_tail_tokens)
    largest_usable = workload.max_input_len - compute_floor
    if workload.min_cache_len == 0:
        largest_usable = max(0, largest_usable)
    maximum = (
        largest_usable if workload.max_cache_len is None else workload.max_cache_len
    )
    if maximum > largest_usable:
        raise PlanningError(
            "workload.max_cache_len must leave the required compute and commit tail"
        )

    if workload.cache_lengths:
        values = workload.cache_lengths
        if workload.random_cache_count:
            raise PlanningError(
                "cache_lengths and random_cache_count are mutually exclusive"
            )
        if len(values) != len(set(values)):
            raise PlanningError("workload.cache_lengths must be unique")
        for value in values:
            if value < workload.min_cache_len or value > maximum:
                raise PlanningError(
                    f"cache length {value} is outside the usable cache range"
                )
            if value % cache_step:
                raise PlanningError(
                    f"cache length {value} must be aligned to {cache_step}"
                )
        return tuple(values)

    if workload.random_cache_count <= 0:
        raise PlanningError(
            "prefill workload requires cache_lengths or positive random_cache_count"
        )
    first = (workload.min_cache_len + cache_step - 1) // cache_step * cache_step
    candidates = list(range(first, maximum + 1, cache_step))
    if workload.random_cache_count > len(candidates):
        raise PlanningError(
            f"requested {workload.random_cache_count} random cache lengths, "
            f"but only {len(candidates)} aligned values exist"
        )
    rng = _stable_rng(workload.seed, "prefill-cache-lengths")
    return tuple(sorted(rng.sample(candidates, workload.random_cache_count)))


def _plan_generated_cases(
    workload: PrefillWorkloadConfig, engine: EngineConfig, *, dp_size: int = 1
) -> Tuple[Dict[str, object], ...]:
    """Expand fixed/random cache slices and compute refinements into identities."""
    if dp_size > 1 and any(batch > 1 for batch in workload.batch_sizes):
        raise PlanningError("prefill DP>1 requires batch_size=1")
    cache_step = _validate_contract(workload, engine)
    cache_lengths = _cache_lengths(workload, cache_step)
    cases: List[Dict[str, object]] = []
    seen = set()
    compute_floor = max(workload.min_compute_len, workload.commit_tail_tokens)
    for batch_size in workload.batch_sizes:
        for cache_len in cache_lengths:
            maximum_compute = workload.max_input_len - cache_len
            first_compute = compute_floor if cache_len else workload.min_compute_len
            for refinement_level, compute_step in enumerate(workload.compute_steps):
                for compute_len in range(
                    first_compute, maximum_compute + 1, compute_step
                ):
                    input_len = cache_len + compute_len
                    geometry = (batch_size, input_len, cache_len)
                    if geometry in seen:
                        continue
                    seen.add(geometry)
                    cases.append(
                        {
                            "measurement_contract": PREFILL_MEASUREMENT_CONTRACT,
                            "batch_size": batch_size,
                            "input_len": input_len,
                            "cache_len": cache_len,
                            "compute_len": compute_len,
                            "measure_runs": workload.measure_runs,
                            "seed": workload.seed,
                            "prefix_policy": (
                                "shared_seed" if batch_size == 1 else "independent"
                            ),
                            "refinement_level": refinement_level,
                            "compute_step": compute_step,
                        }
                    )
    if not cases:
        raise PlanningError("prefill workload generated no valid cases")
    if workload.max_cases is not None and len(cases) > workload.max_cases:
        raise PlanningError(
            f"generated {len(cases)} cases, exceeding max_cases={workload.max_cases}"
        )
    return tuple(cases)


GROUPED_PREFILL_MEASUREMENT_CONTRACT = "concurrent_grouped_batch_v1"


def expected_probe_block_size(
    workload: PrefillWorkloadConfig, engine: EngineConfig
) -> int:
    """Return the configured physical block expectation used by preflight."""
    return resolve_cache_block_size(
        workload.probe_block_size, workload.cache_alignment, engine.args, engine.env
    )


def _plan_grouped_cases(
    workload: PrefillWorkloadConfig, engine: EngineConfig, dp_size: int
):
    if workload.workspace_tokens is None:
        raise PlanningError("workload.workspace_tokens is required for request_batches")
    block_size = expected_probe_block_size(workload, engine)
    alignment = math.lcm(workload.cache_alignment, block_size)
    if workload.commit_tail_tokens % workload.cache_alignment:
        raise PlanningError(
            "workload.commit_tail_tokens must be aligned to cache_alignment"
        )
    cases = []
    for batch in workload.request_batches:
        groups = batch.request_groups
        batch_size = sum(group.count for group in groups)
        if dp_size > 1 and batch_size > 1:
            raise PlanningError("prefill DP>1 requires batch_size=1")
        if batch_size > engine.concurrency_limit:
            raise PlanningError("grouped batch exceeds engine.concurrency_limit")
        for group in groups:
            if group.input_len + 1 > engine.max_seq_len:
                raise PlanningError(
                    "group input_len plus one output token exceeds engine.max_seq_len"
                )
            if group.cache_len % alignment:
                raise PlanningError(
                    "group cache_len must be aligned to physical cache blocks"
                )
            if group.cache_len > group.input_len or (
                group.cache_len
                and group.input_len - group.cache_len < workload.commit_tail_tokens
            ):
                raise PlanningError(
                    "group cache_len must leave the required commit tail"
                )
        packed = sum(group.count * (group.input_len + 1) for group in groups)
        padded = sum(
            group.count
            * ((group.input_len + 1 + block_size - 1) // block_size * block_size)
            for group in groups
        )
        if packed > engine.max_batch_tokens:
            raise PlanningError(
                "grouped input plus output reserve exceeds engine.max_batch_tokens"
            )
        if padded > workload.workspace_tokens:
            raise PlanningError(
                "block-padded grouped batch exceeds workload.workspace_tokens"
            )
        # Check individual seed feasibility; the runner forms concurrent batches
        # bounded by max_batch_tokens and block-padded workspace_tokens.
        for group in groups:
            if group.cache_len:
                seed_tokens = group.cache_len + workload.commit_tail_tokens + 1
                if (
                    seed_tokens > engine.max_seq_len
                    or seed_tokens > engine.max_batch_tokens
                ):
                    raise PlanningError("individual seed exceeds engine token limits")
                if (
                    seed_tokens + block_size - 1
                ) // block_size * block_size > workload.workspace_tokens:
                    raise PlanningError(
                        "individual seed exceeds workload.workspace_tokens"
                    )
        input_len = max(group.input_len for group in groups)
        cache_len = max(group.cache_len for group in groups)
        cases.append(
            {
                **(
                    {"sample_index": batch.sample_index}
                    if batch.sample_index is not None
                    else {}
                ),
                "measurement_contract": GROUPED_PREFILL_MEASUREMENT_CONTRACT,
                "batch_size": batch_size,
                "input_len": input_len,
                "cache_len": cache_len,
                "compute_len": max(
                    group.input_len - group.cache_len for group in groups
                ),
                "measure_runs": workload.measure_runs,
                "seed": workload.seed,
                "prefix_policy": "independent",
                "request_groups": [asdict(group) for group in groups],
            }
        )
    return tuple(cases)


def plan_prefill_cases(
    workload: PrefillWorkloadConfig, engine: EngineConfig, *, dp_size: int = 1
) -> Tuple[Dict[str, object], ...]:
    """Plan flat workloads unchanged, or named grids in declaration order."""
    block_size = expected_probe_block_size(workload, engine)
    for probe_input in (block_size + workload.commit_tail_tokens, 2 * block_size):
        if probe_input + 1 > engine.max_seq_len:
            raise PlanningError(
                "cache probe plus output reserve exceeds engine.max_seq_len"
            )
        if probe_input + 1 > engine.max_batch_tokens:
            raise PlanningError(
                "cache probe plus output reserve exceeds engine.max_batch_tokens"
            )
        if (
            workload.workspace_tokens is not None
            and (probe_input + 1 + block_size - 1) // block_size * block_size
            > workload.workspace_tokens
        ):
            raise PlanningError("cache probe exceeds workload.workspace_tokens")
    if workload.grids:
        cases = tuple(
            {**case, "grid_name": grid.name}
            for grid in workload.grids
            for case in plan_prefill_cases(grid.workload, engine, dp_size=dp_size)
        )
    elif workload.random_request_batches is not None:
        from rtp_llm.test.perf_test.prefill.random_batch import (
            sample_random_request_batches,
        )

        batches = sample_random_request_batches(workload, engine)
        cases = _plan_grouped_cases(
            replace(workload, request_batches=batches), engine, dp_size
        )
    elif workload.request_batches:
        cases = _plan_grouped_cases(workload, engine, dp_size)
    else:
        return _plan_generated_cases(workload, engine, dp_size=dp_size)
    if workload.max_cases is not None and len(cases) > workload.max_cases:
        raise PlanningError(
            f"generated {len(cases)} cases, exceeding max_cases={workload.max_cases}"
        )
    return cases


def require_concurrent_prefill_plan(plan: ExecutionPlan) -> None:
    """Reject persisted measurements whose batch semantics cannot be resumed."""
    if any(
        case.identity.get("measurement_contract")
        not in (PREFILL_MEASUREMENT_CONTRACT, GROUPED_PREFILL_MEASUREMENT_CONTRACT)
        for case in plan.cases
    ):
        raise PlanningError(
            f"prefill resume/profile requires {PREFILL_MEASUREMENT_CONTRACT}; "
            "rerun from config in a new directory for new measurements. "
            "Existing directories remain available for offline reports."
        )
