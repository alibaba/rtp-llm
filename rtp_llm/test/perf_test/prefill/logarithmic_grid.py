"""Deterministic B1 logarithmic input/compute anchors and cache-ratio slices."""

import math

from rtp_llm.test.perf_test.config_schema import (
    EngineConfig,
    PrefillWorkloadConfig,
    RequestBatchConfig,
    RequestGroupConfig,
)
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.prefill.block_size import resolve_cache_block_size


def _aligned_anchors(
    minimum: int, maximum: int, density: int, alignment: int
) -> tuple[int, ...]:
    low = (minimum + alignment - 1) // alignment * alignment
    high = maximum // alignment * alignment
    if low > high:
        raise PlanningError("logarithmic range contains no aligned lengths")
    values = {low, high}
    steps = math.ceil(math.log2(high / low) * density)
    for index in range(steps):
        value = low * 2 ** (index / density)
        aligned = math.floor(value / alignment + 0.5) * alignment
        values.add(min(high, max(low, aligned)))
    return tuple(sorted(values))


def generate_logarithmic_request_batches(
    workload: PrefillWorkloadConfig, engine: EngineConfig
) -> tuple[RequestBatchConfig, ...]:
    """Return sorted unique B1 shapes; grouped planning validates admission."""
    spec = workload.logarithmic_request_batches
    if spec is None or workload.workspace_tokens is None:
        raise PlanningError(
            "logarithmic grids require configuration and workspace_tokens"
        )
    if spec.batch_sizes != (1,):
        raise PlanningError("logarithmic grids support only batch_size=1")
    if spec.max_input_tokens + 1 > engine.max_seq_len:
        raise PlanningError(
            "logarithmic maximum plus one output token exceeds engine.max_seq_len"
        )
    block = resolve_cache_block_size(
        workload.probe_block_size, workload.cache_alignment, engine.args, engine.env
    )
    alignment = math.lcm(workload.cache_alignment, block)
    if spec.input_alignment % alignment:
        raise PlanningError(
            "logarithmic input_alignment must be a multiple of cache "
            "and physical block alignment"
        )
    lengths = _aligned_anchors(
        spec.min_input_tokens,
        spec.max_input_tokens,
        spec.input_points_per_doubling,
        spec.input_alignment,
    )
    compute_floor = max(spec.min_compute_tokens, workload.commit_tail_tokens)
    compute_floor = (compute_floor + alignment - 1) // alignment * alignment
    if compute_floor > lengths[0]:
        raise PlanningError(
            "logarithmic min_compute_tokens/commit tail exceeds the smallest input"
        )
    computes = _aligned_anchors(
        compute_floor, lengths[-1], spec.compute_points_per_doubling, alignment
    )
    geometry = set()
    for length in lengths:
        geometry.add((length, 0))
        geometry.update(
            (length, length - compute) for compute in computes if compute <= length
        )
        max_cache = (length - compute_floor) // alignment * alignment
        for ratio in spec.cache_hit_ratios:
            cache = min(math.floor(length * ratio / alignment) * alignment, max_cache)
            geometry.add((length, cache))
    if workload.max_cases is not None and len(geometry) > workload.max_cases:
        raise PlanningError(
            f"generated {len(geometry)} cases, exceeding max_cases={workload.max_cases}"
        )
    return tuple(
        RequestBatchConfig((RequestGroupConfig(1, length, cache),))
        for length, cache in sorted(geometry)
    )
