"""Deterministic, bounded sampling of heterogeneous prefill batches."""

import itertools
import math
import random

from rtp_llm.test.perf_test.config_schema import (
    EngineConfig,
    PrefillWorkloadConfig,
    RequestBatchConfig,
    RequestGroupConfig,
)
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.prefill.block_size import resolve_cache_block_size


def sample_random_request_batches(
    workload: PrefillWorkloadConfig, engine: EngineConfig
) -> tuple[RequestBatchConfig, ...]:
    """Sample distinct unordered member geometries subject to runtime budgets."""
    spec = workload.random_request_batches
    if spec is None or workload.workspace_tokens is None:
        raise PlanningError(
            "random request batches require configuration and workspace_tokens"
        )
    block = resolve_cache_block_size(
        workload.probe_block_size, workload.cache_alignment, engine.args, engine.env
    )
    alignment = math.lcm(workload.cache_alignment, block)
    if workload.commit_tail_tokens % workload.cache_alignment:
        raise PlanningError(
            "workload.commit_tail_tokens must be aligned to cache_alignment"
        )
    if any(size > engine.concurrency_limit for size in spec.batch_sizes):
        raise PlanningError("random batch exceeds engine.concurrency_limit")
    if spec.num_cases < len(spec.batch_sizes):
        raise PlanningError("num_cases must cover every requested batch size")
    rng = random.Random(workload.seed)
    results = []
    seen = set()
    completions = {}

    def emit(shape):
        seen.add(shape)
        members = list(shape)
        rng.shuffle(members)
        return RequestBatchConfig(
            tuple(RequestGroupConfig(1, length, cache) for length, cache in members)
        )

    def finite_shapes(size, low, high):
        # Bound enumeration by member visits, not just number of distributions.
        # Combinations with replacement enumerate unordered shapes exactly once.
        max_shapes = 100000 // size
        if high - low + 1 > max_shapes:
            return None
        geometries = []
        for units in range(low, high + 1):
            length = units * spec.input_alignment
            max_cache = max(0, (length - workload.commit_tail_tokens) // alignment)
            allow_cold = (
                spec.cold_probability > 0
                or max_cache == 0
                or spec.kv_budget_tokens is not None
            )
            cache_count = max_cache if spec.cold_probability < 1 else 0
            if len(geometries) + allow_cold + cache_count > max_shapes:
                return None
            if allow_cold:
                geometries.append((length, 0))
            geometries.extend(
                (length, cache * alignment) for cache in range(1, cache_count + 1)
            )
        if math.comb(len(geometries) + size - 1, size) > max_shapes:
            return None
        return itertools.combinations_with_replacement(geometries, size)

    def complete_size(size, low):
        shapes = completions[size]
        if shapes is None:
            return None
        min_input = low * spec.input_alignment
        min_padded = (min_input + block) // block * block
        for shape in shapes:
            if shape in seen or (size > 1 and len(set(shape)) < 2):
                continue
            raw = sum(length + 1 for length, _ in shape)
            padded = sum((length + block) // block * block for length, _ in shape)
            kv = padded + sum(
                workload.commit_tail_tokens for _, cache in shape if cache
            )
            if (
                raw > engine.max_batch_tokens
                or padded > workload.workspace_tokens
                or (spec.kv_budget_tokens is not None and kv > spec.kv_budget_tokens)
            ):
                continue
            if spec.cold_probability == 0 and spec.kv_budget_tokens is not None:
                # At p(cold)=0, cacheable cold members require the existing KV
                # fallback. Put all other members first to consume their budget,
                # then try the largest cold members first. If even the largest
                # cannot be forced cold, no ordering can generate this shape.
                forced_cold = sorted(
                    (length + block) // block * block
                    for length, cache in shape
                    if cache == 0 and length - workload.commit_tail_tokens >= alignment
                )
                kv_left = spec.kv_budget_tokens - kv + sum(forced_cold)
                for remaining, cost in enumerate(reversed(forced_cold)):
                    reserve = (len(forced_cold) - remaining - 1) * min_padded
                    if cost + workload.commit_tail_tokens <= kv_left - reserve:
                        break
                    kv_left -= cost
                else:
                    return emit(shape)
                continue
            return emit(shape)
        return None

    def exhaustion_error(sizes):
        if any(completions.get(size) is None for size in sizes):
            return PlanningError(
                "random batch search budget exhausted; candidate shape space exceeds "
                "the completion budget (feasibility is unknown)"
            )
        return PlanningError(
            "impossible or insufficient distinct random batch shape space "
            f"for batch sizes {list(sizes)}"
        )

    def sample_size(size):
        low = (spec.min_input_tokens + spec.input_alignment - 1) // spec.input_alignment
        high = (
            min(
                spec.max_input_tokens,
                engine.max_seq_len - 1,
                spec.batch_input_limits.get(str(size), spec.max_input_tokens),
            )
            // spec.input_alignment
        )
        if low > high:
            raise PlanningError(f"impossible random input bounds for batch size {size}")
        if size in completions:
            return complete_size(size, low)
        # Keep weighted sampling for the common path; retry exhaustion alone
        # cannot prove that the finite shape space is insufficient.
        for _ in range(10000):
            members = []
            raw_left = engine.max_batch_tokens
            workspace_left = workload.workspace_tokens
            kv_left = spec.kv_budget_tokens
            for member in range(size):
                remaining = size - member - 1
                min_input = low * spec.input_alignment
                min_padded = (min_input + 1 + block - 1) // block * block
                upper = min(
                    high,
                    (raw_left - remaining * (min_input + 1) - 1)
                    // spec.input_alignment,
                    ((workspace_left - remaining * min_padded) // block * block - 1)
                    // spec.input_alignment,
                )
                if kv_left is not None:
                    upper = min(
                        upper,
                        ((kv_left - remaining * min_padded) // block * block - 1)
                        // spec.input_alignment,
                    )
                if upper < low:
                    break
                length = rng.randint(low, upper) * spec.input_alignment
                padded = (length + 1 + block - 1) // block * block
                if padded > workspace_left - remaining * min_padded or (
                    kv_left is not None and padded > kv_left - remaining * min_padded
                ):
                    break
                max_cache = max(0, (length - workload.commit_tail_tokens) // alignment)
                cached = max_cache > 0 and rng.random() >= spec.cold_probability
                if (
                    cached
                    and kv_left is not None
                    and padded + workload.commit_tail_tokens
                    > kv_left - remaining * min_padded
                ):
                    cached = False
                cache = rng.randint(1, max_cache) * alignment if cached else 0
                members.append((length, cache))
                raw_left -= length + 1
                workspace_left -= padded
                if kv_left is not None:
                    kv_left -= padded + (workload.commit_tail_tokens if cache else 0)
            shape = tuple(sorted(members))
            if (
                len(members) != size
                or (size > 1 and len(set(shape)) < 2)
                or shape in seen
            ):
                continue
            # Preserve the sampled order before shuffling on the random path.
            seen.add(shape)
            rng.shuffle(members)
            return RequestBatchConfig(
                tuple(RequestGroupConfig(1, length, cache) for length, cache in members)
            )
        completions[size] = finite_shapes(size, low, high)
        return complete_size(size, low)

    # Coverage is mandatory; subsequent cases can use any size with capacity.
    for size in spec.batch_sizes:
        batch = sample_size(size)
        if batch is None:
            raise exhaustion_error((size,))
        results.append(batch)
    while len(results) < spec.num_cases:
        sizes = list(spec.batch_sizes)
        rng.shuffle(sizes)
        for size in sizes:
            batch = sample_size(size)
            if batch is not None:
                results.append(batch)
                break
        else:
            raise exhaustion_error(spec.batch_sizes)
    return tuple(results)
