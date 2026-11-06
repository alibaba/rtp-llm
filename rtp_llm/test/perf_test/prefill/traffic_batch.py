"""Sample grouped prefill geometries from a traffic length distribution."""

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping

from rtp_llm.test.perf_test.config_schema import RequestBatchConfig, RequestGroupConfig
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.prefill.block_size import resolve_cache_block_size
from rtp_llm.test.perf_test.prefill.traffic_distribution import LengthSampler


@dataclass(frozen=True)
class TrafficSample:
    batches: tuple[RequestBatchConfig, ...]
    summary: dict


def sample_traffic_batches(workload, engine, distribution, *, dp_size):
    spec = workload.traffic_request_batches
    if spec is None or workload.workspace_tokens is None:
        raise PlanningError(
            "traffic sampling requires configuration and workspace_tokens"
        )
    block = resolve_cache_block_size(
        workload.probe_block_size, workload.cache_alignment, engine.args, engine.env
    )
    alignment = math.lcm(workload.cache_alignment, block)
    if workload.commit_tail_tokens % alignment:
        raise PlanningError(
            "commit_tail_tokens must be aligned to physical cache blocks"
        )
    if any(size > engine.concurrency_limit for size in spec.batch_sizes):
        raise PlanningError("traffic batch exceeds engine.concurrency_limit")
    if dp_size > 1 and any(size > 1 for size in spec.batch_sizes):
        raise PlanningError("prefill DP>1 requires batch_size=1")
    maximum = min(
        spec.max_input_tokens or engine.max_seq_len - 1, engine.max_seq_len - 1
    )
    try:
        sampler = LengthSampler(
            distribution, spec.min_input_tokens, maximum, workload.seed
        )
    except ValueError as error:
        raise PlanningError(str(error)) from error
    batches = []
    draws = rejected = projected = 0
    per_size = {str(size): 0 for size in spec.batch_sizes}
    lengths = []
    reuse_lengths = []
    caches = []
    for index in range(spec.num_cases):
        size = spec.batch_sizes[index % len(spec.batch_sizes)]
        for _ in range(spec.retry_limit):
            pairs = [sampler.draw() for _ in range(size)]
            draws += size
            groups = []
            for length, reuse in pairs:
                cache = min(
                    reuse // alignment * alignment,
                    max(
                        0,
                        (length - workload.commit_tail_tokens) // alignment * alignment,
                    ),
                )
                groups.append(RequestGroupConfig(1, length, cache))
            packed = sum(g.input_len + 1 for g in groups)
            padded = sum((g.input_len + block) // block * block for g in groups)
            kv = padded + sum(
                workload.commit_tail_tokens for g in groups if g.cache_len
            )
            seeds_fit = all(
                not g.cache_len
                or (
                    g.cache_len + workload.commit_tail_tokens + 1
                    <= min(engine.max_seq_len, engine.max_batch_tokens)
                    and (g.cache_len + workload.commit_tail_tokens + block)
                    // block
                    * block
                    <= workload.workspace_tokens
                )
                for g in groups
            )
            if (
                packed <= engine.max_batch_tokens
                and padded <= workload.workspace_tokens
                and (spec.kv_budget_tokens is None or kv <= spec.kv_budget_tokens)
                and seeds_fit
            ):
                batches.append(RequestBatchConfig(tuple(groups), index + 1))
                per_size[str(size)] += 1
                lengths.extend(length for length, _ in pairs)
                reuse_lengths.extend(reuse for _, reuse in pairs)
                caches.extend(g.cache_len for g in groups)
                projected += sum(
                    reuse != g.cache_len for (_, reuse), g in zip(pairs, groups)
                )
                break
            rejected += 1
        else:
            raise PlanningError(
                f"traffic sampling retry limit exhausted for batch size {size}; increase budgets or retry_limit"
            )

    def stats(values):
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    # A separate fixed seed keeps this estimate independent of the workload RNG.
    # LengthSampler weights eligible cells by their clipped source mass.
    baseline_sampler = LengthSampler(
        distribution, spec.min_input_tokens, maximum, 20261104
    )
    baseline_pairs = [baseline_sampler.draw() for _ in range(4096)]
    baseline_caches = [
        min(
            reuse // alignment * alignment,
            max(0, (length - workload.commit_tail_tokens) // alignment * alignment),
        )
        for length, reuse in baseline_pairs
    ]
    return TrafficSample(
        tuple(batches),
        {
            "source_rows": distribution.source_rows,
            "eligible_source_fraction": sampler.eligible_source_fraction,
            "draw_count": draws,
            "rejected_batches": rejected,
            "cases_per_batch_size": per_size,
            "cache_projected_members": projected,
            "accepted_input_tokens": stats(lengths),
            "accepted_reuse_tokens": stats(reuse_lengths),
            "accepted_cache_tokens": stats(caches),
            "eligible_source_estimate": {
                "method": "independent_weighted_draws",
                "sample_count": len(baseline_pairs),
                "input_tokens": stats([length for length, _ in baseline_pairs]),
                "reuse_tokens": stats([reuse for _, reuse in baseline_pairs]),
                "cache_tokens": stats(baseline_caches),
            },
        },
    )


def materialize_traffic_source(
    source: Mapping[str, Any], resolved_workload: Mapping[str, Any]
) -> dict:
    result = copy.deepcopy(source)

    def materialize(raw, resolved):
        if "traffic_request_batches" in raw:
            raw.pop("traffic_request_batches")
            raw["request_batches"] = [
                {
                    **copy.deepcopy(batch),
                    "request_groups": list(copy.deepcopy(batch["request_groups"])),
                }
                for batch in resolved["request_batches"]
            ]
            raw["seed"] = resolved["seed"]
        for child, resolved_child in zip(
            raw.get("grids", ()), resolved.get("grids", ())
        ):
            materialize(child, resolved_child["workload"])

    materialize(result["workload"], resolved_workload)
    return result
