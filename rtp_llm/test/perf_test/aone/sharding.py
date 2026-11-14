"""Deterministic cost-aware sharding for immutable AOne execution plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import (
    AOneResourcePlan,
    ExecutionPlan,
    PlannedCase,
    thaw,
)

SHARD_PLAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ShardAssignment:
    shard_id: str
    case_ids: Tuple[str, ...]
    estimated_cost: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "case_ids": list(self.case_ids),
            "estimated_cost": self.estimated_cost,
        }


@dataclass(frozen=True)
class AOneShardPlan:
    schema_version: int
    config_fingerprint: str
    workload_fingerprint: str
    shards: Tuple[ShardAssignment, ...]
    total_estimated_cost: int

    @property
    def ownership(self) -> Tuple[Tuple[str, ...], ...]:
        return tuple(shard.case_ids for shard in self.shards)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_fingerprint": self.config_fingerprint,
            "workload_fingerprint": self.workload_fingerprint,
            "total_estimated_cost": self.total_estimated_cost,
            "shards": [shard.to_dict() for shard in self.shards],
        }


def _case_identity(case: Union[PlannedCase, Mapping[str, Any]]) -> Mapping[str, Any]:
    if isinstance(case, PlannedCase):
        return thaw(case.identity)
    if isinstance(case, Mapping):
        return case
    raise PlanningError("AOne case must be a planned case or identity object")


def case_weight(case: Union[PlannedCase, Mapping[str, Any]]) -> int:
    """Estimate prefill attention work using the legacy cache-grid formula."""
    identity = _case_identity(case)
    if "request_groups" in identity:
        groups = identity["request_groups"]
        if not isinstance(groups, (list, tuple)) or not groups:
            raise PlanningError("AOne request_groups must be a non-empty array")
        total = 0
        for group in groups:
            if not isinstance(group, Mapping):
                raise PlanningError("AOne request group must be an object")
            count = group.get("count")
            if type(count) is not int or count <= 0:
                raise PlanningError(
                    "AOne request group count must be a positive integer"
                )
            total += case_weight(
                {
                    "batch_size": count,
                    "input_len": group.get("input_len"),
                    "cache_len": group.get("cache_len"),
                }
            )
        return total
    values = []
    for name in ("batch_size", "input_len", "cache_len"):
        value = identity.get(name)
        if type(value) is not int or value < 0 or name != "cache_len" and value == 0:
            raise PlanningError(f"AOne case {name} must be a valid integer")
        values.append(value)
    batch_size, input_len, cache_len = values
    if cache_len > input_len:
        raise PlanningError("AOne case cache_len cannot exceed input_len")
    return batch_size * input_len * max(1, input_len - cache_len)


def plan_shards(plan: ExecutionPlan) -> AOneShardPlan:
    """Assign longest cases first to the currently lightest planned shard."""
    resource = plan.resource_plan
    if (
        plan.mode != "prefill"
        or plan.action != "run"
        or not isinstance(resource, AOneResourcePlan)
    ):
        raise PlanningError("AOne sharding requires an AOne prefill run plan")
    if resource.machine_count > 1 or resource.shard_count is None:
        raise PlanningError("AOne case sharding cannot be used for a gang")
    if resource.shard_count <= 0:
        raise PlanningError("AOne shard count must be positive")
    if resource.shard_count > len(plan.cases):
        raise PlanningError("AOne shard count cannot exceed planned case count")

    assignments: List[List[PlannedCase]] = [[] for _ in range(resource.shard_count)]
    costs = [0] * resource.shard_count
    ordered = sorted(plan.cases, key=lambda case: (-case_weight(case), case.case_id))
    for case in ordered:
        index = min(range(resource.shard_count), key=lambda item: (costs[item], item))
        assignments[index].append(case)
        costs[index] += case_weight(case)

    shards = tuple(
        ShardAssignment(
            shard_id=f"shard-{index:04d}",
            case_ids=tuple(sorted(case.case_id for case in cases)),
            estimated_cost=costs[index],
        )
        for index, cases in enumerate(assignments)
    )
    return AOneShardPlan(
        schema_version=SHARD_PLAN_SCHEMA_VERSION,
        config_fingerprint=plan.config_fingerprint,
        workload_fingerprint=plan.workload_fingerprint,
        shards=shards,
        total_estimated_cost=sum(costs),
    )
