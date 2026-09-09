"""Selection and deterministic LPT planning over compiled instance metadata."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from .resource_plan import JavaMockBudget


class InstancePlanError(ValueError):
    pass


def _names(raw: str | None, flag: str) -> list[str] | None:
    if raw is None:
        return None
    values = list(
        dict.fromkeys(value.strip() for value in raw.split(",") if value.strip())
    )
    if not values:
        raise InstancePlanError(f"{flag} contains no nonempty entries")
    return values


@dataclass(frozen=True)
class Instance:
    id: str
    category: str
    profile: str
    source: str
    duration_s: float
    budget: JavaMockBudget
    metadata: dict


def parse_catalog(payload: dict, *, source: str, profile: str) -> list[Instance]:
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        raise InstancePlanError("unsupported instance list schema_version")
    rows = payload.get("instances")
    if not isinstance(rows, list):
        raise InstancePlanError("instance list must contain an instances array")
    seen = set()
    instances = []
    for row in rows:
        if not isinstance(row, dict):
            raise InstancePlanError("instance metadata must be an object")
        for key in [
            "id",
            "scenario_id",
            "variant_id",
            "profile",
            "category",
            "source_path",
            "source",
        ]:
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise InstancePlanError(f"instance requires nonempty {key}")
        identity = row["id"]
        if "," in identity or any(c.isspace() for c in identity):
            raise InstancePlanError(
                f"instance id cannot contain comma or whitespace: {identity!r}"
            )
        if identity in seen:
            raise InstancePlanError(f"duplicate instance id: {identity}")
        seen.add(identity)
        if row["source"] != source or row["profile"] != profile:
            raise InstancePlanError(f"instance source/profile mismatch: {identity}")
        for key in ["tags", "requires"]:
            if not isinstance(row.get(key), list) or any(
                not isinstance(item, str) for item in row[key]
            ):
                raise InstancePlanError(
                    f"instance {identity} requires a string list for {key}"
                )
        duration = row.get("estimated_duration_s")
        if (
            type(duration) not in (int, float)
            or not math.isfinite(duration)
            or duration <= 0
        ):
            raise InstancePlanError(
                f"instance {identity} has invalid estimated_duration_s"
            )
        if source == "yaml":
            budget = JavaMockBudget.from_metadata(row.get("resource_budget"))
            execution = row.get("execution")
            if not isinstance(execution, dict) or set(execution) != {
                "timeout_s",
                "cleanup_timeout_s",
            }:
                raise InstancePlanError(
                    f"instance {identity} requires execution time budgets"
                )
            if any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
                for value in execution.values()
            ):
                raise InstancePlanError(
                    f"instance {identity} has invalid execution time budgets"
                )
        else:
            raise InstancePlanError(f"unsupported instance source: {source}")
        instances.append(
            Instance(
                identity,
                row["category"],
                profile,
                source,
                float(duration),
                budget,
                {
                    key: row[key]
                    for key in (
                        "id",
                        "scenario_id",
                        "variant_id",
                        "profile",
                        "grade",
                        "category",
                        "source",
                        "source_path",
                        "tags",
                        "requires",
                        "estimated_duration_s",
                        "resource_budget",
                        "execution",
                    )
                    if key in row
                },
            )
        )
    return instances


def select_instances(
    instances: Sequence[Instance],
    *,
    exact_ids: str | None = None,
    categories: str | None = None,
) -> list[Instance]:
    ids = _names(exact_ids, "--instances")
    cats = _names(categories, "--categories")
    if len({row.id for row in instances}) != len(instances):
        raise InstancePlanError("duplicate instance id across sources")
    if ids is not None:
        missing = set(ids) - {row.id for row in instances}
        if missing:
            raise InstancePlanError(
                f"unknown or profile-excluded instances: {sorted(missing)}"
            )
    if cats is not None:
        cats = ["engine_fault" if c == "engine-fault" else c for c in cats]
        missing = set(cats) - {row.category for row in instances}
        if missing:
            raise InstancePlanError(
                f"unknown or profile-empty categories: {sorted(missing)}"
            )
    selected = [
        row
        for row in instances
        if (ids is None or row.id in ids) and (cats is None or row.category in cats)
    ]
    if not selected:
        raise InstancePlanError("selection contains no executable instances")
    return selected


def plan_instances(
    instances: Sequence[Instance],
    parallel: int,
    timings: Mapping[str, float] | None = None,
) -> list[list[Instance]]:
    if type(parallel) is not int or parallel < 1:
        raise InstancePlanError("parallel must be a positive integer")
    if not instances:
        raise InstancePlanError("cannot plan zero instances")
    if len({row.id for row in instances}) != len(instances):
        raise InstancePlanError("duplicate instance id in plan")
    costs = {}
    for row in instances:
        cost = (timings or {}).get(row.id, row.duration_s)
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost <= 0:
            raise InstancePlanError(f"invalid timing for {row.id}")
        costs[row.id] = cost
    lanes: list[list[Instance]] = [[] for _ in range(min(parallel, len(instances)))]
    loads = [0.0] * len(lanes)
    for row in sorted(instances, key=lambda row: (-costs[row.id], row.id)):
        lane = min(range(len(lanes)), key=lambda i: (loads[i], i))
        lanes[lane].append(row)
        loads[lane] += costs[row.id]
    order = {row.id: i for i, row in enumerate(instances)}
    for lane in lanes:
        lane.sort(key=lambda row: order[row.id])
    return lanes
