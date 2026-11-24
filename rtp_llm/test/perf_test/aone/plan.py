"""Public AOne resource and gang topology planning."""

from __future__ import annotations

from typing import Any, Mapping

from rtp_llm.test.perf_test.decode.scheduler import requires_distributed_decode
from rtp_llm.test.perf_test.errors import (
    ConfigurationError,
    PlanningError,
    ResourceError,
)
from rtp_llm.test.perf_test.plan import AOneResourcePlan, derive_gang_topology


def _aone_resource_plan(
    resolved_config: Mapping[str, Any], world_size: int, case_count: int
) -> AOneResourcePlan:
    execution = resolved_config["execution"]
    if execution["resource"] not in AONE_RESOURCES:
        supported = ", ".join(AONE_RESOURCES)
        raise ResourceError(
            f"unsupported AOne resource {execution['resource']!r}; "
            f"supported resources: {supported}"
        )
    max_parallel = execution["max_parallel"]
    shard_count = execution.get("shard_count")
    gpus_per_machine = execution.get("gpus_per_machine")
    gang = derive_gang_topology(
        world_size, world_size if gpus_per_machine is None else gpus_per_machine
    )
    capacity = AONE_GPU_HOSTS.get(execution["gpu_type"])
    if capacity is not None and gang.local_world_size > capacity:
        raise ResourceError(
            f"GPU type {execution['gpu_type']!r} has host capacity {capacity}; "
            f"requested {gang.local_world_size} GPUs per machine"
        )
    if gang.machine_count > 1:
        # A gang shard always requests the full host, so the GPU type must
        # have a registered per-host GPU count before planning continues.
        if execution["gpu_type"] not in AONE_GPU_HOSTS:
            supported = ", ".join(sorted(AONE_GPU_HOSTS))
            raise ResourceError(
                f"unsupported GPU type {execution['gpu_type']!r} for gang "
                "execution: gang shards request full hosts, so the GPU type "
                "must have a per-host GPU count in AONE_GPU_HOSTS; "
                f"supported GPU types: {supported}"
            )
        if shard_count is not None:
            raise PlanningError(
                "execution.shard_count cannot be combined with "
                "execution.gpus_per_machine: case-shard and gang semantics "
                "are mutually exclusive"
            )
        if max_parallel < gang.machine_count:
            raise ResourceError(
                f"execution.max_parallel {max_parallel} cannot run a gang of "
                f"machine_count {gang.machine_count} machines"
            )
    else:
        if execution["gpu_type"] not in AONE_GPU_PLATFORMS:
            supported = ", ".join(sorted(AONE_GPU_PLATFORMS))
            raise ResourceError(
                f"unsupported GPU type {execution['gpu_type']!r}; "
                f"supported GPU types: {supported}"
            )
        if shard_count is None:
            raise PlanningError(
                "execution.shard_count is required when "
                "execution.gpus_per_machine derives machine_count == 1"
            )
        if shard_count > case_count:
            raise ResourceError(
                f"execution.shard_count {shard_count} exceeds case count {case_count}"
            )
        if max_parallel > shard_count:
            raise ResourceError(
                "execution.max_parallel cannot exceed execution.shard_count"
            )
    return AOneResourcePlan(
        backend="aone",
        resource=execution["resource"],
        gpu_type=execution["gpu_type"],
        gpus_per_shard=world_size,
        shard_count=shard_count,
        max_parallel=max_parallel,
        gpus_per_machine=gang.gpus_per_machine,
        machine_count=gang.machine_count,
        local_world_size=gang.local_world_size,
    )


def plan_resources(
    *, mode: str, resolved_config: Mapping[str, Any], world_size: int, case_count: int
) -> AOneResourcePlan:
    if resolved_config["execution"]["backend"] != "aone":
        raise ResourceError("AOne resource planning requires backend=aone")
    if mode == "prefill" and len(resolved_config["workload"].get("grids", ())) > 1:
        raise PlanningError("AOne prefill supports only one grid per run")
    resource = _aone_resource_plan(resolved_config, world_size, case_count)
    if mode == "decode":
        requires_distributed_decode(
            resolved_config["model"]["topology"], resource.local_world_size
        )
    if mode == "decode" and resource.machine_count == 1:
        raise PlanningError(
            "decode AOne execution requires a gang: execution."
            "gpus_per_machine must be smaller than the model topology "
            f"world_size {world_size} so the derived machine_count "
            "exceeds one"
        )
    aone_exec_properties(
        resource, resolved_config["execution"].get("exec_properties", {})
    )
    return resource


def aone_exec_properties(resource, extra=None):
    """Freeze scheduler properties, preserving full-host reservations for gangs."""
    from rtp_llm.test.perf_test.launch_config import execution_properties

    properties = execution_properties(dict(extra or {}))
    required = {
        "gpu": resource.gpu_type,
        "gpu_count": str(
            AONE_GPU_HOSTS[resource.gpu_type]
            if resource.machine_count > 1
            else resource.gpus_per_shard
        ),
    }
    for key, value in required.items():
        if key in properties and properties[key] != value:
            raise ConfigurationError(
                f"execution.exec_properties.{key} conflicts with planned value {value!r}"
            )
        properties[key] = value
    return properties


AONE_GPU_PLATFORMS = {
    "H20": "cuda13",
    "H20_CU13": "cuda13",
    "L20D_DEV": "cuda13",
    "L20_CU13": "cuda13",
    "RTX_5000_PRO_CU13": "cuda13",
    "SM100_ARM_CU13": "cuda13_arm",
    "SM103_CU13": "cuda13",
}

AONE_GPU_HOSTS = {
    "H20": 8,
    "H20_CU13": 8,
    "L20D_DEV": 8,
    "L20_CU13": 8,
    "RTX_5000_PRO_CU13": 8,
    "SM100_ARM_CU13": 4,
    "SM103_CU13": 8,
}

AONE_RESOURCES = ("daily", "online")
