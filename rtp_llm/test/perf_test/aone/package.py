"""Generate self-contained AOne case-shard and machine-gang packages."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from rtp_llm.test.perf_test.aone.plan import AONE_GPU_HOSTS
from rtp_llm.test.perf_test.errors import PlanningError
from rtp_llm.test.perf_test.plan import AOneResourcePlan, ExecutionPlan, thaw
from rtp_llm.test.perf_test.storage import atomic_write_json

PACKAGE_SCHEMA_VERSION = 1

_PACKAGE_NAME_PATTERN = re.compile(r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")


def _validated_package_name(package_name: object) -> str:
    """Reject names that cannot become a generated Bazel package label."""
    if not isinstance(package_name, str) or not _PACKAGE_NAME_PATTERN.fullmatch(
        package_name
    ):
        raise PlanningError(f"invalid generated Bazel package: {package_name!r}")
    return package_name


@dataclass(frozen=True)
class GeneratedAOnePackage:
    root: Path
    resolved_config: Path
    resolved_workload: Path
    shard_plan: Path
    build_file: Path
    targets: Tuple[str, ...]


def _validated_ownership(
    plan: ExecutionPlan, ownership: Sequence[Sequence[str]]
) -> Tuple[Tuple[str, ...], ...]:
    resource = plan.resource_plan
    if (
        plan.mode != "prefill"
        or plan.action != "run"
        or not isinstance(resource, AOneResourcePlan)
        or resource.machine_count != 1
    ):
        raise PlanningError("AOne package requires an AOne prefill run plan")
    if len(ownership) != resource.shard_count:
        raise PlanningError("AOne ownership count must match the planned shard count")
    normalized: List[Tuple[str, ...]] = []
    flattened: List[str] = []
    for index, case_ids in enumerate(ownership):
        shard = tuple(case_ids)
        if not shard:
            raise PlanningError(f"AOne shard {index} cannot be empty")
        if len(set(shard)) != len(shard):
            raise PlanningError(f"AOne shard {index} contains duplicate cases")
        normalized.append(tuple(sorted(shard)))
        flattened.extend(shard)
    if len(set(flattened)) != len(flattened):
        raise PlanningError("AOne case ownership overlaps across shards")
    planned = {case.case_id for case in plan.cases}
    owned = set(flattened)
    if owned != planned:
        missing = sorted(planned - owned)
        unknown = sorted(owned - planned)
        raise PlanningError(
            f"AOne ownership does not cover the plan: missing={missing}, unknown={unknown}"
        )
    return tuple(normalized)


def _build_file(resource: AOneResourcePlan, ownership: Iterable[Sequence[str]]) -> str:
    lines = [
        'load("//rtp_llm/test/perf_test:aone_defs.bzl", "perf_test_aone_shard")',
        "",
    ]
    for index, _case_ids in enumerate(ownership):
        lines.extend(
            [
                "perf_test_aone_shard(",
                f'    name = "shard_{index:04d}",',
                '    resolved_config = "resolved_config.json",',
                '    resolved_workload = "resolved_workload.json",',
                '    shard_plan = "shard_plan.json",',
                f'    shard_id = "shard-{index:04d}",',
                f'    gpu_type = "{resource.gpu_type}",',
                f"    gpu_count = {resource.gpus_per_shard},",
                ")",
                "",
            ]
        )
    targets = ", ".join(f'":shard_{index:04d}"' for index, _ in enumerate(ownership))
    lines.extend(
        [
            "test_suite(",
            '    name = "all",',
            f"    tests = [{targets}],",
            '    tags = ["manual"],',
            ")",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, value: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    replaced = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        replaced = True
    finally:
        if not replaced:
            temporary.unlink(missing_ok=True)


def generate_aone_package(
    root: Union[str, Path],
    plan: ExecutionPlan,
    *,
    run_id: str,
    ownership: Sequence[Sequence[str]],
    package: Optional[str] = None,
) -> GeneratedAOnePackage:
    """Materialize normalized inputs without retaining submit-host source paths."""
    if not isinstance(run_id, str) or not run_id:
        raise PlanningError("AOne run_id must be a non-empty string")
    normalized_ownership = _validated_ownership(plan, ownership)
    resource = plan.resource_plan
    assert isinstance(resource, AOneResourcePlan)
    cases = {case.case_id: case.to_dict() for case in plan.cases}
    shard_payload = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": plan.mode,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "shards": [
            {
                "shard_id": f"shard-{index:04d}",
                "cases": [cases[case_id] for case_id in case_ids],
            }
            for index, case_ids in enumerate(normalized_ownership)
        ],
    }

    return _write_package(
        root,
        plan,
        package=package,
        payload=shard_payload,
        payload_name="shard_plan.json",
        build_text=_build_file(resource, normalized_ownership),
        target_names=tuple(
            f"shard_{index:04d}" for index in range(len(normalized_ownership))
        ),
        description="AOne",
    )


def _write_package(
    root: Union[str, Path],
    plan: ExecutionPlan,
    *,
    package: Optional[str],
    payload: Dict[str, Any],
    payload_name: str,
    build_text: str,
    target_names: Sequence[str],
    description: str,
    omit_null_shard_count: bool = False,
) -> GeneratedAOnePackage:
    """Write the frozen inputs and generated BUILD."""
    resolved = thaw(plan.resolved_config)
    workload = resolved.pop("workload")
    if omit_null_shard_count and resolved["execution"].get("shard_count") is None:
        resolved["execution"].pop("shard_count", None)
    output = Path(root)
    package_name = _validated_package_name(output.name if package is None else package)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise PlanningError(
            f"generated {description} package already exists: {output}"
        ) from error
    except OSError as error:
        raise PlanningError(
            f"cannot create generated {description} package {output}: {error}"
        ) from error
    resolved_config = output / "resolved_config.json"
    resolved_workload = output / "resolved_workload.json"
    shard_plan = output / payload_name
    build_file = output / "BUILD.bazel"
    atomic_write_json(resolved_config, resolved)
    atomic_write_json(resolved_workload, workload)
    atomic_write_json(shard_plan, payload)
    _atomic_write_text(build_file, build_text)
    return GeneratedAOnePackage(
        root=output,
        resolved_config=resolved_config,
        resolved_workload=resolved_workload,
        shard_plan=shard_plan,
        build_file=build_file,
        targets=tuple(f"//{package_name}:{name}" for name in target_names),
    )


GANG_PLAN_SCHEMA_VERSION = 1

GANG_BASE_PORT = 12333


def _per_host_gpu_count(gpu_type: str) -> int:
    """Return the registered full-host GPU count for ``gpu_type``."""
    per_host = AONE_GPU_HOSTS.get(gpu_type)
    if per_host is None:
        supported = ", ".join(sorted(AONE_GPU_HOSTS))
        raise PlanningError(
            f"unsupported GPU type {gpu_type!r} for gang execution: gang "
            "shards request full hosts, so the GPU type must have a "
            "per-host GPU count in AONE_GPU_HOSTS; "
            f"supported GPU types: {supported}"
        )
    return per_host


def build_gang_plan_payload(plan: ExecutionPlan, run_id: str) -> Dict[str, Any]:
    """Build the ``gang_plan.json`` payload shared by every gang member."""
    if not isinstance(run_id, str) or not run_id:
        raise PlanningError("AOne run_id must be a non-empty string")
    resource = plan.resource_plan
    if not isinstance(resource, AOneResourcePlan):
        raise PlanningError("gang plan requires an AOne resource plan")
    return {
        "schema_version": GANG_PLAN_SCHEMA_VERSION,
        "run_id": run_id,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "world_size": resource.machine_count * resource.local_world_size,
        "machine_count": resource.machine_count,
        "local_world_size": resource.local_world_size,
        "gpus_per_machine": resource.gpus_per_machine,
        "gpus_per_host": _per_host_gpu_count(resource.gpu_type),
        "gpu_type": resource.gpu_type,
        "topology": thaw(plan.resolved_config)["model"]["topology"],
        "master_shard_index": 0,
        "port_range": {
            "start": GANG_BASE_PORT,
            "count": resource.local_world_size,
        },
    }


def _gang_build_file(resource: AOneResourcePlan) -> str:
    lines = [
        'load("//rtp_llm/test/perf_test:aone_defs.bzl", "perf_test_gang_shard")',
        "",
        "perf_test_gang_shard(",
        '    name = "gang",',
        '    resolved_config = "resolved_config.json",',
        '    resolved_workload = "resolved_workload.json",',
        '    gang_plan = "gang_plan.json",',
        f'    gpu_type = "{resource.gpu_type}",',
        f"    gpus_per_host = {_per_host_gpu_count(resource.gpu_type)},",
        f"    machine_count = {resource.machine_count},",
        ")",
        "",
        "test_suite(",
        '    name = "all",',
        '    tests = [":gang"],',
        '    tags = ["manual"],',
        ")",
        "",
    ]
    return "\n".join(lines)


def generate_gang_package(
    root: Union[str, Path],
    plan: ExecutionPlan,
    *,
    run_id: str,
    package: Optional[str] = None,
) -> GeneratedAOnePackage:
    """Materialize the frozen gang inputs without submit-host source paths."""
    if not isinstance(run_id, str) or not run_id:
        raise PlanningError("AOne run_id must be a non-empty string")
    resource = plan.resource_plan
    if not isinstance(resource, AOneResourcePlan):
        raise PlanningError("gang package requires an AOne resource plan")
    gang_payload = build_gang_plan_payload(plan, run_id)
    return _write_package(
        root,
        plan,
        package=package,
        payload=gang_payload,
        payload_name="gang_plan.json",
        build_text=_gang_build_file(resource),
        target_names=("gang",),
        description="gang",
        omit_null_shard_count=True,
    )
