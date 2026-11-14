"""Unified perf-test command entry: strict input, planning, and dispatch."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TextIO

from rtp_llm.test.perf_test.capabilities import (
    validate_aone_capability,
    validate_capability,
)
from rtp_llm.test.perf_test.case_planning import plan_cases as _planned_cases
from rtp_llm.test.perf_test.cli import CliRequest, parse_cli
from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.decode.planner import validate_decode_frontend_lengths
from rtp_llm.test.perf_test.errors import (
    CaseExecutionError,
    ConfigurationError,
    PerfTestError,
    PlanningError,
)
from rtp_llm.test.perf_test.plan import (
    AOneResourcePlan,
    ExecutionPlan,
    LocalResourcePlan,
    build_execution_plan,
)
from rtp_llm.test.perf_test.prefill.planner import require_concurrent_prefill_plan
from rtp_llm.test.perf_test.run_directory import (
    initialize_run_directory,
    inspect_run_directory,
)


class Orchestrator:
    def __init__(self, executors: Mapping[str, Callable[[ExecutionPlan], Any]]):
        self._executors = dict(executors)

    def execute(self, plan: ExecutionPlan) -> Any:
        executor = self._executors.get(plan.backend)
        if executor is None:
            raise PlanningError(f"no executor registered for backend {plan.backend!r}")
        return executor(plan)


# Renderers return backend-specific commands and optional placement details.
# Shared identity/configuration fields cannot be replaced by an adapter.
PreviewRenderer = Callable[[ExecutionPlan], Dict[str, Any]]


def create_preview(
    plan: ExecutionPlan,
    *,
    preview_renderer: Optional[PreviewRenderer] = None,
) -> Dict[str, Any]:
    """Render all resolved plan inputs without executing or writing them."""
    plan_dict = plan.to_dict()
    preview = {
        "schema_version": 1,
        "dry_run": True,
        "mode": plan.mode,
        "action": plan.action,
        "backend": plan.backend,
        "result_directory": plan.result_directory,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "normalized_config": plan_dict["resolved_config"],
        "cases": {
            "count": len(plan.cases),
            "case_ids": [case.case_id for case in plan.cases],
            "items": [case.to_dict() for case in plan.cases],
        },
        "resource_summary": plan.resource_plan.to_dict(),
    }
    details = (preview_renderer or backend_preview)(plan)
    if "commands" not in details or set(details).intersection(preview):
        raise PlanningError(
            "preview renderer must supply commands and cannot replace shared fields"
        )
    preview.update(details)
    return preview


def backend_preview(plan: ExecutionPlan) -> Dict[str, Any]:
    """Render local execution; historical remote plans remain read-only."""
    if plan.backend != "local":
        raise PlanningError("non-local preview requires an explicit renderer")
    plan_path = str(PurePath(plan.result_directory) / "execution_plan.json")
    return {
        "commands": [
            {"stage": "local-run", "argv": ["perf_test_worker", "--plan", plan_path]}
        ]
    }


def dispatch_plan(
    plan: ExecutionPlan,
    *,
    dry_run: bool,
    execute: Callable[[ExecutionPlan], Any],
    preview_renderer: Optional[PreviewRenderer] = None,
) -> Any:
    """Keep dry-run on a branch that cannot invoke the execution callback."""
    if dry_run:
        return create_preview(plan, preview_renderer=preview_renderer)
    return execute(plan)


ExecutorFactory = Callable[[], Mapping[str, Callable[[ExecutionPlan], Any]]]


def _run_id(mode: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{mode}-{now}"


def _source_config(path: str) -> Dict[str, Any]:
    source_path = Path(path)
    try:
        value = json.loads(source_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as error:
        raise ConfigurationError(f"config {source_path} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"config {source_path} is not valid JSON: {error.msg}"
        ) from error
    except OSError as error:
        raise ConfigurationError(
            f"cannot read config {source_path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ConfigurationError(f"config {source_path} must contain an object")
    return value


def _new_plan(request: CliRequest) -> tuple[ExecutionPlan, Dict[str, Any]]:
    assert request.config is not None
    source = _source_config(request.config)
    raw_execution = source.get("execution")
    is_aone = (
        isinstance(raw_execution, Mapping) and raw_execution.get("backend") == "aone"
    )
    parser = None
    planner = None
    validator = validate_capability
    if is_aone:
        from rtp_llm.test.perf_test.aone.config_schema import parse_execution
        from rtp_llm.test.perf_test.aone.plan import plan_resources

        parser = parse_execution
        planner = plan_resources
        validator = validate_aone_capability
    resolved = load_resolved_config(request.config, execution_parser=parser)
    cases = _planned_cases(request.mode, source, resolved, execution_parser=parser)
    root = Path(resolved.resolved_config["output"]["result_root"]).resolve()
    result_directory = str(root / _run_id(request.mode))
    plan = build_execution_plan(
        mode=request.mode,
        action=request.action,
        resolved=resolved,
        cases=cases,
        result_directory=result_directory,
        resource_planner=planner,
        capability_validator=validator,
    )
    if request.mode == "decode":
        topology = resolved.resolved_config["model"]["topology"]
        if isinstance(plan.resource_plan, LocalResourcePlan):
            local_dp_size = topology["dp"]
        elif isinstance(plan.resource_plan, AOneResourcePlan):
            local_dp_size = plan.resource_plan.local_world_size // topology["tp"]
        else:
            raise PlanningError("decode planning requires a known frontend layout")
        validate_decode_frontend_lengths(
            cases, dp_size=topology["dp"], local_dp_size=local_dp_size
        )
    return plan, source


def _persisted_plan(request: CliRequest) -> tuple[ExecutionPlan, Dict[str, Any]]:
    assert request.run_dir is not None
    snapshot = inspect_run_directory(request.run_dir)
    plan = snapshot.execution_plan
    if request.mode != plan.mode:
        raise ConfigurationError(
            f"--mode={request.mode} does not match persisted mode {plan.mode}"
        )
    validator = (
        validate_aone_capability if plan.backend == "aone" else validate_capability
    )
    validator(request.mode, request.action, plan.backend)
    if request.mode == "prefill" and request.action in ("resume", "profile"):
        require_concurrent_prefill_plan(plan)
    if request.cases:
        known = {item.case_id for item in plan.cases}
        unknown = sorted(set(request.cases) - known)
        if unknown:
            raise ConfigurationError(f"unknown --cases value: {unknown[0]}")
    return replace(plan, action=request.action), snapshot.source_config


def build_request_plan(request: CliRequest) -> tuple[ExecutionPlan, Dict[str, Any]]:
    """Resolve a CLI request without mutating its result directory."""
    return _new_plan(request) if request.action == "run" else _persisted_plan(request)


def _default_executors(
    request: Optional[CliRequest] = None,
    *,
    backend: str = "local",
) -> Mapping[str, Callable[[ExecutionPlan], Any]]:
    if backend == "aone":
        from rtp_llm.test.perf_test.execution.aone_bazel import (
            create_default_aone_executor,
        )

        return {"aone": create_default_aone_executor}
    from rtp_llm.test.perf_test.execution.local import create_default_local_executor

    selected = () if request is None else request.cases
    return {
        "local": create_default_local_executor(selected_case_ids=selected),
    }


def run_request(
    request: CliRequest,
    *,
    executor_factory: ExecutorFactory = _default_executors,
) -> Any:
    """Plan first; keep dry-run outside every filesystem/execution side effect."""
    plan, source = build_request_plan(request)
    if request.dry_run:
        preview = create_preview(
            plan, preview_renderer=aone_preview if plan.backend == "aone" else None
        )
        if request.cases:
            preview["selected_case_ids"] = list(request.cases)
        return preview
    if request.action == "run":
        initialize_run_directory(
            plan.result_directory,
            source_config=source,
            plan=plan,
            run_id=Path(plan.result_directory).name,
            code_version=os.environ.get("BUILD_SCM_REVISION", "unknown"),
            repository_revisions=json.loads(
                os.environ.get("PERF_TEST_REPOSITORY_REVISIONS", "{}")
            ),
        )
    executors = (
        _default_executors(request, backend=plan.backend)
        if executor_factory is _default_executors
        else executor_factory()
    )
    orchestrator = Orchestrator(executors)
    return orchestrator.execute(plan)


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
    executor_factory: ExecutorFactory = _default_executors,
) -> int:
    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    try:
        request = parse_cli(sys.argv[1:] if argv is None else argv)
        result = run_request(request, executor_factory=executor_factory)
        if isinstance(result, Mapping):
            output.write(json.dumps(result, sort_keys=True, indent=2) + "\n")
            if request.action == "profile" and result.get("status") == "failed":
                return CaseExecutionError.exit_code
        exit_code = getattr(result, "exit_code", 0)
        return int(exit_code)
    except PerfTestError as error:
        errors.write(f"perf_test: {error}\n")
        return error.exit_code
    except (ValueError, OSError) as error:
        errors.write(f"perf_test: {error}\n")
        return 1


GANG_TARGET = "//generated/perf_test:gang"


def _is_gang_resource(resource: AOneResourcePlan) -> bool:
    return resource.machine_count > 1


def _gang_member_layout(resource: AOneResourcePlan) -> List[Dict[str, Any]]:
    return [
        {
            "shard_index": shard_index,
            "world_ranks": [
                shard_index * resource.local_world_size + offset
                for offset in range(resource.local_world_size)
            ],
            "is_master": shard_index == 0,
        }
        for shard_index in range(resource.machine_count)
    ]


def _command_preview(plan: ExecutionPlan) -> List[Dict[str, Any]]:
    from rtp_llm.test.perf_test.aone.plan import AONE_GPU_PLATFORMS

    if plan.action == "report":
        return []
    resource = plan.resource_plan
    if isinstance(resource, AOneResourcePlan):
        if _is_gang_resource(resource):
            targets = [GANG_TARGET]
            # This permits concurrent scheduling of the members; it does not
            # provide atomic reservation of all gang machines.
            jobs = max(resource.max_parallel, resource.machine_count)
        else:
            targets = [
                f"//generated/perf_test:shard_{shard:04d}"
                for shard in range(resource.shard_count)
            ]
            jobs = resource.max_parallel
        common = [f"--config={AONE_GPU_PLATFORMS[resource.gpu_type]}", "--config=cicd"]
        return [
            {
                "stage": "build",
                "argv": [
                    "bazelisk",
                    "build",
                    *targets,
                    *common,
                    f"--config={resource.resource}_aone_bazel_cache",
                ],
            },
            {
                "stage": "remote-test",
                "argv": [
                    "bazelisk",
                    "test",
                    *targets,
                    *common,
                    f"--config={resource.resource}_aone_bazel_remote",
                    "--strategy=TestRunner=remote",
                    f"--jobs={jobs}",
                ],
            },
        ]
    raise PlanningError("AOne preview requires an AOne resource plan")


def aone_preview(plan: ExecutionPlan) -> Dict[str, Any]:
    """Render AOne commands and placement using shared immutable plan data."""
    from rtp_llm.test.perf_test.aone.sharding import plan_shards

    preview = {"commands": _command_preview(plan)}
    resource = plan.resource_plan
    if isinstance(resource, AOneResourcePlan) and _is_gang_resource(resource):
        # A gang plan has no case shards: every machine hosts the full plan's
        # rank group, so the preview prints the gang topology instead.
        preview["gang"] = {
            "machine_count": resource.machine_count,
            "gpus_per_machine": resource.gpus_per_machine,
            "local_world_size": resource.local_world_size,
            "world_size": resource.machine_count * resource.local_world_size,
            "master_shard_index": 0,
            "target": GANG_TARGET,
            "members": _gang_member_layout(resource),
        }
    elif isinstance(resource, AOneResourcePlan) and plan.action == "run":
        shard_plan = plan_shards(plan)
        preview["shards"] = {
            "count": len(shard_plan.shards),
            "total_estimated_cost": shard_plan.total_estimated_cost,
            "items": [
                {
                    **shard.to_dict(),
                    "target": f"//generated/perf_test:shard_{index:04d}",
                }
                for index, shard in enumerate(shard_plan.shards)
            ],
        }
    return preview
