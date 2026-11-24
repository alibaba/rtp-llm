"""Unified perf-test command entry: strict input, planning, and dispatch."""

from __future__ import annotations

import json
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
from rtp_llm.test.perf_test.config_resolver import resolve_config
from rtp_llm.test.perf_test.decode.planner import validate_decode_frontend_lengths
from rtp_llm.test.perf_test.errors import (
    CaseExecutionError,
    ConfigurationError,
    PerfTestError,
    PlanningError,
)
from rtp_llm.test.perf_test.launch_config import prepare_config
from rtp_llm.test.perf_test.plan import (
    AOneResourcePlan,
    ExecutionPlan,
    LocalResourcePlan,
    SshResourcePlan,
    build_execution_plan,
)
from rtp_llm.test.perf_test.prefill.planner import require_concurrent_prefill_plan
from rtp_llm.test.perf_test.run_context import read_run_context
from rtp_llm.test.perf_test.run_directory import (
    initialize_run_directory,
    inspect_run_directory,
)

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
    """Render local execution; other backends provide placement renderers."""
    if plan.backend != "local":
        raise PlanningError("non-local preview requires an explicit renderer")
    from rtp_llm.test.perf_test.execution.local_bazel import local_preview

    return local_preview(plan)


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


def infer_mode(source: Mapping[str, Any], assertion: Optional[str] = None) -> str:
    workload = source.get("workload")
    if not isinstance(workload, Mapping):
        raise ConfigurationError("workload: expected an object")
    kind = workload.get("kind")
    mode = {"cache_grid": "prefill", "grid": "decode", "distribution": "decode"}.get(
        kind
    )
    if mode is None:
        raise ConfigurationError(f"unsupported workload.kind: {kind!r}")
    if assertion is not None and assertion != mode:
        raise ConfigurationError(
            f"--mode={assertion} conflicts with workload.kind={kind!r}"
        )
    return mode


def plan_from_source(
    source: Mapping[str, Any],
    *,
    source_dir: Path,
    result_directory: Optional[str] = None,
    mode: Optional[str] = None,
    action: str = "run",
    external_inputs: Optional[dict] = None,
) -> tuple[ExecutionPlan, Dict[str, Any]]:
    """Plan in memory once; callers may freeze every experiment task before run."""
    from rtp_llm.test.perf_test.launch_config import prepare_config

    source = prepare_config(source)
    if source["execution"].get("backend") == "local":
        from rtp_llm.test.perf_test.launch_config import anchor_local_model_paths

        source = anchor_local_model_paths(source)
    mode = infer_mode(source, mode)
    execution = source.get("execution")
    if not isinstance(execution, Mapping):
        raise ConfigurationError("execution: expected an object")
    backend = execution.get("backend")
    parser = None
    planner = None
    validator = validate_capability
    if backend == "aone":
        from rtp_llm.test.perf_test.aone.config_schema import parse_execution
        from rtp_llm.test.perf_test.aone.plan import plan_resources

        parser, planner, validator = (
            parse_execution,
            plan_resources,
            validate_aone_capability,
        )
    elif backend == "ssh":
        from rtp_llm.test.perf_test.ssh.config_schema import parse_execution
        from rtp_llm.test.perf_test.ssh.plan import plan_resources

        parser, planner = parse_execution, plan_resources
    resolved = resolve_config(
        source,
        source_dir=source_dir,
        execution_parser=parser,
        external_inputs=external_inputs,
    )
    cases = _planned_cases(mode, source, resolved, execution_parser=parser)
    root = Path(resolved.resolved_config["output"]["result_root"]).resolve()
    plan = build_execution_plan(
        mode=mode,
        action=action,
        resolved=resolved,
        cases=cases,
        result_directory=result_directory or str(root / _run_id(mode)),
        resource_planner=planner,
        capability_validator=validator,
    )
    if mode == "decode":
        topology = resolved.resolved_config["model"]["topology"]
        local_dp = (
            topology["dp"]
            if isinstance(plan.resource_plan, LocalResourcePlan)
            else max(1, plan.resource_plan.local_world_size // topology["tp"])
        )
        validate_decode_frontend_lengths(
            cases, dp_size=topology["dp"], local_dp_size=local_dp
        )
    return plan, dict(source)


def _new_plan(request: CliRequest) -> tuple[ExecutionPlan, Dict[str, Any]]:
    assert request.config is not None
    return plan_from_source(
        _source_config(request.config),
        source_dir=Path(request.config).resolve().parent,
        mode=request.mode,
        action=request.action,
    )


def _persisted_plan(request: CliRequest) -> tuple[ExecutionPlan, Dict[str, Any]]:
    assert request.run_dir is not None
    snapshot = inspect_run_directory(request.run_dir)
    plan = snapshot.execution_plan
    if request.mode is not None and request.mode != plan.mode:
        raise ConfigurationError(
            f"--mode={request.mode} does not match persisted mode {plan.mode}"
        )
    if request.skip_test and plan.mode != "prefill":
        raise ConfigurationError("--skip-test requires a prefill run")
    validator = (
        validate_aone_capability if plan.backend == "aone" else validate_capability
    )
    validator(plan.mode, request.action, plan.backend)
    if (
        plan.mode == "prefill"
        and request.action in ("resume", "profile")
        and not plan.resolved_config["workload"].get("pressure")
    ):
        require_concurrent_prefill_plan(plan)
    if request.cases and plan.resolved_config["workload"].get("pressure"):
        raise ConfigurationError(
            "continuous pressure profile/resume requires the whole grid"
        )
    if (
        request.action == "profile"
        and plan.resolved_config["workload"].get("pressure")
        and plan.resolved_config["workload"]["profiling"]["backend"] == "nsys"
        and not plan.resolved_config["workload"]["pressure"]["nsys_duration_s"]
    ):
        raise ConfigurationError("Nsight pressure capture requires nsys_duration_s")
    if request.cases:
        unknown = sorted(set(request.cases) - {item.case_id for item in plan.cases})
        if unknown:
            raise ConfigurationError(f"unknown --cases value: {unknown[0]}")
    return replace(plan, action=request.action), snapshot.source_config


def build_request_plan(request: CliRequest) -> tuple[ExecutionPlan, Dict[str, Any]]:
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

        return {
            "aone": lambda plan: create_default_aone_executor(
                plan,
                startup_args=request.bazel_startup_args if request else (),
                common_args=request.bazel_args if request else (),
                broker_host=request.broker_host if request else None,
            )
        }
    if backend == "ssh":
        from rtp_llm.test.perf_test.execution.ssh import create_default_ssh_executor

        return {"ssh": lambda plan: create_default_ssh_executor(plan)(plan)}
    from rtp_llm.test.perf_test.execution.local import create_default_local_executor

    selected = () if request is None else request.cases
    return {
        "local": create_default_local_executor(selected_case_ids=selected),
    }


def execute_plan(
    plan: ExecutionPlan,
    source: Mapping[str, Any],
    request: Optional[CliRequest] = None,
    *,
    executor_factory: ExecutorFactory = _default_executors,
) -> Any:
    """Execute an already frozen plan; never re-resolve config or workload."""
    if (
        plan.backend not in ("local", "aone")
        and request is not None
        and (request.bazel_startup_args or request.bazel_args or request.broker_host)
    ):
        raise ConfigurationError(
            "Bazel options and --broker-host require the local or AOne backend"
        )
    if plan.backend == "local" and request is not None and request.broker_host:
        raise ConfigurationError("--broker-host requires the AOne backend")
    if request is not None and plan.backend in ("local", "aone"):
        from rtp_llm.test.perf_test.launch_config import validate_bazel_arguments

        validate_bazel_arguments(request.bazel_args, local=plan.backend == "local")
    if request is not None and request.dry_run:
        renderer = aone_preview if plan.backend == "aone" else None
        if plan.backend == "local":
            from rtp_llm.test.perf_test.execution.local_bazel import local_preview

            renderer = lambda active: local_preview(active, request)
        if plan.backend == "ssh":
            from rtp_llm.test.perf_test.ssh.plan import ssh_preview

            renderer = ssh_preview
        preview = create_preview(plan, preview_renderer=renderer)
        if request.cases:
            preview["selected_case_ids"] = list(request.cases)
        if plan.backend == "aone":
            for command in preview["commands"]:
                command["argv"][1:1] = request.bazel_startup_args
                command["argv"].extend(request.bazel_args)
                if command["stage"] == "remote-test":
                    command["argv"].append(
                        f"--test_timeout={plan.resolved_config['execution']['test_timeout_seconds']}"
                    )
            if request.broker_host and "gang" in preview:
                preview["gang"]["broker_host"] = request.broker_host
        return preview
    if (
        plan.backend == "local"
        and plan.action != "report"
        and executor_factory is _default_executors
    ):
        from rtp_llm.test.perf_test.execution.local_bazel import execute_local_bazel

        return execute_local_bazel(plan, source, request)
    if plan.action == "run":
        run_context = read_run_context(request.run_context if request else None)
        revisions = run_context.get("repository_revisions", {})
        initialize_run_directory(
            plan.result_directory,
            source_config=source,
            plan=plan,
            run_id=Path(plan.result_directory).name,
            code_version=revisions.get("public", "unknown"),
            repository_revisions=revisions,
        )
        if request is not None and plan.backend in ("local", "aone"):
            from rtp_llm.test.perf_test.storage import atomic_write_json

            root = Path(plan.result_directory)
            atomic_write_json(
                root / "input_config.json", request.input_config or dict(source)
            )
            atomic_write_json(
                root / "launch_request.json",
                {
                    "overrides": list(request.overrides),
                    "bazel_args": list(request.bazel_args),
                    "bazel_startup_args": list(request.bazel_startup_args),
                },
            )
    executors = (
        _default_executors(request, backend=plan.backend)
        if executor_factory is _default_executors
        else executor_factory()
    )
    executor = executors.get(plan.backend)
    if executor is None:
        raise PlanningError(f"no executor registered for backend {plan.backend!r}")
    return executor(plan)


def run_request(
    request: CliRequest, *, executor_factory: ExecutorFactory = _default_executors
) -> Any:
    if request.maintenance is not None:
        from rtp_llm.test.perf_test.execution.ssh import maintain

        return maintain(request.run_dir, request.maintenance, dry_run=request.dry_run)
    if request.config:
        source = _source_config(request.config)
        request = replace(request, input_config=source)
        if "experiment_schema_version" in source:
            if (
                request.mode
                or request.action != "run"
                or request.cases
                or request.broker_host
            ):
                raise ConfigurationError(
                    "experiment accepts --config, --dry-run and --run-context only"
                )
            from rtp_llm.test.perf_test.experiment import run_experiment

            return run_experiment(
                request.config,
                dry_run=request.dry_run,
                run_context=request.run_context,
                overrides=request.overrides,
                bazel_args=request.bazel_args,
                bazel_startup_args=request.bazel_startup_args,
            )
        plan, source = plan_from_source(
            prepare_config(source, request.overrides),
            source_dir=Path(request.config).resolve().parent,
            mode=request.mode,
            action=request.action,
        )
    else:
        plan, source = build_request_plan(request)
    return execute_plan(plan, source, request, executor_factory=executor_factory)


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
        exit_code = (
            result.get("exit_code", 0)
            if isinstance(result, Mapping)
            else getattr(result, "exit_code", 0)
        )
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
    from rtp_llm.test.perf_test.aone.plan import aone_exec_properties

    preview["exec_properties"] = aone_exec_properties(
        resource, plan.resolved_config["execution"].get("exec_properties", {})
    )
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
