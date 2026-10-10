"""Plan cache-perf workflows in memory before writing inputs or launching Bazel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from rtp_llm.test.perf_test.cache_grid.config.grid_config import dedupe_cache_grid_cases
from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    is_sensitive_environment_name,
    profile_environment,
)
from rtp_llm.test.perf_test.cache_grid.config.run_request import (
    ResolvedRunRequest,
    resolve_run_request,
    to_prepare_namespace,
)
from rtp_llm.test.perf_test.cache_grid.runner import cache_perf as cli
from rtp_llm.test.perf_test.cache_grid.runner import cache_perf_ci as ci
from rtp_llm.test.perf_test.cache_grid.runner import unified_pipeline

REPO = Path(__file__).resolve().parents[5]
LOCAL_PLATFORM_CONFIGS = {
    "x86_64": "cuda13",
    "aarch64": "cuda13_arm",
}
AONE_RESOURCE_CONFIGS = {
    "daily": ("cicd", "daily_aone_bazel_cache", "daily_aone_bazel_remote"),
    "online": ("cicd", "online_aone_bazel_cache", "online_aone_bazel_remote"),
}
GPU_PLATFORMS = {
    "SM100_ARM_CU13": "cuda13_arm",
    "SM103_CU13": "cuda13",
    "L20_CU13": "cuda13",
    "L20D_DEV": "cuda13",
}


def resolve_gpu_platform(gpu_type: str) -> str:
    try:
        return GPU_PLATFORMS[gpu_type]
    except KeyError as error:
        supported = ", ".join(sorted(GPU_PLATFORMS))
        raise ValueError(
            f"unsupported GPU type {gpu_type!r}; supported GPU types: {supported}"
        ) from error


@dataclass(frozen=True)
class RunPaths:
    result_root: Path
    inputs_dir: Path
    generated_dir: Optional[Path]


@dataclass
class OrchestrationPlan:
    mode: str
    paths: RunPaths
    resolved: ResolvedRunRequest
    request_cases: List[Dict[str, Any]]
    artifacts: Dict[Path, bytes]
    commands: Dict[str, List[str]]
    process_env: Dict[str, str]
    summary: Dict[str, Any]
    ci_plan: Optional[Dict[str, Any]] = None


def _new_or_empty(directory: Path, option: str) -> Path:
    directory = directory.resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f"{option} must be new or empty")
    return directory


def make_run_paths(
    args: argparse.Namespace,
    *,
    repo: Path,
    mode: str,
    run_id: str,
    model_label: str,
) -> RunPaths:
    """Reserve names without creating directories or reparsing the request."""
    label = re.sub(r"[^A-Za-z0-9._-]", "_", model_label)
    repo = repo.resolve()
    result_root = _new_or_empty(
        args.result_root or repo / f"cache_grid_{mode}_results_{label}_{run_id}",
        "--result-root",
    )
    if mode == "local":
        return RunPaths(result_root, result_root, None)
    inputs_dir = _new_or_empty(
        repo / f"cache_grid_{mode}_inputs_{label}_{run_id}", "inputs directory"
    )
    generated_dir = _new_or_empty(
        repo / f"cache_grid_{mode}_generated_{label}_{run_id}",
        "generated directory",
    )
    return RunPaths(result_root, inputs_dir, generated_dir)


def _engine_summary(engine: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy the display view without changing arguments passed to the engine."""
    return {
        **engine,
        "args": {
            name: "<redacted>" if is_sensitive_environment_name(name) else value
            for name, value in engine["args"].items()
        },
    }


def build_local_plan(
    args: argparse.Namespace,
    *,
    repo: Path,
    run_id: str | None = None,
    machine: str | None = None,
    inherited: Mapping[str, str] | None = None,
) -> OrchestrationPlan:
    resolved = resolve_run_request(
        args.model_json,
        args.engine_json,
        args.grid_json,
        args.engine_env_json,
        args.runtime_env_json,
    )
    bazel_config = LOCAL_PLATFORM_CONFIGS.get(machine or platform.machine())
    if bazel_config is None:
        supported = ", ".join(sorted(LOCAL_PLATFORM_CONFIGS))
        raise ValueError(
            f"unsupported local architecture; expected one of: {supported}"
        )
    devices = [value.strip() for value in args.devices.split(",")]
    if any(re.fullmatch(r"[0-9]+", value) is None for value in devices):
        raise ValueError("--devices must be a comma-separated list of GPU indices")
    devices = [str(int(value)) for value in devices]
    if len(devices) != len(set(devices)):
        raise ValueError("--devices entries must be unique")
    if len(devices) != resolved.gpu_count:
        raise ValueError(
            f"topology requires {resolved.gpu_count} devices, got {len(devices)}"
        )
    paths = make_run_paths(
        args,
        repo=repo,
        mode="local",
        run_id=run_id if run_id is not None else uuid.uuid4().hex,
        model_label=resolved.model["label"],
    )
    input_bundle = ci.build_input_artifacts(
        to_prepare_namespace(
            resolved,
            inputs_dir=paths.inputs_dir,
            configs=[bazel_config],
            bazel=args.bazel,
        )
    )
    request_cases = dedupe_cache_grid_cases(
        input_bundle["grid_payload"]["cases"], resolved.grid["cache_alignment"]
    )
    runner_args = [
        f"--profile={paths.result_root / 'profile.snapshot.json'}",
        f"--cache_grid_json={paths.result_root / 'grid.snapshot.json'}",
        f"--result_dir={paths.result_root}",
        "--partial=2",
        "--decode_test_length=1",
        "--batch_size=1",
        "--cache_request_transport=dashsc_input_ids",
        f"--cache_measure_runs={resolved.grid['measure_runs']}",
        f"--cache_commit_tail_tokens={resolved.grid['commit_tail_tokens']}",
    ]
    effective_env = profile_environment(
        {"runtime_env": resolved.runtime_env, "engine_env": resolved.engine_env}
    )
    effective_env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    runner_args.extend(
        f"--engine_env={key}={value}" for key, value in sorted(effective_env.items())
    )
    command = [
        args.bazel,
        "run",
        cli.TARGET,
        f"--config={bazel_config}",
        "--run_under=//rtp_llm/test/utils:gpu_lock",
        *args.bazel_arg,
        "--",
        *runner_args,
    ]
    snapshot_bytes = {
        "profile.snapshot.json": input_bundle["artifacts"][ci.PROFILE_NAME],
        "grid.snapshot.json": input_bundle["artifacts"][ci.GRID_NAME],
    }
    launch = {
        "schema_version": 2,
        "mode": "run",
        "profile_file": "profile.snapshot.json",
        "grid": str(paths.result_root / "grid.snapshot.json"),
        "env": effective_env,
        "runtime_env": resolved.runtime_env,
        "engine_env": resolved.engine_env,
        "runner_args": [
            value for value in runner_args if not value.startswith("--engine_env=")
        ],
        "bazel": {
            "command": "run",
            "configs": [bazel_config],
            "args": list(args.bazel_arg),
            "executable": args.bazel,
        },
        "devices": devices,
        "snapshots": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in snapshot_bytes.items()
        },
    }
    artifacts = {
        paths.result_root / name: content for name, content in snapshot_bytes.items()
    }
    artifacts[paths.result_root / cli.MANIFEST] = cli.encoded(launch)
    process_env = dict(os.environ if inherited is None else inherited)
    process_env.update(effective_env)
    summary = {
        "mode": "local-run",
        "result_root": str(paths.result_root),
        "effective_case_count": len(request_cases),
        "gpu_count": resolved.gpu_count,
        "devices": devices,
        "world_size": resolved.world_size,
        "model": resolved.model,
        "engine": _engine_summary(resolved.engine),
        "grid": resolved.grid,
        "engine_env": resolved.engine_env,
        "runtime_env": resolved.runtime_env,
        "commands": {"run": command},
    }
    return OrchestrationPlan(
        mode="local-run",
        paths=paths,
        resolved=resolved,
        request_cases=request_cases,
        artifacts=artifacts,
        commands={"run": command},
        process_env=process_env,
        summary=summary,
    )


def build_aone_plan(
    args: argparse.Namespace,
    *,
    repo: Path,
    run_id: str | None = None,
    inherited: Mapping[str, str] | None = None,
) -> OrchestrationPlan:
    build_jobs = (
        args.build_jobs if args.build_jobs is not None else max(1, os.cpu_count() or 1)
    )
    if build_jobs <= 0:
        raise ValueError("--build-jobs must be positive")
    if args.max_parallel <= 0:
        raise ValueError("--max-parallel must be positive")
    platform_config = resolve_gpu_platform(args.gpu_type)
    try:
        resource_configs = AONE_RESOURCE_CONFIGS[args.resource]
    except KeyError as error:
        supported = ", ".join(sorted(AONE_RESOURCE_CONFIGS))
        raise ValueError(
            f"unsupported resource {args.resource!r}; supported resources: {supported}"
        ) from error
    configs = (platform_config, *resource_configs)
    resolved = resolve_run_request(
        args.model_json,
        args.engine_json,
        args.grid_json,
        args.engine_env_json,
        args.runtime_env_json,
    )
    paths = make_run_paths(
        args,
        repo=repo,
        mode="aone",
        run_id=run_id if run_id is not None else uuid.uuid4().hex,
        model_label=resolved.model["label"],
    )
    assert paths.generated_dir is not None
    input_bundle = ci.build_input_artifacts(
        to_prepare_namespace(
            resolved,
            inputs_dir=paths.inputs_dir,
            configs=configs,
            bazel=args.bazel,
        )
    )
    ci_plan = ci.build_plan_artifacts(
        input_bundle["profile_payload"],
        input_bundle["grid_payload"],
        grid_bytes=input_bundle["artifacts"][ci.GRID_NAME],
        output_dir=paths.generated_dir,
        package=paths.generated_dir.relative_to(repo.resolve()).as_posix(),
        gpu_type=args.gpu_type,
        gpu_count=resolved.gpu_count,
        shard_count=None,
        measure_runs=resolved.grid["measure_runs"],
        profile_source=str(paths.inputs_dir / ci.PROFILE_NAME),
        grid_source=str(paths.inputs_dir / ci.GRID_NAME),
    )
    request_cases = dedupe_cache_grid_cases(
        input_bundle["grid_payload"]["cases"], resolved.grid["cache_alignment"]
    )
    artifacts = {
        paths.inputs_dir / name: content
        for name, content in input_bundle["artifacts"].items()
    }
    artifacts.update(
        {
            paths.generated_dir / name: content
            for name, content in ci_plan["artifacts"].items()
        }
    )
    common = [f"--config={value}" for value in configs]
    build_command = [
        args.bazel,
        "build",
        ci_plan["suite_target"],
        *common,
        f"--jobs={build_jobs}",
        "--remote_download_outputs=minimal",
        "--keep_going",
        *args.bazel_arg,
    ]
    test_command = [
        args.bazel,
        "test",
        ci_plan["suite_target"],
        *common,
        "--strategy=TestRunner=remote",
        f"--jobs={args.max_parallel}",
        "--test_timeout=7000",
        "--test_output=errors",
        "--keep_going",
        "--remote_download_outputs=all",
        "--nocache_test_results",
        "--run_under=//rtp_llm/test/utils:gpu_lock",
        *args.bazel_arg,
    ]
    commands = {"build": build_command, "test": test_command}
    process_env = dict(os.environ if inherited is None else inherited)
    process_env.update(resolved.runtime_env)
    summary = {
        "mode": "aone-run",
        "result_root": str(paths.result_root),
        "inputs_dir": str(paths.inputs_dir),
        "generated_dir": str(paths.generated_dir),
        "effective_case_count": len(request_cases),
        "gpu_type": args.gpu_type,
        "gpu_count": resolved.gpu_count,
        "resource": args.resource,
        "world_size": resolved.world_size,
        "model": resolved.model,
        "engine": _engine_summary(resolved.engine),
        "grid": resolved.grid,
        "shard_count": ci_plan["shard_count"],
        "build_jobs": build_jobs,
        "max_parallel": args.max_parallel,
        "engine_env": resolved.engine_env,
        "runtime_env": resolved.runtime_env,
        "commands": commands,
    }
    return OrchestrationPlan(
        mode="aone-run",
        paths=paths,
        resolved=resolved,
        request_cases=request_cases,
        artifacts=artifacts,
        commands=commands,
        process_env=process_env,
        summary=summary,
        ci_plan=ci_plan,
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Run complete cache perf workflows")
    modes = root.add_subparsers(dest="mode", required=True)
    local = modes.add_parser("local-run", help="run on explicitly selected local GPUs")
    aone = modes.add_parser("aone-run", help="build, test and collect AOne shards")
    for command in (local, aone):
        command.add_argument("--model-json", required=True)
        command.add_argument("--engine-json", required=True)
        command.add_argument("--grid-json", required=True)
        command.add_argument("--engine-env-json", default="{}")
        command.add_argument("--runtime-env-json", default="{}")
        command.add_argument("--result-root", type=Path)
        command.add_argument("--bazel", default="bazelisk")
        command.add_argument(
            "--bazel-arg",
            action="append",
            default=[],
            help="repeatable Bazel override; use --bazel-arg=--flag=value",
        )
        command.add_argument("--dry-run", action="store_true")
    local.add_argument("--devices", required=True)
    aone.add_argument(
        "--resource", required=True, choices=sorted(AONE_RESOURCE_CONFIGS)
    )
    aone.add_argument("--gpu-type", required=True)
    aone.add_argument("--build-jobs", type=int)
    aone.add_argument("--max-parallel", type=int, default=1)
    return root


def main(argv: Optional[Sequence[str]] = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        plan = (
            build_local_plan(args, repo=REPO)
            if args.mode == "local-run"
            else build_aone_plan(args, repo=REPO)
        )
    except ValueError as error:
        argument_parser.error(str(error))
    if args.dry_run:
        print(json.dumps(plan.summary, ensure_ascii=False, indent=2, sort_keys=True))
        for stage, command in plan.commands.items():
            print(f"{stage}: {shlex.join(command)}")
        return 0
    if args.mode == "local-run":
        return execute_local_plan(plan)
    return execute_aone_plan(plan)


def write_artifacts(artifacts: Mapping[Path, bytes]) -> None:
    """Preflight every destination, then exclusively create frozen inputs."""
    for path in artifacts:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"artifact already exists: {path}")
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(content)


def postprocess_namespace(
    result_dir: Path, *, partial: bool, batch_size: int = 1
) -> argparse.Namespace:
    return argparse.Namespace(
        mode="pipeline",
        profile=None,
        grid=None,
        result_dir=result_dir,
        cases=None,
        runs=None,
        env=[],
        config=None,
        output_base=None,
        bazel=None,
        trace_timeout=180,
        profile_backend="kineto",
        nsys_path="nsys",
        nsys_session=None,
        nsys_tail_seconds=0.1,
        skip_reuse_validation=False,
        dry_run=False,
        test_mode="run",
        skip_test=True,
        batch_size=batch_size,
        grid_dir=None,
        result_root=None,
        skip_fit=partial,
        partial=partial,
        cards=None,
        estimator="median",
        formula_output_dir=None,
        svg_output=None,
        cold_svg_output=None,
        html_output=None,
        oss_destination=None,
    )


def execute_local_plan(plan: OrchestrationPlan) -> int:
    write_artifacts(plan.artifacts)
    completed = subprocess.run(
        plan.commands["run"], cwd=REPO, env=plan.process_env, check=False
    )
    if completed.returncode:
        return completed.returncode
    return unified_pipeline.run(
        postprocess_namespace(
            result_dir=plan.paths.result_root,
            partial=False,
            batch_size=plan.resolved.grid["batch_size"],
        )
    )


def collect_namespace(plan: OrchestrationPlan, *, partial: bool) -> argparse.Namespace:
    assert plan.paths.generated_dir is not None
    return argparse.Namespace(
        plan=plan.paths.generated_dir / ci.PLAN_NAME,
        bazel_testlogs=None,
        result_root=plan.paths.result_root,
        partial=partial,
        batch_size=plan.resolved.grid["batch_size"],
        cards=plan.resolved.gpu_count,
        estimator="median",
        skip_fit=partial,
        formula_output_dir=None,
        svg_output=None,
        cold_svg_output=None,
        html_output=None,
        oss_destination=None,
    )


def write_run_state(path: Path, state: Dict[str, Any]) -> None:
    """Publish a complete stage manifest with an atomic same-directory rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _launch_stage(plan: OrchestrationPlan, stage: str) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            plan.commands[stage], cwd=REPO, env=plan.process_env, check=False
        )
    except KeyboardInterrupt as error:
        return {"returncode": 130, "status": "interrupted", "error": repr(error)}
    except Exception as error:
        return {"returncode": 1, "status": "failed", "error": repr(error)}
    return {"returncode": completed.returncode}


def execute_aone_plan(plan: OrchestrationPlan) -> int:
    """Collect available test outputs while preserving the earliest stage failure."""
    state = {
        "schema_version": 1,
        "mode": "aone-run",
        "status": "running",
        "paths": {
            "inputs": str(plan.paths.inputs_dir),
            "generated": str(plan.paths.generated_dir),
            "results": str(plan.paths.result_root),
        },
        "commands": plan.commands,
        "stages": {},
        "summary": plan.summary,
    }
    state_path = plan.paths.result_root / "cache_perf_run.json"
    write_artifacts(plan.artifacts)

    build = _launch_stage(plan, "build")
    state["stages"]["build"] = build
    if build["returncode"]:
        state["status"] = build.get("status", "failed")
        write_run_state(state_path, state)
        return build["returncode"]

    test = _launch_stage(plan, "test")
    state["stages"]["test"] = test
    collect_args = collect_namespace(plan, partial=test["returncode"] != 0)
    collect_code = 0
    pipeline_code = 0
    try:
        collect_manifest, collect_code = ci.collect_results(collect_args, repo=REPO)
        state["stages"]["collect"] = {
            "returncode": collect_code,
            "missing_shards": collect_manifest["missing_shards"],
        }
    except KeyboardInterrupt as error:
        collect_code = 130
        state["stages"]["collect"] = {
            "returncode": 130,
            "status": "interrupted",
            "error": repr(error),
        }
    except Exception as error:
        collect_code = 1
        state["stages"]["collect"] = {"returncode": 1, "error": repr(error)}
    if collect_code == 0:
        try:
            pipeline_code = unified_pipeline.run(ci.pipeline_namespace(collect_args))
            state["stages"]["pipeline"] = {"returncode": pipeline_code}
        except KeyboardInterrupt as error:
            pipeline_code = 130
            state["stages"]["pipeline"] = {
                "returncode": 130,
                "status": "interrupted",
                "error": repr(error),
            }
        except Exception as error:
            pipeline_code = 1
            state["stages"]["pipeline"] = {"returncode": 1, "error": repr(error)}
    result = test["returncode"] or collect_code or pipeline_code
    state["status"] = "complete"
    for stage in ("test", "collect", "pipeline"):
        outcome = state["stages"].get(stage, {})
        if outcome.get("returncode"):
            state["status"] = outcome.get("status", "failed")
            break
    write_run_state(state_path, state)
    return result
