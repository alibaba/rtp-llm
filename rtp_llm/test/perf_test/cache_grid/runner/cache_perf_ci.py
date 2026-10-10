"""AOne/Bazel sharding and collection for long cache-grid runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shlex
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    fingerprint,
    load_profile,
    profile_environment,
    validate_profile,
)
from rtp_llm.test.perf_test.cache_grid.config.topology import profile_cards
from rtp_llm.test.perf_test.cache_grid.runner.cache_grid_runner import is_grouped_case
from rtp_llm.test.perf_test.cache_grid.runner.generate_cache_grid import build_grid

SCHEMA_VERSION = 1
AONE_MAX_ACTION_SECONDS = 7200
CI_TEST_TIMEOUT_SECONDS = 7000
PLAN_NAME = "cache_perf_ci_plan.json"
BUILD_NAME = "BUILD.bazel"
ENV_NAME = "environment.sh"
PROFILE_NAME = "profile.json"
GRID_NAME = "grid.json"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _encoded(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _new_top_level_dir(directory: Path, repo: Path, option: str) -> Path:
    resolved_repo = repo.resolve()
    resolved = directory.resolve()
    try:
        relative = resolved.relative_to(resolved_repo)
    except ValueError as error:
        raise ValueError(f"{option} must be inside the Bazel workspace") from error
    if len(relative.parts) != 1 or relative.parts[0] in ("", ".", ".."):
        raise ValueError(f"{option} must be a top-level directory in the workspace")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise ValueError(f"{option} must be new or empty")
    return resolved


def _scalar(value: str) -> Any:
    if re.fullmatch(r"[+-]?[0-9]+", value):
        return int(value)
    if re.fullmatch(
        r"[+-]?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
        value,
    ):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("setting values must be finite")
        return parsed
    return value


def _settings(
    values: Sequence[str],
    *,
    option: str,
    initial: Dict[str, Any] | None = None,
    strip_dashes: bool = False,
) -> Dict[str, Any]:
    result = dict(initial or {})
    for item in values:
        name, separator, value = item.partition("=")
        if strip_dashes:
            name = name.lstrip("-")
        if not separator or not _NAME.fullmatch(name):
            raise ValueError(f"{option} requires a valid KEY=VALUE")
        if name in result:
            raise ValueError(f"{option} repeats or overrides {name}")
        result[name] = _scalar(value)
    return result


def create_inputs(args: argparse.Namespace, *, repo: Path) -> Dict[str, Any]:
    """Generate validated profile/grid inputs from CI-friendly scalar options."""
    output_dir = _new_top_level_dir(args.inputs_dir, repo, "--inputs-dir")
    tokenizer_path = args.tokenizer_path or args.checkpoint_path
    if not args.model_label or not args.model_type or not args.checkpoint_path:
        raise ValueError(
            "model label, model type and checkpoint path must be non-empty"
        )
    sizes = {
        "tp_size": args.tp_size,
        "dp_size": args.dp_size,
        "world_size": args.world_size,
    }
    if args.ep_size is not None:
        sizes["ep_size"] = args.ep_size
    if args.pp_size is not None:
        sizes["pp_size"] = args.pp_size
    if any(value <= 0 for value in sizes.values()):
        raise ValueError("topology sizes must be positive")
    engine = _settings(
        args.engine_setting,
        option="--engine-setting",
        initial={
            "model_type": args.model_type,
            "checkpoint_path": args.checkpoint_path,
            "tokenizer_path": tokenizer_path,
            **sizes,
        },
    )
    engine_args = _settings(
        args.engine_arg,
        option="--engine-arg",
        strip_dashes=True,
    )
    engine_env = _settings(
        args.engine_env_setting,
        option="--engine-env-setting",
    )
    runtime_env = _settings(
        args.runtime_env_setting,
        option="--runtime-env-setting",
    )
    configs = list(args.config or ["cuda13_arm"])
    if not configs or any(not value for value in configs):
        raise ValueError("--config values must be non-empty")
    measure_runs = args.runs if args.runs is not None else 3
    profile = {
        "schema_version": 1,
        "model_label": args.model_label,
        "engine": engine,
        "engine_args": engine_args,
        "cache_grid": {
            "cache_alignment": args.cache_alignment,
            "commit_tail_tokens": args.commit_tail_tokens,
            "measure_runs": measure_runs,
        },
        "engine_env": engine_env,
        "runtime_env": runtime_env,
        "bazel": {
            "executable": args.bazel or "bazelisk",
            "configs": configs,
            "test_timeout": CI_TEST_TIMEOUT_SECONDS,
        },
    }
    validate_profile(profile)
    cards = profile_cards(profile)
    topology_cards = args.tp_size * args.dp_size * (args.pp_size or 1)
    if cards != topology_cards:
        raise ValueError(
            f"--world-size={cards} conflicts with TP x DP x PP ({topology_cards})"
        )
    if measure_runs <= 0:
        raise ValueError("--runs must be positive")

    grid_args = argparse.Namespace(
        grid_mode=args.grid_mode,
        min_input_len=args.min_input_len,
        max_input_len=args.max_input_len,
        alignment=args.alignment,
        cache_alignment=args.cache_alignment,
        commit_tail_tokens=args.commit_tail_tokens,
        input_mode=args.input_mode,
        input_points=args.input_points,
        cache_points_per_input=args.cache_points_per_input,
        cache_ratio_points=args.cache_ratio_points,
        fixed_cache_len=args.fixed_cache_len,
        random_cache_count=args.random_cache_count,
        min_cache_len=args.min_cache_len,
        max_cache_len=args.max_cache_len,
        compute_step=args.compute_step,
        min_compute_len=args.min_compute_len,
        batch_size=args.batch_size if args.batch_size is not None else 1,
        measure_runs=measure_runs,
        seed=args.seed,
        max_cases=args.max_cases,
        allow_large_grid=args.allow_large_grid,
    )
    grid = build_grid(grid_args)
    profile_sha256 = fingerprint(profile)
    grid["profile"] = profile
    grid["profile_sha256"] = profile_sha256
    artifacts = {
        PROFILE_NAME: _encoded(profile),
        GRID_NAME: _encoded(grid),
    }
    return {
        "output_dir": str(output_dir),
        "profile": str(output_dir / PROFILE_NAME),
        "grid": str(output_dir / GRID_NAME),
        "profile_fingerprint": profile_sha256,
        "grid_sha256": _sha(artifacts[GRID_NAME]),
        "grid_summary": grid["summary"],
        "artifacts": artifacts,
    }


def write_inputs(inputs: Dict[str, Any]) -> None:
    output_dir = Path(inputs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in inputs["artifacts"].items():
        with (output_dir / name).open("xb") as stream:
            stream.write(content)


def inputs_summary(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: inputs[key]
        for key in (
            "profile",
            "grid",
            "profile_fingerprint",
            "grid_sha256",
            "grid_summary",
        )
    }


def case_weight(case: Dict[str, Any]) -> int:
    """Estimate prefill attention work while retaining a positive cold-start cost."""
    if is_grouped_case(case):
        requests = case["request_groups"]
    else:
        requests = [
            {
                "count": case.get("batch_size", 1),
                "input_len": case["input_len"],
                "cache_len": case["cache_len"],
            }
        ]
    return sum(
        int(request.get("count", 1))
        * int(request["input_len"])
        * max(1, int(request["input_len"]) - int(request["cache_len"]))
        for request in requests
    )


def split_cases(
    cases: Sequence[Dict[str, Any]], shard_count: int
) -> List[Tuple[List[Dict[str, Any]], int]]:
    """Deterministic longest-processing-time-first partitioning."""
    if shard_count <= 0:
        raise ValueError("--shards must be positive")
    if shard_count > len(cases):
        raise ValueError("--shards cannot exceed the effective case count")
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(shard_count)]
    weights = [0] * shard_count
    ordered = sorted(cases, key=lambda case: (-case_weight(case), case["case_id"]))
    for case in ordered:
        index = min(range(shard_count), key=lambda i: (weights[i], i))
        shards[index].append(case)
        weights[index] += case_weight(case)
    for shard in shards:
        shard.sort(key=lambda case: case["case_id"])
    return list(zip(shards, weights))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() not in ("", "0", "false", "none")


def _reject_shared_seed(profile: Dict[str, Any]) -> None:
    for section_name in ("engine", "engine_args"):
        section = profile.get(section_name, {})
        for key, value in section.items():
            if key.lstrip("-") == "cache_shared_seed" and _truthy(value):
                raise ValueError(
                    "CI sharding does not support cache_shared_seed: each shard "
                    "runs in an independent model process"
                )


def _validate_output_dir(output_dir: Path, repo: Path) -> Tuple[Path, str]:
    resolved = _new_top_level_dir(output_dir, repo, "--output-dir")
    relative = resolved.relative_to(repo.resolve())
    return resolved, relative.as_posix()


def _build_file(plan: Dict[str, Any], env: Dict[str, str]) -> bytes:
    lines = [
        'load("//rtp_llm/test/perf_test:defs.bzl", "cache_grid_ci_test")',
        "",
    ]
    env_literal = json.dumps(env, sort_keys=True, ensure_ascii=True)
    for shard in plan["shards"]:
        lines.extend(
            [
                "cache_grid_ci_test(",
                f'    name = "{shard["name"]}",',
                '    profile = "profile.snapshot.json",',
                f'    grid = "{shard["grid_file"]}",',
                f'    gpu_type = "{plan["gpu_type"]}",',
                f'    gpu_count = {plan["gpu_count"]},',
                f"    envs = {env_literal},",
                f'    measure_runs = {plan["measure_runs"]},',
                ")",
                "",
            ]
        )
    tests = ", ".join(f'":{shard["name"]}"' for shard in plan["shards"])
    lines.extend(
        [
            "test_suite(",
            '    name = "all",',
            f"    tests = [{tests}],",
            '    tags = ["manual"],',
            ")",
            "",
        ]
    )
    return "\n".join(lines).encode()


def _environment_file(env: Dict[str, str]) -> bytes:
    lines = ["# Generated by cache_perf ci-plan; values are profile literals."]
    lines.extend(
        f"export {name}={shlex.quote(value)}" for name, value in sorted(env.items())
    )
    return ("\n".join(lines) + "\n").encode()


def create_plan(args: argparse.Namespace, *, repo: Path) -> Dict[str, Any]:
    from rtp_llm.test.perf_test.cache_grid.runner.cache_perf import load_cases

    if args.gpu_count <= 0:
        raise ValueError("--gpu-count must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_+.-]+", args.gpu_type):
        raise ValueError("--gpu-type contains unsupported characters")
    if args.estimated_total_seconds is not None and args.estimated_total_seconds <= 0:
        raise ValueError("--estimated-total-seconds must be positive")
    if args.startup_seconds < 0:
        raise ValueError("--startup-seconds cannot be negative")
    if args.startup_seconds and args.estimated_total_seconds is None:
        raise ValueError("--startup-seconds requires --estimated-total-seconds")
    if (
        args.estimated_total_seconds is not None
        and args.startup_seconds >= args.estimated_total_seconds
    ):
        raise ValueError("--startup-seconds must be below --estimated-total-seconds")
    if args.target_seconds is not None and args.target_seconds <= 0:
        raise ValueError("--target-seconds must be positive")
    if (
        args.target_seconds is not None
        and args.target_seconds >= CI_TEST_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"--target-seconds must be below the configured "
            f"{CI_TEST_TIMEOUT_SECONDS}s test timeout (AOne maximum: "
            f"{AONE_MAX_ACTION_SECONDS}s)"
        )
    if args.target_seconds is not None and args.estimated_total_seconds is None:
        raise ValueError("--target-seconds requires --estimated-total-seconds")

    output_dir, package = _validate_output_dir(args.output_dir, repo)
    profile = load_profile(args.profile)
    _reject_shared_seed(profile)
    cards = profile_cards(profile)
    if cards is not None and cards != args.gpu_count:
        raise ValueError(
            f"--gpu-count={args.gpu_count} conflicts with profile topology ({cards})"
        )
    env = profile_environment(profile)
    if "WORLD_SIZE" in env:
        try:
            world_size = float(env["WORLD_SIZE"])
        except ValueError as error:
            raise ValueError("profile WORLD_SIZE must be a positive integer") from error
        if (
            not math.isfinite(world_size)
            or world_size <= 0
            or not world_size.is_integer()
        ):
            raise ValueError("profile WORLD_SIZE must be a positive integer")
        if int(world_size) != args.gpu_count:
            raise ValueError(
                f"--gpu-count={args.gpu_count} conflicts with profile WORLD_SIZE="
                f"{env['WORLD_SIZE']}"
            )
    grid_payload, cases = load_cases(args.grid)
    source_case_ids = sorted(case["case_id"] for case in grid_payload["cases"])
    effective_case_ids = sorted(case["case_id"] for case in cases)
    partitions = split_cases(cases, args.shards)
    measure_runs = args.runs
    if measure_runs is None:
        measure_runs = int(profile.get("cache_grid", {}).get("measure_runs", 3))
    if measure_runs <= 0:
        raise ValueError("--runs must be positive")

    total_weight = sum(weight for _, weight in partitions)
    shards = []
    artifacts: Dict[str, bytes] = {
        "profile.snapshot.json": _encoded(profile),
    }
    for index, (shard_cases, weight) in enumerate(partitions):
        name = f"shard_{index:02d}"
        filename = name + ".json"
        payload = copy.deepcopy(grid_payload)
        payload["cases"] = shard_cases
        summary = copy.deepcopy(payload.get("summary", {}))
        summary.update(
            {
                "case_count": len(shard_cases),
                "ci_shard_index": index,
                "ci_shard_count": len(partitions),
            }
        )
        payload["summary"] = summary
        content = _encoded(payload)
        artifacts[filename] = content
        estimated_seconds = None
        if args.estimated_total_seconds is not None:
            workload_seconds = args.estimated_total_seconds - args.startup_seconds
            estimated_seconds = (
                args.startup_seconds + workload_seconds * weight / total_weight
            )
        shards.append(
            {
                "index": index,
                "name": name,
                "target": f"//{package}:{name}",
                "grid_file": filename,
                "grid_sha256": _sha(content),
                "case_ids": [case["case_id"] for case in shard_cases],
                "weight": weight,
                "estimated_seconds": estimated_seconds,
            }
        )
    if args.target_seconds is not None:
        longest = max(shard["estimated_seconds"] for shard in shards)
        if longest > args.target_seconds:
            raise ValueError(
                f"estimated longest shard is {math.ceil(longest)}s, above "
                f"--target-seconds={args.target_seconds}; increase --shards"
            )

    plan = {
        "schema_version": SCHEMA_VERSION,
        "package": package,
        "suite_target": f"//{package}:all",
        "profile_source": str(args.profile.resolve()),
        "profile_snapshot_sha256": _sha(artifacts["profile.snapshot.json"]),
        "profile_fingerprint": fingerprint(profile),
        "grid_source": str(args.grid.resolve()),
        "grid_sha256": _sha(args.grid.read_bytes()),
        "source_case_count": len(source_case_ids),
        "case_ids": effective_case_ids,
        "deduplicated_case_ids": sorted(set(source_case_ids) - set(effective_case_ids)),
        "shard_count": len(shards),
        "gpu_type": args.gpu_type,
        "gpu_count": args.gpu_count,
        "measure_runs": measure_runs,
        "estimated_total_seconds": args.estimated_total_seconds,
        "startup_seconds": args.startup_seconds,
        "target_seconds": args.target_seconds,
        "shards": shards,
    }
    artifacts[PLAN_NAME] = _encoded(plan)
    artifacts[BUILD_NAME] = _build_file(plan, env)
    artifacts[ENV_NAME] = _environment_file(env)
    plan["output_dir"] = str(output_dir)
    plan["artifacts"] = artifacts
    return plan


def write_plan(plan: Dict[str, Any]) -> None:
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in plan["artifacts"].items():
        path = output_dir / name
        with path.open("xb") as stream:
            stream.write(content)


def plan_summary(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "suite_target": plan["suite_target"],
        "source_case_count": plan["source_case_count"],
        "effective_case_count": len(plan["case_ids"]),
        "deduplicated_case_ids": plan["deduplicated_case_ids"],
        "shard_count": plan["shard_count"],
        "shards": [
            {
                "target": shard["target"],
                "case_count": len(shard["case_ids"]),
                "weight": shard["weight"],
                "estimated_seconds": shard["estimated_seconds"],
            }
            for shard in plan["shards"]
        ],
    }


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    try:
        with zipfile.ZipFile(archive) as stream:
            for info in stream.infolist():
                target = (destination / info.filename).resolve()
                try:
                    target.relative_to(root)
                except ValueError as error:
                    raise ValueError(
                        f"unsafe path in {archive}: {info.filename}"
                    ) from error
            stream.extractall(destination)
    except zipfile.BadZipFile as error:
        raise ValueError(
            f"invalid Bazel undeclared-outputs archive: {archive}"
        ) from error


def _copy_outputs(source: Path, destination: Path) -> bool:
    archive = source / "outputs.zip"
    if archive.is_file():
        _safe_extract(archive, destination)
        return True
    if source.is_dir():
        files = [path for path in source.rglob("*") if path.is_file()]
        if files:
            destination.mkdir(parents=True, exist_ok=True)
            for path in files:
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            return True
    return False


def _metric_ids(payload: Dict[str, Any], source: Path) -> List[int]:
    metrics = payload.get("metrics")
    if not isinstance(metrics, list):
        raise ValueError(f"{source}: metrics must be a list")
    ids = []
    for index, metric in enumerate(metrics):
        if not isinstance(metric, dict) or type(metric.get("case_id")) is not int:
            raise ValueError(f"{source}: metric {index} has no integer case_id")
        ids.append(metric["case_id"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source}: duplicate case IDs")
    return ids


def collect_results(
    args: argparse.Namespace, *, repo: Path
) -> Tuple[Dict[str, Any], int]:
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported CI plan schema")
    generated = plan_path.parent
    package = Path(plan.get("package", ""))
    if package.is_absolute() or not package.parts or ".." in package.parts:
        raise ValueError("invalid generated Bazel package in CI plan")
    profile_snapshot = generated / "profile.snapshot.json"
    if _sha(profile_snapshot.read_bytes()) != plan.get("profile_snapshot_sha256"):
        raise ValueError("generated profile snapshot fingerprint mismatch")
    planned_ids = [case_id for shard in plan["shards"] for case_id in shard["case_ids"]]
    if plan.get("shard_count") != len(plan["shards"]):
        raise ValueError("CI plan shard_count does not match its shard list")
    if len(planned_ids) != len(set(planned_ids)) or sorted(planned_ids) != plan.get(
        "case_ids"
    ):
        raise ValueError("CI plan has duplicate or inconsistent case IDs")
    for shard in plan["shards"]:
        if not re.fullmatch(r"shard_[0-9]+", shard.get("name", "")):
            raise ValueError("invalid shard name in CI plan")
        if shard.get("grid_file") != shard["name"] + ".json":
            raise ValueError(f"{shard['name']}: invalid grid filename in CI plan")
        grid = generated / shard["grid_file"]
        if _sha(grid.read_bytes()) != shard["grid_sha256"]:
            raise ValueError(f"{shard['name']}: generated grid fingerprint mismatch")
    root = args.result_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("--result-root must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    testlogs = (args.bazel_testlogs or repo / "bazel-testlogs").resolve()
    seen = set()
    runs = []
    missing = []
    for shard in plan["shards"]:
        destination = root / shard["name"]
        outputs = testlogs / package / shard["name"] / "test.outputs"
        if not _copy_outputs(outputs, destination):
            missing.append(shard["name"])
            runs.append(
                {
                    "grid_json": str((generated / shard["grid_file"]).resolve()),
                    "result_dir": str(destination),
                    "status": "missing",
                    "grid_sha256": shard["grid_sha256"],
                }
            )
            continue
        shutil.copy2(profile_snapshot, destination)
        shutil.copy2(generated / shard["grid_file"], destination / "grid.snapshot.json")
        result = destination / "cache_grid_results.json"
        if not result.is_file():
            if args.partial:
                missing.append(shard["name"])
                runs.append(
                    {
                        "grid_json": str((generated / shard["grid_file"]).resolve()),
                        "result_dir": str(destination),
                        "status": "missing",
                        "grid_sha256": shard["grid_sha256"],
                    }
                )
                continue
            raise ValueError(f"{shard['name']}: cache_grid_results.json is missing")
        payload = json.loads(result.read_text(encoding="utf-8"))
        if payload.get("grid_sha256") != shard["grid_sha256"]:
            raise ValueError(f"{shard['name']}: result grid fingerprint mismatch")
        if payload.get("profile_sha256") != plan["profile_fingerprint"]:
            raise ValueError(f"{shard['name']}: result profile fingerprint mismatch")
        if payload.get("measure_runs") != plan["measure_runs"]:
            raise ValueError(f"{shard['name']}: result measure_runs mismatch")
        ids = set(_metric_ids(payload, result))
        expected = set(shard["case_ids"])
        if not ids <= expected:
            raise ValueError(
                f"{shard['name']}: unexpected case IDs {sorted(ids - expected)}"
            )
        if seen & ids:
            raise ValueError(f"duplicate case IDs across shards: {sorted(seen & ids)}")
        if not args.partial and (not payload.get("complete") or ids != expected):
            raise ValueError(
                f"{shard['name']}: incomplete result, missing {sorted(expected - ids)}"
            )
        seen.update(ids)
        runs.append(
            {
                "grid_json": str((generated / shard["grid_file"]).resolve()),
                "result_dir": str(destination),
                "status": "completed" if payload.get("complete") else "partial",
                "grid_sha256": shard["grid_sha256"],
                "case_ids": sorted(ids),
            }
        )
    expected_all = set(plan["case_ids"])
    if not args.partial and (missing or seen != expected_all):
        raise ValueError(
            f"incomplete CI result: missing_shards={missing}, "
            f"missing_cases={sorted(expected_all - seen)}"
        )
    if not seen:
        raise ValueError("no cache-grid results were collected")
    manifest = {
        "schema_version": 2,
        "source": "cache_perf_ci",
        "plan": str(plan_path),
        "expected_case_ids": sorted(expected_all),
        "collected_case_ids": sorted(seen),
        "missing_shards": missing,
        "runs": runs,
    }
    from rtp_llm.test.perf_test.cache_grid.runner.cache_perf import _write_manifest

    _write_manifest(root / "batch_runs.json", manifest)
    return manifest, 0


def pipeline_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Build the existing pipeline's skip-test namespace from ci-collect options."""
    return argparse.Namespace(
        mode="pipeline",
        profile=None,
        grid=None,
        result_dir=None,
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
        batch_size=args.batch_size,
        grid_dir=None,
        result_root=args.result_root,
        skip_fit=args.skip_fit,
        partial=args.partial,
        cards=args.cards,
        estimator=args.estimator,
        formula_output_dir=args.formula_output_dir,
        svg_output=args.svg_output,
        cold_svg_output=args.cold_svg_output,
        html_output=args.html_output,
        oss_destination=args.oss_destination,
    )
