"""Mode-specific frontend to cache_grid_perf_test, with immutable run snapshots."""

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    is_sensitive_environment_name,
    load_profile,
    profile_environment,
)
from rtp_llm.test.perf_test.cache_grid.runner.workspace_budget import (
    grid_token_budget,
    grid_workspace_tokens,
    validate_token_budget,
    workspace_capacity,
)

TARGET = "//rtp_llm/test/perf_test:cache_grid_perf_test"
REPO = Path(__file__).resolve().parents[5]
MANIFEST = "cache_perf_launch.json"
ENV_NAMES = {
    "PATH",
    "LD_LIBRARY_PATH",
    "CC",
    "CXX",
    "CUDAHOSTCXX",
    "NVCC_PREPEND_FLAGS",
    "CUDA_VISIBLE_DEVICES",
    "TOKENIZERS_PARALLELISM",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "START_PORT",
}
ENV_PREFIXES = (
    "CACHE_",
    "DG_JIT_",
    "DSV4_",
    "PERF_",
    "PREFILL_",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TILELANG_",
    "TRITON_",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def managed_env(name):
    return name in ENV_NAMES or name.startswith(ENV_PREFIXES)


def environment(profile, overrides, inherited):
    explicit = {}
    for item in overrides:
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError("--env requires NAME=VALUE")
        explicit[name] = value
    profile_environment({"engine_env": explicit})
    # Credentials may be inherited by the child process, but never serialize
    # them into a launch manifest or a Bazel command line.
    values = {
        k: v
        for k, v in inherited.items()
        if managed_env(k) and not is_sensitive_environment_name(k)
    }
    profile_environment({"engine_env": values})
    values.update(profile_environment(profile))
    values.update(explicit)
    return values


def replace_args(argv, updates, flags=()):
    owned = set(updates) | set(flags)
    result, i = [], 0
    while i < len(argv):
        item = argv[i]
        name = item.split("=", 1)[0].lstrip("-")
        i += 1
        if name not in owned:
            result.append(item)
            continue
        if name in flags or "=" in item:
            continue
        if name == "cache_profile_case_ids":
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
        elif i < len(argv) and not argv[i].startswith("--"):
            i += 1
    result.extend(f"--{k}={v}" for k, v in updates.items() if v is not None)
    return result


def load_saved(directory):
    path = directory / MANIFEST
    if not path.is_file():
        raise ValueError(f"current run requires {MANIFEST}")
    saved = json.loads(path.read_text())
    if not isinstance(saved, dict) or saved.get("schema_version") != 2:
        raise ValueError("launch manifest schema v2 required")
    required = {"snapshots", "profile_file", "grid", "env", "runner_args"}
    if required - saved.keys():
        raise ValueError("incomplete launch manifest schema v2")
    if (
        saved["grid"] != str(directory.resolve() / "grid.snapshot.json")
        or "grid.snapshot.json" not in saved["snapshots"]
    ):
        raise ValueError("current launch requires a fingerprinted local grid snapshot")
    info_path = directory / "test_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if info.get("schema_version") != 4:
            raise ValueError("launch-managed test_info schema v4 required")
        if sha(path.read_bytes()) != info["config_file_sha256"][MANIFEST]:
            raise ValueError("saved configuration changed: " + MANIFEST)
    for filename, digest in saved["snapshots"].items():
        if sha((directory / filename).read_bytes()) != digest:
            raise ValueError(f"saved configuration changed: {filename}")
    profile_file = saved["profile_file"]
    if profile_file not in saved["snapshots"]:
        raise ValueError("profile snapshot is not fingerprinted")
    saved["profile"] = load_profile(directory / profile_file)
    saved["runner_args"] = list(saved["runner_args"]) + [
        "--engine_env=" + k + "=" + v for k, v in sorted(saved["env"].items())
    ]
    return saved


def load_cases(path):
    from rtp_llm.test.perf_test.batch_decode_test import (
        _dedupe_cache_grid_cases,
        _load_cache_grid_cases,
        _resolve_cache_block_size,
    )

    payload = json.loads(path.read_text())
    cases = _load_cache_grid_cases(str(path))
    block = _resolve_cache_block_size(payload)
    return payload, _dedupe_cache_grid_cases(cases, block)


def build_plan(args, inherited=None):
    inherited = dict(os.environ if inherited is None else inherited)
    if args.profile_backend == "nsys" and args.mode != "profile":
        raise ValueError("--profile-backend=nsys requires profile mode")
    if not 0 <= args.nsys_tail_seconds <= 60:
        raise ValueError("--nsys-tail-seconds must be between 0 and 60")
    source = args.result_dir.resolve()
    replay = args.mode in ("retest", "profile")
    if (
        args.mode == "run"
        and source.exists()
        and any(source.iterdir())
        and not getattr(args, "allow_restart_without_checkpoint", False)
    ):
        raise ValueError("run requires an empty/new result directory; use resume")
    if args.mode == "resume" and not (source / "cache_grid_results.json").is_file():
        raise ValueError("resume requires an existing cache_grid_results.json")
    if args.mode == "resume" and (
        args.profile
        or args.grid
        or args.runs is not None
        or args.env
        or args.skip_reuse_validation
    ):
        raise ValueError(
            "resume uses frozen configuration; profile/grid/runs/env overrides are forbidden"
        )
    if replay != bool(args.cases):
        raise ValueError("--cases is required only for retest/profile")
    if (args.profile is None) != (args.grid is None):
        raise ValueError("supply both --profile and --grid, or neither")
    if args.mode == "run" and not args.profile:
        raise ValueError("run requires --profile and --grid")
    if args.runs is not None and args.runs <= 0 or args.trace_timeout <= 0:
        raise ValueError("runs/trace-timeout must be positive")
    saved = load_saved(source) if not args.profile else None
    profile = (
        load_profile(args.profile) if args.profile else copy.deepcopy(saved["profile"])
    )
    grid = args.grid.resolve() if args.grid else Path(saved["grid"])
    payload, cases = load_cases(grid)
    ids = sorted(set(int(s) for s in args.cases.split(","))) if args.cases else []
    missing = set(ids) - {c["case_id"] for c in cases}
    if missing:
        raise ValueError(f"unknown/deduplicated case IDs: {sorted(missing)}")
    if ids:
        cases = [c for c in cases if c["case_id"] in ids]
    if args.mode == "profile" and any(
        c.get("batch_size", 1) != 1 or "request_groups" in c for c in cases
    ):
        raise ValueError("profile currently supports ungrouped batch=1 only")
    env = environment(profile, args.env, inherited) if not saved else dict(saved["env"])
    if saved and args.env:
        overrides = environment({}, args.env, {})
        env.update(overrides)
    bazel = copy.deepcopy((saved or {}).get("bazel") or profile.get("bazel", {}))
    configs = args.config or bazel.get("configs", [])
    if not isinstance(configs, list) or not all(isinstance(v, str) for v in configs):
        raise ValueError("bazel.configs must be a list of strings")
    output_base = args.output_base or bazel.get("output_base")
    bazel = dict(
        configs=configs,
        output_base=str(Path(output_base).resolve()) if output_base else None,
        executable=args.bazel or bazel.get("executable", "bazelisk"),
        test_timeout=int(bazel.get("test_timeout", 345600)),
    )
    dest = (
        source / "cache_perf_replays" / (args.mode + "_" + uuid.uuid4().hex)
        if replay
        else source
    )
    artifacts = {}
    baseline = (
        list(saved["runner_args"])
        if saved
        else [
            "--partial=2",
            "--decode_test_length=1",
            "--batch_size=1",
            "--cache_request_transport=dashsc_input_ids",
        ]
    )
    if args.mode == "resume":
        runner = replace_args(
            baseline, {}, ("require_cache_resume", "allow_resume_mismatch")
        ) + ["--require_cache_resume"]
    else:
        if replay:
            payload = {
                "schema_version": 2,
                "cases": cases,
                "generator": payload.get("generator", {}),
                "summary": {
                    "case_count": len(cases),
                    "input_count": len({c["input_len"] for c in cases}),
                },
            }
        artifacts["grid.snapshot.json"] = (
            encoded(payload) if replay else grid.read_bytes()
        )
        artifacts["profile.snapshot.json"] = encoded(profile)
        updates = dict(
            profile=str(dest / "profile.snapshot.json"),
            cache_grid_json=str(dest / "grid.snapshot.json"),
            result_dir=str(dest),
            cache_profile_runs=0,
            cache_profile_case_ids=None,
            cache_profile_backend="kineto",
            cache_nsys_session=None,
            cache_nsys_path=None,
            cache_nsys_tail_seconds=None,
            profile_runs=0,
        )
        if args.runs is not None and args.mode != "profile":
            updates["cache_measure_runs"] = args.runs
        elif not saved:
            updates["cache_measure_runs"] = int(
                profile.get("cache_grid", {}).get("measure_runs", 3)
            )
        if not saved:
            updates["cache_commit_tail_tokens"] = profile.get("cache_grid", {}).get(
                "commit_tail_tokens", 4096
            )
        workspace = grid_workspace_tokens(payload)
        if workspace is not None:
            # Freeze effective capacities in argv, not just in mutable grid metadata.
            updates.update(
                cache_workspace_tokens=workspace,
                max_seq_len=workspace_capacity(
                    workspace, payload["generator"]["cache_alignment"]
                ),
                max_batch_tokens_size=min(workspace, grid_token_budget(payload)),
            )
        runner = replace_args(
            baseline,
            updates,
            (
                "cache_profile_only",
                "cache_profile_flat_output",
                "require_cache_resume",
                "allow_resume_mismatch",
            ),
        )
        if args.skip_reuse_validation and "--cache_skip_reuse_validation" not in runner:
            runner.append("--cache_skip_reuse_validation")
        if workspace is not None:
            tail = next(
                (
                    int(a.split("=", 1)[1])
                    for a in runner
                    if a.startswith("--cache_commit_tail_tokens=")
                ),
                4096,
            )
            validate_token_budget(
                cases,
                workspace_tokens=workspace,
                block=payload["generator"]["cache_alignment"],
                commit_tail=tail,
                token_budget=grid_token_budget(payload),
            )
        if args.mode == "profile":
            if any(a.split("=")[0] == "--cache_shared_seed" for a in runner):
                raise ValueError(
                    "profile cannot replay shared-seed mode; supply independent --profile and --grid"
                )
            runner = replace_args(
                runner,
                {
                    "cache_profile_runs": args.runs or 1,
                    "cache_profile_trace_timeout": args.trace_timeout,
                    "cache_profile_backend": args.profile_backend,
                    "cache_nsys_path": (
                        args.nsys_path if args.profile_backend == "nsys" else None
                    ),
                    "cache_nsys_session": (
                        args.nsys_session or ("cacheperf_" + uuid.uuid4().hex)
                        if args.profile_backend == "nsys"
                        else None
                    ),
                    "cache_nsys_tail_seconds": (
                        args.nsys_tail_seconds
                        if args.profile_backend == "nsys"
                        else None
                    ),
                },
            )
            runner += [
                "--cache_profile_only",
                "--cache_profile_flat_output",
                "--cache_profile_case_ids",
            ] + list(map(str, ids))
            if args.profile_backend == "nsys":
                env["GEN_TIMELINE_SYNC"] = "0"
                runner = replace_args(runner, {"gen_timeline_sync": "False"})
        env["PERF_PROFILE_RUNS"] = "0"
        # Make all resolved names part of test_info and the existing resume guard.
        # Bazel --test_env supplies the values before imports; these defaults only
        # register names with the runner's reproduction/environment capture.
        runner = [a for a in runner if not a.startswith("--engine_env=")]
        runner += ["--engine_env=" + k + "=" + v for k, v in sorted(env.items())]
        launch = dict(
            schema_version=2,
            mode=args.mode,
            profile_file="profile.snapshot.json",
            grid=str(dest / "grid.snapshot.json"),
            env=env,
            runner_args=[a for a in runner if not a.startswith("--engine_env=")],
            bazel=bazel,
            source_result_dir=str(source) if replay else None,
            selected_case_ids=ids,
            snapshots={k: sha(v) for k, v in artifacts.items()},
        )
        artifacts[MANIFEST] = encoded(launch)
    command = [bazel["executable"]]
    if bazel["output_base"]:
        command.append("--output_base=" + bazel["output_base"])
    command += ["test", TARGET] + ["--config=" + c for c in configs]
    command += [
        f"--test_timeout={bazel['test_timeout']}",
        "--test_output=streamed",
        "--nocache_test_results",
    ]
    command += ["--test_env=" + k + "=" + v for k, v in sorted(env.items())]
    command += [
        "--test_env=" + name
        for name in sorted(inherited)
        if managed_env(name) and is_sensitive_environment_name(name)
    ]
    command += ["--test_arg=" + arg for arg in runner]
    if args.mode == "profile" and args.profile_backend == "nsys":
        session = next(
            a.split("=", 1)[1] for a in runner if a.startswith("--cache_nsys_session=")
        )
        command.append(
            "--run_under="
            + shlex.join(
                [
                    args.nsys_path,
                    "launch",
                    "--show-output=true",
                    "--session-new=" + session,
                    "--trace=cuda,nvtx,osrt",
                    "--wait=all",
                ]
            )
        )
    process_env = {
        k: v
        for k, v in inherited.items()
        if not managed_env(k) or is_sensitive_environment_name(k)
    }
    process_env.update(env)
    return dict(
        command=command,
        process_env=process_env,
        artifacts=artifacts,
        destination=dest,
        summary=dict(
            mode=args.mode,
            planned_cases=len(cases),
            selected_case_ids=ids,
            profiler=args.mode == "profile",
            profile_backend=args.profile_backend if args.mode == "profile" else None,
            skip_reuse_validation="--cache_skip_reuse_validation" in runner,
            reads_checkpoint=args.mode == "resume",
            output=str(dest),
            environment=env,
            timing=(
                "profiler diagnostic, not formal latency"
                if args.mode == "profile"
                else "formal client wall and engine prefill recorded separately"
            ),
        ),
    )


def execute_plan(plan, *, allow_existing=True):
    """Persist a validated launch plan and execute its Bazel test."""
    dest = plan["destination"]
    dest.mkdir(parents=True, exist_ok=allow_existing)
    for name, content in plan["artifacts"].items():
        with (dest / name).open("xb") as stream:
            stream.write(content)
    return subprocess.run(
        plan["command"], cwd=REPO, env=plan["process_env"], check=False
    ).returncode


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run_pipeline(args):
    from rtp_llm.test.perf_test.cache_grid.runner.unified_pipeline import run

    return run(args)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["run", "resume", "retest", "profile", "pipeline"])
    p.add_argument("--profile", type=Path, help="JSON or commented JSONC profile")
    p.add_argument("--grid", type=Path)
    p.add_argument("--result-dir", type=Path)
    p.add_argument("--cases", help="comma-separated original case IDs")
    p.add_argument("--runs", type=int)
    p.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--config", action="append", help="repeatable Bazel configuration")
    p.add_argument("--output-base")
    p.add_argument("--bazel")
    p.add_argument("--trace-timeout", type=int, default=180)
    p.add_argument("--profile-backend", choices=("kineto", "nsys"), default="kineto")
    p.add_argument(
        "--nsys-path", default="nsys", help="nsys executable in the test environment"
    )
    p.add_argument("--nsys-session", help="new nsys session (default: unique name)")
    p.add_argument(
        "--nsys-tail-seconds",
        type=float,
        default=0.1,
        help="extra capture after request completion; not a GPU barrier (0..60)",
    )
    p.add_argument(
        "--skip-reuse-validation",
        action="store_true",
        help=(
            "Keep cases whose observed reuse differs from the grid expectation; "
            "the actual values remain recorded"
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true", help="print only; no writes/Bazel/GPU work"
    )
    pipeline = p.add_argument_group("pipeline: test, fit and charts")
    pipeline.add_argument("--test-mode", choices=("run", "resume"), default="run")
    pipeline.add_argument(
        "--skip-test",
        action="store_true",
        help="Post-process an existing complete result without launching a test",
    )
    pipeline.add_argument(
        "--batch-size", type=int, default=None, help="Optional batch filter"
    )
    pipeline.add_argument("--grid-dir", type=Path)
    pipeline.add_argument("--result-root", type=Path)
    pipeline.add_argument("--skip-fit", action="store_true")
    pipeline.add_argument("--partial", action="store_true")
    pipeline.add_argument("--cards", type=int, help="Override card count for TPM")
    pipeline.add_argument(
        "--estimator", choices=("median", "min", "trimmed"), default="median"
    )
    pipeline.add_argument("--formula-output-dir", type=Path)
    pipeline.add_argument("--svg-output", type=Path)
    pipeline.add_argument("--cold-svg-output", type=Path)
    pipeline.add_argument("--html-output", type=Path)
    pipeline.add_argument(
        "--oss-destination",
        help="Upload a tar.gz archive of the completed result directory to this OSS object URI",
    )
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.mode == "pipeline":
            return run_pipeline(args)
        if (
            args.skip_test
            or args.test_mode != "run"
            or args.batch_size is not None
            or args.grid_dir
            or args.result_root
            or args.skip_fit
            or args.partial
            or args.cards is not None
            or args.estimator != "median"
            or args.formula_output_dir
            or args.svg_output
            or args.cold_svg_output
            or args.html_output
            or args.oss_destination
        ):
            raise ValueError("pipeline options require pipeline mode")
        if args.result_dir is None:
            raise ValueError("--result-dir is required")
        plan = build_plan(args)
        print(json.dumps(plan["summary"], ensure_ascii=False, indent=2), flush=True)
        print(shlex.join(plan["command"]), flush=True)
        if args.dry_run:
            return 0
        return execute_plan(plan, allow_existing=args.mode in ("run", "resume"))
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        p.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
