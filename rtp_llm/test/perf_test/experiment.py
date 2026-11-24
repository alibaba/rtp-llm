"""Single-file experiments expand into independently frozen ordinary runs."""

from __future__ import annotations

import copy
import itertools
import json
import re
import uuid
from pathlib import Path
from typing import Any, Mapping

from rtp_llm.test.perf_test.errors import ConfigurationError
from rtp_llm.test.perf_test.plan import thaw
from rtp_llm.test.perf_test.storage import atomic_write_json

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_MAX_TASKS = 4096


def _object(value, name, allowed):
    if not isinstance(value, dict) or set(value) - allowed:
        raise ConfigurationError(
            f"{name}: expected an object with fields {sorted(allowed)}"
        )
    return value


def _name(value, field):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ConfigurationError(f"{field}: expected a safe nonempty name")
    return value


def _paths(overrides, zipped, product):
    paths = list(overrides) + list(zipped) + list(product)
    for i, path in enumerate(paths):
        if not isinstance(path, str) or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part)
            for part in path.split(".")
        ):
            raise ConfigurationError(
                "experiment override paths must be dotted field names"
            )
        if not path.startswith(("model.", "engine.", "workload.")) and path not in {
            "execution.hosts",
            "execution.gpus_per_machine",
            "execution.start_port",
        }:
            raise ConfigurationError(f"experiment cannot override {path}")
        if any(
            path == other
            or path.startswith(other + ".")
            or other.startswith(path + ".")
            for other in paths[i + 1 :]
        ):
            raise ConfigurationError(f"overlapping experiment override: {path}")


def _set(config, path, value):
    parts = path.split(".")
    parent = config
    for part in parts[:-1]:
        if part not in parent and part in {"args", "env", "topology"}:
            parent[part] = {}
        if not isinstance(parent.get(part), dict):
            raise ConfigurationError(f"unknown experiment override path: {path}")
        parent = parent[part]
    parent[parts[-1]] = copy.deepcopy(value)


def expand_experiment(document: Mapping[str, Any]):
    """Expand zip axes then Cartesian axes without mutating the source."""
    raw = _object(
        document,
        "experiment",
        {"experiment_schema_version", "name", "base_config", "benchmarks"},
    )
    if (
        type(raw.get("experiment_schema_version")) is not int
        or raw["experiment_schema_version"] != 1
    ):
        raise ConfigurationError("experiment_schema_version: expected 1")
    _name(raw.get("name"), "experiment.name")
    if not isinstance(raw.get("base_config"), dict):
        raise ConfigurationError(
            "experiment.base_config must be an inline config object"
        )
    benchmarks = raw.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ConfigurationError("experiment.benchmarks must be a nonempty array")
    result, names = [], set()
    for benchmark in benchmarks:
        item = _object(
            benchmark, "benchmark", {"name", "overrides", "zip_axes", "product_axes"}
        )
        name = _name(item.get("name"), "benchmark.name")
        if name in names:
            raise ConfigurationError(f"duplicate benchmark name: {name}")
        names.add(name)
        overrides, zipped, product = (
            item.get(k, {}) for k in ("overrides", "zip_axes", "product_axes")
        )
        if not all(isinstance(x, dict) for x in (overrides, zipped, product)):
            raise ConfigurationError("overrides and axes must be objects")
        _paths(overrides, zipped, product)
        if any(
            not isinstance(v, list) or not v
            for v in [*zipped.values(), *product.values()]
        ):
            raise ConfigurationError("experiment axes must contain nonempty arrays")
        if len({len(v) for v in zipped.values()}) > 1:
            raise ConfigurationError("zip_axes arrays must have equal lengths")
        iterations = list(zip(*zipped.values())) if zipped else [()]
        for index_values in iterations:
            for product_values in itertools.product(*product.values()):
                if len(result) >= _MAX_TASKS:
                    raise ConfigurationError(f"experiment exceeds {_MAX_TASKS} tasks")
                config = copy.deepcopy(raw["base_config"])
                selected = {
                    **overrides,
                    **dict(zip(zipped, index_values)),
                    **dict(zip(product, product_values)),
                }
                for path, value in selected.items():
                    _set(config, path, value)
                result.append(
                    {
                        "name": f"{name}-{len(result):04d}",
                        "config": config,
                        "parameters": selected,
                    }
                )
    return result


def run_experiment(
    path,
    *,
    dry_run=False,
    run_context=None,
    overrides=(),
    bazel_args=(),
    bazel_startup_args=(),
):
    """Preflight every task, then execute exactly those plans in order."""
    from rtp_llm.test.perf_test.cli import CliRequest
    from rtp_llm.test.perf_test.entry import execute_plan, plan_from_source
    from rtp_llm.test.perf_test.launch_config import (
        prepare_config,
        validate_bazel_arguments,
    )

    path = Path(path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ConfigurationError(f"cannot read experiment {path}: {error}") from error
    original_document = copy.deepcopy(document)
    for expression in overrides:
        if not expression.startswith("base_config."):
            raise ConfigurationError(
                "experiment --set paths must start with base_config."
            )
    document["base_config"] = prepare_config(
        document["base_config"],
        tuple(expression[len("base_config.") :] for expression in overrides),
    )
    expanded = expand_experiment(document)
    output = document["base_config"].get("output", {})
    if not isinstance(output, dict) or not isinstance(
        output.get("result_root", "perf_test_results"), str
    ):
        raise ConfigurationError("output.result_root: expected a path")
    invocation_id = uuid.uuid4().hex
    root = (
        Path(output.get("result_root", "perf_test_results")).resolve()
        / f"experiment-{document['name']}-{invocation_id}"
    )
    snapshots, planned = {}, []
    fixed_source, mode = None, None
    for task in expanded:
        source = task["config"]
        if fixed_source is not None:
            source["execution"]["source"] = copy.deepcopy(fixed_source)
        plan, source = plan_from_source(
            source,
            source_dir=path.parent,
            result_directory=str(
                root / "runs" / f"{task['name'].replace('.', '_')}-{invocation_id}"
            ),
            external_inputs=snapshots,
        )
        if plan.backend == "ssh" and (bazel_args or bazel_startup_args):
            raise ConfigurationError("Bazel options require the local or AOne backend")
        validate_bazel_arguments(bazel_args, local=plan.backend == "local")
        if mode is not None and mode != plan.mode:
            raise ConfigurationError("an experiment cannot mix prefill and decode")
        mode = plan.mode
        if plan.backend == "ssh":
            fixed_source = thaw(plan.resolved_config["execution"]["source"])
        planned.append((task, plan, source))
    summary = {
        "experiment_schema_version": 1,
        "name": document["name"],
        "directory": str(root),
        "status": "planned" if dry_run else "completed",
        "runs": [],
        "exit_code": 0,
    }
    if not dry_run:
        root.mkdir(parents=True, exist_ok=False)
        atomic_write_json(
            root / "experiment_plan.json",
            {
                "name": document["name"],
                "tasks": [
                    {
                        "name": task["name"],
                        "run_id": Path(plan.result_directory).name,
                        "source_config": source,
                        "plan": plan.to_dict(),
                    }
                    for task, plan, source in planned
                ],
            },
        )
    for task, plan, source in planned:
        record = {
            "name": task["name"],
            "run_id": Path(plan.result_directory).name,
            "directory": plan.result_directory,
            "parameters": task["parameters"],
            "case_count": len(plan.cases),
            "config_fingerprint": plan.config_fingerprint,
            "workload_fingerprint": plan.workload_fingerprint,
            "status": "planned",
        }
        if not dry_run:
            try:
                outcome = execute_plan(
                    plan,
                    source,
                    CliRequest(
                        mode=plan.mode,
                        action="run",
                        config=None,
                        run_dir=None,
                        cases=(),
                        dry_run=False,
                        run_context=run_context,
                        bazel_args=tuple(bazel_args),
                        bazel_startup_args=tuple(bazel_startup_args),
                        overrides=tuple(overrides),
                        input_config=original_document,
                    ),
                )
                code = (
                    outcome.get("exit_code", 0)
                    if isinstance(outcome, Mapping)
                    else getattr(outcome, "exit_code", 0)
                )
                code = int(code)
                if code < 0:
                    code = 128 - code
                record.update(
                    status="completed" if code == 0 else "failed", exit_code=int(code)
                )
                outcome_state = (
                    outcome.get("state", outcome.get("status"))
                    if isinstance(outcome, Mapping)
                    else getattr(outcome, "state", None)
                )
                if (
                    getattr(outcome_state, "value", outcome_state) == "interrupted"
                    or code == 130
                ):
                    record.update(status="interrupted", exit_code=int(code) or 130)
            except Exception as error:
                record.update(status="failed", exit_code=1, error=str(error))
            except KeyboardInterrupt:
                record.update(status="interrupted", exit_code=130)
            if record["status"] == "interrupted":
                summary["runs"].append(record)
                summary.update(status="interrupted", exit_code=record["exit_code"])
                atomic_write_json(root / "experiment_summary.json", summary)
                return summary
            if record["exit_code"]:
                summary.update(
                    status="failed",
                    exit_code=summary["exit_code"] or record["exit_code"],
                )
        summary["runs"].append(record)
        if not dry_run:
            atomic_write_json(root / "experiment_summary.json", summary)
    return summary
