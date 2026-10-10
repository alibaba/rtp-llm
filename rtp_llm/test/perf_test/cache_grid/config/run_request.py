"""Normalize categorized cache-perf inputs for local and AOne planners."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Sequence, Set

from rtp_llm.test.perf_test.cache_grid.config.perf_profile import (
    ProfileError,
    is_sensitive_environment_name,
    profile_environment,
)

MODEL_FIELDS = {"label", "type", "checkpoint", "tokenizer", "topology"}
TOPOLOGY_FIELDS = {"tp", "dp", "pp", "ep"}
ENGINE_FIELDS = {
    "max_seq_len",
    "seq_size_per_block",
    "kernel_seq_size_per_block",
    "max_batch_tokens_size",
    "max_context_batch_size",
    "concurrency_limit",
    "args",
}
GRID_FIELDS = {
    "max_input_len",
    "cache_alignment",
    "commit_tail_tokens",
    "compute_step",
    "min_compute_len",
    "min_cache_len",
    "max_cache_len",
    "fixed_cache_len",
    "random_cache_count",
    "measure_runs",
    "batch_size",
    "seed",
    "max_cases",
}
_PROTECTED_ENGINE_ARGS = (
    MODEL_FIELDS
    | TOPOLOGY_FIELDS
    | ENGINE_FIELDS
    | GRID_FIELDS
    | {
        "world_size",
        "gpu_count",
        "tokenizers_parallelism",
        "tp_size",
        "dp_size",
        "pp_size",
        "ep_size",
        "world_rank",
        "local_world_size",
        "model_label",
        "model_type",
        "checkpoint_path",
        "tokenizer_path",
    }
)


@dataclass(frozen=True)
class ResolvedRunRequest:
    model: Dict[str, Any]
    engine: Dict[str, Any]
    grid: Dict[str, Any]
    engine_env: Dict[str, str]
    runtime_env: Dict[str, str]
    world_size: int
    gpu_count: int


class _ObjectPairs(list):
    """Retain JSON object entries until paths and duplicate keys are checked."""


def _restore_json(value: Any, path: str) -> Any:
    if isinstance(value, _ObjectPairs):
        result = {}
        for name, item in value:
            item_path = f"{path}.{name}"
            if name in result:
                raise ValueError(f"{item_path}: duplicate JSON key")
            result[name] = _restore_json(item, item_path)
        return result
    if isinstance(value, list):
        return [
            _restore_json(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    return value


def _json_object(raw: str, option: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_ObjectPairs)
    except json.JSONDecodeError as error:
        raise ValueError(f"{option} must be valid JSON: {error.msg}") from error
    value = _restore_json(value, option)
    if not isinstance(value, dict):
        raise ValueError(f"{option} must be a JSON object")
    return value


def _reject_unknown(section: str, value: Mapping[str, Any], allowed: Set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{section}.{unknown[0]} is not supported")


def _positive_int(path: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _scalar_map(path: str, value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a JSON object")
    normalized = {}
    for name, item in value.items():
        if is_sensitive_environment_name(name):
            raise ValueError(f"{path}.{name}: credential names are not allowed")
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise ValueError(f"{path}.{name}: use a string or finite number")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{path}.{name}: use a finite number")
        normalized[name] = str(item)
        try:
            profile_environment({path: {name: normalized[name]}})
        except ProfileError as error:
            raise ValueError(f"{path}.{name}: {error}") from error
    return normalized


def resolve_run_request(
    model_json: str,
    engine_json: str,
    grid_json: str,
    engine_env_json: str,
    runtime_env_json: str,
) -> ResolvedRunRequest:
    """Validate five JSON objects and derive shared model/grid settings."""
    raw_model = _json_object(model_json, "model")
    raw_engine = _json_object(engine_json, "engine")
    raw_grid = _json_object(grid_json, "grid")
    engine_env = _scalar_map("engine_env", _json_object(engine_env_json, "engine_env"))
    runtime_env = _scalar_map(
        "runtime_env", _json_object(runtime_env_json, "runtime_env")
    )
    _reject_unknown("model", raw_model, MODEL_FIELDS)
    _reject_unknown("engine", raw_engine, ENGINE_FIELDS)
    _reject_unknown("grid", raw_grid, GRID_FIELDS)

    raw_topology = raw_model.get("topology", {})
    if not isinstance(raw_topology, dict):
        raise ValueError("model.topology must be a JSON object")
    _reject_unknown("model.topology", raw_topology, TOPOLOGY_FIELDS)
    raw_args = raw_engine.get("args")
    if not isinstance(raw_args, dict):
        raise ValueError("engine.args must be a JSON object")
    for path, value in (
        ("model.type", raw_model.get("type")),
        ("model.checkpoint", raw_model.get("checkpoint")),
        ("engine.args.act_type", raw_args.get("act_type")),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{path} must be a non-empty string")
    for name, value in raw_args.items():
        if name.lstrip("-").lower() in _PROTECTED_ENGINE_ARGS:
            raise ValueError(
                f"engine.args.{name} is reserved or belongs to another category"
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, (str, int, float))
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise ValueError(f"engine.args.{name} must be a string or finite number")
    max_seq_len = _positive_int("engine.max_seq_len", raw_engine.get("max_seq_len"))
    _positive_int("engine.seq_size_per_block", raw_engine.get("seq_size_per_block"))
    compute_steps = raw_grid.get("compute_step")
    if (
        not isinstance(compute_steps, list)
        or not compute_steps
        or any(type(value) is not int or value <= 0 for value in compute_steps)
        or any(left <= right for left, right in zip(compute_steps, compute_steps[1:]))
    ):
        raise ValueError(
            "grid.compute_step must be positive integers in coarse-to-fine order"
        )
    cache_alignment = _positive_int(
        "grid.cache_alignment", raw_grid.get("cache_alignment")
    )
    if "fixed_cache_len" in raw_grid and "random_cache_count" in raw_grid:
        raise ValueError("grid.fixed_cache_len conflicts with grid.random_cache_count")

    topology = {
        key: _positive_int(f"model.topology.{key}", raw_topology.get(key, 1))
        for key in ("tp", "dp", "pp", "ep")
    }
    world_size = topology["tp"] * topology["dp"] * topology["pp"]
    checkpoint = raw_model["checkpoint"]
    model = {
        "label": raw_model.get("label", PurePosixPath(checkpoint.rstrip("/")).name),
        "type": raw_model["type"],
        "checkpoint": checkpoint,
        "tokenizer": raw_model.get("tokenizer", checkpoint),
        "topology": topology,
    }
    for field in ("label", "tokenizer"):
        if not isinstance(model[field], str) or not model[field]:
            raise ValueError(f"model.{field} must be a non-empty string")

    engine = {
        **raw_engine,
        "max_batch_tokens_size": raw_engine.get("max_batch_tokens_size", max_seq_len),
        "max_context_batch_size": raw_engine.get("max_context_batch_size", 1),
        "concurrency_limit": raw_engine.get("concurrency_limit", 1),
        "args": {
            "reuse_cache": 1,
            "enable_device_cache": 1,
            "enable_cuda_graph": 0,
            **raw_args,
        },
    }
    for field in (
        "max_batch_tokens_size",
        "max_context_batch_size",
        "concurrency_limit",
    ):
        _positive_int(f"engine.{field}", engine[field])
    if "kernel_seq_size_per_block" in engine:
        _positive_int(
            "engine.kernel_seq_size_per_block", engine["kernel_seq_size_per_block"]
        )

    grid = {
        "max_input_len": raw_grid.get("max_input_len", max_seq_len - 1),
        "cache_alignment": cache_alignment,
        "commit_tail_tokens": raw_grid.get("commit_tail_tokens", cache_alignment),
        "compute_step": compute_steps,
        "min_compute_len": raw_grid.get("min_compute_len", compute_steps[-1]),
        "min_cache_len": raw_grid.get("min_cache_len", 0),
        "max_cache_len": raw_grid.get("max_cache_len"),
        "fixed_cache_len": raw_grid.get("fixed_cache_len"),
        "random_cache_count": raw_grid.get("random_cache_count", 8),
        "measure_runs": raw_grid.get("measure_runs", 3),
        "batch_size": raw_grid.get("batch_size", 1),
        "seed": raw_grid.get("seed", 104729),
        "max_cases": raw_grid.get("max_cases", 200),
    }
    for field in (
        "max_input_len",
        "commit_tail_tokens",
        "min_compute_len",
        "random_cache_count",
        "measure_runs",
        "batch_size",
        "max_cases",
    ):
        _positive_int(f"grid.{field}", grid[field])
    for field in ("min_cache_len", "max_cache_len", "fixed_cache_len", "seed"):
        value = grid[field]
        if field in raw_grid and (type(value) is not int or value < 0):
            raise ValueError(f"grid.{field} must be a non-negative integer")
    if grid["max_input_len"] + 1 > max_seq_len:
        raise ValueError("grid.max_input_len + 1 must not exceed engine.max_seq_len")
    if grid["fixed_cache_len"] is not None:
        grid["random_cache_count"] = None

    overlap = set(engine_env) & set(runtime_env)
    if overlap:
        name = sorted(overlap)[0]
        raise ValueError(
            f"environment variable appears in both sections: "
            f"engine_env.{name} and runtime_env.{name}"
        )
    # These names control GPU locking/visibility or server process topology.
    reserved = {
        "WORLD_SIZE",
        "GPU_COUNT",
        "LOCAL_WORLD_SIZE",
        "WORLD_RANK",
        "TP_SIZE",
        "DP_SIZE",
        "EP_SIZE",
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "TOKENIZERS_PARALLELISM",
    }
    for section, values in (("engine_env", engine_env), ("runtime_env", runtime_env)):
        managed = reserved & set(values)
        if managed:
            name = sorted(managed)[0]
            raise ValueError(f"{section}.{name} is derived and cannot be overridden")
    engine_env.update(
        {
            "WORLD_SIZE": str(world_size),
            "GPU_COUNT": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "WORLD_RANK": "0",
            "TP_SIZE": str(topology["tp"]),
            "DP_SIZE": str(topology["dp"]),
            "EP_SIZE": str(topology["ep"]),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return ResolvedRunRequest(
        model=model,
        engine=engine,
        grid=grid,
        engine_env=engine_env,
        runtime_env=runtime_env,
        world_size=world_size,
        gpu_count=world_size,
    )


def to_prepare_namespace(
    resolved: ResolvedRunRequest,
    *,
    inputs_dir: Path,
    configs: Sequence[str],
    bazel: str = "bazelisk",
) -> argparse.Namespace:
    """Adapt a resolved request to ``cache_perf_ci.build_input_artifacts``."""
    topology = resolved.model["topology"]
    engine_settings = {
        key: value for key, value in resolved.engine.items() if key != "args"
    }
    return argparse.Namespace(
        inputs_dir=inputs_dir,
        model_label=resolved.model["label"],
        model_type=resolved.model["type"],
        checkpoint_path=resolved.model["checkpoint"],
        tokenizer_path=resolved.model["tokenizer"],
        tp_size=topology["tp"],
        dp_size=topology["dp"],
        ep_size=topology["ep"],
        pp_size=topology["pp"],
        world_size=resolved.world_size,
        engine_setting=[f"{key}={value}" for key, value in engine_settings.items()],
        engine_arg=[f"{key}={value}" for key, value in resolved.engine["args"].items()],
        engine_env_setting=[],
        runtime_env_setting=[],
        extra_engine_args_json="{}",
        extra_engine_env_json=json.dumps(resolved.engine_env),
        extra_runtime_env_json=json.dumps(resolved.runtime_env),
        config=list(configs),
        bazel=bazel,
        runs=resolved.grid["measure_runs"],
        grid_mode="fixed-cache-sweep",
        min_input_len=256,
        max_input_len=resolved.grid["max_input_len"],
        alignment=128,
        cache_alignment=resolved.grid["cache_alignment"],
        commit_tail_tokens=resolved.grid["commit_tail_tokens"],
        input_mode="stratified",
        input_points=1024,
        cache_points_per_input=16,
        cache_ratio_points=7,
        fixed_cache_len=(
            None
            if resolved.grid["fixed_cache_len"] is None
            else [resolved.grid["fixed_cache_len"]]
        ),
        random_cache_count=resolved.grid["random_cache_count"],
        min_cache_len=resolved.grid["min_cache_len"],
        max_cache_len=resolved.grid["max_cache_len"],
        compute_step=resolved.grid["compute_step"],
        min_compute_len=resolved.grid["min_compute_len"],
        batch_size=resolved.grid["batch_size"],
        seed=resolved.grid["seed"],
        max_cases=resolved.grid["max_cases"],
        allow_large_grid=False,
    )
