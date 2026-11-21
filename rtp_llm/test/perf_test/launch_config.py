"""Select one execution profile and apply explicit, typed configuration edits."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from rtp_llm.test.perf_test.errors import ConfigurationError

_COMMON = {"backend", "platform", "test_timeout_seconds"}
_LOCAL = {"devices"}
_AONE = {
    "resource",
    "gpu_type",
    "shard_count",
    "max_parallel",
    "gpus_per_machine",
    "exec_properties",
}
DEFAULT_TEST_TIMEOUT_SECONDS = 7000


def _check_fields(value, allowed, path):
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path}: expected an object")
    unknown = set(value) - allowed
    if unknown:
        raise ConfigurationError(f"{path}.{sorted(unknown)[0]}: unknown field")


def _edit(document, expression):
    path, separator, raw = expression.partition("=")
    parts = path.split(".")
    if path == "schema_version" or path.endswith(".schema_version"):
        raise ConfigurationError("--set cannot change schema_version")
    if (
        not separator
        or not raw
        or any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", part) is None for part in parts)
    ):
        raise ConfigurationError(
            f"--set: expected dotted.path=value, got {expression!r}"
        )
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    current = document
    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise ConfigurationError(f"--set {path}: parent must be an object")
        current = current.setdefault(part, {})
    if not isinstance(current, dict):
        raise ConfigurationError(f"--set {path}: parent must be an object")
    current[parts[-1]] = value


def prepare_config(source: Mapping[str, Any], overrides: Sequence[str] = ()) -> dict:
    """Project v2 profiles onto the existing v1 execution contracts.

    Legacy SSH documents bypass profile conversion. Legacy local/AOne inputs
    remain byte-semantically unchanged unless explicit overrides are supplied.
    """
    from rtp_llm.test.perf_test.config_schema import _reject_unsupported_forms

    document = copy.deepcopy(source)
    if not isinstance(document, dict):
        raise ConfigurationError("config: expected an object")
    version = document.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        raise ConfigurationError("schema_version: expected integer 1 or 2")
    execution = document.get("execution")
    if not isinstance(execution, dict):
        raise ConfigurationError("execution: expected an object")
    old_backend = execution.get("backend")
    if version == 1 and old_backend == "ssh":
        for expression in overrides:
            _edit(document, expression)
        if not isinstance(document.get("execution"), dict):
            raise ConfigurationError("execution: expected an object")
        if document["execution"].get("backend") != "ssh":
            raise ConfigurationError(
                "SSH backend switching is outside the shared profile contract"
            )
        return document
    if version == 1 and not overrides:
        return document
    if version == 1:
        if old_backend not in ("local", "aone"):
            raise ConfigurationError("execution.backend: expected local or aone")
        common = {key: value for key, value in execution.items() if key in _COMMON}
        if old_backend == "aone" and "platform" not in common:
            from rtp_llm.test.perf_test.aone.plan import AONE_GPU_PLATFORMS

            common["platform"] = AONE_GPU_PLATFORMS.get(execution.get("gpu_type"))
        if old_backend == "local":
            common.setdefault("platform", "cuda13")
        common[old_backend] = {
            key: value for key, value in execution.items() if key not in _COMMON
        }
        document["execution"] = common
    for expression in overrides:
        _edit(document, expression)
    _reject_unsupported_forms(document)
    execution = document["execution"]
    _check_fields(execution, _COMMON | {"local", "aone"}, "execution")
    for backend, fields in (("local", _LOCAL), ("aone", _AONE)):
        if backend in execution:
            _check_fields(execution[backend], fields, f"execution.{backend}")
    backend = execution.get("backend")
    if backend not in ("local", "aone"):
        raise ConfigurationError("execution.backend: version 2 supports local or aone")
    active = dict(execution.get(backend, {}))
    active.update({key: execution[key] for key in _COMMON if key in execution})
    active.setdefault("test_timeout_seconds", DEFAULT_TEST_TIMEOUT_SECONDS)
    if backend == "local" and not active.get("platform"):
        raise ConfigurationError("execution.platform: required for local execution")
    if active.get("shard_count", ...) is None:
        active.pop("shard_count")
    document["schema_version"] = 1
    document["execution"] = active
    return document


def anchor_local_model_paths(source):
    """Resolve submit-side filesystem paths before changing to runfiles cwd."""
    from pathlib import Path

    result = copy.deepcopy(source)
    model = result.get("model", {})
    if not isinstance(model, dict):
        return result
    for key in ("checkpoint", "tokenizer"):
        value = model.get(key)
        if not isinstance(value, str) or "://" in value:
            continue
        candidate = Path(value).expanduser()
        if candidate.exists() or value.startswith(("./", "../", "~")):
            model[key] = str(candidate.resolve())
    return result


def platform_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_+-]+", value):
        raise ConfigurationError(
            "execution.platform: expected a Bazel platform config name"
        )
    return value


def execution_properties(value):
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise ConfigurationError(
            "execution.exec_properties: expected string keys and values"
        )
    return dict(value)


def validate_bazel_arguments(arguments, *, local=False):
    """Keep generated platform, worker input, device and timeout flags authoritative."""
    values = list(arguments)
    for index, argument in enumerate(values):
        option, separator, value = argument.partition("=")
        if not separator and index + 1 < len(values):
            value = values[index + 1]
        if option in {"--config", "--run_under", "--test_arg", "--test_timeout"}:
            raise ConfigurationError(
                f"{option} is managed by perf_test; use execution configuration"
            )
        if option in {"--test_env", "--action_env"} and value.split("=", 1)[0] in {
            "GPU_COUNT",
            "WORLD_SIZE",
            "RTP_TEST_WATCHDOG_SECONDS",
        }:
            raise ConfigurationError(
                f"{option} {value}: device allocation and watchdog are managed by perf_test"
            )
        if local and (
            (
                option == "--strategy"
                and value.startswith("TestRunner=")
                and value != "TestRunner=local"
            )
            or (option == "--test_strategy" and value != "local")
        ):
            raise ConfigurationError(
                "local execution requires the local TestRunner strategy"
            )
