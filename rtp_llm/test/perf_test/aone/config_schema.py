"""Strict AOne execution configuration, independent of runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from rtp_llm.test.perf_test.config_schema import ConfigError


def _object(path: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected an object")
    return value


def _reject_unknown(path: str, value: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        field_path = unknown[0] if not path else f"{path}.{unknown[0]}"
        expected = ", ".join(sorted(allowed))
        raise ConfigError(f"{field_path}: unknown field; expected one of: {expected}")


def _required(path: str, value: Mapping[str, Any], name: str) -> Any:
    if name not in value:
        field_path = name if not path else f"{path}.{name}"
        raise ConfigError(f"{field_path}: required field is missing")
    return value[name]


def _non_empty_string(path: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path}: expected a non-empty string")
    return value


def _positive_int(path: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{path}: expected a positive integer")
    return value


@dataclass(frozen=True)
class AOneExecutionConfig:
    backend: str
    resource: str
    gpu_type: str
    max_parallel: int
    shard_count: Optional[int]
    gpus_per_machine: Optional[int] = None


def parse_execution(value: Any) -> AOneExecutionConfig:
    raw = _object("execution", value)
    backend = _non_empty_string(
        "execution.backend", _required("execution", raw, "backend")
    )
    if backend == "aone":
        _reject_unknown(
            "execution",
            raw,
            {
                "backend",
                "resource",
                "gpu_type",
                "max_parallel",
                "shard_count",
                "gpus_per_machine",
            },
        )
        gpus_per_machine = (
            None
            if "gpus_per_machine" not in raw
            else _positive_int("execution.gpus_per_machine", raw["gpus_per_machine"])
        )
        if gpus_per_machine is None:
            shard_count = _positive_int(
                "execution.shard_count", _required("execution", raw, "shard_count")
            )
        else:
            shard_count = (
                None
                if "shard_count" not in raw
                else _positive_int("execution.shard_count", raw["shard_count"])
            )
        return AOneExecutionConfig(
            backend=backend,
            resource=_non_empty_string(
                "execution.resource", _required("execution", raw, "resource")
            ),
            gpu_type=_non_empty_string(
                "execution.gpu_type", _required("execution", raw, "gpu_type")
            ),
            max_parallel=_positive_int(
                "execution.max_parallel",
                _required("execution", raw, "max_parallel"),
            ),
            shard_count=shard_count,
            gpus_per_machine=gpus_per_machine,
        )
    raise ConfigError("execution.backend: expected aone")
