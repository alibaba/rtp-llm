"""Strict AOne execution configuration, independent of runtime dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from rtp_llm.test.perf_test.config_schema import (
    ConfigError,
    _non_empty_string,
    _object,
    _positive_int,
    _reject_unknown,
    _required,
)

DEFAULT_AONE_TEST_TIMEOUT_SECONDS = 7000


@dataclass(frozen=True)
class AOneExecutionConfig:
    backend: str
    resource: str
    gpu_type: str
    max_parallel: int
    shard_count: Optional[int]
    gpus_per_machine: Optional[int] = None
    test_timeout_seconds: int = DEFAULT_AONE_TEST_TIMEOUT_SECONDS
    platform: Optional[str] = None
    exec_properties: Dict[str, str] = field(default_factory=dict)


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
                "test_timeout_seconds",
                "platform",
                "exec_properties",
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
        from rtp_llm.test.perf_test.aone.plan import AONE_GPU_PLATFORMS
        from rtp_llm.test.perf_test.launch_config import (
            execution_properties,
            platform_name,
        )

        gpu_type = _non_empty_string(
            "execution.gpu_type", _required("execution", raw, "gpu_type")
        )
        platform = (
            raw["platform"] if "platform" in raw else AONE_GPU_PLATFORMS.get(gpu_type)
        )
        if "platform" in raw:
            platform_name(platform)
        if platform is not None:
            platform_name(platform)
        expected = AONE_GPU_PLATFORMS.get(gpu_type)
        if expected is not None and platform != expected:
            raise ConfigError(
                f"execution.platform {platform!r} does not match GPU type {gpu_type!r} ({expected})"
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
            test_timeout_seconds=_positive_int(
                "execution.test_timeout_seconds",
                raw.get("test_timeout_seconds", DEFAULT_AONE_TEST_TIMEOUT_SECONDS),
            ),
            platform=platform,
            exec_properties=execution_properties(raw.get("exec_properties", {})),
        )
    raise ConfigError("execution.backend: expected aone")
