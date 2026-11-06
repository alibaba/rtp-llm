"""Strict, side-effect-free configuration models for the unified perf test."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple, Union

from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchConfig,
    DecodeProfilingConfig,
    FixedBatchConfig,
    TargetTPOTConfig,
)
from rtp_llm.test.perf_test.errors import ConfigurationError

SCHEMA_VERSION = 1
DEFAULT_PREFILL_TRANSPORT = "dashsc_input_ids"


class ConfigError(ConfigurationError, ValueError):
    """Raised when a unified perf-test config violates its schema."""


def _is_sensitive_name(name: str) -> bool:
    lower = name.lower()
    return (
        any(
            fragment in lower
            for fragment in (
                "password",
                "secret",
                "access_key",
                "api_key",
                "credential",
            )
        )
        or lower == "token"
        or lower.endswith("_token")
    )


def _reject_unsupported_forms(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            if not isinstance(name, str):
                raise ConfigError(f"{path}: object keys must be strings")
            item_path = name if path == "config" else f"{path}.{name}"
            lower = name.lower()
            if lower in {"include", "includes", "$include"}:
                raise ConfigError(f"{item_path}: include directives are not supported")
            if _is_sensitive_name(name):
                raise ConfigError(
                    f"{item_path}: credentials and secret values must remain external"
                )
            _reject_unsupported_forms(item, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_unsupported_forms(item, f"{path}[{index}]")
    elif isinstance(value, str):
        if "$" in value or "`" in value:
            raise ConfigError(
                f"{path}: interpolation and shell evaluation are not supported"
            )
        if value.lower().startswith(("secret:", "vault:", "env:")):
            raise ConfigError(f"{path}: secret references are not supported")


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


def _non_negative_int(path: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ConfigError(f"{path}: expected a non-negative integer")
    return value


def _positive_number(path: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a positive finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ConfigError(f"{path}: expected a positive finite number")
    return parsed


def _boolean(path: str, value: Any) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{path}: expected a boolean")
    return value


def _int_tuple(
    path: str, value: Any, *, allow_zero: bool = False, allow_empty: bool = False
) -> Tuple[int, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{path}: expected an array of integers")
    if not value and not allow_empty:
        raise ConfigError(f"{path}: expected at least one value")
    parser = _non_negative_int if allow_zero else _positive_int
    return tuple(parser(f"{path}[{index}]", item) for index, item in enumerate(value))


def _extension_map(path: str, value: Any, *, scalar_only: bool) -> Dict[str, Any]:
    raw = _object(path, value)
    result: Dict[str, Any] = {}
    for name, item in raw.items():
        item_path = f"{path}.{name}"
        if scalar_only and (
            isinstance(item, bool)
            or not isinstance(item, (str, int, float))
            or isinstance(item, float)
            and not math.isfinite(item)
        ):
            raise ConfigError(f"{item_path}: expected a string or finite number")
        result[name] = copy.deepcopy(item)
    return result


@dataclass(frozen=True)
class TopologyConfig:
    tp: int = 1
    dp: int = 1
    pp: int = 1
    ep: int = 1

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.tp, self.dp, self.pp, self.ep


@dataclass(frozen=True)
class ModelConfig:
    label: str
    type: str
    checkpoint: str
    tokenizer: str
    topology: TopologyConfig


@dataclass(frozen=True)
class EngineConfig:
    max_seq_len: int
    max_batch_tokens: int
    concurrency_limit: int
    args: Dict[str, Any] = field(default_factory=dict)
    env: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestGroupConfig:
    count: int
    input_len: int
    cache_len: int


@dataclass(frozen=True)
class RequestBatchConfig:
    request_groups: Tuple[RequestGroupConfig, ...]
    sample_index: Optional[int] = None


@dataclass(frozen=True)
class TrafficRequestBatchesConfig:
    distribution_file: str
    num_cases: int
    batch_sizes: Tuple[int, ...]
    min_input_tokens: int
    max_input_tokens: Optional[int] = None
    kv_budget_tokens: Optional[int] = None
    retry_limit: int = 10000


@dataclass(frozen=True)
class RandomRequestBatchesConfig:
    num_cases: int = 100
    batch_sizes: Tuple[int, ...] = (2, 4, 8)
    min_input_tokens: int = 8192
    max_input_tokens: int = 262144
    input_alignment: int = 256
    cold_probability: float = 0.2
    kv_budget_tokens: Optional[int] = None
    batch_input_limits: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PrefillProfilingConfig:
    backend: str = "kineto"
    trace_timeout_seconds: float = 120.0
    nsys_path: Optional[str] = None


@dataclass(frozen=True)
class PrefillGridConfig:
    name: str
    workload: "PrefillWorkloadConfig"


@dataclass(frozen=True)
class PrefillWorkloadConfig:
    kind: str
    max_input_len: int
    cache_alignment: int
    commit_tail_tokens: int
    compute_steps: Tuple[int, ...]
    batch_sizes: Tuple[int, ...]
    measure_runs: int = 3
    min_compute_len: int = 1
    min_cache_len: int = 0
    max_cache_len: Optional[int] = None
    cache_lengths: Tuple[int, ...] = ()
    random_cache_count: int = 8
    seed: int = 1
    max_cases: Optional[int] = None
    request_batches: Tuple[RequestBatchConfig, ...] = ()
    random_request_batches: Optional[RandomRequestBatchesConfig] = None
    traffic_request_batches: Optional[TrafficRequestBatchesConfig] = None
    grids: Tuple[PrefillGridConfig, ...] = ()
    transport: str = DEFAULT_PREFILL_TRANSPORT
    profiling: PrefillProfilingConfig = field(default_factory=PrefillProfilingConfig)
    probe_block_size: Optional[int] = None
    workspace_tokens: Optional[int] = None


@dataclass(frozen=True)
class DecodeGridWorkloadConfig:
    kind: str
    input_lengths: Tuple[int, ...]
    output_lengths: Tuple[int, ...]
    batch: DecodeBatchConfig
    measure_runs: int = 3
    warmup_runs: int = 1
    generate_config: Dict[str, Any] = field(default_factory=dict)
    profiling: DecodeProfilingConfig = field(default_factory=DecodeProfilingConfig)


@dataclass(frozen=True)
class DecodeDistributionWorkloadConfig:
    kind: str
    distribution_file: str
    output_length: int
    batch: DecodeBatchConfig
    measure_runs: int = 3
    warmup_runs: int = 1
    generate_config: Dict[str, Any] = field(default_factory=dict)
    profiling: DecodeProfilingConfig = field(default_factory=DecodeProfilingConfig)


WorkloadConfig = Union[
    PrefillWorkloadConfig,
    DecodeGridWorkloadConfig,
    DecodeDistributionWorkloadConfig,
]


@dataclass(frozen=True)
class LocalExecutionConfig:
    backend: str
    devices: Tuple[int, ...] = ()


class ExecutionConfig(Protocol):
    """Backend configuration contract; implementations must be dataclasses.

    The resolver serializes the returned dataclass with ``asdict``. Backend
    parsers own strict field validation and normalization of their section.
    """

    backend: str


ExecutionParser = Callable[[Any], ExecutionConfig]


@dataclass(frozen=True)
class OutputConfig:
    result_root: str = "perf_test_results"


@dataclass(frozen=True)
class RegressionConfig:
    enabled: bool = False
    baseline: Optional[str] = None
    threshold_pct: float = 10.0
    generate_candidate: bool = False


@dataclass(frozen=True)
class PublisherConfig:
    enabled: bool = False
    required: bool = False
    destination: Optional[str] = None
    artifacts: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PublishConfig:
    oss: PublisherConfig = field(default_factory=PublisherConfig)
    odps: PublisherConfig = field(default_factory=PublisherConfig)


@dataclass(frozen=True)
class PerfTestConfig:
    schema_version: int
    model: ModelConfig
    engine: EngineConfig
    workload: WorkloadConfig
    execution: ExecutionConfig
    output: OutputConfig
    regression: RegressionConfig
    publish: PublishConfig


def _parse_topology(value: Any) -> TopologyConfig:
    raw = _object("model.topology", value)
    _reject_unknown("model.topology", raw, {"tp", "dp", "pp", "ep"})
    sizes = {
        name: _positive_int(f"model.topology.{name}", raw.get(name, 1))
        for name in ("tp", "dp", "pp")
    }
    # Match the engine's automatic EP derivation within one PP stage, and
    # persist the effective value so launches and fingerprints agree.
    stage_size = sizes["tp"] * sizes["dp"]
    ep = _positive_int("model.topology.ep", raw.get("ep", stage_size))
    if ep != stage_size and not (ep == 1 and sizes["dp"] == 1):
        raise ConfigError(
            f"model.topology.ep: expected tp * dp ({stage_size}), or 1 when dp is 1"
        )
    return TopologyConfig(
        **sizes,
        ep=ep,
    )


def _parse_model(value: Any) -> ModelConfig:
    raw = _object("model", value)
    _reject_unknown(
        "model", raw, {"label", "type", "checkpoint", "tokenizer", "topology"}
    )
    checkpoint = _non_empty_string(
        "model.checkpoint", _required("model", raw, "checkpoint")
    )
    return ModelConfig(
        label=_non_empty_string("model.label", _required("model", raw, "label")),
        type=_non_empty_string("model.type", _required("model", raw, "type")),
        checkpoint=checkpoint,
        tokenizer=_non_empty_string(
            "model.tokenizer", raw.get("tokenizer", checkpoint)
        ),
        topology=_parse_topology(raw.get("topology", {})),
    )


def _parse_engine(value: Any) -> EngineConfig:
    raw = _object("engine", value)
    _reject_unknown(
        "engine",
        raw,
        {"max_seq_len", "max_batch_tokens", "concurrency_limit", "args", "env"},
    )
    return EngineConfig(
        max_seq_len=_positive_int(
            "engine.max_seq_len", _required("engine", raw, "max_seq_len")
        ),
        max_batch_tokens=_positive_int(
            "engine.max_batch_tokens",
            _required("engine", raw, "max_batch_tokens"),
        ),
        concurrency_limit=_positive_int(
            "engine.concurrency_limit",
            _required("engine", raw, "concurrency_limit"),
        ),
        args=_extension_map("engine.args", raw.get("args", {}), scalar_only=True),
        env=_extension_map("engine.env", raw.get("env", {}), scalar_only=True),
    )


def _parse_request_batches(value: Any) -> Tuple[RequestBatchConfig, ...]:
    path = "workload.request_batches"
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path}: expected a non-empty array")
    batches = []
    for index, item in enumerate(value):
        batch_path = f"{path}[{index}]"
        raw = _object(batch_path, item)
        _reject_unknown(batch_path, raw, {"request_groups", "sample_index"})
        groups = _required(batch_path, raw, "request_groups")
        if not isinstance(groups, list) or not groups:
            raise ConfigError(
                f"{batch_path}.request_groups: expected a non-empty array"
            )
        parsed = []
        for group_index, group in enumerate(groups):
            group_path = f"{batch_path}.request_groups[{group_index}]"
            group = _object(group_path, group)
            _reject_unknown(group_path, group, {"count", "input_len", "cache_len"})
            parsed.append(
                RequestGroupConfig(
                    count=_positive_int(
                        f"{group_path}.count", _required(group_path, group, "count")
                    ),
                    input_len=_positive_int(
                        f"{group_path}.input_len",
                        _required(group_path, group, "input_len"),
                    ),
                    cache_len=_non_negative_int(
                        f"{group_path}.cache_len",
                        _required(group_path, group, "cache_len"),
                    ),
                )
            )
        batches.append(
            RequestBatchConfig(
                tuple(parsed),
                (
                    None
                    if "sample_index" not in raw
                    else _positive_int(
                        f"{batch_path}.sample_index", raw["sample_index"]
                    )
                ),
            )
        )
    return tuple(batches)


def _parse_random_request_batches(value: Any) -> RandomRequestBatchesConfig:
    path = "workload.random_request_batches"
    raw = _object(path, value)
    _reject_unknown(
        path,
        raw,
        {
            "num_cases",
            "batch_sizes",
            "min_input_tokens",
            "max_input_tokens",
            "input_alignment",
            "cold_probability",
            "kv_budget_tokens",
            "batch_input_limits",
        },
    )
    num_cases = _positive_int(f"{path}.num_cases", raw.get("num_cases", 100))
    batch_sizes = _int_tuple(f"{path}.batch_sizes", raw.get("batch_sizes", [2, 4, 8]))
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ConfigError(f"{path}.batch_sizes: values must be unique")
    if num_cases < len(batch_sizes):
        raise ConfigError(f"{path}.num_cases: must cover every batch size")
    minimum = _positive_int(
        f"{path}.min_input_tokens", raw.get("min_input_tokens", 8192)
    )
    maximum = _positive_int(
        f"{path}.max_input_tokens", raw.get("max_input_tokens", 262144)
    )
    if minimum > maximum:
        raise ConfigError(f"{path}.max_input_tokens: must be at least min_input_tokens")
    probability = raw.get("cold_probability", 0.2)
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ConfigError(
            f"{path}.cold_probability: expected a finite number between 0 and 1"
        )
    limits = _object(f"{path}.batch_input_limits", raw.get("batch_input_limits", {}))
    parsed_limits = {}
    for key, value in limits.items():
        if (
            not isinstance(key, str)
            or not key.isascii()
            or not key.isdecimal()
            or str(int(key)) != key
            or int(key) <= 0
        ):
            raise ConfigError(
                f"{path}.batch_input_limits: keys must be positive batch-size strings"
            )
        if int(key) not in batch_sizes:
            raise ConfigError(
                f"{path}.batch_input_limits.{key}: batch size is not requested"
            )
        parsed_limits[key] = _positive_int(f"{path}.batch_input_limits.{key}", value)
    budget = raw.get("kv_budget_tokens")
    return RandomRequestBatchesConfig(
        num_cases=num_cases,
        batch_sizes=batch_sizes,
        min_input_tokens=minimum,
        max_input_tokens=maximum,
        input_alignment=_positive_int(
            f"{path}.input_alignment", raw.get("input_alignment", 256)
        ),
        cold_probability=float(probability),
        kv_budget_tokens=(
            None
            if budget is None
            else _positive_int(f"{path}.kv_budget_tokens", budget)
        ),
        batch_input_limits=parsed_limits,
    )


def _parse_traffic_request_batches(value: Any) -> TrafficRequestBatchesConfig:
    path = "workload.traffic_request_batches"
    raw = _object(path, value)
    _reject_unknown(
        path,
        raw,
        {
            "distribution_file",
            "num_cases",
            "batch_sizes",
            "min_input_tokens",
            "max_input_tokens",
            "kv_budget_tokens",
            "retry_limit",
        },
    )
    sizes = _int_tuple(f"{path}.batch_sizes", _required(path, raw, "batch_sizes"))
    count = _positive_int(f"{path}.num_cases", _required(path, raw, "num_cases"))
    minimum = _positive_int(
        f"{path}.min_input_tokens", _required(path, raw, "min_input_tokens")
    )
    maximum = (
        None
        if raw.get("max_input_tokens") is None
        else _positive_int(f"{path}.max_input_tokens", raw["max_input_tokens"])
    )
    if len(set(sizes)) != len(sizes) or count < len(sizes):
        raise ConfigError(
            f"{path}: batch sizes must be unique and covered by num_cases"
        )
    if maximum is not None and maximum < minimum:
        raise ConfigError(f"{path}.max_input_tokens: must be at least min_input_tokens")
    return TrafficRequestBatchesConfig(
        _non_empty_string(
            f"{path}.distribution_file", _required(path, raw, "distribution_file")
        ),
        count,
        sizes,
        minimum,
        maximum,
        (
            None
            if raw.get("kv_budget_tokens") is None
            else _positive_int(f"{path}.kv_budget_tokens", raw["kv_budget_tokens"])
        ),
        _positive_int(f"{path}.retry_limit", raw.get("retry_limit", 10000)),
    )


def _parse_prefill_workload(raw: Mapping[str, Any]) -> PrefillWorkloadConfig:
    shared_fields = {
        "kind",
        "cache_alignment",
        "commit_tail_tokens",
        "measure_runs",
        "transport",
        "profiling",
        "probe_block_size",
        "workspace_tokens",
        "max_cases",
        "seed",
    }
    generated_fields = {
        "max_input_len",
        "compute_steps",
        "batch_sizes",
        "min_compute_len",
        "min_cache_len",
        "max_cache_len",
        "cache_lengths",
        "random_cache_count",
        "seed",
    }
    _reject_unknown(
        "workload",
        raw,
        shared_fields
        | generated_fields
        | {
            "grids",
            "request_batches",
            "random_request_batches",
            "traffic_request_batches",
        },
    )
    transport = raw.get("transport", DEFAULT_PREFILL_TRANSPORT)
    if transport not in ("http_text", "dashsc_input_ids"):
        raise ConfigError("workload.transport: expected http_text or dashsc_input_ids")
    profiling = _object("workload.profiling", raw.get("profiling", {}))
    _reject_unknown(
        "workload.profiling",
        profiling,
        {"backend", "trace_timeout_seconds", "nsys_path"},
    )
    backend = profiling.get("backend", "kineto")
    if backend not in ("kineto", "nsys"):
        raise ConfigError("workload.profiling.backend: expected kineto or nsys")
    settings = dict(
        transport=transport,
        profiling=PrefillProfilingConfig(
            backend=backend,
            trace_timeout_seconds=_positive_number(
                "workload.profiling.trace_timeout_seconds",
                profiling.get("trace_timeout_seconds", 120.0),
            ),
            nsys_path=(
                None
                if profiling.get("nsys_path") is None
                else _non_empty_string(
                    "workload.profiling.nsys_path", profiling["nsys_path"]
                )
            ),
        ),
        probe_block_size=(
            None
            if "probe_block_size" not in raw
            else _positive_int("workload.probe_block_size", raw["probe_block_size"])
        ),
        workspace_tokens=(
            None
            if "workspace_tokens" not in raw
            else _positive_int("workload.workspace_tokens", raw["workspace_tokens"])
        ),
    )
    if "grids" in raw:
        if set(raw) & (
            (generated_fields - {"seed"})
            | {"request_batches", "random_request_batches", "traffic_request_batches"}
        ):
            raise ConfigError(
                "workload.grids conflicts with top-level generated fields or request_batches"
            )
        grids = raw["grids"]
        if not isinstance(grids, list) or not grids:
            raise ConfigError("workload.grids: expected a non-empty array")
        parsed_grids = []
        names = set()
        for index, item in enumerate(grids):
            path = f"workload.grids[{index}]"
            item = _object(path, item)
            _reject_unknown(
                path,
                item,
                generated_fields
                | {
                    "name",
                    "request_batches",
                    "random_request_batches",
                    "traffic_request_batches",
                },
            )
            name = _non_empty_string(f"{path}.name", _required(path, item, "name"))
            if name in names:
                raise ConfigError("workload.grids names must be unique")
            names.add(name)
            child = {key: value for key, value in raw.items() if key in shared_fields}
            child.update({key: value for key, value in item.items() if key != "name"})
            parsed_grids.append(PrefillGridConfig(name, _parse_prefill_workload(child)))
        base = parsed_grids[0].workload
        return replace(
            base,
            grids=tuple(parsed_grids),
            request_batches=(),
            random_request_batches=None,
            traffic_request_batches=None,
            **settings,
        )
    if "traffic_request_batches" in raw:
        if set(raw) & (
            (generated_fields - {"seed"})
            | {"request_batches", "random_request_batches"}
        ):
            raise ConfigError(
                "workload.traffic_request_batches conflicts with other case sources"
            )
        if settings["workspace_tokens"] is None:
            raise ConfigError(
                "workload.workspace_tokens: required for traffic_request_batches"
            )
        traffic = _parse_traffic_request_batches(raw["traffic_request_batches"])
        base = {
            key: value
            for key, value in raw.items()
            if key in shared_fields - set(settings)
        }
        base.update(
            max_input_len=traffic.max_input_tokens or traffic.min_input_tokens,
            compute_steps=[1],
            batch_sizes=[1],
            cache_lengths=[0],
            seed=raw.get("seed", 20261009),
        )
        return replace(
            _parse_generated_prefill_workload(base),
            traffic_request_batches=traffic,
            **settings,
        )
    if "random_request_batches" in raw:
        if set(raw) & ((generated_fields - {"seed"}) | {"request_batches"}):
            raise ConfigError(
                "workload.random_request_batches conflicts with request_batches or generated grid fields"
            )
        if settings["workspace_tokens"] is None:
            raise ConfigError(
                "workload.workspace_tokens: required for random_request_batches"
            )
        random = _parse_random_request_batches(raw["random_request_batches"])
        base = {
            key: value
            for key, value in raw.items()
            if key in shared_fields - set(settings)
        }
        base.update(
            max_input_len=random.max_input_tokens,
            compute_steps=[1],
            batch_sizes=[1],
            cache_lengths=[0],
            seed=raw.get("seed", 20260917),
        )
        return replace(
            _parse_generated_prefill_workload(base),
            random_request_batches=random,
            **settings,
        )
    if "request_batches" in raw:
        if set(raw) & (generated_fields - {"seed"}):
            raise ConfigError(
                "workload.request_batches conflicts with generated grid fields"
            )
        batches = _parse_request_batches(raw["request_batches"])
        if settings["workspace_tokens"] is None:
            raise ConfigError("workload.workspace_tokens: required for request_batches")
        base = {
            key: value
            for key, value in raw.items()
            if key in shared_fields - set(settings)
        }
        base.update(
            max_input_len=max(
                group.input_len for batch in batches for group in batch.request_groups
            ),
            compute_steps=[1],
            batch_sizes=[1],
            cache_lengths=[0],
            seed=raw.get("seed", 1),
        )
        return replace(
            _parse_generated_prefill_workload(base), request_batches=batches, **settings
        )
    base = {key: value for key, value in raw.items() if key not in settings}
    return replace(_parse_generated_prefill_workload(base), **settings)


def _parse_generated_prefill_workload(raw: Mapping[str, Any]) -> PrefillWorkloadConfig:
    allowed = {
        "kind",
        "max_input_len",
        "cache_alignment",
        "commit_tail_tokens",
        "compute_steps",
        "batch_sizes",
        "measure_runs",
        "min_compute_len",
        "min_cache_len",
        "max_cache_len",
        "cache_lengths",
        "random_cache_count",
        "seed",
        "max_cases",
    }
    _reject_unknown("workload", raw, allowed)
    compute_steps = _int_tuple(
        "workload.compute_steps", _required("workload", raw, "compute_steps")
    )
    if len(set(compute_steps)) != len(compute_steps) or any(
        left <= right for left, right in zip(compute_steps, compute_steps[1:])
    ):
        raise ConfigError(
            "workload.compute_steps: expected unique coarse-to-fine values"
        )
    max_cache_len = raw.get("max_cache_len")
    max_cases = raw.get("max_cases")
    cache_lengths = _int_tuple(
        "workload.cache_lengths",
        raw.get("cache_lengths", []),
        allow_zero=True,
        allow_empty=True,
    )
    random_cache_count = _non_negative_int(
        "workload.random_cache_count",
        raw.get("random_cache_count", 0 if cache_lengths else 8),
    )
    if cache_lengths and random_cache_count:
        raise ConfigError(
            "workload.cache_lengths conflicts with workload.random_cache_count"
        )
    return PrefillWorkloadConfig(
        kind="cache_grid",
        max_input_len=_positive_int(
            "workload.max_input_len", _required("workload", raw, "max_input_len")
        ),
        cache_alignment=_positive_int(
            "workload.cache_alignment",
            _required("workload", raw, "cache_alignment"),
        ),
        commit_tail_tokens=_positive_int(
            "workload.commit_tail_tokens",
            _required("workload", raw, "commit_tail_tokens"),
        ),
        compute_steps=compute_steps,
        batch_sizes=_int_tuple(
            "workload.batch_sizes", _required("workload", raw, "batch_sizes")
        ),
        measure_runs=_positive_int("workload.measure_runs", raw.get("measure_runs", 3)),
        min_compute_len=_positive_int(
            "workload.min_compute_len", raw.get("min_compute_len", compute_steps[-1])
        ),
        min_cache_len=_non_negative_int(
            "workload.min_cache_len", raw.get("min_cache_len", 0)
        ),
        max_cache_len=(
            None
            if max_cache_len is None
            else _non_negative_int("workload.max_cache_len", max_cache_len)
        ),
        cache_lengths=cache_lengths,
        random_cache_count=random_cache_count,
        seed=_non_negative_int("workload.seed", raw.get("seed", 1)),
        max_cases=(
            None
            if max_cases is None
            else _positive_int("workload.max_cases", max_cases)
        ),
    )


def _parse_decode_batch(value: Any) -> DecodeBatchConfig:
    raw = _object("workload.batch", value)
    strategy = _non_empty_string(
        "workload.batch.strategy", _required("workload.batch", raw, "strategy")
    )
    if strategy == "fixed":
        _reject_unknown("workload.batch", raw, {"strategy", "sizes", "unit"})
        unit = raw.get("unit", "global")
        if not isinstance(unit, str) or unit not in {"global", "per_dp_rank"}:
            raise ConfigError("workload.batch.unit: expected global or per_dp_rank")
        return FixedBatchConfig(
            strategy="fixed",
            sizes=_int_tuple(
                "workload.batch.sizes", _required("workload.batch", raw, "sizes")
            ),
            unit=unit,
        )
    if strategy == "target_tpot":
        _reject_unknown(
            "workload.batch",
            raw,
            {"strategy", "max_size", "candidate_step", "target_tpot_ms"},
        )
        return TargetTPOTConfig(
            strategy="target_tpot",
            max_size=_positive_int(
                "workload.batch.max_size",
                _required("workload.batch", raw, "max_size"),
            ),
            candidate_step=_positive_int(
                "workload.batch.candidate_step",
                _required("workload.batch", raw, "candidate_step"),
            ),
            target_tpot_ms=_positive_number(
                "workload.batch.target_tpot_ms",
                _required("workload.batch", raw, "target_tpot_ms"),
            ),
        )
    raise ConfigError("workload.batch.strategy: expected one of: fixed, target_tpot")


def _parse_decode_profiling(value: Any) -> DecodeProfilingConfig:
    defaults = DecodeProfilingConfig()
    if value is None:
        return defaults
    raw = _object("workload.profiling", value)
    fields = {"enabled", "runs", "steps", "enable_all_ranks", "flush_timeout_seconds"}
    _reject_unknown("workload.profiling", raw, fields)
    return DecodeProfilingConfig(
        enabled=_boolean(
            "workload.profiling.enabled", raw.get("enabled", defaults.enabled)
        ),
        runs=_positive_int("workload.profiling.runs", raw.get("runs", defaults.runs)),
        steps=_positive_int(
            "workload.profiling.steps", raw.get("steps", defaults.steps)
        ),
        enable_all_ranks=_boolean(
            "workload.profiling.enable_all_ranks",
            raw.get("enable_all_ranks", defaults.enable_all_ranks),
        ),
        flush_timeout_seconds=_positive_int(
            "workload.profiling.flush_timeout_seconds",
            raw.get("flush_timeout_seconds", defaults.flush_timeout_seconds),
        ),
    )


def _decode_common(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "batch": _parse_decode_batch(_required("workload", raw, "batch")),
        "measure_runs": _positive_int(
            "workload.measure_runs", raw.get("measure_runs", 3)
        ),
        "warmup_runs": _non_negative_int(
            "workload.warmup_runs", raw.get("warmup_runs", 1)
        ),
        "profiling": _parse_decode_profiling(raw.get("profiling")),
        "generate_config": _extension_map(
            "workload.generate_config",
            raw.get("generate_config", {}),
            scalar_only=False,
        ),
    }


def _parse_workload(value: Any) -> WorkloadConfig:
    raw = _object("workload", value)
    kind = _non_empty_string("workload.kind", _required("workload", raw, "kind"))
    if kind == "cache_grid":
        return _parse_prefill_workload(raw)
    common = {
        "kind",
        "batch",
        "measure_runs",
        "warmup_runs",
        "profiling",
        "generate_config",
    }
    if kind == "grid":
        _reject_unknown("workload", raw, common | {"input_lengths", "output_lengths"})
        return DecodeGridWorkloadConfig(
            kind=kind,
            input_lengths=_int_tuple(
                "workload.input_lengths",
                _required("workload", raw, "input_lengths"),
            ),
            output_lengths=_int_tuple(
                "workload.output_lengths", _required("workload", raw, "output_lengths")
            ),
            **_decode_common(raw),
        )
    if kind == "distribution":
        _reject_unknown(
            "workload", raw, common | {"distribution_file", "output_length"}
        )
        decode_common = _decode_common(raw)
        if getattr(decode_common["batch"], "unit", "global") == "per_dp_rank":
            raise ConfigError("workload.batch.unit: per_dp_rank requires grid")
        return DecodeDistributionWorkloadConfig(
            kind=kind,
            distribution_file=_non_empty_string(
                "workload.distribution_file",
                _required("workload", raw, "distribution_file"),
            ),
            output_length=_positive_int(
                "workload.output_length", _required("workload", raw, "output_length")
            ),
            **decode_common,
        )
    raise ConfigError("workload.kind: expected one of: cache_grid, distribution, grid")


def _parse_execution(value: Any) -> ExecutionConfig:
    raw = _object("execution", value)
    backend = _non_empty_string(
        "execution.backend", _required("execution", raw, "backend")
    )
    if backend == "local":
        _reject_unknown("execution", raw, {"backend", "devices"})
        return LocalExecutionConfig(
            backend=backend,
            devices=_int_tuple(
                "execution.devices",
                raw.get("devices", []),
                allow_zero=True,
                allow_empty=True,
            ),
        )
    raise ConfigError("execution.backend: expected local")


def _parse_output(value: Any) -> OutputConfig:
    raw = _object("output", value)
    _reject_unknown("output", raw, {"result_root"})
    return OutputConfig(
        result_root=_non_empty_string(
            "output.result_root", raw.get("result_root", "perf_test_results")
        )
    )


def _parse_regression(value: Any) -> RegressionConfig:
    raw = _object("regression", value)
    _reject_unknown(
        "regression",
        raw,
        {"enabled", "baseline", "threshold_pct", "generate_candidate"},
    )
    baseline = raw.get("baseline")
    config = RegressionConfig(
        enabled=_boolean("regression.enabled", raw.get("enabled", False)),
        baseline=(
            None
            if baseline is None
            else _non_empty_string("regression.baseline", baseline)
        ),
        threshold_pct=_positive_number(
            "regression.threshold_pct", raw.get("threshold_pct", 10.0)
        ),
        generate_candidate=_boolean(
            "regression.generate_candidate", raw.get("generate_candidate", False)
        ),
    )
    if config.enabled and config.baseline is None:
        raise ConfigError("regression.baseline: required when regression is enabled")
    return config


def _parse_publisher(path: str, value: Any) -> PublisherConfig:
    raw = _object(path, value)
    _reject_unknown(path, raw, {"enabled", "required", "destination", "artifacts"})
    destination = raw.get("destination")
    artifacts = raw.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ConfigError(f"{path}.artifacts: expected an array of strings")
    normalized_artifacts = tuple(
        _non_empty_string(f"{path}.artifacts[{index}]", item)
        for index, item in enumerate(artifacts)
    )
    return PublisherConfig(
        enabled=_boolean(f"{path}.enabled", raw.get("enabled", False)),
        required=_boolean(f"{path}.required", raw.get("required", False)),
        destination=(
            None
            if destination is None
            else _non_empty_string(f"{path}.destination", destination)
        ),
        artifacts=normalized_artifacts,
    )


def _parse_publish(value: Any) -> PublishConfig:
    raw = _object("publish", value)
    _reject_unknown("publish", raw, {"oss", "odps"})
    return PublishConfig(
        oss=_parse_publisher("publish.oss", raw.get("oss", {})),
        odps=_parse_publisher("publish.odps", raw.get("odps", {})),
    )


def parse_config(
    value: Any, *, execution_parser: Optional[ExecutionParser] = None
) -> PerfTestConfig:
    """Validate an already-decoded config without reading files or environment."""
    raw = _object("config", value)
    _reject_unsupported_forms(raw)
    allowed = {
        "schema_version",
        "model",
        "engine",
        "workload",
        "execution",
        "output",
        "regression",
        "publish",
    }
    _reject_unknown("", raw, allowed)
    version = _required("", raw, "schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ConfigError(
            f"schema_version: expected integer {SCHEMA_VERSION}, got {version!r}"
        )
    config = PerfTestConfig(
        schema_version=version,
        model=_parse_model(_required("", raw, "model")),
        engine=_parse_engine(_required("", raw, "engine")),
        workload=_parse_workload(_required("", raw, "workload")),
        execution=(execution_parser or _parse_execution)(
            _required("", raw, "execution")
        ),
        output=_parse_output(_required("", raw, "output")),
        regression=_parse_regression(_required("", raw, "regression")),
        publish=_parse_publish(_required("", raw, "publish")),
    )
    if (
        isinstance(config.workload, DecodeGridWorkloadConfig)
        and getattr(config.workload.batch, "unit", "global") == "per_dp_rank"
        and config.model.topology.pp != 1
    ):
        raise ConfigError("workload.batch.unit: per_dp_rank decode requires PP=1")

    return config
