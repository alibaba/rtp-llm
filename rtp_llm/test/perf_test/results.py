"""Versioned common result envelope with discriminated mode metrics."""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from rtp_llm.test.perf_test.decode.metrics import (
    summarize_decode_measurements,
    validate_decode_measurement,
)
from rtp_llm.test.perf_test.decode.model import DecodeBatchMeasurement
from rtp_llm.test.perf_test.prefill.measurement import (
    validate_concurrent_prefill_metrics,
)
from rtp_llm.test.perf_test.state import CaseState, RunState

RESULT_SCHEMA_VERSION = 2
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")


class ResultSchemaError(ValueError):
    """Raised when a persisted result does not match the common envelope."""


def _object(path: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResultSchemaError(f"{path}: expected an object")
    return value


def _required(path: str, value: Mapping[str, Any], fields: set[str]) -> None:
    missing = sorted(fields - set(value))
    if missing:
        raise ResultSchemaError(f"{path}.{missing[0]}: required field is missing")


def _exact_fields(path: str, value: Mapping[str, Any], fields: set[str]) -> None:
    _required(path, value, fields)
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ResultSchemaError(f"{path}.{unknown[0]}: unknown field")


def _string(path: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ResultSchemaError(f"{path}: expected a non-empty string")
    return value


def _count(path: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ResultSchemaError(f"{path}: expected a non-negative integer")
    return value


def _positive_int(path: str, value: Any) -> int:
    parsed = _count(path, value)
    if parsed == 0:
        raise ResultSchemaError(f"{path}: expected a positive integer")
    return parsed


def _number(path: str, value: Any, *, positive: bool = False) -> Union[int, float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultSchemaError(f"{path}: expected a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or positive and parsed == 0:
        raise ResultSchemaError(f"{path}: expected a non-negative finite number")
    return value


def _record_list(path: str, value: Any) -> Tuple[Dict[str, Any], ...]:
    if not isinstance(value, list):
        raise ResultSchemaError(f"{path}: expected an array")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ResultSchemaError(f"{path}[{index}]: expected an object")
        result.append(copy.deepcopy(item))
    return tuple(result)


@dataclass(frozen=True)
class PrefillMetrics:
    input_length: int
    expected_reuse_length: int
    observed_reuse_length: int
    compute_length: int
    latency_ms: Union[int, float]
    throughput_tokens_per_second: Union[int, float]
    validation: Dict[str, bool]
    seed: int
    prefix_policy: Dict[str, Any]
    measurements: Tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_length": self.input_length,
            "expected_reuse_length": self.expected_reuse_length,
            "observed_reuse_length": self.observed_reuse_length,
            "compute_length": self.compute_length,
            "latency_ms": self.latency_ms,
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "validation": dict(self.validation),
            "seed": self.seed,
            "prefix_policy": copy.deepcopy(self.prefix_policy),
            "measurements": copy.deepcopy(list(self.measurements)),
        }


@dataclass(frozen=True)
class DecodeMetrics:
    workload_kind: str
    input_lengths: Tuple[int, ...]
    output_length: int
    global_batch_size: int
    topology: Dict[str, int]
    successful_requests: int
    failed_requests: int
    latency_ms: Dict[str, Union[int, float]]
    throughput_tokens_per_second: Union[int, float]
    measurements: Tuple[Dict[str, Any], ...]
    search: Optional[Dict[str, Any]]
    distribution: Optional[Dict[str, Any]]
    profiling: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workload_kind": self.workload_kind,
            "input_lengths": list(self.input_lengths),
            "output_length": self.output_length,
            "global_batch_size": self.global_batch_size,
            "topology": dict(self.topology),
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "latency_ms": dict(self.latency_ms),
            "throughput_tokens_per_second": self.throughput_tokens_per_second,
            "measurements": copy.deepcopy(list(self.measurements)),
            "search": copy.deepcopy(self.search),
            "distribution": copy.deepcopy(self.distribution),
            "profiling": copy.deepcopy(self.profiling),
        }


ModeMetrics = Union[PrefillMetrics, DecodeMetrics]


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    identity: Dict[str, Any]
    status: CaseState
    runs: int
    errors: Tuple[Dict[str, Any], ...]
    metrics: Optional[ModeMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "identity": copy.deepcopy(self.identity),
            "status": self.status.value,
            "runs": self.runs,
            "errors": copy.deepcopy(list(self.errors)),
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class ResultEnvelope:
    schema_version: int
    run_id: str
    mode: str
    backend: str
    status: RunState
    config_fingerprint: str
    workload_fingerprint: str
    model: Dict[str, Any]
    summary: Dict[str, int]
    cases: Tuple[CaseResult, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "mode": self.mode,
            "backend": self.backend,
            "status": self.status.value,
            "config_fingerprint": self.config_fingerprint,
            "workload_fingerprint": self.workload_fingerprint,
            "model": copy.deepcopy(self.model),
            "summary": dict(self.summary),
            "cases": [case.to_dict() for case in self.cases],
        }


def _parse_prefill_metrics(path: str, value: Any) -> PrefillMetrics:
    raw = _object(path, value)
    fields = {
        "input_length",
        "expected_reuse_length",
        "observed_reuse_length",
        "compute_length",
        "latency_ms",
        "throughput_tokens_per_second",
        "validation",
        "seed",
        "prefix_policy",
        "measurements",
    }
    _exact_fields(path, raw, fields)
    input_length = _positive_int(f"{path}.input_length", raw["input_length"])
    expected = _count(f"{path}.expected_reuse_length", raw["expected_reuse_length"])
    observed = _count(f"{path}.observed_reuse_length", raw["observed_reuse_length"])
    compute = _positive_int(f"{path}.compute_length", raw["compute_length"])
    if expected + compute != input_length:
        raise ResultSchemaError(f"{path}: reuse plus compute must equal input length")
    validation = _object(f"{path}.validation", raw["validation"])
    _exact_fields(f"{path}.validation", validation, {"shape", "reuse", "timing"})
    if any(type(item) is not bool for item in validation.values()):
        raise ResultSchemaError(f"{path}.validation: values must be boolean")
    prefix_policy = _object(f"{path}.prefix_policy", raw["prefix_policy"])
    return PrefillMetrics(
        input_length=input_length,
        expected_reuse_length=expected,
        observed_reuse_length=observed,
        compute_length=compute,
        latency_ms=_number(f"{path}.latency_ms", raw["latency_ms"], positive=True),
        throughput_tokens_per_second=_number(
            f"{path}.throughput_tokens_per_second",
            raw["throughput_tokens_per_second"],
            positive=True,
        ),
        validation=dict(validation),
        seed=_count(f"{path}.seed", raw["seed"]),
        prefix_policy=copy.deepcopy(dict(prefix_policy)),
        measurements=_record_list(f"{path}.measurements", raw["measurements"]),
    )


def _decode_identity(path: str, value: Mapping[str, Any]) -> None:
    _exact_fields(
        path,
        value,
        {
            "workload_kind",
            "input_lengths",
            "output_length",
            "batch_policy",
            "measure_runs",
            "warmup_runs",
            "generate_config",
            "profiling",
            "distribution",
        },
    )
    if value["workload_kind"] not in {"grid", "distribution"}:
        raise ResultSchemaError(f"{path}.workload_kind: expected grid or distribution")
    lengths = value["input_lengths"]
    if not isinstance(lengths, list) or not lengths:
        raise ResultSchemaError(f"{path}.input_lengths: expected a non-empty array")
    for index, item in enumerate(lengths):
        _positive_int(f"{path}.input_lengths[{index}]", item)
    if value["workload_kind"] == "grid" and len(set(lengths)) != 1:
        raise ResultSchemaError(f"{path}.input_lengths: grid inputs must be uniform")
    _positive_int(f"{path}.output_length", value["output_length"])
    _positive_int(f"{path}.measure_runs", value["measure_runs"])
    _count(f"{path}.warmup_runs", value["warmup_runs"])
    _object(f"{path}.generate_config", value["generate_config"])
    policy_path = f"{path}.batch_policy"
    policy = _object(policy_path, value["batch_policy"])
    if policy.get("strategy") == "fixed":
        _exact_fields(policy_path, policy, {"strategy", "size"})
        size = _positive_int(f"{policy_path}.size", policy["size"])
    elif policy.get("strategy") == "target_tpot":
        _exact_fields(
            policy_path,
            policy,
            {"strategy", "max_size", "candidate_step", "target_tpot_ms"},
        )
        size = _positive_int(f"{policy_path}.max_size", policy["max_size"])
        _positive_int(f"{policy_path}.candidate_step", policy["candidate_step"])
        _number(
            f"{policy_path}.target_tpot_ms", policy["target_tpot_ms"], positive=True
        )
    else:
        raise ResultSchemaError(f"{policy_path}.strategy: unknown batch strategy")
    if len(lengths) != size:
        raise ResultSchemaError(f"{path}.input_lengths: must match batch policy size")
    profile_path = f"{path}.profiling"
    profile = _object(profile_path, value["profiling"])
    _exact_fields(
        profile_path,
        profile,
        {"enabled", "runs", "steps", "enable_all_ranks", "flush_timeout_seconds"},
    )
    for name in ("enabled", "enable_all_ranks"):
        if type(profile[name]) is not bool:
            raise ResultSchemaError(f"{profile_path}.{name}: expected boolean")
    for name in ("runs", "steps", "flush_timeout_seconds"):
        _positive_int(f"{profile_path}.{name}", profile[name])
    distribution = value["distribution"]
    if value["workload_kind"] == "distribution":
        _object(f"{path}.distribution", distribution)
    elif distribution is not None:
        raise ResultSchemaError(f"{path}.distribution: expected null for grid")


def _parse_search(path: str, value: Any) -> Dict[str, Any]:
    raw = _object(path, value)
    _exact_fields(
        path,
        raw,
        {
            "target_tpot_ms",
            "max_size",
            "candidate_step",
            "outcome",
            "best_batch_size",
            "observations",
        },
    )
    target = _number(f"{path}.target_tpot_ms", raw["target_tpot_ms"], positive=True)
    maximum = _positive_int(f"{path}.max_size", raw["max_size"])
    step = _positive_int(f"{path}.candidate_step", raw["candidate_step"])
    _string(f"{path}.outcome", raw["outcome"])
    if raw["outcome"] not in {"found", "no_satisfying_candidate"}:
        raise ResultSchemaError(f"{path}.outcome: unknown search outcome")
    observations = _record_list(f"{path}.observations", raw["observations"])
    if not observations:
        raise ResultSchemaError(f"{path}.observations: expected at least one candidate")
    seen, satisfied = set(), []
    for index, observation in enumerate(observations):
        observation_path = f"{path}.observations[{index}]"
        _required(observation_path, observation, {"batch_size", "outcome"})
        allowed = {
            "batch_size",
            "outcome",
            "tpot_ms",
            "success_rate",
            "throughput_tokens_per_second",
            "error",
        }
        _exact_fields(observation_path, observation, set(observation) & allowed)
        batch = _positive_int(
            f"{observation_path}.batch_size", observation["batch_size"]
        )
        if (
            batch > maximum
            or (batch not in (1, maximum) and batch % step != 0)
            or batch in seen
        ):
            raise ResultSchemaError(
                f"{observation_path}.batch_size: invalid or duplicate candidate"
            )
        seen.add(batch)
        outcome = _string(f"{observation_path}.outcome", observation["outcome"])
        if outcome not in {
            "satisfied",
            "target_exceeded",
            "capacity_failed",
            "measurement_failed",
        }:
            raise ResultSchemaError(
                f"{observation_path}.outcome: unknown candidate outcome"
            )
        for name in ("tpot_ms", "success_rate", "throughput_tokens_per_second"):
            if observation.get(name) is not None:
                _number(f"{observation_path}.{name}", observation[name])
        if (
            observation.get("success_rate") is not None
            and observation["success_rate"] > 1
        ):
            raise ResultSchemaError(
                f"{observation_path}.success_rate: must not exceed one"
            )
        if "error" in observation:
            _string(f"{observation_path}.error", observation["error"])
        if outcome == "satisfied":
            tpot = observation.get("tpot_ms", target)
            if tpot is None or tpot > target or observation.get("success_rate", 1) != 1:
                raise ResultSchemaError(
                    f"{observation_path}: satisfied candidate must meet target with full success"
                )
            satisfied.append(batch)
    best = raw["best_batch_size"]
    if raw["outcome"] == "found":
        _positive_int(f"{path}.best_batch_size", best)
        if not satisfied or best != max(satisfied):
            raise ResultSchemaError(
                f"{path}.best_batch_size: must select largest satisfied candidate"
            )
    elif best is not None or satisfied:
        raise ResultSchemaError(
            f"{path}.best_batch_size: failed search cannot select a batch"
        )
    return copy.deepcopy(dict(raw))


def _parse_profiling(path: str, value: Any) -> Dict[str, Any]:
    raw = _object(path, value)
    _required(path, raw, {"status", "manifest_path", "error"})
    # Profiles may embed the complete manifest or just its status and reference.
    allowed = {
        "status",
        "manifest_path",
        "error",
        "schema_version",
        "case_id",
        "batch_size",
        "input_lengths",
        "output_length",
        "runs",
        "steps",
        "enable_all_ranks",
        "trace_name",
        "artifacts",
    }
    _exact_fields(path, raw, set(raw) & allowed)
    if raw["status"] not in {"disabled", "completed", "failed"}:
        raise ResultSchemaError(f"{path}.status: unknown profiling status")
    if raw["status"] != "disabled":
        _string(f"{path}.manifest_path", raw["manifest_path"])
    elif raw["manifest_path"] is not None:
        raise ResultSchemaError(f"{path}.manifest_path: disabled profile must be null")
    if raw["error"] is not None:
        _string(f"{path}.error", raw["error"])
    return copy.deepcopy(dict(raw))


def _parse_decode_metrics(path: str, value: Any) -> DecodeMetrics:
    raw = _object(path, value)
    fields = {
        "workload_kind",
        "input_lengths",
        "output_length",
        "global_batch_size",
        "topology",
        "successful_requests",
        "failed_requests",
        "latency_ms",
        "throughput_tokens_per_second",
        "measurements",
        "search",
        "distribution",
        "profiling",
    }
    _exact_fields(path, raw, fields)
    kind = raw["workload_kind"]
    if kind not in {"grid", "distribution"}:
        raise ResultSchemaError(f"{path}.workload_kind: expected grid or distribution")
    input_lengths = raw["input_lengths"]
    if not isinstance(input_lengths, list) or not input_lengths:
        raise ResultSchemaError(f"{path}.input_lengths: expected a non-empty array")
    parsed_inputs = tuple(
        _positive_int(f"{path}.input_lengths[{index}]", item)
        for index, item in enumerate(input_lengths)
    )
    global_batch = _positive_int(f"{path}.global_batch_size", raw["global_batch_size"])
    if len(parsed_inputs) != global_batch:
        raise ResultSchemaError(
            f"{path}.global_batch_size: must equal the number of input lengths"
        )
    successful = _count(f"{path}.successful_requests", raw["successful_requests"])
    failed = _count(f"{path}.failed_requests", raw["failed_requests"])
    measurements = _record_list(f"{path}.measurements", raw["measurements"])
    if not measurements or successful + failed != global_batch * len(measurements):
        raise ResultSchemaError(
            f"{path}: request outcomes must equal global_batch_size times measurement rounds"
        )
    topology = _object(f"{path}.topology", raw["topology"])
    _exact_fields(f"{path}.topology", topology, {"tp", "dp", "pp", "ep"})
    parsed_topology = {
        name: _positive_int(f"{path}.topology.{name}", topology[name])
        for name in ("tp", "dp", "pp", "ep")
    }
    latency = _object(f"{path}.latency_ms", raw["latency_ms"])
    latency_fields = {"wait", "prefill", "decode", "tpot", "total"}
    _exact_fields(f"{path}.latency_ms", latency, latency_fields)
    parsed_latency = {
        name: _number(f"{path}.latency_ms.{name}", latency[name])
        for name in sorted(latency_fields)
    }
    output_length = _positive_int(f"{path}.output_length", raw["output_length"])
    throughput = _number(
        f"{path}.throughput_tokens_per_second", raw["throughput_tokens_per_second"]
    )
    try:
        normalized_measurements = validate_decode_measurement(
            {
                "input_lengths": list(parsed_inputs),
                "output_length": output_length,
                "measure_runs": len(measurements),
            },
            DecodeBatchMeasurement(global_batch, parsed_inputs, (), measurements),
        )
        summary = summarize_decode_measurements(normalized_measurements)
    except ValueError as error:
        raise ResultSchemaError(f"{path}.measurements: {error}") from error
    if (
        successful != summary["successful_requests"]
        or failed != summary["failed_requests"]
    ):
        raise ResultSchemaError(f"{path}: request totals must match raw measurements")
    for name, latency_value in parsed_latency.items():
        if not math.isclose(latency_value, summary["latency_ms"][name]):
            raise ResultSchemaError(
                f"{path}.latency_ms.{name}: must match raw measurements"
            )
    if not math.isclose(throughput, summary["throughput_tokens_per_second"]):
        raise ResultSchemaError(
            f"{path}.throughput_tokens_per_second: must match raw measurements"
        )
    optional_records = {}
    for name in ("search", "distribution"):
        item = raw[name]
        if item is not None and not isinstance(item, dict):
            raise ResultSchemaError(f"{path}.{name}: expected an object or null")
        optional_records[name] = copy.deepcopy(item)
    return DecodeMetrics(
        workload_kind=kind,
        input_lengths=parsed_inputs,
        output_length=output_length,
        global_batch_size=global_batch,
        topology=parsed_topology,
        successful_requests=successful,
        failed_requests=failed,
        latency_ms=parsed_latency,
        throughput_tokens_per_second=throughput,
        measurements=tuple(normalized_measurements),
        search=(
            None
            if raw["search"] is None
            else _parse_search(f"{path}.search", raw["search"])
        ),
        distribution=optional_records["distribution"],
        profiling=_parse_profiling(f"{path}.profiling", raw["profiling"]),
    )


def parse_case_result(index: int, mode: str, value: Any) -> CaseResult:
    """Validate one case independently, allowing diagnostic consumers to isolate failures."""
    path = f"cases[{index}]"
    raw = _object(path, value)
    fields = {"case_id", "identity", "status", "runs", "errors", "metrics"}
    _exact_fields(path, raw, fields)
    try:
        status = CaseState(raw["status"])
    except ValueError as error:
        raise ResultSchemaError(f"{path}.status: unknown case state") from error
    identity = _object(f"{path}.identity", raw["identity"])
    if mode == "decode":
        _decode_identity(f"{path}.identity", identity)
    errors = _record_list(f"{path}.errors", raw["errors"])
    parsed_errors = []
    for error_index, item in enumerate(errors):
        error_path = f"{path}.errors[{error_index}]"
        error_fields = {"code", "message"}
        if mode == "decode" and "details" in item:
            error_fields.add("details")
        _exact_fields(error_path, item, error_fields)
        parsed_errors.append(
            {
                "code": _string(f"{error_path}.code", item["code"]),
                "message": _string(f"{error_path}.message", item["message"]),
            }
        )
        if "details" in item:
            details_path = f"{error_path}.details"
            details = _object(details_path, item["details"])
            _exact_fields(details_path, details, {"search"})
            parsed_errors[-1]["details"] = {
                "search": _parse_search(f"{details_path}.search", details["search"])
            }
    metrics_value = raw["metrics"]
    if status == CaseState.OK and metrics_value is None:
        raise ResultSchemaError(f"{path}.metrics: required for ok case")
    metrics = None
    if metrics_value is not None:
        metrics = (
            _parse_prefill_metrics(f"{path}.metrics", metrics_value)
            if mode == "prefill"
            else _parse_decode_metrics(f"{path}.metrics", metrics_value)
        )
    runs = _count(f"{path}.runs", raw["runs"])
    if mode == "prefill" and status == CaseState.OK:
        try:
            validate_concurrent_prefill_metrics(identity, metrics_value, runs)
        except ValueError as error:
            raise ResultSchemaError(f"{path}.metrics: {error}") from error
    if mode == "decode":
        if runs != identity["measure_runs"] or (
            metrics is not None and len(metrics.measurements) != runs
        ):
            raise ResultSchemaError(
                f"{path}.runs: must match planned and measured rounds"
            )
        policy = identity["batch_policy"]
        search = metrics.search if metrics is not None else None
        if status == CaseState.TARGET_TPOT_NOT_MET:
            if metrics is not None:
                raise ResultSchemaError(
                    f"{path}.metrics: failed search cannot have formal metrics"
                )
            if not parsed_errors or "details" not in parsed_errors[0]:
                raise ResultSchemaError(
                    f"{path}.errors[0].details.search: required for failed search"
                )
            search = parsed_errors[0]["details"]["search"]
            if search["outcome"] != "no_satisfying_candidate":
                raise ResultSchemaError(
                    f"{path}.errors[0].details.search.outcome: expected failed search"
                )
        if search is not None:
            if policy["strategy"] != "target_tpot" or any(
                search[name] != policy[name]
                for name in ("target_tpot_ms", "max_size", "candidate_step")
            ):
                raise ResultSchemaError(
                    f"{path}: search must match identity.batch_policy"
                )
        if metrics is not None:
            if status == CaseState.OK and metrics.failed_requests:
                raise ResultSchemaError(
                    f"{path}.status: ok requires zero failed requests"
                )
            batch = metrics.global_batch_size
            if (
                metrics.workload_kind != identity["workload_kind"]
                or list(metrics.input_lengths) != identity["input_lengths"][:batch]
                or metrics.output_length != identity["output_length"]
            ):
                raise ResultSchemaError(
                    f"{path}.metrics: selected shape must match identity"
                )
            if metrics.workload_kind == "grid" and metrics.distribution is not None:
                raise ResultSchemaError(
                    f"{path}.metrics.distribution: expected null for grid"
                )
            if policy["strategy"] == "fixed":
                if batch != policy["size"] or search is not None:
                    raise ResultSchemaError(
                        f"{path}.metrics: fixed batch must match policy without search"
                    )
            elif (
                search is None
                or search["outcome"] != "found"
                or search["best_batch_size"] != batch
            ):
                raise ResultSchemaError(
                    f"{path}.metrics.search: must describe selected best batch"
                )
            if policy["strategy"] == "target_tpot":
                if (
                    metrics.latency_ms["tpot"] > policy["target_tpot_ms"]
                    or metrics.failed_requests != 0
                ):
                    raise ResultSchemaError(
                        f"{path}.metrics: selected batch must satisfy target with full success"
                    )
                selected_observation = next(
                    observation
                    for observation in search["observations"]
                    if observation["batch_size"] == batch
                )
                selected_values = {
                    "tpot_ms": metrics.latency_ms["tpot"],
                    "success_rate": 1.0,
                    "throughput_tokens_per_second": metrics.throughput_tokens_per_second,
                }
                for name, measured_value in selected_values.items():
                    observed_value = selected_observation.get(name)
                    if observed_value is not None and not math.isclose(
                        observed_value, measured_value
                    ):
                        raise ResultSchemaError(
                            f"{path}.metrics.search: selected observation {name} must match formal metrics"
                        )
    return CaseResult(
        case_id=_string(f"{path}.case_id", raw["case_id"]),
        identity=copy.deepcopy(dict(identity)),
        status=status,
        runs=runs,
        errors=tuple(parsed_errors),
        metrics=metrics,
    )


def parse_result_envelope(value: Any) -> ResultEnvelope:
    raw = _object("result", value)
    fields = {
        "schema_version",
        "run_id",
        "mode",
        "backend",
        "status",
        "config_fingerprint",
        "workload_fingerprint",
        "model",
        "summary",
        "cases",
    }
    _exact_fields("", raw, fields)
    if raw["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ResultSchemaError("schema_version: unsupported result schema")
    mode = raw["mode"]
    if mode not in {"prefill", "decode"}:
        raise ResultSchemaError("mode: expected prefill or decode")
    backend = raw["backend"]
    # Both modes accept the aone backend: a gang run reports through the same
    # envelope as a local run (single-machine decode+aone never reaches one).
    if backend not in {"local", "aone"}:
        raise ResultSchemaError("backend: expected local or aone")
    try:
        status = RunState(raw["status"])
    except ValueError as error:
        raise ResultSchemaError("status: unknown run state") from error
    if status not in {
        RunState.COMPLETED,
        RunState.PARTIAL,
        RunState.FAILED,
        RunState.INTERRUPTED,
    }:
        raise ResultSchemaError("status: result requires a terminal run state")
    for name in ("config_fingerprint", "workload_fingerprint"):
        if not isinstance(raw[name], str) or not _FINGERPRINT.fullmatch(raw[name]):
            raise ResultSchemaError(f"{name}: expected a lowercase SHA-256 fingerprint")
    model = _object("model", raw["model"])
    _exact_fields("model", model, {"label", "type", "checkpoint", "topology"})
    topology = _object("model.topology", model["topology"])
    _exact_fields("model.topology", topology, {"tp", "dp", "pp", "ep"})
    parsed_model = {
        "label": _string("model.label", model["label"]),
        "type": _string("model.type", model["type"]),
        "checkpoint": _string("model.checkpoint", model["checkpoint"]),
        "topology": {
            name: _positive_int(f"model.topology.{name}", topology[name])
            for name in ("tp", "dp", "pp", "ep")
        },
    }
    summary = _object("summary", raw["summary"])
    summary_fields = {"planned_cases", "successful_cases", "failed_cases"}
    if mode == "decode":
        summary_fields.add("profiling_failed_cases")
    _exact_fields("summary", summary, summary_fields)
    parsed_summary = {
        name: _count(f"summary.{name}", summary[name]) for name in summary_fields
    }
    cases_value = raw["cases"]
    if not isinstance(cases_value, list):
        raise ResultSchemaError("cases: expected an array")
    cases = tuple(
        parse_case_result(index, mode, item) for index, item in enumerate(cases_value)
    )
    if parsed_summary["planned_cases"] != len(cases):
        raise ResultSchemaError("summary.planned_cases: must equal case count")
    if parsed_summary["successful_cases"] + parsed_summary["failed_cases"] != len(
        cases
    ):
        raise ResultSchemaError("summary: successful plus failed must equal case count")
    if mode == "decode":
        successful = sum(case.status == CaseState.OK for case in cases)
        profiling_failed = sum(
            case.metrics is not None and case.metrics.profiling["status"] == "failed"
            for case in cases
        )
        if parsed_summary["successful_cases"] != successful:
            raise ResultSchemaError(
                "summary.successful_cases: must match case outcomes"
            )
        if parsed_summary["profiling_failed_cases"] != profiling_failed:
            raise ResultSchemaError(
                "summary.profiling_failed_cases: must match profile outcomes"
            )
    return ResultEnvelope(
        schema_version=RESULT_SCHEMA_VERSION,
        run_id=_string("run_id", raw["run_id"]),
        mode=mode,
        backend=backend,
        status=status,
        config_fingerprint=raw["config_fingerprint"],
        workload_fingerprint=raw["workload_fingerprint"],
        model=parsed_model,
        summary=parsed_summary,
        cases=cases,
    )
