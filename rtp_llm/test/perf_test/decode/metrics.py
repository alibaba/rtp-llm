"""Validation and aggregation of raw per-request decode measurements."""

from __future__ import annotations

import copy
import math
import statistics
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Sequence

from rtp_llm.test.perf_test.decode.model import DecodeBatchMeasurement

_LATENCIES = ("wait", "prefill", "decode", "tpot", "total")


class _MeasurementError(ValueError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status


def _number(path: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _MeasurementError("invalid_timing", f"{path} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise _MeasurementError(
            "invalid_timing", f"{path} must be a non-negative finite number"
        )
    return parsed


def _identity(case: Mapping[str, Any]) -> Dict[str, Any]:
    from rtp_llm.test.perf_test.plan import freeze, thaw

    return thaw(
        freeze(
            {
                key: copy.deepcopy(value)
                for key, value in case.items()
                if key != "case_id"
            }
        )
    )


def _failure(
    case: Mapping[str, Any],
    status: str,
    message: str,
    search: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    error = {"code": status, "message": message}
    if search is not None:
        error["details"] = {"search": copy.deepcopy(dict(search))}
    return {
        "case_id": case["case_id"],
        "identity": _identity(case),
        "status": status,
        "runs": int(case["measure_runs"]),
        "errors": [error],
        "metrics": None,
    }


def _planned_lengths(selected: DecodeBatchMeasurement) -> List[int]:
    lengths = list(selected.input_lengths)
    if selected.batch_size <= 0 or len(lengths) != selected.batch_size:
        raise _MeasurementError(
            "invalid_shape", "selected input lengths do not match global batch"
        )
    return lengths


def validate_decode_measurement(
    case: Mapping[str, Any],
    selected: DecodeBatchMeasurement,
    *,
    batch_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Validate a candidate against its planned prefix and normalize its rounds."""
    expected_lengths = _planned_lengths(selected)
    expected_batch = selected.batch_size if batch_size is None else batch_size
    if selected.batch_size != expected_batch or expected_lengths != list(
        case["input_lengths"][:expected_batch]
    ):
        raise _MeasurementError(
            "invalid_shape", "measurement input shape does not match candidate"
        )
    measurements = selected.measurements
    expected_runs = int(case["measure_runs"])
    if (
        not isinstance(measurements, (list, tuple))
        or len(measurements) != expected_runs
    ):
        raise _MeasurementError(
            "invalid_shape", "measurement count does not match planned runs"
        )
    output_length = int(case["output_length"])
    normalized_measurements = []
    for run_index, measurement in enumerate(measurements):
        if not isinstance(measurement, Mapping):
            raise _MeasurementError(
                "invalid_shape", f"measurements[{run_index}] must be an object"
            )
        if set(measurement) != {
            "run",
            "requests",
            "throughput_tokens_per_second",
            "worker_allocations",
        }:
            raise _MeasurementError(
                "invalid_shape",
                "measurement requires exactly run, requests, throughput_tokens_per_second and worker_allocations",
            )
        if type(measurement["run"]) is not int or measurement["run"] != run_index:
            raise _MeasurementError(
                "invalid_shape", "measurement run indices must be contiguous"
            )
        requests = measurement.get("requests")
        if not isinstance(requests, list) or len(requests) != len(expected_lengths):
            raise _MeasurementError(
                "invalid_shape", "request count does not match global batch"
            )
        normalized_requests = []
        seen_indices = set()
        for request in requests:
            if not isinstance(request, Mapping):
                raise _MeasurementError(
                    "invalid_shape", "request outcome must be an object"
                )
            request_index = request.get("request_index")
            if type(request_index) is not int or request_index not in range(
                len(expected_lengths)
            ):
                raise _MeasurementError(
                    "invalid_shape", "request index is outside the global batch"
                )
            if request_index in seen_indices:
                raise _MeasurementError(
                    "invalid_shape", "request index appears more than once"
                )
            seen_indices.add(request_index)
            if (
                type(request.get("input_length")) is not int
                or type(request.get("output_length")) is not int
                or request.get("input_length") != expected_lengths[request_index]
                or request.get("output_length") != output_length
            ):
                raise _MeasurementError(
                    "invalid_shape", "request input/output shape does not match plan"
                )
            if type(request.get("success")) is not bool:
                raise _MeasurementError(
                    "invalid_shape", "request success must be boolean"
                )
            normalized_request = copy.deepcopy(dict(request))
            if request["success"] or "latency_ms" in request:
                latency = request.get("latency_ms")
                if not isinstance(latency, Mapping) or set(latency) != set(_LATENCIES):
                    raise _MeasurementError(
                        "invalid_timing",
                        "successful request requires all latency fields",
                    )
                normalized_request["latency_ms"] = {
                    name: _number(f"latency_ms.{name}", latency[name])
                    for name in _LATENCIES
                }
            normalized_requests.append(normalized_request)
        if seen_indices != set(range(len(expected_lengths))):
            raise _MeasurementError(
                "invalid_shape", "measurement does not cover every request"
            )
        normalized_measurement = copy.deepcopy(dict(measurement))
        allocations = measurement["worker_allocations"]
        if not isinstance(allocations, list) or any(
            not isinstance(worker, Mapping) for worker in allocations
        ):
            raise _MeasurementError(
                "invalid_shape", "worker allocations must be an array of objects"
            )
        normalized_measurement["requests"] = normalized_requests
        normalized_measurement["throughput_tokens_per_second"] = _number(
            "throughput_tokens_per_second",
            measurement.get("throughput_tokens_per_second"),
        )
        normalized_measurements.append(normalized_measurement)
    return normalized_measurements


def summarize_decode_measurements(
    measurements: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate validated rounds identically for producers and persisted readers."""
    requests = [request for run in measurements for request in run["requests"]]
    successful = [request for request in requests if request["success"]]
    return {
        "successful_requests": len(successful),
        "failed_requests": len(requests) - len(successful),
        "latency_ms": {
            name: (
                statistics.fmean(request["latency_ms"][name] for request in successful)
                if successful
                else 0.0
            )
            for name in _LATENCIES
        },
        "throughput_tokens_per_second": statistics.fmean(
            run["throughput_tokens_per_second"] for run in measurements
        ),
    }


def build_decode_case_result(
    case: Mapping[str, Any],
    *,
    selected: DecodeBatchMeasurement,
    topology: Mapping[str, int],
    search: Optional[Mapping[str, Any]] = None,
    profiling: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate raw runs and produce one common-schema decode case result."""
    try:
        expected_lengths = _planned_lengths(selected)
        normalized_measurements = validate_decode_measurement(case, selected)
        parsed_topology = dict(topology)
        if set(parsed_topology) != {"tp", "dp", "pp", "ep"} or any(
            type(value) is not int or value <= 0 for value in parsed_topology.values()
        ):
            raise _MeasurementError(
                "invalid_shape", "decode topology requires positive tp/dp/pp/ep"
            )
    except _MeasurementError as error:
        return _failure(case, error.status, str(error), search)

    requests = [
        request for run in normalized_measurements for request in run["requests"]
    ]
    failed = [request for request in requests if not request["success"]]
    distribution_details = None
    if case["workload_kind"] == "distribution":
        distribution_details = copy.deepcopy(dict(case.get("distribution") or {}))
        distribution_details["actual_sequence_lengths"] = list(expected_lengths)
    metrics = {
        "workload_kind": case["workload_kind"],
        "input_lengths": list(expected_lengths),
        "output_length": int(case["output_length"]),
        "global_batch_size": selected.batch_size,
        "topology": parsed_topology,
        **summarize_decode_measurements(normalized_measurements),
        "measurements": normalized_measurements,
        "search": None if search is None else copy.deepcopy(dict(search)),
        "distribution": distribution_details,
    }
    if profiling is not None:
        metrics["profiling"] = copy.deepcopy(dict(profiling))
    status = "request_failed" if failed else "ok"
    return {
        "case_id": case["case_id"],
        "identity": _identity(case),
        "status": status,
        "runs": len(normalized_measurements),
        "errors": [
            {
                "code": "request_failed",
                "message": str(request.get("error", "request failed")),
            }
            for request in failed
        ],
        "metrics": metrics,
    }
