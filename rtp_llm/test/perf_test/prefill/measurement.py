"""Versioned semantics for prefill measurement identities."""

import math
import statistics
from typing import Any, Mapping

PREFILL_MEASUREMENT_CONTRACT: str = "concurrent_batch_v1"


def validate_concurrent_prefill_metrics(
    identity: Mapping[str, Any], metrics: Mapping[str, Any], runs: int
) -> None:
    """Validate persisted concurrent rounds; legacy identities remain readable."""
    contract = identity.get("measurement_contract")
    grouped = contract == "concurrent_grouped_batch_v1"
    if contract not in (PREFILL_MEASUREMENT_CONTRACT, "concurrent_grouped_batch_v1"):
        return

    def positive_number(value: Any) -> bool:
        return type(value) in (int, float) and math.isfinite(value) and value > 0

    batch = identity.get("batch_size")
    if type(batch) is not int or batch <= 0 or type(runs) is not int or runs <= 0:
        raise ValueError("concurrent prefill requires positive batch size and runs")
    planned_runs = identity.get("measure_runs")
    if type(planned_runs) is not int or planned_runs <= 0 or runs != planned_runs:
        raise ValueError("concurrent prefill runs must match planned measure_runs")
    geometry = []
    if grouped:
        from rtp_llm.test.perf_test.prefill.prompts import _normalize_grouped_case

        normalized = _normalize_grouped_case(dict(identity))
        if normalized["prefix_policy"] != "independent":
            raise ValueError("grouped concurrent prefill requires independent prefixes")
        for group_index, group in enumerate(normalized["request_groups"]):
            geometry.extend(
                [(group["input_len"], group["cache_len"], group_index)] * group["count"]
            )
    expected = (
        sum(item[1] for item in geometry) if grouped else identity.get("cache_len")
    )
    input_length = (
        sum(item[0] for item in geometry) if grouped else identity.get("input_len")
    )
    if (
        type(expected) is not int
        or expected < 0
        or metrics.get("input_length") != input_length
        or metrics.get("expected_reuse_length") != expected
        or metrics.get("observed_reuse_length") != expected
    ):
        raise ValueError("concurrent prefill lengths must match identity")
    measurements = metrics.get("measurements")
    if not isinstance(measurements, (list, tuple)) or len(measurements) != batch * runs:
        raise ValueError(
            "concurrent prefill requires exactly batch size members per round"
        )
    seen = set()
    elapsed_by_run = {}
    latencies = []
    round_latencies = {}
    for member in measurements:
        if not isinstance(member, Mapping):
            raise ValueError("concurrent prefill measurement must be an object")
        run = member.get("run")
        index = member.get("request_index")
        if (
            type(run) is not int
            or not 0 <= run < runs
            or type(index) is not int
            or not 0 <= index < batch
            or (run, index) in seen
        ):
            raise ValueError(
                "concurrent prefill requires unique planned round/member indices"
            )
        seen.add((run, index))
        if grouped:
            length, cached, group_index = geometry[index]
            for field, value in (
                ("input_length", length),
                ("expected_reuse_length", cached),
                ("compute_length", length - cached),
                ("group_index", group_index),
            ):
                if type(member.get(field)) is not int or member[field] != value:
                    raise ValueError(
                        "grouped prefill member geometry must match identity"
                    )
        elapsed = member.get("round_elapsed_ms")
        latency = member.get("latency_ms")
        reuse = member.get("observed_reuse_length")
        if (
            member.get("measurement_contract") != contract
            or type(reuse) is not int
            or reuse != (geometry[index][1] if grouped else expected)
            or not positive_number(latency)
            or not positive_number(elapsed)
        ):
            raise ValueError(
                "concurrent prefill member has invalid marker, reuse, or timing"
            )
        if run in elapsed_by_run and elapsed_by_run[run] != elapsed:
            raise ValueError(
                "concurrent prefill round elapsed must be identical for all members"
            )
        elapsed_by_run[run] = elapsed
        latencies.append(latency)
        round_latencies[run] = max(round_latencies.get(run, 0), latency)
    compute = metrics.get("compute_length")
    if not positive_number(compute):
        raise ValueError("concurrent prefill compute length must be positive")
    if grouped and compute != input_length - expected:
        raise ValueError("grouped prefill compute must equal aggregate uncached length")
    throughput = statistics.mean(
        (1 if grouped else batch) * compute * 1000.0 / elapsed_by_run[run]
        for run in range(runs)
    )
    for field, expected_value in (
        (
            "latency_ms",
            statistics.median(list(round_latencies.values()) if grouped else latencies),
        ),
        ("throughput_tokens_per_second", throughput),
    ):
        actual = metrics.get(field)
        if not positive_number(actual) or not math.isclose(actual, expected_value):
            raise ValueError(f"concurrent prefill {field} must match raw measurements")
