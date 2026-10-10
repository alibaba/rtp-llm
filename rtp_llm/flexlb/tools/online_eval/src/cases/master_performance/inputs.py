"""Validate the case gate inputs and their declared metric bindings."""

import re


METRIC = re.compile(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*\Z")

OBSERVATION_FIELDS = frozenset({"benchmark_id", "warmup_s", "measure_s", "sample_s", "max_gap_s"})
CHECK_FIELDS = frozenset({
    "qps_tolerance", "min_requests", "max_pacing_lag_ms", "min_input_tps",
    "min_output_tps", "min_goodput_rps", "min_slo_fraction", "max_error_rate",
    "max_ttft_p99_ms", "max_e2e_p99_ms", "max_tpot_p99_ms", "slo_ttft_ms",
    "slo_e2e_ms", "slo_tpot_ms", "max_inflight_growth_rps",
})


def engine_tps(spec, bounds=None):
    if (
        not isinstance(spec, dict)
        or set(spec) != {"source", "metric_roles"}
        or spec["source"] != "metric_store"
    ):
        raise ValueError("engine TPS input requires metric_store and metric_roles")
    roles = spec["metric_roles"]
    if (
        not isinstance(roles, dict)
        or len(roles) != 3
        or any(
            not isinstance(k, str)
            or not METRIC.fullmatch(k)
            or v not in ("prefill", "decode")
            for k, v in roles.items()
        )
        or list(roles.values()).count("prefill") != 2
        or list(roles.values()).count("decode") != 1
        or (bounds is not None and set(bounds) != set(roles))
    ):
        raise ValueError("engine TPS floors must match YAML metric_roles")
    return spec
