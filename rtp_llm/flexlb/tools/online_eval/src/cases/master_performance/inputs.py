"""Validate the case gate inputs and their declared metric bindings."""

from cases.metric_inputs import metric_fields
from input_contract import finite_number as finite
from cases.numeric_parameters import (
    SOURCE_NUMBERS, OUTPUT_DISTRIBUTION_NUMBERS, JAVA_FLOW_NUMBERS, CAPTURE_NUMBERS,
    NONNEGATIVE, FRACTION, COUNT, OFFSET,
    number_fields,
)

OBSERVATION_FIELDS = frozenset({"sample_s", "max_gap_s"})
CHECK_FIELDS = frozenset({
    "qps_tolerance", "min_requests", "max_pacing_lag_ms", "min_input_tps",
    "min_output_tps", "min_goodput_rps", "min_slo_fraction", "max_error_rate",
    "max_ttft_p99_ms", "max_e2e_p99_ms", "max_tpot_p99_ms", "slo_ttft_ms",
    "slo_e2e_ms", "slo_tpot_ms", "max_inflight_growth_rps",
})


NUMERIC_PARAMETERS = {
    **SOURCE_NUMBERS,
    **OUTPUT_DISTRIBUTION_NUMBERS,
    **JAVA_FLOW_NUMBERS,
    **CAPTURE_NUMBERS,
    **number_fields(NONNEGATIVE,
        'checks.e2e.expected',
        'checks.engine_tps.rtp_llm_context_tps.expected',
        'checks.engine_tps.rtp_llm_context_tps_with_cache.expected',
        'checks.engine_tps.rtp_llm_generate_tps.expected',
        'checks.goodput.expected',
        'checks.inflight_growth.expected',
        'checks.input_tps.expected',
        'checks.output_tps.expected',
        'checks.pacing_lag.expected',
        'checks.tpot.expected',
        'checks.ttft.expected',
        'observation.max_gap_s',
        'observation.sample_s',
        'traffic.client.playback.ramp_up_seconds',
    ),
    **number_fields(FRACTION,
        'checks.errors.expected',
        'checks.offered_load.expected',
        'checks.slo_fraction.expected',
    ),
    **number_fields(COUNT,
        'checks.requests.expected',
        'observation.slo.e2e_ms',
        'observation.slo.tpot_ms',
        'observation.slo.ttft_ms',
        'traffic.source.parameters.max_input_tokens',
    ),
    **number_fields(OFFSET,
        'observation.windows.measurement.from.offset_s',
        'observation.windows.measurement.until.offset_s',
    ),
}


def engine_tps(spec, bounds=None):
    bindings = metric_fields(spec)
    roles = {binding["metric"]: binding["labels"].get("role") for binding in bindings.values()}
    if (
        len(roles) != 3
        or any(
            v not in ("prefill", "decode")
            for v in roles.values()
        )
        or list(roles.values()).count("prefill") != 2
        or list(roles.values()).count("decode") != 1
        or (bounds is not None and set(bounds) != set(roles))
    ):
        raise ValueError("engine TPS floors must match YAML metric fields")
    if len(bindings) != 3 or any(set(binding["labels"]) != {"role"} for binding in bindings.values()):
        raise ValueError("engine TPS fields require one role selector each")
    return spec


def engine_roles(spec, bounds=None):
    engine_tps(spec, bounds)
    return {binding["metric"]: binding["labels"]["role"] for binding in spec["fields"].values()}


# Public comparisons bind exact metrics, units and cohorts. The analyzer keeps
# its compact numerical criteria; compilation owns this projection.
RULES = {
    'offered_load': ('qps_tolerance', 'offered_qps_deviation', 'le', 'ratio', 'measurement'),
    'requests': ('min_requests', 'cohort_requests', 'ge', 'requests', 'measurement'),
    'pacing_lag': ('max_pacing_lag_ms', 'pacing_lag_max_ms', 'le', 'ms', 'measurement'),
    'input_tps': ('min_input_tps', 'input_tps', 'ge', 'tokens/s', 'measurement'),
    'output_tps': ('min_output_tps', 'output_tps', 'ge', 'tokens/s', 'measurement'),
    'goodput': ('min_goodput_rps', 'goodput_rps', 'ge', 'requests/s', 'measurement'),
    'slo_fraction': ('min_slo_fraction', 'slo_fraction', 'ge', 'ratio', 'measurement'),
    'errors': ('max_error_rate', 'error_rate', 'le', 'ratio', 'full_run'),
    'ttft': ('max_ttft_p99_ms', 'ttft_p99_ms', 'le', 'ms', 'measurement'),
    'e2e': ('max_e2e_p99_ms', 'e2e_p99_ms', 'le', 'ms', 'measurement'),
    'tpot': ('max_tpot_p99_ms', 'tpot_p99_ms', 'le', 'ms', 'measurement'),
    'inflight_growth': ('max_inflight_growth_rps', 'inflight_growth_rps', 'le', 'requests/s', 'measurement'),
}


def compile_checks(case, checks, inputs):
    from cases.inputs import fields
    from cases.check_inputs import metric_criterion, compile_metric_checks

    bindings = metric_fields(engine_tps(inputs))
    fields(checks, set(RULES) | {'engine_tps'}, 'parameters.checks')
    result = {}
    result.update(compile_metric_checks(case, checks, RULES, namespace='performance_gate'))
    fields(checks['engine_tps'], set(bindings), 'parameters.checks.engine_tps')
    floors, overrides = {}, {}
    for name, binding in bindings.items():
        identity = binding['metric']
        floor, profiles = metric_criterion(case, checks['engine_tps'][name],
            metric='performance_gate/'+identity.split('/')[1]+'_scrape_engine_mean',
            unit='tokens/s', window='measurement', op='ge',
            path='parameters.checks.engine_tps.'+name, profiles=True)
        floors[identity] = floor
        for profile, value in profiles.items():
            overrides.setdefault(profile, {})[identity] = value
    result['engine_tps'] = floors
    if overrides:
        result['engine_tps_by_profile'] = overrides
    return result


def observation_contract(data):
    from cases.windows import anchored_windows
    from runtime.observation import capture_limits
    bounds = anchored_windows(data["windows"], {
        "measurement": "observation_start",
    })["measurement"]
    if bounds["from"] < 0:
        raise ValueError("measurement cannot start before observation")
    capture_limits(data["capture"], "parameters.observation.capture")
    return dict(warmup_s=bounds["from"], measure_s=bounds["until"]-bounds["from"])


NUMERIC = OBSERVATION_FIELDS | CHECK_FIELDS | {"qps", "warmup_s", "measure_s"}


def validate(criteria, gate_input=None):
    if not isinstance(criteria, dict) or set(criteria) - {"engine_tps", "engine_tps_by_profile"} != NUMERIC:
        raise ValueError(
            "performance criteria must explicitly supply every contract field"
        )
    for k in NUMERIC:
        if not finite(criteria[k]) or criteria[k] < 0:
            raise ValueError(k + " must be finite and nonnegative")
    for k in (
        "qps",
        "measure_s",
        "min_requests",
        "min_input_tps",
        "min_output_tps",
        "min_goodput_rps",
        "slo_ttft_ms",
        "slo_e2e_ms",
        "slo_tpot_ms",
        "max_ttft_p99_ms",
        "max_e2e_p99_ms",
        "max_tpot_p99_ms",
    ):
        if criteria[k] <= 0:
            raise ValueError(k + " must be positive")
    if type(criteria["min_requests"]) is not int:
        raise ValueError("min_requests must be an integer")
    for k in ("qps_tolerance", "min_slo_fraction", "max_error_rate"):
        if criteria[k] > 1:
            raise ValueError(k + " must be a fraction")
    if not 0 < criteria["sample_s"] <= criteria["max_gap_s"] <= criteria["measure_s"]:
        raise ValueError("invalid sample coverage budget")
    if criteria["max_error_rate"] != 0:
        raise ValueError("performance gate requires 100% request success")
    if "engine_tps" in criteria:
        bounds = criteria["engine_tps"]
        if (not isinstance(bounds, dict) or not bounds
                or any(not finite(v) or v <= 0 for v in bounds.values())):
            raise ValueError("engine_tps requires positive absolute floors")
        engine_tps(gate_input, bounds)
    overrides = criteria.get("engine_tps_by_profile", {})
    if (not isinstance(overrides, dict)
            or any(type(k) is not str or not k for k in overrides)):
        raise ValueError("engine_tps_by_profile requires registered profiles")
    for bounds in overrides.values():
        if (not isinstance(bounds, dict) or not bounds
                or "engine_tps" not in criteria or set(bounds) - set(criteria["engine_tps"])
                or any(not finite(v) or v <= 0 for v in bounds.values())):
            raise ValueError("invalid profile engine TPS floors")
    return criteria


def for_profile(criteria, profile, gate_input=None):
    """Freeze effective floors before observation and evidence collection."""
    import copy
    result = copy.deepcopy(validate(criteria, gate_input))
    overrides = result.pop("engine_tps_by_profile", {})
    if profile in overrides:
        result["engine_tps"].update(overrides[profile])
    return validate(result, gate_input)
