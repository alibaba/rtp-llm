"""Validate the case gate inputs and their declared metric bindings."""

from cases.metric_inputs import metric_fields

OBSERVATION_FIELDS = frozenset({"benchmark_id", "warmup_s", "measure_s", "sample_s", "max_gap_s"})
CHECK_FIELDS = frozenset({
    "qps_tolerance", "min_requests", "max_pacing_lag_ms", "min_input_tps",
    "min_output_tps", "min_goodput_rps", "min_slo_fraction", "max_error_rate",
    "max_ttft_p99_ms", "max_e2e_p99_ms", "max_tpot_p99_ms", "slo_ttft_ms",
    "slo_e2e_ms", "slo_tpot_ms", "max_inflight_growth_rps",
})


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
    from cases.check_inputs import metric_criterion

    bindings = metric_fields(engine_tps(inputs))
    fields(checks, set(RULES) | {'engine_tps'}, 'parameters.checks')
    result = {}
    for name, (key, metric, op, unit, window) in RULES.items():
        result[key], _ = metric_criterion(case, checks[name], metric='performance_gate/'+metric,
            unit=unit, window=window, op=op, path='parameters.checks.'+name)
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
