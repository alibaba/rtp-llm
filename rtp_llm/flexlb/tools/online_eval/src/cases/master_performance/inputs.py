"""Validate the case gate inputs and their declared metric bindings."""

from case_api.metric_inputs import metric_fields
from case_api.criteria import CriteriaContract
from input_contract import finite_number as finite
from case_api.numeric_parameters import OUTPUT_DISTRIBUTION_NUMBERS
from case_api.numeric_parameters import CAPTURE_NUMBERS
from case_api.numeric_parameters import NONNEGATIVE
from case_api.numeric_parameters import FRACTION
from case_api.numeric_parameters import COUNT
from case_api.numeric_parameters import OFFSET

OBSERVATION_INPUTS = {'sample_s': NONNEGATIVE, 'max_gap_s': NONNEGATIVE}
SLO_INPUTS = dict.fromkeys(('ttft_ms', 'e2e_ms', 'tpot_ms'), COUNT)


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
from case_api.check_inputs import CheckSet, CheckSpec

CHECKS = CheckSet('performance_gate', {
    'offered_load': CheckSpec('qps_tolerance', 'offered_qps_deviation', 'le', 'ratio', ('measurement',), number=FRACTION),
    'requests': CheckSpec('min_requests', 'cohort_requests', 'ge', 'requests', ('measurement',), number=COUNT),
    'pacing_lag': CheckSpec('max_pacing_lag_ms', 'pacing_lag_max_ms', 'le', 'ms', ('measurement',), number=NONNEGATIVE),
    'input_tps': CheckSpec('min_input_tps', 'input_tps', 'ge', 'tokens/s', ('measurement',), number=NONNEGATIVE),
    'output_tps': CheckSpec('min_output_tps', 'output_tps', 'ge', 'tokens/s', ('measurement',), number=NONNEGATIVE),
    'goodput': CheckSpec('min_goodput_rps', 'goodput_rps', 'ge', 'requests/s', ('measurement',), number=NONNEGATIVE),
    'slo_fraction': CheckSpec('min_slo_fraction', 'slo_fraction', 'ge', 'ratio', ('measurement',), number=FRACTION),
    'errors': CheckSpec('max_error_rate', 'error_rate', 'le', 'ratio', ('full_run',), number=FRACTION),
    'ttft': CheckSpec('max_ttft_p99_ms', 'ttft_p99_ms', 'le', 'ms', ('measurement',), number=NONNEGATIVE),
    'e2e': CheckSpec('max_e2e_p99_ms', 'e2e_p99_ms', 'le', 'ms', ('measurement',), number=NONNEGATIVE),
    'tpot': CheckSpec('max_tpot_p99_ms', 'tpot_p99_ms', 'le', 'ms', ('measurement',), number=NONNEGATIVE),
    'inflight_growth': CheckSpec('max_inflight_growth_rps', 'inflight_growth_rps', 'le', 'requests/s', ('measurement',), number=NONNEGATIVE),
})
RULES = CHECKS.rules

CRITERIA = CriteriaContract({
    'observation': OBSERVATION_INPUTS,
    'windows': dict.fromkeys(('warmup_s', 'measure_s'), NONNEGATIVE),
    'checks': CHECKS.criteria_fields,
    'slo': {'slo_' + name: NONNEGATIVE for name in SLO_INPUTS},
    'runtime': {'qps': NONNEGATIVE},
}, optional={'engine_tps': None, 'engine_tps_by_profile': None})


def compile_checks(case, checks, inputs):
    from case_api.inputs import fields
    from case_api.check_inputs import metric_criterion

    bindings = metric_fields(engine_tps(inputs))
    fields(checks, set(RULES) | {'engine_tps'}, 'parameters.checks')
    result = {}
    result.update(CHECKS.compile(case, checks))
    fields(checks['engine_tps'], set(bindings), 'parameters.checks.engine_tps')
    floors, overrides = {}, {}
    for name, binding in bindings.items():
        identity = binding['metric']
        floor, profiles = metric_criterion(case, checks['engine_tps'][name],
            metric='performance_gate/'+identity.split('/')[1]+'_scrape_engine_mean',
            unit='tokens/s', windows=('measurement',), op='ge',
            path='parameters.checks.engine_tps.'+name, profiles=True)
        floors[identity] = floor
        for profile, value in profiles.items():
            overrides.setdefault(profile, {})[identity] = value
    result['engine_tps'] = floors
    if overrides:
        result['engine_tps_by_profile'] = overrides
    return result


def observation_contract(data):
    from case_api.windows import anchored_windows
    from runtime.observation import capture_limits
    bounds = anchored_windows(data["windows"], {
        "measurement": "observation_start",
    })["measurement"]
    if bounds["from"] < 0:
        raise ValueError("measurement cannot start before observation")
    capture_limits(data["capture"], "parameters.observation.capture")
    return dict(warmup_s=bounds["from"], measure_s=bounds["until"]-bounds["from"])


NUMERIC = CRITERIA.fields


def validate(criteria, gate_input=None):
    CRITERIA.read(criteria)
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
    return effective_profile(INPUTS.admit_projection(criteria, gate_input), profile)


def effective_profile(criteria, profile):
    """Freeze effective floors before observation and evidence collection."""
    import copy
    result = copy.deepcopy(criteria)
    overrides = result.pop("engine_tps_by_profile", {})
    if profile in overrides:
        result["engine_tps"].update(overrides[profile])
    return result


from case_api.input_contract import InputContract, section_fields
from case_api.traffic_inputs import java_flow_inputs

INPUTS = InputContract.from_fields({
    'traffic': java_flow_inputs(numbers={
        **OUTPUT_DISTRIBUTION_NUMBERS,
        'traffic.client.playback.ramp_up_seconds': NONNEGATIVE,
        'traffic.source.parameters.max_input_tokens': COUNT,
    }),
    'procedure': {'analysis_timeout_s': NONNEGATIVE},
    'observation': {
        **OBSERVATION_INPUTS, 'inputs': None,
        'windows': {'measurement': {boundary: {'offset_s': OFFSET} for boundary in ('from', 'until')}},
        'capture': section_fields('observation', {'capture'}, CAPTURE_NUMBERS)['capture'],
    },
    'analysis': {'slo': SLO_INPUTS},
    'checks': {**CHECKS.input_fields, 'engine_tps': {
        name: {'expected': NONNEGATIVE} for name in
        ('rtp_llm_context_tps', 'rtp_llm_context_tps_with_cache', 'mock_decode_wall_tps')
    }},
}, projection=validate)
