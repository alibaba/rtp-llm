"""Validate the case gate inputs and their declared metric bindings."""

from cases.cache_scale_in.constants import SHARED_WARMUP_AFFINITY_MS

from input_contract import NumberRule
from case_api.numeric_parameters import CAPTURE_NUMBERS
from case_api.numeric_parameters import FRACTION
from case_api.numeric_parameters import NONNEGATIVE
from case_api.numeric_parameters import COUNT
from case_api.numeric_parameters import OFFSET
from scenario.parameters import validate_fields
from case_api.metric_inputs import metric_fields
from case_api.criteria import CriteriaContract

PROCEDURE_INPUTS = {
    'target_p': COUNT, 'removal_mode': None, 'drain_timeout_ms': COUNT,
    'topology_timeout_s': NONNEGATIVE, 'warmup_min_s': NONNEGATIVE,
    'post_affinity_ms': NumberRule(integer=True, minimum=0, maximum=9223372036854775807),
    'analysis_timeout_s': NONNEGATIVE, 'startup_timeout_s': NONNEGATIVE,
}
OBSERVATION_INPUTS = dict.fromkeys(('warmup_timeout_s', 'sample_s', 'window_s', 'step_s', 'max_gap_s'), NONNEGATIVE)
PROCEDURE_CRITERIA = {name: rule for name, rule in PROCEDURE_INPUTS.items()
                      if name not in {'analysis_timeout_s', 'startup_timeout_s'}}
COLLAPSE_INPUTS = dict.fromkeys(('absolute_min_hit', 'max_drop', 'sustain_s'), NONNEGATIVE)


ENGINE_FIELDS = {"worker_status_qps": "requests/s", "worker_status_rt_us": "us",
        "running": "requests", "waiting": "requests", "cache_evictions": "events",
        "prefill_ms_avg": "ms", "prefill_batches": "batches", "prefill_batch_requests": "requests",
        "hit_tokens_total": "tokens", "context_tokens_total": "tokens", "context_requests_total": "requests",
        "cache_key_hits": "keys", "cache_keys_requested": "keys", "admission_open": "boolean",
        "admitted_rpcs_total": "requests", "rejected_rpcs_total": "requests"}


def engine_counters(spec):
    from cases.cache_scale_in.analysis import COUNTERS

    required = set(COUNTERS) | {"waiting", "running", "cache_evictions", "prefill_ms_avg", "admission_open"}
    fields = metric_fields(spec)
    if not required <= set(fields):
        raise ValueError("parameters.observation.inputs.engine_counters: missing or invalid metric bindings; missing required fields " + str(sorted(required - set(fields))))
    if (
        set(fields) - set(ENGINE_FIELDS)
        or any(
            v["labels"] != {"role": "prefill"}
            for v in fields.values()
        )
        or len({v["metric"] for v in fields.values()}) != len(fields)
    ):
        raise ValueError("engine counter input has missing or invalid metric bindings")
    return spec


from case_api.check_inputs import CheckSet, CheckSpec

CHECKS = CheckSet('cache_gate', {
    'offered_load': CheckSpec('qps_tolerance', 'offered_qps_deviation', 'le', 'ratio', ('baseline', 'post'), number=FRACTION),
    'baseline_hit': CheckSpec('baseline_min_hit', 'baseline_min_half_hit', 'ge', 'ratio', ('baseline',), number=FRACTION),
    'baseline_stability': CheckSpec('baseline_max_spread', 'baseline_half_spread', 'le', 'ratio', ('baseline',), number=FRACTION),
    'completed': CheckSpec('min_completed', 'min_window_completed', 'ge', 'requests', ('baseline', 'post'), number=COUNT),
})
RULES = CHECKS.rules

CRITERIA = CriteriaContract({
    'procedure': PROCEDURE_CRITERIA,
    'observation': OBSERVATION_INPUTS,
    'windows': dict.fromkeys(('baseline_s', 'observe_s'), NONNEGATIVE),
    'checks': {**CHECKS.criteria_fields, **COLLAPSE_INPUTS, 'expected_behavior': None},
    'runtime': {'qps': NONNEGATIVE},
})


def compile_checks(case, checks, policy):
    from case_api.inputs import fields
    from case_api.check_inputs import metric_criterion

    fields(checks, set(RULES) | {'collapse'}, 'parameters.checks')
    fields(policy, {'absolute_min_hit', 'max_drop', 'sustain_s'}, 'parameters.analysis.collapse')
    result = dict(policy)
    result.update(CHECKS.compile(case, checks))
    expected, _ = metric_criterion(case, checks['collapse'], metric='cache_gate/collapse_detected',
        unit='boolean', windows=('post',), op='eq', path='parameters.checks.collapse')
    if type(expected) is not int or expected not in (0, 1):
        raise ValueError('cache collapse expectation must be 0 or 1')
    result['expected_behavior'] = 'collapse' if expected else 'stable'
    return result


def observation_contract(data):
    from case_api.windows import anchored_windows
    from runtime.observation import capture_limits
    windows = anchored_windows(data["windows"], {
        "baseline": "baseline_ready", "post": "target_topology_observed",
    })
    base, post = windows["baseline"], windows["post"]
    if base["until"] != 0 or base["from"] >= 0 or post["from"] != 0:
        raise ValueError("cache windows must end at baseline readiness and start at target readiness")
    capture_limits(data["capture"], "parameters.observation.capture")
    return dict(baseline_s=-base["from"], observe_s=post["until"])


FIELDS = CRITERIA.fields | {"flow"}


def validate_criteria(params, plan):
    p = validate_fields(
        params, plan, FIELDS | {"gate_input"}, FIELDS | {"gate_input"}
    )
    from cases.cache_scale_in.inputs import engine_counters
    engine_counters(p["gate_input"])
    if p["removal_mode"] not in ("graceful", "abrupt"):
        raise ValueError("removal_mode must be graceful or abrupt")
    plan.reference(p["flow"], "java_flow")
    if plan.env.discovery != "discovery_file":
        raise ValueError("scale-in requires dynamic discovery_file")
    CRITERIA.read({key: value for key, value in p.items() if key in CRITERIA.fields})
    if not 0 <= p["warmup_min_s"] <= p["warmup_timeout_s"]:
        raise ValueError("warmup_min_s must fit the warmup budget")
    if type(p["post_affinity_ms"]) is not int or not 0 <= p["post_affinity_ms"] <= 9223372036854775807:
        raise ValueError("post_affinity_ms must be a nonnegative signed long")
    if p["expected_behavior"] not in ("stable", "collapse"):
        raise ValueError("unknown expected cache behavior")
    if p["expected_behavior"] == "collapse" and p["post_affinity_ms"] != 0:
        raise ValueError("collapse control requires post_affinity_ms=0")
    if not plan.env.master_test_control or plan.env.master_layout != "single":
        raise ValueError("affinity intervention requires one opted-in Master")
    for k in ("target_p", "min_completed"):
        if type(p[k]) is not int or p[k] < 1:
            raise ValueError(k + " must be a positive integer")
    if not 1 <= p["target_p"] < plan.env.n_prefill <= 512:
        raise ValueError("scale-in requires fewer target P and at most 512 initial P")
    for k in (
        "qps_tolerance",
        "baseline_min_hit",
        "baseline_max_spread",
        "absolute_min_hit",
        "max_drop",
    ):
        if p[k] > 1:
            raise ValueError(k + " must be a fraction")
    if not 0 < p["sample_s"] <= p["step_s"] <= p["window_s"] <= p["baseline_s"] / 2:
        raise ValueError("sampling/window/baseline durations are inconsistent")
    if p["max_gap_s"] < p["sample_s"] or p["warmup_timeout_s"] < p["baseline_s"]:
        raise ValueError("insufficient warmup or sample gap budget")
    if p["observe_s"] < p["window_s"] + p["sustain_s"] or p["qps"] <= 0:
        raise ValueError("observation cannot cover sustained collapse")
    if (type(p["drain_timeout_ms"]) is not int
            or not 0 <= p["drain_timeout_ms"] <= 30000
            or (p["removal_mode"] == "graceful" and p["drain_timeout_ms"] == 0)
            or p["topology_timeout_s"] <= 0):
        raise ValueError("removal must have bounded drain and topology budgets")

    for profile in plan.profiles:
        resolved = plan.env.resolved_config(profile)
        if resolved["router"]["roles"]["prefill"]["cacheAffinity"]["maxExtraTtftMs"] != SHARED_WARMUP_AFFINITY_MS:
            raise ValueError("shared warmup requires cache affinity 1000000000")
        stale_ms = resolved["workerRegistry"]["health"]["statusStaleAfterMs"]
        if p["topology_timeout_s"] * 1000 <= stale_ms:
            raise ValueError(
                "topology budget must exceed Master status staleness"
            )
    return p


from case_api.input_contract import InputContract, section_fields
from case_api.traffic_inputs import java_flow_inputs

INPUTS = InputContract.from_fields({
    'traffic': java_flow_inputs(max_requests=True),
    'procedure': PROCEDURE_INPUTS,
    'observation': {
        **OBSERVATION_INPUTS, 'inputs': None,
        'windows': {name: {boundary: {'offset_s': OFFSET} for boundary in ('from', 'until')}
                    for name in ('baseline', 'post')},
        'capture': section_fields('observation', {'capture'}, CAPTURE_NUMBERS)['capture'],
    },
    'analysis': {'collapse': COLLAPSE_INPUTS},
    'checks': {**CHECKS.input_fields, 'collapse': {'expected': NONNEGATIVE}},
})
