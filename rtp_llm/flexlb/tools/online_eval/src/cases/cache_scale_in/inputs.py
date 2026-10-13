"""Validate the case gate inputs and their declared metric bindings."""

from cases.cache_scale_in.constants import SHARED_WARMUP_AFFINITY_MS

import math
from input_contract import finite_number, NumberRule
from case_api.numeric_parameters import SOURCE_NUMBERS
from case_api.numeric_parameters import JAVA_FLOW_NUMBERS
from case_api.numeric_parameters import CAPTURE_NUMBERS
from case_api.numeric_parameters import FRACTION
from case_api.numeric_parameters import NONNEGATIVE
from case_api.numeric_parameters import COUNT
from case_api.numeric_parameters import OFFSET
from case_api.numeric_parameters import number_fields
from scenario.parameters import validate_fields
from case_api.metric_inputs import metric_fields

PROCEDURE_FIELDS = frozenset({"target_p", "removal_mode", "drain_timeout_ms", "topology_timeout_s", "warmup_min_s", "post_affinity_ms"})
OBSERVATION_FIELDS = frozenset({"warmup_timeout_s", "sample_s",
                               "window_s", "step_s", "max_gap_s"})



ENGINE_FIELDS = {"running": "requests", "waiting": "requests", "cache_evictions": "events",
        "prefill_ms_avg": "ms", "prefill_batches": "batches", "prefill_batch_requests": "requests",
        "hit_tokens_total": "tokens", "context_tokens_total": "tokens", "context_requests_total": "requests",
        "cache_key_hits": "keys", "cache_keys_requested": "keys", "admission_open": "boolean",
        "admitted_rpcs_total": "requests", "rejected_rpcs_total": "requests"}


NUMERIC_PARAMETERS = {
    **SOURCE_NUMBERS,
    "procedure.post_affinity_ms": NumberRule(integer=True, minimum=0, maximum=9223372036854775807),
    "traffic.max_requests": NumberRule(integer=True, minimum=1, maximum=1000000),
    **JAVA_FLOW_NUMBERS,
    **CAPTURE_NUMBERS,
    **number_fields(NONNEGATIVE,
        'checks.collapse.expected',
        'analysis.collapse.absolute_min_hit',
        'analysis.collapse.max_drop',
        'analysis.collapse.sustain_s',
        'observation.max_gap_s',
        'observation.sample_s',
        'observation.step_s',
        'observation.warmup_timeout_s',
        'observation.window_s',
        'procedure.analysis_timeout_s',
        'procedure.topology_timeout_s',
        'procedure.warmup_min_s',
        'procedure.startup_timeout_s',
    ),
    **number_fields(COUNT,
        'procedure.drain_timeout_ms',
        'procedure.target_p',
    ),
    **number_fields(OFFSET,
        'observation.windows.baseline.from.offset_s',
        'observation.windows.baseline.until.offset_s',
        'observation.windows.post.from.offset_s',
        'observation.windows.post.until.offset_s',
    ),
}


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
NUMERIC_PARAMETERS.update(CHECKS.numeric_parameters)

CHECK_FIELDS = CHECKS.fields | {'sustain_s', 'absolute_min_hit', 'max_drop'}


def compile_checks(case, checks, policy):
    from case_api.inputs import fields
    from case_api.check_inputs import metric_criterion
    from case_api.check_inputs import compile_metric_checks

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


FIELDS = PROCEDURE_FIELDS | OBSERVATION_FIELDS | {"baseline_s", "observe_s"} | CHECK_FIELDS | {"flow", "qps", "expected_behavior"}


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
    for k in FIELDS - {"flow", "removal_mode", "expected_behavior"}:
        if not finite_number(p[k]) or p[k] < 0:
            raise ValueError(k + " must be finite and nonnegative")
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


from case_api.input_contract import InputContract, InputGroup, numeric_fields
from adapters.java_flow import JAVA_FLOW_INPUT_FIELDS

INPUTS = InputContract({
    'traffic': InputGroup(JAVA_FLOW_INPUT_FIELDS | {"kind", "max_requests"}),
    'procedure': InputGroup(PROCEDURE_FIELDS | {"analysis_timeout_s", "startup_timeout_s"}),
    'observation': InputGroup(OBSERVATION_FIELDS | {"inputs", "windows", "capture"}),
    'analysis': InputGroup({"collapse"}),
    'checks': InputGroup(set(RULES) | {"collapse"}),
}, numeric_fields(NUMERIC_PARAMETERS))
