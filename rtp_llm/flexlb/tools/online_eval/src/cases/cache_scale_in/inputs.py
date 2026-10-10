"""Validate the case gate inputs and their declared metric bindings."""

import math
from input_contract import finite_number
from cases.numeric_parameters import (
    SOURCE_NUMBERS, JAVA_FLOW_NUMBERS, CAPTURE_NUMBERS, FRACTION,
    NONNEGATIVE, COUNT, OFFSET, number_fields,
)
from scenario.parameters import validate_fields
from cases.metric_inputs import metric_fields

PROCEDURE_FIELDS = frozenset({"target_p", "removal_mode", "drain_timeout_ms", "topology_timeout_s"})
OBSERVATION_FIELDS = frozenset({"warmup_timeout_s", "sample_s",
                               "window_s", "step_s", "max_gap_s"})



ENGINE_FIELDS = {"running": "requests", "waiting": "requests", "cache_evictions": "events",
        "prefill_ms_avg": "ms", "prefill_batches": "batches", "prefill_batch_requests": "requests",
        "hit_tokens_total": "tokens", "context_tokens_total": "tokens", "context_requests_total": "requests",
        "cache_key_hits": "keys", "cache_keys_requested": "keys", "admission_open": "boolean",
        "admitted_rpcs_total": "requests", "rejected_rpcs_total": "requests"}


NUMERIC_PARAMETERS = {
    **SOURCE_NUMBERS,
    **JAVA_FLOW_NUMBERS,
    **CAPTURE_NUMBERS,
    **number_fields(FRACTION,
        'checks.baseline_hit.expected',
        'checks.baseline_stability.expected',
        'checks.offered_load.expected',
    ),
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
    ),
    **number_fields(COUNT,
        'checks.completed.expected',
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


def engine_metric_snapshot(monitor, fields, timeout=5):
    """Strict field projection from the declared, TSDB-owned raw metric IDs."""
    import math
    bindings = {name: spec["metric"] for name, spec in fields.items()}
    rows = monitor.metric_snapshot(bindings.values(), source="mock", timeout=timeout)
    reverse = {identity: field for field, identity in bindings.items()}
    engines = {}
    for observation in rows:
        labels = observation["metric"]
        if labels.get("role") != "prefill":
            continue
        identity = labels.get("engine_name")
        if not identity or not labels.get("engine_incarnation"):
            raise ValueError("metric lacks engine identity/incarnation")
        if not labels.get("engine_ip") or not labels.get("grpc_port"):
            raise ValueError("engine gate metric lacks endpoint identity labels")
        value = float(observation["value"][1])
        if not math.isfinite(value):
            raise ValueError("non-finite engine metric")
        row = engines.setdefault(identity, dict(engine_incarnation=labels["engine_incarnation"],
                      grpc_addr=labels["engine_ip"] + ":" + labels["grpc_port"]))
        field = reverse[observation["metric_id"]]
        if field in row or row["engine_incarnation"] != labels["engine_incarnation"]:
            raise ValueError("duplicate engine metric or incarnation")
        row[field] = value
    if not engines or any(set(fields) - set(row) for row in engines.values()):
        raise ValueError("incomplete engine monitoring contract")
    return engines


RULES = {
    'offered_load': ('qps_tolerance', 'offered_qps_deviation', 'le', 'ratio', ('baseline', 'post')),
    'baseline_hit': ('baseline_min_hit', 'baseline_min_half_hit', 'ge', 'ratio', ('baseline',)),
    'baseline_stability': ('baseline_max_spread', 'baseline_half_spread', 'le', 'ratio', ('baseline',)),
    'completed': ('min_completed', 'min_window_completed', 'ge', 'requests', ('baseline', 'post')),
}

CHECK_FIELDS = frozenset(rule[0] for rule in RULES.values()) | {'sustain_s', 'absolute_min_hit', 'max_drop'}


def compile_checks(case, checks, policy):
    from cases.inputs import fields
    from cases.check_inputs import metric_criterion, compile_metric_checks

    fields(checks, set(RULES) | {'collapse'}, 'parameters.checks')
    fields(policy, {'absolute_min_hit', 'max_drop', 'sustain_s'}, 'parameters.analysis.collapse')
    result = dict(policy)
    result.update(compile_metric_checks(case, checks, RULES, namespace='cache_gate'))
    expected, _ = metric_criterion(case, checks['collapse'], metric='cache_gate/collapse_detected',
        unit='boolean', windows=('post',), op='eq', path='parameters.checks.collapse')
    if expected != 0:
        raise ValueError('cache gate requires absence of sustained collapse')
    return result


def observation_contract(data):
    from cases.windows import anchored_windows
    from runtime.observation import capture_limits
    windows = anchored_windows(data["windows"], {
        "baseline": "baseline_ready", "post": "target_topology_observed",
    })
    base, post = windows["baseline"], windows["post"]
    if base["until"] != 0 or base["from"] >= 0 or post["from"] != 0:
        raise ValueError("cache windows must end at baseline readiness and start at target readiness")
    capture_limits(data["capture"], "parameters.observation.capture")
    return dict(baseline_s=-base["from"], observe_s=post["until"])


FIELDS = PROCEDURE_FIELDS | OBSERVATION_FIELDS | {"baseline_s", "observe_s"} | CHECK_FIELDS | {"flow", "qps"}


def validate_criteria(params, plan):
    p = validate_fields(
        params, plan, FIELDS | {"gate_input"}, FIELDS | {"gate_input"}
    )
    from cases.cache_scale_in.inputs import engine_counters
    engine_counters(p["gate_input"])
    if p["removal_mode"] not in ("graceful", "abrupt"):
        raise ValueError("removal_mode must be graceful or abrupt")
    plan.reference(p["flow"], "java_flow")
    if plan.environment.get("discovery") != "discovery_file":
        raise ValueError("scale-in requires dynamic discovery_file")
    for k in FIELDS - {"flow", "removal_mode"}:
        if not finite_number(p[k]) or p[k] < 0:
            raise ValueError(k + " must be finite and nonnegative")
    for k in ("target_p", "min_completed"):
        if type(p[k]) is not int or p[k] < 1:
            raise ValueError(k + " must be a positive integer")
    if not 1 <= p["target_p"] < plan.environment["n_prefill"] <= 512:
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
    from scenario.compiler import environment

    for profile in plan.profiles:
        resolved = environment(plan.environment, plan.path, profile)["resolved_config"]
        stale_ms = resolved["workerRegistry"]["health"]["statusStaleAfterMs"]
        if p["topology_timeout_s"] * 1000 <= stale_ms:
            raise ValueError(
                "topology budget must exceed Master status staleness"
            )
    return p
