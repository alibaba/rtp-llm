"""Validate the case gate inputs and their declared metric bindings."""

from cases.metric_inputs import metric_fields

PROCEDURE_FIELDS = frozenset({"target_p", "removal_mode", "drain_timeout_ms", "topology_timeout_s"})
OBSERVATION_FIELDS = frozenset({"warmup_timeout_s", "baseline_s", "observe_s", "sample_s",
                               "window_s", "step_s", "max_gap_s"})
CHECK_FIELDS = frozenset({"qps_tolerance", "baseline_min_hit", "baseline_max_spread",
                         "absolute_min_hit", "max_drop", "min_completed", "sustain_s"})


ENGINE_COUNTER_UNITS = {"running": "requests", "waiting": "requests", "cache_evictions": "events",
        "prefill_ms_avg": "ms", "prefill_batches": "batches", "prefill_batch_requests": "requests",
        "hit_tokens_total": "tokens", "context_tokens_total": "tokens", "context_requests_total": "requests",
        "cache_key_hits": "keys", "cache_keys_requested": "keys", "admission_open": "boolean",
        "admitted_rpcs_total": "requests", "rejected_rpcs_total": "requests"}


def engine_counters(spec):
    from cases.cache_scale_in.analysis import COUNTERS

    required = set(COUNTERS) | {"waiting", "running", "cache_evictions", "prefill_ms_avg"}
    fields = metric_fields(spec)
    if (
        not required <= set(fields)
        or any(
            v["labels"] != {"role": "prefill"}
            for v in fields.values()
        )
        or len({v["metric"] for v in fields.values()}) != len(fields)
    ):
        raise ValueError("engine counter input has missing or invalid metric bindings")
    return spec


def engine_snapshot(monitor, fields, timeout=5):
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
    'offered_load': ('qps_tolerance', 'offered_qps_deviation', 'le', 'ratio', 'baseline_and_post'),
    'baseline_hit': ('baseline_min_hit', 'baseline_min_half_hit', 'ge', 'ratio', 'baseline'),
    'baseline_stability': ('baseline_max_spread', 'baseline_half_spread', 'le', 'ratio', 'baseline'),
    'completed': ('min_completed', 'min_window_completed', 'ge', 'requests', 'baseline_and_post'),
}


def compile_checks(case, checks, policy):
    from cases.inputs import fields
    from cases.check_inputs import metric_criterion

    fields(checks, set(RULES) | {'collapse'}, 'parameters.checks')
    fields(policy, {'absolute_min_hit', 'max_drop', 'sustain_s'}, 'parameters.observation.collapse')
    result = dict(policy)
    for name, (key, metric, op, unit, window) in RULES.items():
        result[key], _ = metric_criterion(case, checks[name], metric='cache_gate/'+metric,
            unit=unit, window=window, op=op, path='parameters.checks.'+name)
    expected, _ = metric_criterion(case, checks['collapse'], metric='cache_gate/collapse_detected',
        unit='boolean', window='post', op='eq', path='parameters.checks.collapse')
    if expected != 0:
        raise ValueError('cache gate requires absence of sustained collapse')
    return result
