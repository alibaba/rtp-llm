"""Validate the case gate inputs and their declared metric bindings."""

import re


METRIC = re.compile(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*\Z")

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
    if (
        not isinstance(spec, dict)
        or set(spec) != {"source", "fields"}
        or spec["source"] != "metric_store"
    ):
        raise ValueError("engine counter input requires metric_store and fields")
    fields = spec["fields"]
    if (
        not isinstance(fields, dict)
        or not required <= set(fields)
        or any(
            not isinstance(k, str)
            or not isinstance(v, str)
            or not METRIC.fullmatch(v)
            for k, v in fields.items()
        )
        or len(set(fields.values())) != len(fields)
    ):
        raise ValueError("engine counter input has missing or invalid metric bindings")
    return spec


def engine_snapshot(monitor, fields, timeout=5):
    """Strict field projection from the declared, TSDB-owned raw metric IDs."""
    import math
    bindings = dict(fields)
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
