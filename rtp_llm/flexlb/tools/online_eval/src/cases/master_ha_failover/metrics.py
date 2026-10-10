"""Case-owned HA numeric projections; raw journals remain separate evidence."""

import json
import math
import time
from collections import defaultdict
from pathlib import Path

from cases.master_ha_failover.analysis import prefill_assignment_windows
from analysis.time_buckets import TimeBuckets
from monitoring.metric_store import MetricUnavailable, series_row

STATE_FIELDS = ("http_up", "scheduler_inflight", "prefill_inflight_requests",
                "decode_master_queued", "decode_confirmed_running")


def _artifacts(payload):
    candidates = []
    for stage in payload["stages"]:
        resource = stage.get("output", {}).get("rows", {})
        if resource.get("kind") != "ha_rows" or not stage.get("artifacts"):
            continue
        paths = {Path(path).name: Path(path) for path in stage["artifacts"]}
        if {"client_events.jsonl", "master_states.jsonl"} <= set(paths):
            candidates.append((paths["client_events.jsonl"], paths["master_states.jsonl"],
                               resource["env_epoch"]))
    if len(candidates) != 1:
        raise MetricUnavailable("HA metrics require one completed request/state journal resource")
    return candidates[0]


def _read_rows(path):
    if not path.is_file():
        raise MetricUnavailable("HA metric evidence missing: " + str(path))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _request_series(rows, anchor):
    grid = TimeBuckets(0, 1)
    buckets = defaultdict(lambda: dict(sent=0, success=0, failed=0))
    for row in rows:
        counts = buckets[grid.index(row["send_start_epoch_ms"])]
        counts["sent"] += 1
        counts["success" if row.get("status") == "ok" else "failed"] += 1
    if not buckets:
        return {key: [] for key in ("sent", "success", "failed")}
    return {
        key: [dict(x=second - anchor, y=buckets[second][key])
              for second in range(min(buckets), max(buckets) + 1)]
        for key in ("sent", "success", "failed")
    }


def _state_series(rows, anchor):
    samples = defaultdict(list)
    for row in rows:
        timestamp = row.get("epoch_s")
        name = row.get("master")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or name not in {"A", "B"}:
            raise ValueError("HA Master state row lacks timestamp or identity")
        if row.get("http_up") not in (0, 1) or (row["http_up"] == 1 and (
                row.get("state_error") or any(row.get(field) is None for field in STATE_FIELDS))):
            raise ValueError("healthy HA Master state violates ledger contract")
        for field in STATE_FIELDS:
            value = row.get(field)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("invalid HA Master state value")
            samples[(name, field)].append(dict(x=timestamp - anchor, y=value))
    return samples


def _prefill_balance_series(rows, anchor, fleet_size):
    windows = prefill_assignment_windows(rows)
    values = {field: [] for field in ("prefill_peak_qps", "prefill_mean_qps", "prefill_skew")}
    for second, counts in sorted(windows.items()):
        total = sum(counts.values())
        peak = max(counts.values()) if counts else 0
        current = {
            "prefill_peak_qps": peak / 5,
            "prefill_mean_qps": total / (5 * fleet_size),
            "prefill_skew": peak * fleet_size / total if total >= 200 else None,
        }
        for field, value in current.items():
            values[field].append(dict(x=second - anchor, y=value))
    return values


def metric_contract(producer, identity, calculation):
    from cases.master_ha_failover.analysis import HA_METRICS, measure_client_metric
    from monitoring.measurement import implementation_measurement
    if calculation is not None:
        raise ValueError('HA producer owns its case-specific calculation')
    if producer == 'ha_gates' and identity in {'ha_gate/' + name for name in HA_METRICS}:
        function, source, population = measure_client_metric, 'client_journal', 'declared_request_window_and_selection'
    elif producer == 'ha_evidence':
        key = identity.removeprefix('ha/')
        if not identity.startswith('ha/'):
            raise ValueError('unknown HA evidence metric: ' + identity)
        if key in STATE_FIELDS:
            function, source, population = _state_series, 'debug_api', 'master_instance'
        elif key in {'sent', 'success', 'failed'}:
            function, source, population = _request_series, 'client_journal', 'ha_requests_by_send_time'
        elif key in {'prefill_peak_qps', 'prefill_mean_qps', 'prefill_skew'}:
            function, source, population = _prefill_balance_series, 'client_journal', 'declared_prefill_fleet'
        else:
            raise ValueError('unknown HA evidence metric: ' + identity)
    else:
        raise ValueError('unknown HA metric: ' + identity)
    return dict(source_type=source, measurement=implementation_measurement(function,
        population=population, accuracy='sampled' if source == 'debug_api' else 'request_ledger',
        request_identity=source == 'client_journal'))


def produce(directory, payload):
    from monitoring.metric_store import MetricStore, publish
    store = MetricStore.read(directory)
    anchor = payload["clock_anchor"]["epoch_s"]
    request_path, state_path, epoch = _artifacts(payload)
    requests, states = _read_rows(request_path), _read_rows(state_path)
    configured = (payload.get("configuration") or {}).get("environment", {}).get("n_prefill")
    observed = len({r.get("prefill") for r in requests if r.get("prefill")})
    fleet_size = configured if type(configured) is int and configured > 0 else observed
    values = {key: [(None, points)] for key, points in _request_series(requests, anchor).items()}
    if fleet_size:
        values.update({key: [(None, points)] for key, points in
                       _prefill_balance_series(requests, anchor, fleet_size).items()})
    states_by_key = _state_series(states, anchor)
    for field in STATE_FIELDS:
        values[field] = [(master, states_by_key.get((master, field), [])) for master in ("A", "B")]
    for field in ("prefill_peak_qps", "prefill_mean_qps", "prefill_skew"):
        values.setdefault(field, [(None, [])])
    for field, members in values.items():
        identity = "ha/" + field
        definition = store.document["definitions"][identity]
        publish(store, identity, definition, [series_row([[p["x"] + anchor, p["y"]] for p in points],
                epoch=epoch, source="ha", labels={"master": master} if master else {})
                for master, points in members], producer="ha_evidence",
                evidence=dict(request_events=str(request_path) if request_path else None,
                              master_states=str(state_path) if state_path else None,
                              fleet_size=fleet_size or None))
    store.save(directory)
    return dict(request_events=str(request_path) if request_path else None,
                request_count=len(requests), prefill_fleet_size=fleet_size or None,
                prefill_fleet_size_source="declared configuration" if configured else "observed endpoints",
                master_states=str(state_path) if state_path else None, state_samples=len(states))


def gate_labels(params):
    """The cohort and predicate identify the frozen scalar, not its publish timestamp."""
    return dict(window=json.dumps(params["rows"], sort_keys=True),
                selection=json.dumps({k: params[k] for k in ("target", "route", "error_kind", "code")
                                      if k in params}, sort_keys=True))


def publish_gate(ctx, params, actual, rows):
    """Freeze the computed cohort metric and return the published store for checks."""
    from monitoring.metric_store import export_metrics, publish
    store = export_metrics(ctx.artifact_dir, ctx.monitor.query_plan)
    identity = params["metric"]
    labels = gate_labels(params)
    observations = [row for row in store.document["metrics"].get(identity, [])
                    if row["labels"] != labels]
    current = series_row([[time.time(), actual]], epoch=ctx.env_epoch,
                         source="ha_gate", labels=labels)
    stamps = [row["send_start_epoch_ms"] / 1000 for row in rows if "send_start_epoch_ms" in row]
    publish(store, identity, store.document["definitions"][identity], [current],
            producer="ha_gates", evidence=dict(input_resource=params["rows"], sample_count=len(rows),
                observed_request_bounds=[min(stamps), max(stamps)] if stamps else None, selection=json.loads(labels["selection"]),
                calculation=identity))
    store.document["metrics"][identity] = observations + store.document["metrics"][identity]
    store.save(ctx.artifact_dir)
    return store
