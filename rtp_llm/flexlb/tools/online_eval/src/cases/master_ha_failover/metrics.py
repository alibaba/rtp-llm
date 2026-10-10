"""Case-owned HA numeric projections; raw journals remain separate evidence."""

import json
import math
import time
from collections import defaultdict
from pathlib import Path

from cases.master_ha_failover.analysis import prefill_assignment_windows

STATE_FIELDS = ("http_up", "scheduler_inflight", "prefill_inflight_requests",
                "decode_master_queued", "decode_confirmed_running")

def _artifacts(payload):
    finish = next((stage for stage in payload["stages"] if stage["id"] == "finish"), {})
    paths = {Path(path).name: Path(path) for path in finish.get("artifacts", [])}
    return paths.get("client_events.jsonl"), paths.get("master_states.jsonl")


def _read_rows(path):
    if path is None or not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _request_series(rows, anchor):
    buckets = defaultdict(lambda: {"sent": 0, "success": 0, "failed": 0})
    for row in rows:
        timestamp = row.get("send_start_epoch_ms")
        if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
            raise ValueError("HA request lacks finite send timestamp")
        bucket = math.floor(timestamp / 1000)
        counts = buckets[bucket]
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
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
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



def produce(directory, payload):
    from monitoring.metric_store import MetricStore, publish
    store = MetricStore.read(directory)
    anchor = payload["clock_anchor"]["epoch_s"]
    request_path, state_path = _artifacts(payload)
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
        publish(store, identity, definition, [dict(epoch="1", source="ha",
                labels={"master": master} if master else {},
                points=[[p["x"] + anchor, p["y"]] for p in points])
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
    current = dict(epoch=str(ctx.env_epoch), source="ha_gate",
        labels=labels, points=[[time.time(), actual]])
    stamps = [row["send_start_epoch_ms"] / 1000 for row in rows if "send_start_epoch_ms" in row]
    publish(store, identity, store.document["definitions"][identity], [current],
            producer="ha_gates", evidence=dict(input_resource=params["rows"], sample_count=len(rows),
                observed_request_bounds=[min(stamps), max(stamps)] if stamps else None, selection=json.loads(labels["selection"]),
                calculation=identity))
    store.document["metrics"][identity] = observations + store.document["metrics"][identity]
    store.save(ctx.artifact_dir)
    return store
