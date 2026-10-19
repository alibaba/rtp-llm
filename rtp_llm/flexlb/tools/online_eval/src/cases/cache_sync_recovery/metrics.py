"""Publish case-specific measurements through the shared frozen MetricStore."""

from implementation_identity import implementation_identity
from monitoring.contracts import implementation_measurement


def series(evidence):
    events = {e.get("name", e["id"]): e["epoch_s"] for e in evidence["events"]}
    start = events.get("cache_only_started", float("-inf"))
    end = events.get("performance_observation_finished", float("inf"))
    rows = [r for r in evidence["frames"] if start <= r["epoch_s"] <= end]
    total = evidence["environment"].get("n_prefill", 0) + evidence["environment"].get("n_decode", 0)
    total -= len(evidence.get("victims", []))
    result = {
        "full_gc_pause_ms": [[g["epoch_s"], g["duration_ms"]] for g in evidence.get("gc", [])
                             if g["full"] and start <= g["epoch_s"] <= end],
        "healthy_age_s": [[r["epoch_s"], r["healthy_max_status_age_ms"]/1000
                           if r.get("healthy_max_status_age_ms") is not None else None] for r in rows],
        "visible_workers": [[r["epoch_s"], total-r["missing_workers"]
                            if total and r.get("missing_workers") is not None else None] for r in rows],
    }
    return result


def metric_contract(producer, identity, calculation):
    if producer != "cache_stability" or calculation is not None or identity not in {
            "cache_stability/"+key for key in ("full_gc_pause_ms", "healthy_age_s", "visible_workers")}:
        raise ValueError("unknown cache stability measurement: " + identity)
    return dict(source_type="process_log" if identity.endswith("/full_gc_pause_ms") else "debug_api",
                implementation=implementation_identity(series), measurement=implementation_measurement(
                    series, population="observed_worker_generation_fleet", accuracy="sampled", request_identity=False))


def produce(directory, evidence, result):
    from monitoring.metric_store import MetricStore, series_row
    from monitoring.publication import publish
    store = MetricStore.read(directory)
    for key, points in series(evidence).items():
        identity = "cache_stability/"+key
        if identity not in store.document["definitions"]:
            continue
        publish(store, identity, store.document["definitions"][identity],
                [series_row(points, epoch=evidence["provenance"]["env_epoch"], source="cache_stability", labels={})],
                producer="cache_stability", evidence=dict(path=str(directory)+"/cache_stability-gate-evidence.json",
                                                          validity=result["validity"]))
    store.save(directory)
