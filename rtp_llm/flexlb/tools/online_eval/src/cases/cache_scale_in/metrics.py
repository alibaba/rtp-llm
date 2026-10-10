"""Publish the cache gate's frozen survivor-window measurements."""

from runtime.observation import evidence_origin
from monitoring.metric_store import MetricStore, export_metrics, publish
from monitoring.query_plan import load_plan


def produce(directory, evidence, result):
    export_metrics(directory, load_plan("cache_scale_in.yaml"))
    store = MetricStore.read(directory)
    rows = evidence["samples"]
    anchor = evidence_origin(evidence)
    identity = "derived/survivor_hit_ratio"
    publish(store, identity, store.document["definitions"][identity],
            [dict(epoch="1", source="cache_gate", labels={},
                  points=[[anchor+w["end"], w["hit"] if not w["errors"] else None]
                          for w in result["windows"]])], producer="cache_windows",
            evidence=dict(path=str(directory) + "/cache-gate-evidence.json",
                          calculation="survivor hit/context counter delta; timestamp is window end",
                          survivors=evidence.get("survivors", [])))

    for metric, value in result["gate_metrics"].items():
        identity = "cache_gate/" + metric
        publish(store, identity, store.document["definitions"][identity],
                [dict(epoch="1", source="cache_gate", labels={},
                      points=[[anchor+evidence["post_end"], value]])], producer="cache_windows",
                evidence=dict(path=str(directory) + "/cache-gate-evidence.json",
                              calculation=metric, scope=result["measurement_scope"],
                              measurement_validity="INVALID" if result["verdict"] == "INVALID" else "VALID"))
    store.save(directory)
