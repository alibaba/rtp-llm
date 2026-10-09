"""Publish the cache gate's frozen survivor-window measurements."""

from monitoring.metric_store import MetricStore, export_metrics, publish
from monitoring.query_plan import load_plan


def produce(directory, evidence, result):
    export_metrics(directory, load_plan("cache_scale_in.yaml"))
    store = MetricStore.read(directory)
    rows = evidence["samples"]
    anchor = rows[0]["epoch_s"] - rows[0]["t"] if rows else 0
    identity = "derived/survivor_hit_ratio"
    publish(store, identity, store.document["definitions"][identity],
            [dict(epoch="1", source="cache_gate", labels={},
                  points=[[anchor+w["end"], w["hit"] if not w["errors"] else None]
                          for w in result["windows"]])], producer="cache_windows",
            evidence=dict(path=str(directory) + "/cache-gate-evidence.json",
                          calculation="survivor hit/context counter delta; timestamp is window end",
                          survivors=evidence.get("survivors", [])))

    store.save(directory)
