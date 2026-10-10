"""Publish the cache gate's frozen survivor-window measurements."""

from cases.cache_scale_in.analysis import analyze
from runtime.observation import evidence_origin
from monitoring.metric_store import MetricStore, export_metrics, publish, series_row
from monitoring.query_plan import load_plan


def metric_contract(producer, identity, calculation):
    from monitoring.measurement import implementation_measurement
    if producer != 'cache_windows' or calculation is not None:
        raise ValueError('cache producer owns its survivor-window calculation')
    if identity == 'derived/survivor_hit_ratio':
        source, population, accuracy = 'prometheus', 'surviving_prefill_engines', 'counter_delta'
    elif identity == 'cache_gate/offered_qps_deviation':
        source, population, accuracy = 'client_journal', 'baseline_initial_fleet_and_post_survivors', 'request_ledger'
    elif identity in {'cache_gate/' + key for key in ('baseline_min_half_hit', 'baseline_half_spread',
                                                    'min_window_completed', 'collapse_detected')}:
        source, population, accuracy = 'prometheus', 'baseline_initial_fleet_and_post_survivors', 'counter_delta'
    else:
        raise ValueError('unknown cache metric: ' + identity)
    return dict(source_type=source, measurement=implementation_measurement(analyze,
                population=population, accuracy=accuracy, request_identity=source == 'client_journal'))


def produce(directory, evidence, result):
    export_metrics(directory, load_plan("cache_scale_in.yaml"))
    store = MetricStore.read(directory)
    epoch = evidence["provenance"]["env_epoch"]
    anchor = evidence_origin(evidence)
    identity = "derived/survivor_hit_ratio"
    publish(store, identity, store.document["definitions"][identity],
            [series_row([[anchor+w["end"], w["hit"] if not w["errors"] else None]
                         for w in result["windows"]], epoch=epoch, source="cache_gate", labels={})], producer="cache_windows",
            evidence=dict(path=str(directory) + "/cache-gate-evidence.json",
                          calculation="survivor hit/context counter delta; timestamp is window end",
                          survivors=evidence.get("survivors", [])))

    for metric, value in result["gate_metrics"].items():
        identity = "cache_gate/" + metric
        publish(store, identity, store.document["definitions"][identity],
                [series_row([[anchor+evidence["post_end"], value]],
                            epoch=epoch, source="cache_gate", labels={})], producer="cache_windows",
                evidence=dict(path=str(directory) + "/cache-gate-evidence.json",
                              calculation=metric, scope=result["measurement_scope"],
                              measurement_validity="INVALID" if result["verdict"] == "INVALID" else "VALID"))
    store.save(directory)
