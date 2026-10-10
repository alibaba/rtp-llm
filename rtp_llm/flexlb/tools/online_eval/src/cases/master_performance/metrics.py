"""Request-cohort projections published by the performance case."""

from analysis.request_metrics import RequestLedger, describe_calculation
from cases.master_performance.analysis import analyze, engine_tps_checks
from monitoring.measurement import implementation_measurement
from monitoring.metric_store import series_row


GATE_POPULATIONS = {
    'arrival_cohort': frozenset({'sent_qps', 'cohort_requests', 'success_requests',
        'cohort_error_rate', 'slo_fraction', 'goodput_rps', 'input_goodput', 'output_goodput',
        'offered_qps_deviation', 'pacing_lag_max_ms'}),
    'whole_run_terminals': frozenset({'error_rate'}),
    'completion_window_successes': frozenset({'input_tps', 'output_tps'}),
    'arrival_cohort_successes': frozenset({'ttft_p99_ms', 'e2e_p99_ms', 'tpot_p99_ms'}),
    'full_request_lifetimes': frozenset({'inflight_start', 'inflight_end', 'inflight_growth_rps'}),
}
ENGINE_METRICS = frozenset({'rtp_llm_context_tps', 'rtp_llm_context_tps_with_cache', 'rtp_llm_generate_tps'})


def metric_contract(producer, identity, calculation):
    if producer != 'performance_requests':
        raise ValueError('unsupported performance metric producer')
    if identity.startswith('request/'):
        if calculation is None:
            raise ValueError('request metric requires executable calculation: ' + identity)
        return describe_calculation(calculation, windows={'measurement'})
    if calculation is not None:
        raise ValueError('performance gate owns its frozen calculation')
    key = identity.removeprefix('performance_gate/')
    if not identity.startswith('performance_gate/'):
        raise ValueError('unknown performance metric: ' + identity)
    engine_keys = {name + suffix for name in ENGINE_METRICS for suffix in ('_scrape_engine_mean', '_engine_count')}
    if key in engine_keys:
        return dict(source_type='prometheus', measurement=implementation_measurement(
            engine_tps_checks, population='measurement_window_engine_incarnations',
            accuracy='sampled', request_identity=False))
    for population, keys in GATE_POPULATIONS.items():
        if key in keys:
            return dict(source_type='client_journal', measurement=implementation_measurement(
                analyze, population=population, accuracy='request_ledger', request_identity=True))
    raise ValueError('unknown performance metric: ' + identity)


def values(evidence, definitions=None):
    if definitions is None:
        from monitoring.query_plan import definitions as expand, load_plan
        definitions = expand(load_plan('master_performance.yaml'))
    ledger = RequestLedger(evidence['flow']['records'])
    window = evidence['window']
    windows = {'measurement': (window['start_epoch_ms'], window['end_epoch_ms'])}
    return {identity: ledger.series(spec['calculation'], windows)
            for identity, spec in definitions.items() if spec.get('producer') == 'performance_requests'
            and 'calculation' in spec}


def produce(directory, evidence, result):
    from monitoring.metric_store import MetricStore, export_metrics, publish
    from monitoring.query_plan import load_plan
    export_metrics(directory, load_plan("master_performance.yaml"))
    store = MetricStore.read(directory)
    epoch = evidence["provenance"]["env_epoch"]
    request_values = ({identity: [] for identity, spec in store.document['definitions'].items()
                       if spec.get('producer') == 'performance_requests' and 'calculation' in spec}
                      if result['verdict'] == 'INVALID' and not result['metrics'] and not result['windows']
                      else values(evidence, store.document['definitions']))
    for identity, points in request_values.items():
        publish(store, identity, store.document["definitions"][identity],
                [series_row(points, epoch=epoch, source="client", labels={})],
                producer="performance_requests", evidence=dict(
                    path=str(directory) + "/performance-gate-evidence.json",
                    window=evidence.get("window"), calculation=store.document["definitions"][identity]["calculation"]))

    for name, value in result["metrics"].items():
        key = name.split("/")[-1]
        identity = "performance_gate/" + key
        if name.startswith("mock/") and not key.endswith("_engine_count"):
            identity += "_scrape_engine_mean"
        publish(store, identity, store.document["definitions"][identity],
                [series_row([[evidence["window"]["end_epoch_ms"]/1000, value]],
                            epoch=epoch, source="performance_gate", labels={})],
                producer="performance_requests", evidence=dict(
                    path=str(directory) + "/performance-gate-evidence.json",
                    window=evidence["window"], measurement_validity="INVALID" if result["verdict"] == "INVALID" else "VALID",
                    algorithm="absolute_performance_gate"))
    store.save(directory)
