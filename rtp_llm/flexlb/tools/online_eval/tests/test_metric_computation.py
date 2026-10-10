"""Guard the boundaries between exact journals, sampled inputs and presentation."""

import math
import pytest

from analysis.statistics import percentile_nr
from analysis.time_buckets import TimeBuckets
from cases.master_ha_failover.analysis import row_ts_ms
from cases.master_ha_failover.metrics import _request_series, produce as produce_ha
from cases.master_performance.metrics import produce as produce_performance, values
from monitoring.metric_store import MetricContractError, MetricStore, MetricUnavailable, export_metrics, publish, series_row
from monitoring.query_plan import definitions, load_plan, queries_for_targets


def test_nearest_rank_preserves_precision_zero_and_missing_distinctions():
    assert percentile_nr([], .99) is None
    assert percentile_nr([0], .99) == 0
    assert percentile_nr([1.2345, 2.6789], .99) == 2.6789
    assert percentile_nr(list(range(100)), .99) == 98
    for values_, p in [([math.nan], .99), ([True], .99), ([1], 0), ([1], math.inf)]:
        with pytest.raises(ValueError):
            percentile_nr(values_, p)


def test_bucket_origin_is_not_the_report_origin():
    measurement = TimeBuckets(10.25, 1)
    wall = TimeBuckets(0, 1)
    assert measurement.index(10249) == -1
    assert measurement.index(10250) == 0
    assert measurement.index(11250) == 1
    assert measurement.epoch_s(1) == 11.25
    assert wall.index(11250) == 11
    assert wall.epoch_s(11) == 11
    with pytest.raises(ValueError):
        measurement.index(math.nan)
    with pytest.raises(ValueError):
        TimeBuckets(0, 0)


def test_arrival_cohort_is_not_the_completion_window():
    # The successful request arrives in bucket 0 but terminates in bucket 1.
    issued = dict(rid='one', send_start_epoch_ms=10250, input_len=100)
    terminal = dict(issued, status='ok', total_ms=1250, ttft_ms=123.456,
                    observed_output_tokens=2)
    evidence = dict(window=dict(start_epoch_ms=10250, end_epoch_ms=12250), criteria=dict(measure_s=2),
                    flow=dict(issued=[issued], records=[terminal]))
    perf = values(evidence)
    assert perf['request/ttft_p99_ms'] == [[10.25, 123.456], [11.25, None]]
    ha = _request_series([terminal], 10.25)
    assert ha['success'] == [dict(x=-.25, y=1)]
    assert ha['success'][0]['x'] + 10.25 == 10


def test_ha_requires_issue_time_even_when_terminal_time_is_available():
    assert row_ts_ms(dict(send_start_epoch_ms=1250, wall_clock_ts=99)) == 1250
    for value in (None, math.nan, math.inf, True, 0):
        with pytest.raises(ValueError, match='send_start_epoch_ms'):
            row_ts_ms(dict(send_start_epoch_ms=value, wall_clock_ts=99))


def test_python_prometheus_producer_does_not_become_a_query():
    plan = load_plan('cache_scale_in.yaml')
    metric = 'derived/survivor_hit_ratio'
    assert definitions(plan)[metric]['source_type'] == 'prometheus'
    assert definitions(plan)[metric]['measurement']['accuracy'] == 'counter_delta'
    queries, _ = queries_for_targets(plan, {'mock': ''}, lambda name: '{job="' + name + '"}', 1)
    assert 'mock/survivor_hit_ratio' not in queries
    assert set(queries) == {'mock/up'} | {'mock/' + name for name in plan['sources']['mock']}


@pytest.mark.parametrize('name', ['master_performance', 'master_ha_failover'])
def test_request_ledgers_exclude_sampled_twins_from_actual_query_plan(name):
    plan = load_plan(name + '.yaml')
    queries, _ = queries_for_targets(plan, {'client-flow': '', 'master-A': ''},
                                     lambda name: '{job="' + name + '"}', 1)
    for metric in (('actual_send_qps', 'success_qps', 'error_qps') if name == 'master_ha_failover' else ()) + (
                   'completed_qps', 'ttft_p99_seconds', 'total_p99_seconds'):
        assert 'client-flow/' + metric not in queries
    if name == 'master_ha_failover':
        for metric in ('flexlb_app_flexlb_scheduler_inflight_size',
                       'flexlb_app_flexlb_inflight_request_count',
                       'flexlb_auto_tpm_decode_reserved_count',
                       'flexlb_auto_tpm_decode_running_count'):
            assert 'master-A/' + metric not in queries


def test_produced_null_series_is_absent_with_explicit_environment_identity(tmp_path):
    store = export_metrics(tmp_path, load_plan('master_performance.yaml'))
    metric = 'request/ttft_p99_ms'
    publish(store, metric, store.document['definitions'][metric],
            [series_row([[10, None]], epoch=3, source='client', labels={})],
            producer='performance_requests', evidence={})
    row = store.document['metrics'][metric][0]
    assert row['epoch'] == '3'
    assert row['source'] == 'client'
    assert row['provenance']['source_type'] == 'client_journal'
    assert row['status'] == 'ABSENT'
    with pytest.raises(MetricUnavailable):
        store.select(metric)
    for epoch in (None, '1', 0, True):
        with pytest.raises(MetricContractError, match='epoch'):
            series_row([], epoch=epoch, source='client', labels={})


def test_performance_publication_uses_frozen_environment_epoch(tmp_path):
    from test_performance_gate import evidence
    from cases.master_performance.analysis import analyze
    e = evidence()
    e['provenance']['env_epoch'] = 4
    produce_performance(tmp_path, e, analyze(e))
    rows = MetricStore.read(tmp_path).document['metrics']['request/ttft_p99_ms']
    assert {row['epoch'] for row in rows} == {'4'}


def test_ha_evidence_uses_resource_identity_and_requires_files(tmp_path):
    requests, states = tmp_path / 'client_events.jsonl', tmp_path / 'master_states.jsonl'
    requests.write_text('{"send_start_epoch_ms":1100,"status":"ok"}\n')
    states.write_text('')
    data = dict(clock_anchor=dict(epoch_s=.25), configuration=dict(environment=dict(n_prefill=1)),
                stages=[dict(id='arbitrary_stage_name', output=dict(rows=dict(kind='ha_rows', env_epoch=7)),
                             artifacts=[str(requests), str(states)])])
    export_metrics(tmp_path, load_plan('master_ha_failover.yaml'))
    produce_ha(tmp_path, data)
    row = MetricStore.read(tmp_path).document['metrics']['ha/success'][0]
    assert row['epoch'] == '7'
    assert row['points'] == [[1, 1]]
    states.unlink()
    with pytest.raises(MetricUnavailable, match='evidence missing'):
        produce_ha(tmp_path, data)
    data['stages'] = []
    with pytest.raises(MetricUnavailable, match='journal resource'):
        produce_ha(tmp_path, data)


def test_environment_identity_is_frozen_before_provenance_acquisition():
    from runtime.observation import ObservationClock
    from workload.gate_evidence import new_evidence
    e = new_evidence('performance_evidence_schema_version', ObservationClock(10, 1), {},
                     instance='run', env_epoch=6)
    assert e['provenance'] == dict(instance='run', env_epoch=6)


def test_cache_publication_retains_prometheus_origin_and_environment_epoch(tmp_path):
    from test_cache_scale_gate import CacheGateTest
    from cases.cache_scale_in.analysis import analyze
    from cases.cache_scale_in.metrics import produce
    e = CacheGateTest().evidence()
    e['provenance']['env_epoch'] = 5
    produce(tmp_path, e, analyze(e))
    row = MetricStore.read(tmp_path).document['metrics']['derived/survivor_hit_ratio'][0]
    assert row['epoch'] == '5'
    assert row['provenance']['source_type'] == 'prometheus'
    assert row['provenance']['measurement']['accuracy'] == 'counter_delta'
