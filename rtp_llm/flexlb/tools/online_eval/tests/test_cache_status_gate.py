"""Smooth noisy gauges per engine without hiding outages or missing evidence."""
from copy import deepcopy
import pytest
import test_cache_scale_gate as cache_fixture
from cases.cache_scale_in.analysis import analyze
from cases.cache_scale_in.status_reporting import StatusSeries


def evidence():
    value = cache_fixture.CacheGateTest().evidence()
    for t in range(81, 231):
        row = deepcopy(value['samples'][-1])
        row.update(t=t, epoch_s=1000+t, started=t*10, terminal=t*8)
        row['engines']['p0'].update(hit_tokens_total=16000+.8*1000*(t-20), context_tokens_total=t*1000,
                                   context_requests_total=t*10, cache_evictions=t)
        value['samples'].append(row)
    value['events'] = [dict(id='withdraw_start', t=20, epoch_s=1020)]
    value.update(baseline_start=10, baseline_end=20, post_start=30, post_end=230)
    value['criteria']['observe_s'] = 200
    return value


def test_single_zero_sample_is_smoothed_but_not_replaced():
    data = evidence()
    data['samples'][25]['engines']['p0']['worker_status_qps'] = 0
    result = analyze(data)
    assert result['verdict'] == 'PASS'
    assert result['gate_metrics']['status_min_rolling_qps'] == pytest.approx(45)
    assert data['samples'][25]['engines']['p0']['worker_status_qps'] == 0


def test_transition_outage_is_not_hidden_by_healthy_post():
    data = evidence()
    for row in data['samples'][20:30]:
        row['engines']['p0']['worker_status_qps'] = 0
    assert analyze(data)['verdict'] == 'FAIL'
    assert analyze(data)['gate_metrics']['status_min_rolling_qps'] == 0


def test_short_transition_has_complete_survivor_history():
    data = evidence()
    data.update(post_start=22, post_end=222)
    data['samples'][20]['engines']['p0']['worker_status_qps'] = 0
    result = analyze(data)
    assert result['verdict'] == 'PASS'
    assert result['status_reporting']['transition']['samples'] == 2
    assert result['status_reporting']['transition']['min_rolling_qps'] == 45


def test_irregular_intervals_are_time_weighted():
    rows = [dict(t=t, engines={'p': dict(grpc_addr='p', worker_status_qps=q, worker_status_rt_us=1)})
            for t, q in ((0, 0), (1, 100), (4, 0), (10, 0))]
    assert StatusSeries.read(rows, 'p', 6).mean_qps(10) == 30


def test_percentile_is_per_engine_not_pooled():
    data = evidence()
    for row in data['samples'][10:20]:
        row['engines']['p1']['worker_status_rt_us'] = 20000
    assert analyze(data)['verdict'] == 'FAIL'
    assert analyze(data)['status_reporting']['baseline']['p99_rt_us'] == 20000


def test_single_post_spike_is_diagnostic_not_a_maximum_gate():
    data = evidence()
    data['samples'][100]['engines']['p0']['worker_status_rt_us'] = 50000
    result = analyze(data)
    assert result['verdict'] == 'PASS'
    assert result['gate_metrics']['status_p99_rt_us'] == 100
    assert result['gate_metrics']['status_max_rt_us'] == 50000
    assert all(check['id'] != 'status_max_rt_us' for check in result['checks'])


def test_multiple_spikes_fail_sampled_p99():
    data = evidence()
    for index in (100, 101, 102):
        data['samples'][index]['engines']['p0']['worker_status_rt_us'] = 20000
    assert analyze(data)['gate_metrics']['status_p99_rt_us'] == 20000
    assert analyze(data)['verdict'] == 'FAIL'


def test_small_transition_still_checks_latency():
    data = evidence()
    data['samples'][25]['engines']['p0']['worker_status_rt_us'] = 20000
    assert analyze(data)['verdict'] == 'FAIL'


def test_boundary_sample_belongs_only_to_post():
    data = evidence()
    data['samples'][30]['engines']['p0']['worker_status_rt_us'] = 50000
    result = analyze(data)
    assert result['verdict'] == 'PASS'
    assert result['status_reporting']['transition']['max_rt_us'] == 100
    assert result['status_reporting']['post']['max_rt_us'] == 50000


def test_removed_worker_is_not_required_after_withdrawal():
    data = evidence()
    data['samples'][20]['engines']['p1']['worker_status_qps'] = 0
    assert analyze(data)['status_reporting']['transition']['min_rolling_qps'] == 50


@pytest.mark.parametrize('value', [None, float('nan'), -1, True])
def test_bad_gauge_invalidates_evidence(value):
    data = evidence()
    data['samples'][25]['engines']['p0']['worker_status_qps'] = value
    assert analyze(data)['verdict'] == 'INVALID'
    assert analyze(data)['gate_metrics']['status_p99_rt_us'] is None


def test_incomplete_lookback_is_invalid_not_skipped():
    data = evidence()
    del data['samples'][0]
    assert analyze(data)['verdict'] == 'INVALID'


def test_gap_cannot_be_smoothed_into_a_pass():
    data = evidence()
    del data['samples'][23:27]
    assert analyze(data)['verdict'] == 'INVALID'


def test_changed_incarnation_in_lookback_is_invalid():
    data = evidence()
    data['samples'][5]['engines']['p0']['engine_incarnation'] = 'another'
    assert analyze(data)['verdict'] == 'INVALID'


def test_withdrawal_anchor_is_required():
    data = evidence()
    data['events'] = []
    assert analyze(data)['verdict'] == 'INVALID'


def test_slow_engine_is_not_hidden_by_fleet_mean():
    data = evidence()
    for row in data['samples'][:20]:
        row['engines']['p0']['worker_status_qps'] = 1000
        row['engines']['p1']['worker_status_qps'] = 0
    assert analyze(data)['verdict'] == 'FAIL'
    assert analyze(data)['status_reporting']['baseline']['min_rolling_qps'] == 0


def test_thresholds_are_inclusive_and_unchanged():
    data = evidence()
    for row in data['samples']:
        for engine in row['engines'].values():
            engine.update(worker_status_qps=10, worker_status_rt_us=10000)
    assert analyze(data)['verdict'] == 'PASS'
