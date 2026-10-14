"""Status reporting is a measured survivor gate, including the withdrawal interval."""
import pytest
import test_cache_scale_gate as cache_fixture
from cases.cache_scale_in.analysis import analyze


def evidence():
    value = cache_fixture.CacheGateTest().evidence()
    value['events'] = [dict(id='withdraw_start', t=10, epoch_s=1010)]
    value['baseline_end'] = 10
    value['post_start'] = 20
    return value


@pytest.mark.parametrize('field,value', [('worker_status_qps', 0), ('worker_status_rt_us', 20000)])
def test_transition_threshold_failure_is_not_hidden_by_healthy_post(field, value):
    data = evidence()
    data['samples'][15]['engines']['p0'][field] = value
    result = analyze(data)
    assert result['verdict'] == 'FAIL'
    assert result['status_reporting']['transition']['samples'] > 0
    assert result['status_reporting']['post']['min_qps'] == 50


def test_withdrawn_worker_is_not_expected_to_keep_reporting():
    data = evidence()
    data['samples'][15]['engines']['p1']['worker_status_qps'] = 0
    assert analyze(data)['status_reporting']['transition']['min_qps'] == 50


def test_survivor_metric_missing_is_invalid_instead_of_a_threshold_failure():
    data = evidence()
    del data['samples'][15]['engines']['p0']['worker_status_rt_us']
    result = analyze(data)
    assert result['verdict'] == 'INVALID'
    assert result['gate_metrics']['status_max_rt_us'] is None


def test_missing_withdrawal_anchor_is_invalid():
    data = evidence()
    data["events"] = []
    assert analyze(data)["verdict"] == "INVALID"
