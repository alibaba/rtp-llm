"""Physical filtering and exceptional-source lifecycle are independent of cases."""

import json
from io import BytesIO
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

from monitoring.collectors import HttpJsonAdapter
from monitoring.probe import PrometheusEvidence
from monitoring.collection_plan import collection_plan, physical_metrics, scrape_job
from monitoring.query_plan import load_plan
from monitoring.session import PrometheusSession
from monitoring.sources import evidence_collector
from scenario.loader import ScenarioError


@pytest.mark.parametrize('name', ['cache_scale_in.yaml', 'master_ha_failover.yaml',
                                  'master_performance.yaml'])
def test_decode_rate_query_and_scrape_filter_use_the_same_wall_metric(name):
    plan = load_plan(name)
    names = physical_metrics(plan, 'mock')
    assert 'mock_decode_wall_tps' in names
    assert 'rtp_llm_generate_tps' not in names  # Native gauge counts window tokens.
    queries = [spec for spec in plan['sources']['mock'].values()
               if 'mock_decode_wall_tps' in spec['promql']]
    assert queries
    for spec in queries:
        assert spec['unit'] == 'tokens/s'
        assert 'mock_decode_wall_tps' in spec['exported_metrics']


def test_physical_whitelists_follow_case_plan_and_keep_all_query_dependencies():
    plan = load_plan('master_performance.yaml')
    names = physical_metrics(plan, 'client')
    assert 'flexlb_client_actual_send_total' in names
    assert 'flexlb_client_ttft_seconds_bucket' not in names
    job = scrape_job(plan, 'client-one', 'http://localhost:123/metrics?scope=run', 'client')
    assert job['params'] == {'scope': ['run']}
    assert job['metric_relabel_configs'][0]['source_labels'] == ['__name__']
    assert job['metric_relabel_configs'][0]['action'] == 'keep'
    assert 'rtp_llm_context_tps' in physical_metrics(plan, 'mock')
    assert {'mock_prefill_batch_size_bucket', 'mock_prefill_batch_size_count',
            'mock_prefill_batch_size_sum'} <= set(physical_metrics(plan, 'mock'))
    selected = collection_plan(load_plan('master_ha_failover.yaml'))
    assert set(selected['prometheus']) == {'mock'}
    assert set(selected['evidence']['master_inflight']['fields']) == {
        'http_up', 'scheduler_inflight', 'prefill_inflight_requests',
        'decode_master_queued', 'decode_confirmed_running'}
    assert len(selected['evidence']['master_inflight']['implementation']['sha256']) == 64


def test_unused_targets_are_not_scraped_or_waited_on(tmp_path):
    session = PrometheusSession(tmp_path, {'mock': 'http://local:1/metrics',
        'master-A': 'http://unused:2/metrics'}, query_plan='master_ha_failover.yaml')
    assert set(session.targets) == {'mock'}
    with patch.object(session, 'instant') as read, patch('urllib.request.urlopen') as http:
        session.add_targets({'client-unused': 'http://unused:3/metrics'})
    read.assert_not_called()
    http.assert_not_called()
    assert not session.target_bounds


def test_only_evidence_plan_needs_no_prometheus_binary(tmp_path):
    plan = load_plan('master_ha_failover.yaml')
    plan['sources'] = dict(mock={}, client={}, master={})
    plan['produced'] = {key: spec for key, spec in plan['produced'].items() if 'collection' not in spec}
    with patch('monitoring.query_plan.load_plan', return_value=plan), patch('shutil.which', return_value=None):
        session = PrometheusSession(tmp_path, {}, binary='', query_plan='master_ha_failover.yaml')
    session.start()
    session.stop()
    assert session.process is None
    assert not (tmp_path/'prometheus.json').exists()
    assert json.loads((tmp_path/'queries.json').read_text())['queries'] == {}
    assert 'ha/http_up' not in json.loads((tmp_path/'metrics.json').read_text())['definitions']


@pytest.mark.parametrize('dependency', [None, [], ['another_metric']])
def test_raw_dependency_cannot_disagree_with_query(tmp_path, dependency):
    spec = dict(promql='running${selector}', mode='scrape', unit='requests', value_kind='gauge', labels=[])
    if dependency is not None:
        spec['exported_metrics'] = dependency
    (tmp_path/'case.yaml').write_text(json.dumps(dict(metric_plan_schema_version=5, sources=dict(mock=dict(running=spec)))))
    with patch('monitoring.query_plan.CATALOG', tmp_path), pytest.raises(ScenarioError):
        load_plan('case.yaml')


def test_case_cannot_author_an_unregistered_collection(tmp_path):
    spec = dict(producer='ha_evidence', source_type='debug_api', unit='boolean',
                value_kind='gauge', labels=['master'], collection=dict(source='arbitrary_url', field='x'))
    (tmp_path/'case.yaml').write_text(json.dumps(dict(metric_plan_schema_version=5, produced={'ha/http_up': spec})))
    with patch('monitoring.query_plan.CATALOG', tmp_path), pytest.raises(ScenarioError, match='invalid produced'):
        load_plan('case.yaml')


def test_http_outage_is_explicit_but_bad_data_is_never_unavailability():
    adapter = HttpJsonAdapter('http://unused', lambda data: {'depth': data['depth']},
                              lambda: {'available': False})
    with patch('urllib.request.urlopen', side_effect=OSError('offline')):
        assert adapter(timeout=.1) == {'available': False}
        with pytest.raises(OSError):
            HttpJsonAdapter('http://unused', dict)(timeout=.1)
    with patch('urllib.request.urlopen', return_value=BytesIO(b'not json')):
        with pytest.raises(ValueError):
            adapter(timeout=.1)
    with patch('urllib.request.urlopen', return_value=BytesIO(b'{}')):
        with pytest.raises(KeyError):
            adapter(timeout=.1)


def test_registered_projection_reads_only_selected_fields(tmp_path):
    plan = load_plan('master_ha_failover.yaml')
    plan['produced'] = {'ha/http_up': plan['produced']['ha/http_up']}
    env = NS(master_specs=dict(A=NS(bind_ip='localhost', http_port=123)))
    collector = evidence_collector(plan, 'master_inflight', env, tmp_path/'states.jsonl',
                                  session=NS(interval=1), limits=dict(max_samples=10, max_bytes=10000))
    with patch('urllib.request.urlopen', return_value=BytesIO(b'{}')):
        assert collector.adapters['A'](timeout=.1) == {'master': 'A', 'http_up': 1}
    plan['produced']['ha/scheduler_inflight'] = load_plan('master_ha_failover.yaml')['produced']['ha/scheduler_inflight']
    collector = evidence_collector(plan, 'master_inflight', env, tmp_path/'states.jsonl',
                                  session=NS(interval=1), limits=dict(max_samples=10, max_bytes=10000))
    with patch('urllib.request.urlopen', return_value=BytesIO(b'{}')):
        with pytest.raises(ValueError, match='required ledger'):
            collector.adapters['A'](timeout=.1)


def test_unselected_adapter_cannot_start(tmp_path):
    with pytest.raises(ValueError, match='not selected'):
        evidence_collector(load_plan('master_performance.yaml'), 'master_inflight', object(),
                           tmp_path/'states.jsonl', session=NS(interval=1), limits=dict(max_samples=10, max_bytes=10000))
    assert not list(tmp_path.iterdir())


def test_probe_registration_does_not_read_and_only_scrapes_collect():
    calls = []
    probe = PrometheusEvidence('unused.jsonl', {'one': lambda **kw: calls.append(kw) or {'target': 'one', 'queue': 0}},
        fields={'queue'}, label='target', source='example', session=NS(interval=1),
        limits=dict(max_samples=1, max_bytes=1000))
    assert probe.describe() == [] and calls == []
    assert list(probe.collect())[0].samples[0].value == 0
    assert len(calls) == 1
    with pytest.raises(ValueError, match='budget exceeded'):
        list(probe.collect())
    with pytest.raises(RuntimeError, match='collection failed'):
        probe.stop()


@pytest.mark.parametrize('row', [{'target': 'wrong', 'queue': 1}, {'target': 'one', 'queue': float('nan')}])
def test_bad_probe_data_is_not_published_as_partial_or_zero(row):
    probe = PrometheusEvidence('unused.jsonl', {'one': lambda **kw: row},
        fields={'queue'}, label='target', source='example', session=NS(interval=1),
        limits=dict(max_samples=1, max_bytes=1000))
    with pytest.raises(ValueError):
        list(probe.collect())
    assert probe.error is not None


def test_http_response_budget_does_not_truncate_or_fallback():
    adapter = HttpJsonAdapter('http://unused', dict, lambda: {'available': False}, max_response_bytes=2)
    with patch('urllib.request.urlopen', return_value=BytesIO(b'{"depth": 1}')):
        with pytest.raises(ValueError, match='byte budget'):
            adapter(timeout=.1)


def test_python_source_contract_rejects_unknown_fields():
    plan = load_plan('master_ha_failover.yaml')
    plan['produced']['ha/http_up']['collection']['field'] = 'misspelled'
    with pytest.raises(ValueError, match='unknown collection field'):
        collection_plan(plan)
