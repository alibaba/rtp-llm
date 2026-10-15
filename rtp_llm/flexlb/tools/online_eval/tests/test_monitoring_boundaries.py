"""Failed publication and exhausted archive budgets retain authoritative evidence."""

import json
import subprocess
import sys
from io import BytesIO
from unittest.mock import Mock, patch

import pytest

from monitoring.archive import export_metrics
from monitoring.metric_store import MetricStore
from monitoring.contracts import MetricContractError
from artifacts.json_io import write_json
from monitoring.query_plan import load_plan, queries_for_targets
from monitoring.session import PrometheusSession
from runtime.deadline import Deadline


def test_declaration_and_frozen_reader_imports_do_not_load_runtime():
    script = """
import sys
import monitoring.query_plan
assert 'monitoring.metric_store' not in sys.modules
assert 'monitoring.producers' not in sys.modules
assert 'monitoring.probe' not in sys.modules
assert 'monitoring.session' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', script], check=True)
    subprocess.run([sys.executable, '-c',
                    "import sys; import monitoring.metric_store; "
                    "assert 'case_api.registry' not in sys.modules; "
                    "assert 'monitoring.query_plan' not in sys.modules"], check=True)
    subprocess.run([sys.executable, '-c',
                    "import sys; import monitoring.client; "
                    "assert 'monitoring.metric_store' not in sys.modules; "
                    "assert 'monitoring.session' not in sys.modules; "
                    "assert 'monitoring.probe' not in sys.modules"], check=True)


def test_query_client_is_independent_and_sees_registered_sources():
    from monitoring.client import PrometheusClient
    kinds = {'first': 'mock'}
    client = PrometheusClient('http://owned-tsdb', query_plan=load_plan('default.yaml'),
                              target_kinds=kinds, max_gap_s=5)
    kinds['later'] = 'client'
    assert client.target_kind('later') == 'client'
    assert client.selector('later') == '{job="later"}'
    with patch.object(client, 'api', return_value={'result': []}) as api:
        assert client.query('up', when=12) == []
    assert api.call_args.kwargs['time'] == 12


def test_source_selection_is_independent_of_definition_order():
    from monitoring.sources import selected_sources
    plan = load_plan('master_ha_failover.yaml')
    before = selected_sources(plan)
    plan['produced'] = dict(reversed(list(plan['produced'].items())))
    assert selected_sources(plan) == before


def test_recording_needs_only_a_window_and_query_client(tmp_path):
    from monitoring.recording import ArchiveWindow, record_queries
    from monitoring.client import PrometheusClient
    plan = load_plan('default.yaml')
    client = PrometheusClient('http://owned-tsdb', query_plan=plan,
                              target_kinds={'worker': 'mock'}, max_gap_s=5)
    window = ArchiveWindow(tmp_path, 'default.yaml', {}, 1, 1,
                           {'worker': 'http://exporter/metrics'}, {'worker': [1, 2]})
    with patch.object(client, 'api', return_value={'result': []}):
        result = record_queries(window, client)
    assert result['errors']
    assert result['target_bounds'] == {'worker': [1, 2]}
    assert json.loads((tmp_path / 'queries.json').read_text()) == result
    assert not (tmp_path / 'metrics.json').exists()


def test_probe_failed_export_retains_completed_chunks(tmp_path):
    from monitoring.probe import PrometheusEvidence
    session = Mock(interval=1)
    session.client.query.side_effect = [
        [dict(metric={'__name__': 'online_eval_debug_alive', 'host': 'a'},
              values=[[10, '1']])], TimeoutError('budget exhausted')]
    probe = PrometheusEvidence(tmp_path / 'samples.jsonl', {'a': Mock(url='http://unused/status')},
                               fields=['alive'], label='host', source='debug',
                               session=session, limits={'max_samples': 100, 'max_bytes': 10000}, client=session.client)
    probe.started, probe.ended = 0, 120
    with pytest.raises(TimeoutError, match='budget exhausted'):
        probe._export(Deadline(float('inf')))
    frozen = json.loads((tmp_path / 'samples.prometheus.json').read_text())
    assert len(frozen['queries']) == 1
    assert frozen['error'] == 'budget exhausted'
    assert json.loads(probe.path.read_text()) == {'epoch_s': 10, 'host': 'a', 'alive': 1.0}


def test_invalid_conversion_preserves_previous_frozen_artifact(tmp_path):
    previous = dict(metrics_schema_version=1, definitions={}, metrics={},
                    collection_gaps={}, errors=[], run={'identity': 'previous'})
    write_json(tmp_path / 'metrics.json', previous)
    before = (tmp_path / 'metrics.json').read_bytes()
    (tmp_path / 'queries.json').write_text(json.dumps(dict(
        start=1, end=2, step=1, targets={},
        queries={'/invalid': dict(promql='x', result=[])})))
    with pytest.raises(MetricContractError, match='identity'):
        export_metrics(tmp_path)
    assert (tmp_path / 'metrics.json').read_bytes() == before
    assert MetricStore.read(tmp_path).document == previous


def test_new_targets_require_explicit_source_kind(tmp_path):
    plan = load_plan('default.yaml')
    with pytest.raises(ValueError, match='target kinds'):
        PrometheusSession(tmp_path, {'arbitrary-name': 'http://unused/metrics'}, metric_plan=plan)
    with pytest.raises(ValueError, match='target kinds'):
        queries_for_targets(plan, {'arbitrary-name': ''}, lambda _: '{}', 1)
    session = PrometheusSession(tmp_path, {'arbitrary-name': 'http://unused/metrics'},
                                metric_plan=plan, target_kinds={'arbitrary-name': 'mock'})
    assert session.client.target_kind('arbitrary-name') == 'mock'


def test_archive_budget_keeps_partial_queries_and_still_reaps_process(tmp_path):
    session = PrometheusSession(tmp_path, {'mock': 'http://unused/metrics'},
                                target_kinds={'mock': 'mock'})
    session.started = 1
    session.process = Mock()
    session.process.poll.return_value = None
    session.log = Mock()
    now = [0.0]
    calls = []
    def query(endpoint, **kwargs):
        calls.append(kwargs)
        assert kwargs['deadline'].expires_at == 1.0
        now[0] += .6
        return {'result': []}
    with patch('monitoring.session.time.monotonic', side_effect=lambda: now[0]), \
         patch.object(session.client, 'api', side_effect=query):
        with pytest.raises(TimeoutError):
            session.stop(timeout=1, materialize=False)
    assert len(calls) == 2  # Later chunks and queries do not get fresh budgets.
    archive = json.loads((tmp_path / 'queries.json').read_text())
    assert archive['errors'] and not (tmp_path / 'metrics.json').exists()
    session.process.terminate.assert_called_once()
    session.process.kill.assert_not_called()
    session.process.wait.assert_called_once_with(timeout=pytest.approx(1))
    session.log.close.assert_called_once()
    assert session.closed


def test_query_http_timeout_is_bounded_by_remaining_budget(tmp_path):
    session = PrometheusSession(tmp_path, {}, metric_plan=load_plan('default.yaml'))
    session.client.url = 'http://owned-tsdb'
    deadline = Deadline(.2, clock=lambda: 0)
    with patch('monitoring.client.urllib.request.urlopen',
               return_value=BytesIO(b'{"status":"success","data":{"result":[]}}')) as request:
        assert session.client.api('query', timeout=5, deadline=deadline, query='up') == {'result': []}
    assert request.call_args.kwargs['timeout'] == .2


def test_calculator_identity_is_explicit_and_not_derived_from_method():
    from analysis.request_metrics import describe_calculation
    contract = describe_calculation(dict(calculator='request_rate', window='measurement',
        selection=dict(time_basis='arrival', status='all'), bucket_s=1), windows={'measurement'})
    assert contract['measurement']['method'] == 'request_rate'
    assert contract['implementation'] == dict(module='analysis.request_metrics', function='_request_rate')


@pytest.mark.parametrize('when, expected', [(0, 0), (12, 12), (None, 1234)])
def test_query_timestamp_defaults_only_when_absent(when, expected):
    from monitoring.client import PrometheusClient
    client = PrometheusClient('http://owned-tsdb', query_plan=load_plan('default.yaml'),
                              target_kinds={}, max_gap_s=5)
    with patch.object(client, 'api', return_value={'result': []}) as api, \
         patch('monitoring.client.time.time', return_value=1234):
        client.query('up', when=when)
    assert api.call_args.kwargs['time'] == expected


def raw_archive_client(tmp_path):
    from monitoring.client import PrometheusClient
    from monitoring.recording import ArchiveWindow
    plan = dict(sources=dict(mock={'temperature': dict(
        promql='temperature${selector}', unit='K', value_kind='gauge', labels=['engine'],
        exported_metrics=['temperature'], mode='scrape', required=True)}, master={}, client={}),
        produced={})
    client = PrometheusClient('http://owned-tsdb', query_plan=plan,
                              target_kinds={'worker': 'mock'}, max_gap_s=5)
    window = ArchiveWindow(tmp_path, 'test', {}, 1, 1,
                           {'worker': 'http://exporter/metrics'}, {'worker': [1, 121]})
    return client, window


@pytest.mark.parametrize('failure', [TimeoutError('budget exhausted'),
                                    RuntimeError('query failed'), 'invalid_labels'])
def test_raw_archive_keeps_inner_chunks_without_accepting_failed_metric(tmp_path, failure):
    from monitoring.recording import record_queries
    from monitoring.contracts import MetricUnavailable
    client, window = raw_archive_client(tmp_path)
    first = dict(metric={'__name__': 'temperature', 'engine': 'a'}, values=[[30, '2']])
    second = (dict(metric={'__name__': 'temperature'}, values=[[90, '3']])
              if failure == 'invalid_labels' else failure)
    def samples():
        return [[first], [second] if failure == 'invalid_labels' else second]
    with patch.object(client, 'api', return_value={'result': [dict(
            metric={'__name__': 'up'}, values=[[1, '1'], [121, '1']])]}), \
         patch.object(client, 'query', side_effect=samples()):
        result = record_queries(window, client)
    assert result['queries']['worker/temperature']['result'] == [first]
    assert len(result['errors']) == 1
    assert result['errors'][0]['query'] == 'worker/temperature'
    assert json.loads((tmp_path / 'queries.json').read_text()) == result
    store = export_metrics(tmp_path)
    row = store.document['metrics']['mock/temperature'][0]
    assert row['points'] == [[30, 2.0]] and row['status'] == 'ERROR'
    with pytest.raises(MetricUnavailable, match='query failed'):
        store.select('mock/temperature')
    # The eager gate API must propagate the failure, never return partial data.
    with patch.object(client, 'query', side_effect=samples()), pytest.raises(
            (TimeoutError, RuntimeError, ValueError)):
        client.metric_rows('mock/temperature', source='worker', start=1, end=121)


def publication_store(tmp_path):
    spec = load_plan('master_ha_failover.yaml')['produced']['ha/scheduler_inflight']
    store = MetricStore(dict(metrics_schema_version=1,
                            definitions={'ha/scheduler_inflight': spec}, metrics={}))
    store.save(tmp_path)
    return store, spec


@pytest.mark.parametrize('change', [dict(source=''), dict(source=None), dict(epoch=''),
                                  dict(epoch=None), dict(epoch=0), dict(labels=[]),
                                  dict(points=None), dict(points=[1]), dict(points=[[1, 2, 3]])])
def test_invalid_published_identity_preserves_store_and_artifact(tmp_path, change):
    from monitoring.publication import publish
    store, spec = publication_store(tmp_path)
    previous = (tmp_path / 'metrics.json').read_bytes()
    valid = dict(epoch='1', source='ha', labels={'master': 'A'}, points=[[1, 2]])
    with pytest.raises(MetricContractError):
        publish(store, 'ha/scheduler_inflight', spec, [valid, dict(valid, **change)],
                producer='ha_evidence', evidence={})
    assert store.document['metrics'] == {}
    assert (tmp_path / 'metrics.json').read_bytes() == previous
    assert MetricStore.read(tmp_path).document['metrics'] == {}


def test_invalid_publication_evidence_is_rejected_before_mutation(tmp_path):
    from monitoring.publication import publish
    store, spec = publication_store(tmp_path)
    with pytest.raises(MetricContractError, match='metadata'):
        publish(store, 'ha/scheduler_inflight', spec,
                [dict(epoch=1, source='ha', labels={'master': 'A'}, points=[[1, 2]])],
                producer='ha_evidence', evidence=[])
    assert store.document['metrics'] == {}


def test_save_revalidates_mutated_document_before_replacing_artifact(tmp_path):
    from monitoring.publication import publish
    from monitoring.metric_store import frozen_metric_reads
    store, spec = publication_store(tmp_path)
    publish(store, 'ha/scheduler_inflight', spec,
            [dict(epoch=1, source='ha', labels={'master': 'A'}, points=[[1, 2]])],
            producer='ha_evidence', evidence={})
    store.save(tmp_path)
    previous = (tmp_path / 'metrics.json').read_bytes()
    with frozen_metric_reads():
        frozen = MetricStore.read(tmp_path)
        store.document['metrics']['ha/scheduler_inflight'][0]['source'] = ''
        with pytest.raises(MetricContractError, match='identity'):
            store.save(tmp_path)
        assert (tmp_path / 'metrics.json').read_bytes() == previous
        assert MetricStore.read(tmp_path) is frozen
        store.document['metrics']['ha/scheduler_inflight'][0]['source'] = 'repaired'
        store.save(tmp_path)
        assert MetricStore.read(tmp_path).document['metrics']['ha/scheduler_inflight'][0]['source'] == 'repaired'


def registration_session(tmp_path):
    session = PrometheusSession(tmp_path, {'existing': 'http://exporter/metrics'},
                                target_kinds={'existing': 'mock'})
    session.client.url = 'http://owned-tsdb'
    session.target_bounds['existing'] = [1, None]
    write_json(tmp_path / 'prometheus.json', {'scrape_configs': [{'job_name': 'existing'}]})
    return session


@pytest.mark.parametrize('phase', ['reload', 'ready'])
def test_failed_client_registration_restores_config_and_allows_retry(tmp_path, phase):
    session = registration_session(tmp_path)
    kinds = session.client.target_kinds
    config = json.loads((tmp_path / 'prometheus.json').read_text())
    reload_results = ([OSError('reload failed'), BytesIO(b''), BytesIO(b'')]
                      if phase == 'reload' else [BytesIO(b''), BytesIO(b''), BytesIO(b'')])
    ready_results = [None] if phase == 'reload' else [TimeoutError('not ready'), None]
    with patch('monitoring.session.urllib.request.urlopen', side_effect=reload_results) as reload, \
         patch.object(session, '_wait_targets', side_effect=ready_results):
        with pytest.raises((OSError, TimeoutError)):
            session.add_client_target('new', 'http://client/metrics')
        assert session.targets == {'existing': 'http://exporter/metrics'}
        assert session.target_kinds == {'existing': 'mock'}
        assert session.client.target_kinds is kinds
        assert session.target_bounds == {'existing': [1, None]}
        assert json.loads((tmp_path / 'prometheus.json').read_text()) == config
        assert session.registration_error is None
        session.add_client_target('new', 'http://client/metrics')
    assert reload.call_count == 3
    assert kinds['new'] == 'client'
    assert set(session.target_bounds) == {'existing', 'new'}
    assert [job['job_name'] for job in json.loads((tmp_path / 'prometheus.json').read_text())['scrape_configs']] == ['existing', 'new']


def test_failed_registration_rollback_marks_session_unusable_and_still_cleans_up(tmp_path):
    session = registration_session(tmp_path)
    session.process = Mock()
    session.process.poll.return_value = None
    with patch('monitoring.session.urllib.request.urlopen', side_effect=OSError('reload unavailable')):
        with pytest.raises(RuntimeError, match='rollback failed'):
            session.add_client_target('new', 'http://client/metrics')
    assert isinstance(session.registration_error, OSError)
    assert 'new' not in session.targets and 'new' not in session.target_kinds
    with pytest.raises(RuntimeError, match='rollback failure'):
        session.add_client_target('new', 'http://client/metrics')
    with pytest.raises(ValueError, match='live Prometheus'):
        session.add_probe(Mock(job='probe-new'), 'http://probe/metrics')
    session.stop(export=False)
    session.process.terminate.assert_called_once()
    assert session.closed


def test_registration_rollback_uses_the_original_deadline(tmp_path):
    session = registration_session(tmp_path)
    now = [0.0]
    deadline = Deadline(1, clock=lambda: now[0])
    def not_ready(*args):
        now[0] = 1
        deadline.check()
    with patch('monitoring.session.urllib.request.urlopen', return_value=BytesIO(b'')) as request, \
         patch.object(session, '_wait_targets', side_effect=not_ready):
        with pytest.raises(RuntimeError, match='rollback failed'):
            session.add_client_target('new', 'http://client/metrics', deadline=deadline)
    assert request.call_count == 1  # An expired budget cannot authorize a fresh reload.
    assert isinstance(session.registration_error, TimeoutError)
    assert set(session.targets) == {'existing'}
    assert [job['job_name'] for job in json.loads((tmp_path / 'prometheus.json').read_text())['scrape_configs']] == ['existing']


def test_config_write_failure_restores_inventory_without_reload(tmp_path):
    session = registration_session(tmp_path)
    config = json.loads((tmp_path / 'prometheus.json').read_text())
    def fail_once(path, data):
        if any(job['job_name'] == 'new' for job in data['scrape_configs']):
            raise OSError('disk full')
        return write_json(path, data)
    with patch('monitoring.session.write_json', side_effect=fail_once), \
         patch('monitoring.session.urllib.request.urlopen') as request:
        with pytest.raises(OSError, match='disk full'):
            session.add_client_target('new', 'http://client/metrics')
    request.assert_not_called()
    assert session.registration_error is None
    assert set(session.targets) == set(session.target_kinds) == {'existing'}
    assert json.loads((tmp_path / 'prometheus.json').read_text()) == config


def test_rollback_failure_is_frozen_as_an_archive_error(tmp_path):
    session = registration_session(tmp_path)
    session.started = 1
    session.registration_error = OSError('reload unavailable')
    session.target_bounds['existing'] = [1, 2]
    with patch.object(session.client, 'api', return_value={'result': []}), \
         pytest.raises(RuntimeError, match='incomplete Prometheus archive'):
        session.archive()
    errors = MetricStore.read(tmp_path).document['errors']
    assert any('registration rollback failed' in error['error'] for error in errors)


def test_archive_has_cleanup_budget_and_shutdown_keeps_reserved_time(tmp_path):
    session = PrometheusSession(tmp_path, {'mock': 'http://unused/metrics'},
                                target_kinds={'mock': 'mock'})
    session.process = Mock()
    session.process.poll.return_value = None
    now = [0.0]
    parent = Deadline(30, clock=lambda: now[0])
    observed = []

    def archive(**kwargs):
        observed.append(kwargs['deadline'].expires_at)
        # Query collection legitimately takes longer than process shutdown.
        now[0] = 19
        kwargs['deadline'].check()

    with patch.object(session, 'archive', side_effect=archive):
        session.stop(10, materialize=False, archive_timeout=100, deadline=parent)
    assert observed == [20]  # Outer cleanup budget reserves the final 10s.
    session.process.wait.assert_called_once_with(timeout=10)
    session.process.kill.assert_not_called()


def test_archive_expiry_preserves_error_and_fresh_shutdown_budget(tmp_path):
    session = PrometheusSession(tmp_path, {'mock': 'http://unused/metrics'},
                                target_kinds={'mock': 'mock'})
    session.process = Mock()
    session.process.poll.return_value = None
    now = [0.0]
    parent = Deadline(30, clock=lambda: now[0])

    def archive(**kwargs):
        now[0] = kwargs['deadline'].expires_at
        kwargs['deadline'].check()

    with patch.object(session, 'archive', side_effect=archive), \
         pytest.raises(TimeoutError, match='Prometheus archive') as exc:
        session.stop(10, archive_timeout=100, deadline=parent)
    assert 'Prometheus process' not in str(exc.value)
    session.process.wait.assert_called_once_with(timeout=10)
    session.process.kill.assert_not_called()
    assert session.closed


def test_expired_outer_budget_does_not_refresh_shutdown_allowance(tmp_path):
    session = PrometheusSession(tmp_path, {'mock': 'http://unused/metrics'},
                                target_kinds={'mock': 'mock'})
    session.process = Mock()
    session.process.poll.return_value = None
    parent = Deadline(0, clock=lambda: 1)
    with pytest.raises(TimeoutError, match='shutdown budget'):
        session.stop(10, export=False, deadline=parent)
    session.process.terminate.assert_called_once()
    session.process.kill.assert_called_once()
    session.process.wait.assert_called_once_with(timeout=1)


def test_snapshot_queries_only_requested_metrics_and_labels_at_one_scrape():
    from monitoring.client import PrometheusClient
    client = PrometheusClient('http://owned-tsdb', query_plan=load_plan('cache_scale_in.yaml'),
                              target_kinds={'mock': 'mock'}, max_gap_s=5)
    labels = dict(__name__='mock_context_requests_total', role='prefill',
                  engine_name='p0', engine_incarnation='one', engine_ip='127.0.0.1', grpc_port='7000')
    # A removed engine's older series must not be resurrected through lookback.
    samples = [[dict(metric={'__name__': 'up'}, values=[[10, '1'], [11, '1']])],
               [dict(metric=labels, values=[[10, '3'], [11, '4']]),
                dict(metric=dict(labels, engine_name='removed'), values=[[10, '7']])]]
    with patch.object(client, 'query', side_effect=samples) as query:
        result = client.metric_snapshot(['mock/context_requests_total'], source='mock', labels={'role': 'prefill'})
    assert len(result) == 1 and result[0]['value'] == [11, '4']
    health, values = query.call_args_list
    assert health.args[0] == 'up{job="mock"}[5000ms]'
    assert values.args[0] == '{job="mock",__name__=~"mock_context_requests_total",role="prefill"}[5000ms]'
    assert values.args[1] == 11  # Query pinned to health scrape, even if a new scrape commits meanwhile.


@pytest.mark.parametrize('health, message', [([], 'stale'),
    ([dict(metric={}, values=[[10, '0']])], 'unavailable')])
def test_snapshot_rejects_stale_or_failed_target_before_reading_metrics(health, message):
    from monitoring.client import PrometheusClient
    client = PrometheusClient('http://owned-tsdb', query_plan=load_plan('cache_scale_in.yaml'),
                              target_kinds={'mock': 'mock'}, max_gap_s=5)
    with patch.object(client, 'query', return_value=health) as query, pytest.raises(RuntimeError, match=message):
        client.metric_snapshot(['mock/context_requests_total'], source='mock')
    assert query.call_count == 1
