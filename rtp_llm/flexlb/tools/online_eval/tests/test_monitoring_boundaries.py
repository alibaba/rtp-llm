"""Failed publication and exhausted archive budgets retain authoritative evidence."""

import json
import subprocess
import sys
from io import BytesIO
from unittest.mock import Mock, patch

import pytest

from monitoring.archive import export_metrics
from monitoring.metric_store import MetricStore, MetricContractError, atomic_json
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


def test_probe_failed_export_retains_completed_chunks(tmp_path):
    from monitoring.probe import PrometheusEvidence
    session = Mock(interval=1)
    session.query.side_effect = [
        [dict(metric={'__name__': 'online_eval_debug_alive', 'host': 'a'},
              values=[[10, '1']])], TimeoutError('budget exhausted')]
    probe = PrometheusEvidence(tmp_path / 'samples.jsonl', {'a': Mock(url='http://unused/status')},
                               fields=['alive'], label='host', source='debug',
                               session=session, limits={'max_samples': 100, 'max_bytes': 10000})
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
    atomic_json(tmp_path / 'metrics.json', previous)
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
    assert session.target_kind('arbitrary-name') == 'mock'


def test_archive_budget_keeps_partial_queries_and_still_reaps_process(tmp_path):
    session = PrometheusSession(tmp_path, {'mock': 'http://unused/metrics'},
                                target_kinds={'mock': 'mock'})
    session.started = 1
    session.process = Mock()
    session.process.poll.return_value = None
    session.log = Mock()
    now = [0.0]
    deadline = Deadline(1.0, clock=lambda: now[0])
    calls = []
    def query(endpoint, **kwargs):
        calls.append(kwargs)
        assert kwargs['deadline'] is deadline
        now[0] += .6
        return {'result': []}
    with patch('monitoring.session.Deadline', return_value=deadline), \
         patch.object(session, 'api', side_effect=query):
        with pytest.raises(TimeoutError):
            session.stop(materialize=False)
    assert len(calls) == 2  # Later chunks and queries do not get fresh budgets.
    archive = json.loads((tmp_path / 'queries.json').read_text())
    assert archive['errors'] and not (tmp_path / 'metrics.json').exists()
    session.process.terminate.assert_called_once()
    session.process.kill.assert_called_once()
    session.process.wait.assert_called_once_with(timeout=1)
    session.log.close.assert_called_once()
    assert session.closed


def test_query_http_timeout_is_bounded_by_remaining_budget(tmp_path):
    session = PrometheusSession(tmp_path, {}, metric_plan=load_plan('default.yaml'))
    session.url = 'http://owned-tsdb'
    deadline = Deadline(.2, clock=lambda: 0)
    with patch('monitoring.client.urllib.request.urlopen',
               return_value=BytesIO(b'{"status":"success","data":{"result":[]}}')) as request:
        assert session.api('query', timeout=5, deadline=deadline, query='up') == {'result': []}
    assert request.call_args.kwargs['timeout'] == .2


def test_calculator_identity_is_explicit_and_not_derived_from_method():
    from analysis.request_metrics import describe_calculation
    contract = describe_calculation(dict(calculator='request_rate', window='measurement',
        selection=dict(time_basis='arrival', status='all'), bucket_s=1), windows={'measurement'})
    assert contract['measurement']['method'] == 'request_rate'
    assert contract['implementation'] == dict(module='analysis.request_metrics', function='_request_rate')
