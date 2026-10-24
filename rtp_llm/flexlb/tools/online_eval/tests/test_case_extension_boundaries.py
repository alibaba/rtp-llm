"""Direct case metric callbacks receive detached inputs and explicit deadlines."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from scenario.context import RuntimeContext
from scenario.measurement import WorkloadPolicy
from runtime.deadline import Deadline


@pytest.fixture
def measured(tmp_path, monkeypatch):
    from cases.master_ha_failover import metrics
    store = SimpleNamespace(document={'definitions': {'ha/sent': {'producer': 'ha_evidence'}}},
        save=Mock(), series=lambda *_: ({}, {}, {}, []))
    monkeypatch.setattr('monitoring.archive.export_metrics', lambda *args: store)
    monkeypatch.setattr('monitoring.metric_store.MetricStore.read', lambda *args: store)
    instance = dict(workload_runtime={'capture_metrics': False}, collection_profile='request',
                    execution={'monitoring': {}})
    ctx = RuntimeContext(instance, None, tmp_path, lambda: 0, lambda _: None)
    policy = WorkloadPolicy()
    policy.attach(ctx)
    result = dict(id='fixture', status='PASS', cleanup=[], stages=[dict(id='gate', checks=[dict(id='ok', status='PASS')])],
                  implementation={'configuration': {'case': 'master_ha_failover'}})
    return policy, ctx, result, metrics


def test_direct_finalizer_cannot_mutate_canonical_analysis(measured, monkeypatch):
    policy, ctx, snapshot, metrics = measured
    def finalize(directory, analysis, selected, deadline):
        with pytest.raises(TypeError):
            analysis['status'] = 'FAIL'
        analysis['stages'][0]['checks'][0]['status'] = 'FAIL'
        assert selected == {'ha_evidence'}
        return {'ha_evidence': {'observed': True}}
    monkeypatch.setattr(metrics, 'finalize', finalize)
    finalized = policy.finalize(ctx, snapshot, Deadline(100, lambda: 0))
    assert finalized.analysis['stages'][0]['checks'][0]['status'] == 'PASS'
    assert snapshot['stages'][0]['checks'][0]['status'] == 'PASS'
    assert finalized.analysis['producer_results'] == {'ha_evidence': {'observed': True}}


@pytest.mark.parametrize('metadata', [None, [], {'bad': float('nan')}, {'bad': object()}])
def test_finalizer_metadata_is_json_valid(measured, monkeypatch, metadata):
    policy, ctx, snapshot, metrics = measured
    monkeypatch.setattr(metrics, 'finalize', lambda *args: metadata)
    with pytest.raises((ValueError, TypeError)):
        policy.finalize(ctx, snapshot, Deadline(100, lambda: 0))


def test_expired_measurement_never_calls_case_projection(measured, monkeypatch):
    policy, ctx, snapshot, metrics = measured
    project = Mock()
    monkeypatch.setattr(metrics, 'project', project)
    with pytest.raises(TimeoutError):
        policy.finalize(ctx, snapshot, Deadline(-1, lambda: 0))
    project.assert_not_called()


def test_finalizer_published_metrics_refresh_the_analysis(measured, monkeypatch):
    policy, ctx, snapshot, metrics = measured
    from monitoring.metric_store import MetricStore
    store = MetricStore.read(ctx.artifact_dir)
    series = {}
    store.series = lambda *_: (dict(series), {}, {}, [])
    def finalize(directory, analysis, selected, deadline):
        assert analysis['series'] == {}
        series['derived'] = [[0, 7]]
        return {'ha_evidence': {}}
    monkeypatch.setattr(metrics, 'finalize', finalize)
    finalized = policy.finalize(ctx, snapshot, Deadline(100, lambda: 0))
    assert finalized.analysis['series'] == {'derived': [[0, 7]]}
