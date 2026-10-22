"""Case callbacks cannot mutate canonical results; implementation identity covers siblings."""
from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from case_api.extension_contracts import FinalizationContext
from case_api.registry import MetricProducer, ProducerPhase
from monitoring import producers
from monitoring.contracts import MetricContractError
from scenario.program_compiler import _implementation


def prepare(monkeypatch, callback):
    capability = MetricProducer(lambda *args: {}, callback, ProducerPhase.FINALIZE)
    monkeypatch.setattr(producers, 'registry', lambda: SimpleNamespace(producers={'example': capability}))
    monkeypatch.setattr(producers.MetricStore, 'read', lambda _: SimpleNamespace(
        document={'definitions': {'example/value': {'producer': 'example'}}}))


def test_finalize_callback_has_an_isolated_snapshot(monkeypatch, tmp_path):
    canonical = {'verdict': 'FAIL', 'checks': [{'status': 'FAIL'}]}
    def callback(directory, context):
        assert isinstance(context, FinalizationContext)
        assert context.directory == tmp_path
        with pytest.raises(TypeError):
            context.run['verdict'] = 'PASS'
        context.run['checks'][0]['status'] = 'PASS'
        return {'diagnosis': 'complete'}
    prepare(monkeypatch, callback)
    producers.produce(tmp_path, canonical)
    assert canonical['verdict'] == 'FAIL'
    assert canonical['checks'][0]['status'] == 'FAIL'
    assert canonical['producer_results'] == {'example': {'diagnosis': 'complete'}}


@pytest.mark.parametrize('metadata', [None, {'value': float('nan')}, {'value': object()}])
def test_invalid_metadata_is_not_published(monkeypatch, tmp_path, metadata):
    prepare(monkeypatch, lambda directory, context: metadata)
    canonical = {'verdict': 'FAIL'}
    with pytest.raises((MetricContractError, ValueError, TypeError)):
        producers.produce(tmp_path, canonical)
    assert 'producer_results' not in canonical
    assert canonical['verdict'] == 'FAIL'


def test_both_phases_use_selected_metric_definitions(monkeypatch, tmp_path):
    calls = []
    capabilities = {
        'selected_gate': MetricProducer(lambda *args: {}, lambda directory: calls.append('gate'), ProducerPhase.GATE),
        'unused_gate': MetricProducer(lambda *args: {}, lambda directory: calls.append('unused'), ProducerPhase.GATE),
        'selected_finalize': MetricProducer(lambda *args: {},
            lambda directory, context: calls.append(context.run['verdict']) or {'done': True}, ProducerPhase.FINALIZE),
    }
    monkeypatch.setattr(producers, 'registry', lambda: SimpleNamespace(producers=capabilities))
    monkeypatch.setattr(producers.MetricStore, 'read', lambda _: SimpleNamespace(document={'definitions': {
        'trial/gate': {'producer': 'selected_gate'},
        'trial/gate_duplicate': {'producer': 'selected_gate'},
        'trial/finalize': {'producer': 'selected_finalize'},
    }}))
    producers.produce(tmp_path, {}, phase=ProducerPhase.GATE)
    canonical = {'verdict': 'FAIL'}
    producers.produce(tmp_path, canonical)
    assert calls == ['gate', 'FAIL']
    assert canonical['producer_results'] == {'selected_finalize': {'done': True}}


def test_gate_phase_obeys_deadline_before_callback(monkeypatch, tmp_path):
    def expired():
        raise TimeoutError('expired')
    callbacks = []
    prepare(monkeypatch, lambda directory: callbacks.append(directory))
    monkeypatch.setattr(producers, 'registry', lambda: SimpleNamespace(producers={
        'example': MetricProducer(lambda *args: {}, lambda directory: callbacks.append(directory), ProducerPhase.GATE)}))
    with pytest.raises(TimeoutError):
        producers.produce(tmp_path, {}, phase=ProducerPhase.GATE, deadline=SimpleNamespace(check=expired))
    assert not callbacks


def test_case_fingerprint_covers_business_modules_and_is_deterministic(tmp_path):
    program = tmp_path / 'program.py'
    program.write_text('CASE = None\n')
    analysis = tmp_path / 'analysis.py'
    analysis.write_text('threshold = 1\n')
    config = SimpleNamespace(declaration={})
    module = SimpleNamespace(path=program)
    first = _implementation(config, module, 'example')
    assert first == _implementation(config, module, 'example')
    assert set(first['case_sources']) == {'program.py', 'analysis.py'}
    analysis.write_text('threshold = 2\n')
    changed = _implementation(config, module, 'example')
    assert first['sha256'] == changed['sha256']
    assert first['case_sources_sha256'] != changed['case_sources_sha256']
    json.dumps(changed)


def test_capability_sharing_requires_explicit_declaration():
    from case_api.registry import CaseDefinition, CaseRegistry, RegisteredCase, ReportView
    view = ReportView(lambda path, data, fail: None, lambda directory, analysis, presentation: directory)
    def entry(name, views):
        return RegisteredCase(name, name + '.program', Path(__file__),
                              CaseDefinition(builders={'default': lambda case: None}, report_views=views))
    registry = CaseRegistry.build([entry('owner', {'shared.yaml': view}), entry('foreign', {})])
    with pytest.raises(ValueError, match='not declared by case'):
        registry.capability('view', 'shared.yaml', case='foreign')
    shared = CaseRegistry.build([entry('owner', {'shared.yaml': view}),
                                 entry('consumer', {'shared.yaml': view})])
    assert shared.capability('view', 'shared.yaml', case='consumer') is view


def test_compiler_rejects_another_cases_view():
    import yaml
    from scenario.program_compiler import configure_program
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / 'config/scenarios/master_performance.yaml').read_text())
    data['reports'] = ['master_ha_failover.yaml']
    with pytest.raises(ValueError, match='not declared by case'):
        configure_program(data, 'foreign-view.yaml')
