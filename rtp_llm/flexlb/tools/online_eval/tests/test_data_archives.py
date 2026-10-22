"""Data archives cross behaviors without changing mode identity or hiding failures."""
import copy
from pathlib import Path

import pytest

from document_io import load_document, DocumentError
from scenario.program_compiler import configure_program
from scenario.compiler import compile_scenarios
from scenario.catalog import handlers
from cases.cache_scale_in.report import request_outcomes

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return load_document(ROOT / 'config/scenarios/cache_scale_in.yaml')


def test_both_archives_cross_both_behaviors_and_modes(monkeypatch):
    monkeypatch.setattr('scenario.compiler.VICTIM_OFFSETS', (700, 701, 702))
    monkeypatch.setattr('adapters.java_mock_resources.VICTIM_OFFSETS', (700, 701, 702))
    data = configuration()
    before = copy.deepcopy(data)
    document = configure_program(data, 'cache_scale_in.yaml')
    assert data == before
    variants = {v['id']: v for v in document['variants']}
    assert set(variants) == {'default', 'affinity_disabled', 'glm', 'glm__affinity_disabled'}
    plans = compile_scenarios([('cache.yaml', document)], handlers=handlers())
    assert len(plans) == 8
    for plan in plans:
        vid = plan['variant_id']
        glm = vid.startswith('glm')
        negative = vid.endswith('affinity_disabled')
        env = plan['environment']
        assert (env['n_prefill'], env['n_decode']) == ((125, 536) if glm else (48, 192))
        observe = next(s['params']['criteria'] for s in plan['stages'] if s['id'] == 'scale_in')
        assert observe['target_p'] == (64 if glm else 24)
        assert observe['qps'] == (240 if glm else 900)
        assert observe['baseline_min_hit'] == (0.85 if glm else 0.46)
        assert observe['absolute_min_hit'] == (0.45 if glm else 0)
        assert observe['max_drop'] == (0.25 if glm else 0.1)
        assert observe['warmup_min_s'] == (900 if glm else 240)
        assert observe['warmup_timeout_s'] == (1050 if glm else 300)
        assert observe['post_affinity_ms'] == (0 if negative else 1e9)
        assert observe['expected_behavior'] == ('collapse' if negative else 'stable')
        assert observe['sustain_s'] == 30
        assert observe['observe_s'] == 240
        flow = next(s['params'] for s in plan['stages'] if s['id'] == 'traffic')
        assert flow['source']['parameters']['count'] == (587711 if glm else 844864)
        assert ('glm-5.3' if glm else 'deepseek_v4_flash') in flow['source']['parameters']['path']
        resolved = env['resolved_config']
        if plan['profile'] == 'batch-window':
            assert resolved['dispatcher']['maxInflightPerPrefillWorker'] == 2
            decision = resolved['scheduler']['decision']
            assert decision['maxCollectionWaitMs'] == (10 if glm else 350)
            assert decision['maxPredictedExecutionMs'] == (550 if glm else 500)
            assert decision['maxRequests'] == 32
        assert plan['implementation']['variant_axes'][vid] == dict(
            data_archive='glm' if glm else 'deepseek', variant='affinity_disabled' if negative else 'default')


@pytest.mark.parametrize('archives', [
    [], {}, {'default': 'x', 'entries': {}},
    {'default': 'missing', 'entries': {'x': {}}},
    {'default': 'x', 'entries': {'x': {}, 'bad/id': {}}},
    {'default': 'x', 'entries': {'x': {}, 'glm': {'program': 'other'}}},
    {'default': 'x', 'entries': {'x': {}, 'glm': {'parameters': []}}},
])
def test_archive_schema_is_strict(archives):
    data = configuration()
    data['data_archives'] = archives
    with pytest.raises(DocumentError):
        configure_program(data, 'cache.yaml')


def test_archive_collisions_and_identity_changes_are_rejected(monkeypatch):
    data = configuration()
    data['data_archives']['entries']['affinity_disabled'] = {}
    with pytest.raises(DocumentError, match='duplicate configuration id'):
        configure_program(data, 'cache.yaml')
    data = configuration()
    data['data_archives']['entries']['glm']['environment']['profile_overrides']['batch-window']['decision'] = 'single'
    monkeypatch.setattr('scenario.compiler.VICTIM_OFFSETS', (700, 701, 702))
    monkeypatch.setattr('adapters.java_mock_resources.VICTIM_OFFSETS', (700, 701, 702))
    with pytest.raises(ValueError, match='identity field'):
        compile_scenarios([('cache.yaml', configure_program(data, 'cache.yaml'))], handlers=handlers())


def test_archive_inputs_and_numeric_constraints_are_not_bypassed():
    data = configuration()
    data['data_archives']['entries']['glm']['parameters']['procedure']['target_p'] = -1
    with pytest.raises(ValueError):
        configure_program(data, 'cache.yaml')
    data = configuration()
    data['data_archives']['entries']['glm']['parameters']['misspelled'] = 1
    with pytest.raises(ValueError, match='unknown configuration fields'):
        configure_program(data, 'cache.yaml')


def test_request_failures_include_removed_unknown_and_late_timeouts():
    evidence = dict(client_attribution=dict(complete=True, issued_count=100, terminal_count=100,
        counts=dict(survivor=dict(sent=90, terminal=90, failed=0),
                    removed=dict(sent=6, terminal=6, failed=6),
                    unknown=dict(sent=4, terminal=4, failed=4)),
        failures=[dict(error='queue stopped') for _ in range(6)] +
                 [dict(error='REQUEST_INACTIVE') for _ in range(4)]))
    rows, kpis = request_outcomes(evidence)
    assert rows[0] == ['整轮（完整账本）', 100, 100, 10]
    assert rows[-1] == ['unknown', 4, 4, 4]
    assert [k['value'] for k in kpis] == [10, 4]
    assert all(k['tone'] == 'danger' for k in kpis)
    assert all(k['value'] == '不可确认' for k in request_outcomes({})[1])


def test_authoring_explanation_uses_archive_then_behavior(monkeypatch):
    from scenario.case_validation import validate_cases
    monkeypatch.setattr('scenario.compiler.VICTIM_OFFSETS', (700, 701, 702))
    monkeypatch.setattr('adapters.java_mock_resources.VICTIM_OFFSETS', (700, 701, 702))
    report = validate_cases(ROOT / 'config/scenarios/cache_scale_in.yaml')
    for instance in report['instances']:
        axes = instance['variant_axes']
        parameters = instance['parameters']
        assert parameters['checks']['baseline_hit']['expected'] == (0.85 if axes['data_archive']=='glm' else 0.46)
        assert parameters['traffic']['client']['playback']['qps'] == (240 if axes['data_archive']=='glm' else 900)
        assert parameters['checks']['collapse']['expected'] == (1 if axes['variant']=='affinity_disabled' else 0)
