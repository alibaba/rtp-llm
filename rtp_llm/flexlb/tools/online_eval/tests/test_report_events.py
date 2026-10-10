"""Events are frozen observations, with shared selection and clock semantics."""

import copy
import json
from unittest.mock import patch

import pytest

from reporting.events import attach_events, project_events
from reporting.view_config import view
from scenario.loader import ScenarioError
from scenario.runtime import RuntimeContext
from workload.report import write_views
from test_workload_report_views import payload
from test_cache_scale_gate import CacheGateTest
from cases.cache_scale_in.publication import publish_cache
from cases.cache_scale_in.analysis import analyze
from monitoring.metric_store import MetricStore


def presentation():
    return dict(charts=dict(events={
        'shutdown': dict(label='服务停止', source='stage', stage='stop', boundary='end'),
        'withdraw': dict(label='开始撤出', source='case', event='withdraw_start'),
    }, event_ids=['shutdown'], panels=[
        dict(id='selected', event_ids=['withdraw']), dict(id='hidden', event_ids=[]),
        dict(id='inherited'),
    ]))


def test_stage_boundary_and_custom_event_use_recorded_times_and_explicit_panel_selection():
    spec = dict(panels=[dict(id=identity) for identity in ('selected', 'hidden', 'inherited')])
    phases = [dict(stage='stop', event='start', epoch_s=100),
              dict(stage='stop', event='end', epoch_s=108, status='PASS')]
    events = [dict(id='withdraw_start', epoch_s=104), dict(id='unselected', epoch_s=109)]
    attach_events(spec, presentation(), origin=100, phases=phases, events=events)
    assert [(event['id'], event['t']) for event in spec['events']] == [('withdraw', 4), ('shutdown', 8)]
    assert [event['id'] for event in spec['panels'][0]['events']] == ['withdraw']
    assert spec['panels'][1]['events'] == []
    assert [event['id'] for event in spec['panels'][2]['events']] == ['shutdown']
    assert spec['timeOriginEpochS'] == 100
    assert spec['events'][0]['source']['record'] == events[0]
    assert spec['events'][1]['source']['record']['status'] == 'PASS'
    assert project_events(presentation(), origin=100) == []
    spec['panels'][0]['events'][0]['name'] = 'display mutation'
    assert spec['events'][0]['name'] == '开始撤出'


@pytest.mark.parametrize('epoch', [None, True, float('nan'), '104'])
def test_invalid_recorded_time_fails_instead_of_becoming_zero(epoch):
    with pytest.raises(ValueError, match='finite epoch_s'):
        project_events(presentation(), origin=100,
                       events=[dict(id='withdraw_start', epoch_s=epoch)])


@pytest.mark.parametrize('mutation', ['unknown', 'duplicate', 'bad_boundary', 'timestamp', 'wrong_source'])
def test_event_declarations_are_validated_in_shared_loader(mutation):
    data = copy.deepcopy(view('master_ha_core.yaml'))
    if mutation == 'unknown':
        data['charts']['panels'][0]['event_ids'] = ['missing']
    elif mutation == 'duplicate':
        data['charts']['panels'][0]['event_ids'] = ['kill_a', 'kill_a']
    elif mutation == 'bad_boundary':
        data['charts']['events']['kill_a']['boundary'] = 'success'
    elif mutation == 'timestamp':
        data['charts']['events']['kill_a']['t'] = 12
    else:
        data['charts']['events']['kill_a']['source'] = 'logs'
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError):
            view('master_ha_core.yaml')


def test_runtime_case_event_records_both_clocks_and_isolates_returned_record(tmp_path):
    ctx = RuntimeContext({}, None, tmp_path, clock=lambda: 50, sleeper=lambda _: None)
    with patch('scenario.runtime.time.time', return_value=1000):
        event = ctx.record_event('withdraw_start')
    assert event == dict(id='withdraw_start', epoch_s=1000, monotonic_s=50)
    event['id'] = 'mutated'
    assert ctx.report_events[0]['id'] == 'withdraw_start'
    with pytest.raises(ValueError, match='identity'):
        ctx.record_event('display label')


def test_canonical_reassembly_keeps_native_chart_origin_for_case_events(tmp_path):
    evidence = CacheGateTest().evidence()
    evidence['events'] = [dict(id='withdraw_start', epoch_s=1020, t=20)]
    bundle_spec = publish_cache(tmp_path, evidence, analyze(evidence))
    assert bundle_spec['events'][0]['t'] == 20
    analysis = payload()
    analysis['events'] = evidence['events']
    # Workload and the case's observation window begin at different times.
    analysis['clock_anchor'] = dict(epoch_s=900)
    paths = write_views(tmp_path, analysis, ['cache_scale_in_overview.yaml'])
    spec = json.loads((paths['cache_scale_in_overview.yaml'].parent/'report-spec.json').read_text())
    assert spec['events'][0]['t'] == 20
    assert spec['panels'][0]['events'][0]['t'] == 20
    assert spec['timeOriginEpochS'] == 1000


def test_default_empty_and_all_metrics_share_one_view(tmp_path):
    empty = write_views(tmp_path/'empty', payload())
    assert list(empty) == ['default.yaml']
    spec = json.loads((empty['default.yaml'].parent/'report-spec.json').read_text())
    assert spec['panels'] == []
    with_metrics = payload({
        '1/mock/qps/{}': [[0, 1]], '1/case/derived/hit/{}': [[0, .8]],
    })
    paths = write_views(tmp_path/'metrics', with_metrics)
    spec = json.loads((paths['default.yaml'].parent/'report-spec.json').read_text())
    assert len(spec['panels']) == 2
    assert len(paths) == 1


def test_frozen_inventory_includes_python_producer_metrics():
    store = MetricStore(dict(metrics_schema_version=1, definitions={
        'derived/hit': dict(unit='ratio'),
    }, metrics={'derived/hit': [dict(epoch='1', source='case', metric_id='derived/hit', labels={},
                                       points=[[100, .8]], provenance=dict(source_type='derived'))]},
                            collection_gaps={}, errors=[]))
    series, sources, _, _ = store.series(100)
    assert series == {'1/case/derived/hit/{}': [[0, .8]]}
    assert sources['1/case/derived/hit/{}']['metric_id'] == 'derived/hit'
    assert sources['1/case/derived/hit/{}']['unit'] == 'ratio'


@pytest.mark.parametrize('legacy', ['ha', 'produced', 'checks'])
def test_view_kind_is_shared_instead_of_case_specific(legacy):
    data = copy.deepcopy(view('master_ha_core.yaml'))
    assert data['kind'] == 'selected'
    data['kind'] = legacy
    with patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError, match='default or selected'):
            view('master_ha_core.yaml')


def test_compilation_rejects_event_stage_typo_before_execution():
    from cases.config import configure_program
    from scenario.loader import load_document
    from reporting.view_config import VIEWS
    config = load_document(VIEWS.parent/'scenarios/master_ha_failover.yaml')
    data = copy.deepcopy(view('master_ha_core.yaml'))
    data['charts']['events']['kill_a']['stage'] = 'unknown_stage'
    with patch('reporting.view_config.view', return_value=data):
        with pytest.raises(ScenarioError, match='references unknown stage'):
            configure_program(config, 'fixture')


def test_failed_stage_marker_retains_failure_in_display_and_evidence():
    events = project_events(presentation(), origin=100,
                            phases=[dict(stage='stop', event='end', epoch_s=108, status='ERROR')])
    assert events[0]['name'] == '服务停止 · ERROR'
    assert events[0]['source']['record']['status'] == 'ERROR'
