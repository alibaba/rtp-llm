"""Display clock changes coordinates and coverage, never the gate population."""

import copy
from pathlib import Path

import pytest

from cases.config import configure_program
from reporting.timeline import apply, freeze, traffic_event, validate_configuration
from runtime.client_journal import first_request
from scenario.loader import ScenarioError, load_document

ROOT = Path(__file__).resolve().parents[1]


def declaration(origin=None, start='origin'):
    return dict(time_axis=dict(origin=origin or {'event': 'traffic_started'},
        range={'from': start, 'until': {'stage': 'drain', 'boundary': 'end'}}))


def events():
    return [dict(id='traffic_started', epoch_s=100), dict(id='fault_injected', epoch_s=110)]


def phases():
    return [dict(stage='drain', event='end', epoch_s=130)]


def spec():
    return dict(timeOriginEpochS=120, events=[dict(id='fault', t=-10)],
        panels=[dict(id='timeline', timeX=True, events=[dict(id='fault', t=-10)], series=[
            dict(points=[dict(x=-20, y=1), dict(x=0, y=2), dict(x=10, y=None)])])])


def test_onset_uses_first_issued_including_failed_and_unfinished_requests():
    rows = [dict(rid='later', send_start_epoch_ms=105000, status='ok'),
            dict(rid='failed', send_start_epoch_ms=100000, status='exception'),
            dict(rid='unfinished', send_start_epoch_ms=99000, status='scheduled')]
    stamp = first_request(rows)
    assert stamp['epoch_s'] == 99 and stamp['rid'] == 'unfinished'
    resources = [dict(resource={'id': 'second'}, traffic_start=first_request(rows[:1])),
                 dict(resource={'id': 'first'}, traffic_start=stamp)]
    event = traffic_event(resources)
    assert event['epoch_s'] == 99 and event['source']['resource']['id'] == 'first'
    assert traffic_event([]) is None
    with pytest.raises(ValueError, match='positive finite'):
        first_request([dict(send_start_epoch_ms=float('nan'))])


def test_whole_run_uses_same_origin_for_curves_events_and_range():
    timeline = freeze(declaration(), events=events(), phases=phases())
    frozen = copy.deepcopy(timeline)
    source = spec()
    result = apply(copy.deepcopy(source), dict(status='PASS', report_timeline=timeline))
    assert result['timeOriginEpochS'] == 100
    assert result['timeAxis'] == {'min': 0, 'max': 30}
    assert [p['x'] for p in result['panels'][0]['series'][0]['points']] == [0, 20, 30]
    assert result['events'][0]['t'] == result['panels'][0]['events'][0]['t'] == 10
    assert source == spec() and timeline == frozen


def test_fault_origin_can_keep_pre_fault_warmup_with_negative_coordinates():
    config = declaration({'event': 'fault_injected'}, {'event': 'traffic_started'})
    timeline = freeze(config, events=events(), phases=phases())
    result = apply(spec(), dict(status='PASS', report_timeline=timeline))
    assert result['timeAxis'] == {'min': -10, 'max': 20}
    assert result['panels'][0]['series'][0]['points'][0]['x'] == -10
    assert result['events'][0]['t'] == 0


def test_stage_origin_and_boundary_are_resolved_from_actual_phase_records():
    config = declaration({'stage': 'drain', 'boundary': 'start'})
    validate_configuration(config, {'drain'})
    timeline = freeze(config, events=[], phases=phases() + [dict(stage='drain', event='start', epoch_s=120)])
    assert timeline['origin']['epoch_s'] == 120


def test_missing_anchor_does_not_silently_use_workload_or_measurement_clock():
    timeline = freeze(declaration(), events=[], phases=phases())
    assert timeline['status'] == 'UNAVAILABLE'
    with pytest.raises(ValueError, match='did not occur'):
        apply(spec(), dict(status='PASS', report_timeline=timeline))
    failed = apply(spec(), dict(status='ERROR', report_timeline=timeline))
    assert 'timeOriginEpochS' not in failed and 'timeAxis' not in failed
    assert failed['panels'][0]['series'] == [] and failed['events'] == []
    assert failed['reportTimeline']['missing'] == [{'event': 'traffic_started'}]


def test_duplicate_event_and_reversed_range_fail_loud():
    with pytest.raises(ValueError, match='ambiguous'):
        freeze(declaration(), events=events() + events()[:1], phases=phases())
    with pytest.raises(ValueError, match='increasing'):
        freeze(declaration(), events=events(), phases=[dict(stage='drain', event='end', epoch_s=90)])


@pytest.mark.parametrize('name,end', [('master_performance', 'drain'), ('cache_scale_in', 'drain'),
                                     ('master_ha_failover', 'finish')])
def test_yaml_time_axis_is_frozen_into_every_variant_and_does_not_change_stages(name, end):
    config = load_document(ROOT / f'config/scenarios/{name}.yaml')
    document = configure_program(config, name)
    plain = copy.deepcopy(config)
    del plain['reporting']
    baseline = configure_program(plain, name)
    for variant, original in zip(document['variants'], baseline['variants']):
        assert variant['reporting']['time_axis']['origin'] == {'event': 'traffic_started'}
        assert variant['reporting']['time_axis']['range']['until'] == {'stage': end, 'boundary': 'end'}
        assert variant['stages'] == original['stages']
    config['reporting']['time_axis']['origin'] = {'stage': 'missing', 'boundary': 'start'}
    with pytest.raises(ScenarioError, match='unknown stage'):
        configure_program(config, name)


def test_performance_display_includes_warmup_and_drain_without_changing_measurement():
    from cases.master_performance.metrics import values
    evidence = dict(window={'start_epoch_ms': 110000, 'end_epoch_ms': 120000}, flow={'records': [
        dict(rid='warmup', send_start_epoch_ms=100000, status='ok', total_ms=1000),
        dict(rid='measure', send_start_epoch_ms=115000, status='ok', total_ms=1000),
        dict(rid='drain', send_start_epoch_ms=119000, status='ok', total_ms=5000),
    ]})
    calculation = dict(calculator='request_rate', window='traffic',
                       selection={'time_basis': 'arrival', 'status': 'all'}, bucket_s=1)
    definitions = {'request/sent': {'producer': 'performance_requests', 'calculation': calculation}}
    original = copy.deepcopy(evidence)
    displayed = values(evidence, definitions)['request/sent']
    assert displayed[0] == [100, 1] and displayed[-1][0] == 123
    assert sum(value for _, value in displayed) == 3
    definitions['request/sent']['calculation']['window'] = 'measurement'
    measured = values(evidence, definitions)['request/sent']
    assert measured[0][0] == 110 and measured[-1][0] == 119
    assert sum(value for _, value in measured) == 2
    assert evidence == original


def test_offline_publication_uses_frozen_clock_and_event_records(tmp_path):
    from monitoring.metric_store import MetricStore
    from reporting.timeline import archived
    context = dict(report_timeline=freeze(declaration(), events=events(), phases=phases()),
                   events=events(), phases=phases())
    MetricStore(dict(metrics_schema_version=1, definitions={}, metrics={}, run=context)).save(tmp_path)
    presentation = dict(charts=dict(events={'fault': dict(label='Fault', source='case', event='fault_injected')},
                                    event_ids=['fault']))
    result = archived(spec(), tmp_path, presentation, status='PASS')
    assert result['timeOriginEpochS'] == 100
    assert result['events'][0]['t'] == 10
    assert result['timeAxis'] == {'min': 0, 'max': 30}
    assert result['panels'][0]['events'][0]['epoch_s'] == 110


def test_runtime_freezes_resource_onset_into_evidence_metrics_analysis_and_report(tmp_path):
    import json
    from types import SimpleNamespace
    from runtime.resource_evidence import ResourceEvidence
    from scenario.context import RuntimeContext
    from workload.runtime import WorkloadPolicy

    instance = dict(reporting=declaration(), execution={'monitoring': {}},
                    workload_runtime={'capture_metrics': False}, collection_profile='request')
    ctx = RuntimeContext(instance, None, tmp_path, lambda: 1, lambda _: None)
    stamp = first_request([dict(rid='first-failed', send_start_epoch_ms=100000, status='exception')])
    ctx.register_resource('java_flow', None, evidence=lambda *_: ResourceEvidence(
        requests={'traffic_start': stamp}, producer='client'))
    policy = WorkloadPolicy()
    policy.attach(ctx)
    policy.events = [dict(row, action="java_flow_drain", env_epoch=1) for row in phases()]
    result = dict(id='fixture::default::single-nonbatch', status='PASS', cleanup=[],
                  checks=[dict(stage='gate', id='criterion', status='PASS')],
                  stages=[dict(id='gate', status='PASS', checks=[dict(id='criterion', status='PASS')])])
    deadline = SimpleNamespace(check=lambda: None)
    analysis = policy.finalize(ctx, result, deadline)
    policy.render(ctx, result, analysis, deadline)
    artifacts = [json.loads((tmp_path/name).read_text()) for name in ['workload-evidence.json', 'metrics.json']]
    timelines = [artifacts[0]['report_timeline'], artifacts[1]['run']['report_timeline'], analysis['report_timeline']]
    assert all(timeline == timelines[0] for timeline in timelines)
    assert timelines[0]['origin']['record']['source']['request']['rid'] == 'first-failed'
    html = Path(result['workload']['report'])
    report = json.loads(html.with_name('report-spec.json').read_text())
    assert report['timeOriginEpochS'] == 100 and report['timeAxis'] == {'min': 0, 'max': 30}
    assert result['status'] == 'PASS' and result['workload']['runtime_validity'] == 'VALID'
