"""Gate windows and thresholds survive projection, replay and missing samples."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from analysis.gate_references import definition, freeze_reference, window
from reporting.gate_lines import apply_gate_lines, validate_bindings, check_references
from reporting.view_model import ViewDefinition


def presentation(scale=100):
    return ViewDefinition.read(dict(kind='selected', report={'subtitle': 'test'}, charts=dict(curves={'ratio': dict(
        name='ratio', group='measurements', metric_id='test/ratio', labels={},
        axis='percent', source_unit='ratio', unit='%', scale=scale)}, panels=[dict(
            id='ratio-panel', title='Ratio', curve_ids=['ratio'],
            axes={'percent': dict(title='%', position='left')},
            gate_lines=[dict(check='check', reference='floor', curve_id='ratio')])]), sections={}))


def reference(spans=None, value=.95):
    return freeze_reference(definition('check', 'floor', 'test/ratio', 'ratio',
        description='whole-window statistic; not a per-sample assertion'), value,
        spans or [window('one', 100, 110)], op='ge')


def test_multiple_windows_are_separate_lines_with_source_unit_scale_and_no_mutation():
    view = presentation()
    frozen = reference([window('one', 100, 110), window('two', 120, 130)])
    original = copy.deepcopy(frozen)
    spec = dict(timeOriginEpochS=90, panels=[dict(id='ratio-panel', series=[], caption='no samples')])
    result = apply_gate_lines(spec, view, [frozen])
    assert [s['points'] for s in result['panels'][0]['series']] == [
        [dict(x=10, y=95), dict(x=20, y=95)], [dict(x=30, y=95), dict(x=40, y=95)]]
    assert all(s['source_type'] == 'configuration' for s in result['panels'][0]['series'])
    assert result['panels'][0]['caption'] == 'no samples'
    assert spec['panels'][0]['series'] == [] and frozen == original
    assert result['panels'][0]['series'][0]['provenance']['windows'] == original['windows']


@pytest.mark.parametrize('bad', [dict(value=float('nan')), dict(value=True),
    dict(windows=[window('reversed', 110, 100)]),
    dict(windows=[window('one', 100, 110), window('overlap', 105, 130)]),
    dict(windows=[window('inf', 0, float('inf'))]), dict(windows=[])])
def test_invalid_frozen_references_fail_before_rendering(bad):
    value = dict(reference(), **bad)
    with pytest.raises(ValueError):
        apply_gate_lines(dict(timeOriginEpochS=0, panels=[dict(id='ratio-panel', series=[], caption='')]),
                         presentation(), [value])


def test_unavailable_or_unexecuted_reference_never_becomes_full_timeline_line():
    spec = dict(timeOriginEpochS=90, timeAxis=dict(min=0, max=1000),
                panels=[dict(id='ratio-panel', series=[], caption='no samples')])
    frozen = freeze_reference(definition('check', 'floor', 'test/ratio', 'ratio', description='floor'),
                              .95, [], op='ge', status='UNAVAILABLE', reason='window event missing')
    for refs in ([], [frozen]):
        result = apply_gate_lines(spec, presentation(), refs)
        panel = result['panels'][0]
        assert panel['series'] == []
        assert panel['gate_lines'][0]['status'] == 'UNAVAILABLE'
        assert '不可用' in panel['caption']


@pytest.mark.parametrize('change', [dict(metric_id='wrong/ratio'), dict(unit='ms'), dict(labels={'role': 'wrong'})])
def test_semantic_mismatch_cannot_bind_by_axis_or_display_name(change):
    spec = dict(timeOriginEpochS=0, panels=[dict(id='ratio-panel', series=[], caption='')])
    with pytest.raises(ValueError, match='match'):
        apply_gate_lines(spec, presentation(), [dict(reference(), **change)])


@pytest.mark.parametrize('change', [dict(check='unknown'), dict(reference='unknown'),
                                  dict(curve_id='not-in-panel'), dict(value=.2), dict(hidden='true')])
def test_invalid_binding_is_rejected_by_configuration_or_catalog(change):
    raw = presentation().to_dict()
    raw['charts']['panels'][0]['gate_lines'][0].update(change)
    with pytest.raises(ValueError):
        view = ViewDefinition.read(raw)
        for panel in view.charts.panels:
            panel.validate_bindings(view.charts.curves)
        validate_bindings(view, [definition('check', 'floor', 'test/ratio', 'ratio', description='floor')])


def test_report_clock_projection_moves_references_and_samples_together():
    from reporting.timeline import apply
    spec = apply_gate_lines(dict(timeOriginEpochS=90, panels=[dict(
        id='ratio-panel', timeX=True, series=[dict(points=[dict(x=10, y=93)])], caption='')]),
        presentation(), [reference()])
    result = apply(spec, dict(status='PASS', report_timeline=dict(status='AVAILABLE',
        origin=dict(epoch_s=95, selector={'event': 'traffic_started'}),
        start=dict(epoch_s=95), end=dict(epoch_s=140))))
    assert result['panels'][0]['series'][0]['points'] == [dict(x=5, y=93)]
    assert result['panels'][0]['series'][1]['points'] == [dict(x=5, y=95), dict(x=15, y=95)]
    assert result['timeAxis'] == dict(min=0, max=45)


def test_shared_check_model_retains_references_in_nested_frozen_checks():
    from reporting.run_model import RunPresentation
    frozen = reference()
    run = RunPresentation.read(dict(id='test', checks=[dict(stage='gate', id='outer', status='PASS',
        evidence={'checks': [dict(id='inner', status='PASS', evidence={'gate_references': [frozen]})]})]))
    assert check_references(run.checks) == [frozen]


def test_performance_reference_uses_effective_profile_and_absolute_measurement_window():
    from cases.master_performance.analysis import gate_references
    from cases.master_performance.inputs import effective_profile
    criteria = dict(engine_tps={'mock/rtp_llm_context_tps': 50000},
                    engine_tps_by_profile={'single-nonbatch': {'mock/rtp_llm_context_tps': 45000}},
                    max_ttft_p99_ms=2000)
    effective = effective_profile(criteria, 'single-nonbatch')
    frozen = gate_references(effective, dict(start_epoch_ms=1180000, end_epoch_ms=1360000))
    assert frozen[0]['value'] == 45000
    assert frozen[0]['windows'] == [window('measurement', 1180, 1360)]
    assert frozen[-1]['value'] == 2000 and frozen[-1]['op'] == 'le'


def test_cache_dynamic_floor_and_population_are_frozen_for_their_own_windows():
    from cases.cache_scale_in.analysis import gate_references
    refs = gate_references(dict(observation_origin_epoch_s=1000, criteria=dict(
        baseline_min_hit=.85, sustain_s=30), baseline_start=40, baseline_end=100,
        post_start=120, post_end=360), .62)
    assert refs[0]['metric_id'] == 'derived/baseline_hit_ratio' and refs[0]['value'] == .85
    assert refs[0]['windows'] == [window('baseline', 1040, 1100)]
    assert refs[1]['value'] == .62 and refs[1]['windows'] == [window('post', 1120, 1360)]
    assert refs[1]['op'] == 'lt'  # The detection predicate, independent of positive/negative-control expectation.
    assert 'sustain_s=30' in refs[1]['description']


def test_ha_empty_cohort_keeps_declared_window_not_observed_bounds():
    from cases.master_ha_failover.actions import _client_check
    scope = window('baseline', 100, 120)
    ctx = SimpleNamespace(resource=lambda key, kind: [] if kind == 'ha_rows' else scope)
    result = _client_check(ctx, dict(rows='rows', scope='scope', check_id='baseline_success',
        metric='ha_gate/success_rate', op='ge', expected=.95, min_samples=100, unit='ratio'), Mock())
    check, = result.checks
    assert check.status == 'ERROR'
    assert check.evidence['sample_count'] == 0
    assert check.evidence['gate_references'][0]['windows'] == [scope]


def test_ha_window_selection_retains_resolved_stage_bounds_even_without_requests():
    from cases.master_ha_failover.actions import _window
    resources = {'rows': [], 'scope': window('full_run', 90, 200)}
    def register(kind, value, **kwargs):
        resources[kind] = value
        return kind
    ctx = SimpleNamespace(resource=lambda key, kind: resources[key], resolve=lambda value: value,
                          register_resource=register)
    result = _window(ctx, dict(rows='rows', scope='scope', window_name='b_only',
                              **{'from': 100, 'until': 120}), Mock())
    assert result.output['window'] == 'ha_window'
    assert resources['ha_window'] == window('b_only', 100, 120)


def test_three_case_views_roundtrip_and_declare_explicit_panel_lines():
    from reporting.view_config import view
    for name, count in [('master_performance', 4), ('master_ha_failover', 9), ('cache_scale_in', 4)]:
        first = view(name + '.yaml')
        second = ViewDefinition.read(first.to_dict(), subtitle_fields=first.subtitle.fields)
        assert second.to_dict() == first.to_dict()
        assert sum(len(panel.gate_lines) for panel in second.charts.panels) == count


def test_offline_publication_uses_frozen_thresholds_not_changed_case_criteria(tmp_path):
    from cases.cache_scale_in.analysis import analyze
    from cases.cache_scale_in.program import GATE
    from metric_fixtures import freeze_metrics
    from test_cache_scale_gate import CacheGateTest
    evidence = CacheGateTest().evidence(.8)
    result = analyze(evidence)
    frozen_values = {r['check']: r['value'] for r in result['gate_references']}
    evidence['criteria']['baseline_min_hit'] = .99
    freeze_metrics(tmp_path, 'cache_scale_in')
    bundle = GATE.publish(tmp_path, evidence, result)
    doc = json.loads((bundle / 'report-spec.json').read_text())
    lines = [s for p in doc['panels'] for s in p['series'] if s.get('role') == 'gate_reference']
    assert {s['provenance']['check']: s['provenance']['value'] for s in lines} == frozen_values


def test_missing_explicit_ha_boundary_cannot_fall_back_to_full_run():
    from cases.master_ha_failover.actions import _window
    ctx = SimpleNamespace(resource=lambda key, kind: [] if kind == 'ha_rows' else window('full_run', 90, 200),
                          resolve=lambda value: None, register_resource=Mock())
    with pytest.raises(ValueError, match='boundary did not resolve'):
        _window(ctx, dict(rows='rows', scope='scope', window_name='b_only', **{'from': 'missing'}), Mock())
    ctx.register_resource.assert_not_called()


def test_unknown_reference_is_a_path_qualified_document_error():
    from document_io import DocumentError
    from reporting.view_config import view
    from unittest.mock import patch
    raw = view('cache_scale_in.yaml').to_dict()
    raw['charts']['panels'][0]['gate_lines'][0]['reference'] = 'typo'
    with patch('reporting.view_config.load_document', return_value=raw), \
         pytest.raises(DocumentError, match='cache_scale_in.yaml.*unknown gate reference'):
        view('cache_scale_in.yaml')


def test_worker_status_references_preserve_population_windows_and_skip_instant_geometry():
    from cases.cache_scale_in.analysis import gate_references
    evidence = dict(observation_origin_epoch_s=1000, criteria=dict(baseline_min_hit=.85,
        sustain_s=30, status_min_qps=10, status_max_rt_us=10000), baseline_start=40, baseline_end=100,
        post_start=120, post_end=360)
    spans = {'baseline': dict(start=40, end=100), 'transition': dict(start=110, end=120),
             'post': dict(start=120, end=360)}
    refs = gate_references(evidence, .62, spans)
    assert refs[2]['metric_id'] == 'cache_gate/status_min_qps' and refs[2]['value'] == 10
    assert refs[2]['windows'] == [window('baseline', 1040, 1100), window('transition', 1110, 1120), window('post', 1120, 1360)]
    assert refs[3]['unit'] == 'us' and refs[3]['op'] == 'le'
    spans['transition'] = dict(start=120, end=120)
    refs = gate_references(evidence, .62, spans)
    assert len(refs[2]['windows']) == 2 and '零时长' in refs[2]['description']


def test_unestablished_baseline_publishes_absence_without_duplicate_timestamps(tmp_path):
    from metric_fixtures import freeze_metrics
    from test_cache_scale_gate import CacheGateTest
    from cases.cache_scale_in.metrics import produce
    from monitoring.metric_store import MetricStore
    evidence = CacheGateTest().evidence()
    result = dict(verdict='INVALID', windows=[], gate_metrics={}, measurement_scope={},
                  baseline_windows=[dict(start=0, end=0, hit=None, errors=['missing']),
                                    dict(start=0, end=0, hit=None, errors=['missing'])])
    freeze_metrics(tmp_path, 'cache_scale_in')
    produce(tmp_path, evidence, result)
    rows = MetricStore.read(tmp_path).document['metrics']['derived/baseline_hit_ratio']
    assert rows[0]['points'] == [] and rows[0]['status'] == 'ABSENT'


@pytest.mark.parametrize('case', ['master_performance', 'master_ha_failover', 'cache_scale_in'])
def test_gate_overlay_preserves_all_existing_samples_and_display_clock(case):
    from reporting.view_config import view
    from reporting.curves import materialize, project_panels
    from importlib import import_module
    presentation = view(case + '.yaml')
    curves = [materialize(identity, style, [[1000, 10], [1001, None], [1002, 12]],
                          origin=900, labels=dict(style.labels))
              for identity, style in presentation.charts.curves.items()]
    spec = dict(panels=project_panels(curves, presentation), timeOriginEpochS=900,
                timeAxis=dict(min=50, max=250), timeOriginLabel='frozen acquisition clock')
    original = copy.deepcopy(spec)
    definitions = import_module('cases.' + case + '.analysis').GATE_REFERENCES
    refs = [freeze_reference(d, 5, [window('declared', 990, 1010)], op='ge') for d in definitions]
    result = apply_gate_lines(spec, presentation, refs)
    assert spec == original
    assert result['timeOriginEpochS'] == original['timeOriginEpochS']
    assert result['timeAxis'] == original['timeAxis']
    assert result['timeOriginLabel'] == original['timeOriginLabel']
    for before, after in zip(original['panels'], result['panels']):
        measured = [curve for curve in after['series'] if curve.get('role') != 'gate_reference']
        assert measured == before['series']
        assert after['axes'] == before['axes']


@pytest.mark.parametrize('case,omitted', [('master_ha_failover', {'gate-success','gate-balance','gate-route'}),
                                        ('cache_scale_in', {'worker-status-gate'})])
def test_unexecuted_gate_panels_do_not_add_empty_charts_or_hide_monitor_gaps(case, omitted):
    from reporting.view_config import view
    from reporting.curves import project_panels
    presentation = view(case + '.yaml')
    original = dict(timeOriginEpochS=1000, panels=project_panels([], presentation))
    result = apply_gate_lines(original, presentation, [])
    assert {p['id'] for p in result['panels']} == {p['id'] for p in original['panels']} - omitted
    section = result['sections'][-1]
    assert section['id'] == 'gate.reference.availability'
    assert {row['panel'] for row in section['value']} == omitted
    assert all(row['reason'] for row in section['value'])
    assert original['panels'] == project_panels([], presentation)
