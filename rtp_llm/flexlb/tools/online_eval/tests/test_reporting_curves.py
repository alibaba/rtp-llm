"""Selected reports preserve missing-data semantics independently of case logic."""

import copy
from types import SimpleNamespace
from unittest import mock

import pytest

from cases import registry
from reporting.curves import materialize, project_panels


def presentation():
    return {'charts': {'curves': {
        'a': dict(metric_id='metric/a', name='A', group='G', axis='y', unit='ms', color='blue'),
        'b': dict(metric_id='metric/b', name='B', group='G', axis='y', unit='ms', color='red'),
    }, 'panels': [dict(id='p', title='Panel', caption='Observed', empty_caption='Missing',
        curve_ids=['b', 'a'], axes={'y': dict(title='ms', position='left')})]}}


@pytest.mark.parametrize('values,caption', [([], 'Missing'), ([None], 'Missing'),
    ([float('nan'),float('inf')], 'Missing'), ([0], 'Observed 缺少有效曲线：B。')])
def test_null_nan_and_zero_use_one_empty_panel_rule(values, caption):
    config = presentation()
    curves = [materialize('a', config['charts']['curves']['a'], list(enumerate(values)))]
    assert project_panels(curves, config)[0]['caption'] == caption


def test_materialization_and_projection_keep_labels_scaling_provenance_and_order():
    config = presentation()
    style = dict(config['charts']['curves']['a'], name='{master} · A', group='{master}', scale=1000)
    provenance = {'source_type': 'prometheus'}
    a = materialize('a', style, [[100, .5], [101, None]], origin=100,
                    labels={'master': 'A'}, provenance=provenance)
    b = materialize('b', config['charts']['curves']['b'], [[0, 1]])
    original = copy.deepcopy([a,b])
    panel = project_panels([a,b], config)[0]
    assert [curve['curve_id'] for curve in panel['series']] == ['b', 'a']
    assert a['points'] == [{'x':0,'y':500},{'x':1,'y':None}]
    assert a['name'] == 'A · A' and a['group'] == 'A'
    assert a['provenance'] == provenance
    assert [a,b] == original
    panel['axes']['y']['title'] = 'changed'
    assert config['charts']['panels'][0]['axes']['y']['title'] == 'ms'


def test_configuration_gate_lines_do_not_hide_missing_measurements():
    from cases.master_performance.panels import report_panels
    from reporting.view_config import view

    config = view('master_performance.yaml')
    config['charts']['curves']['mock/rtp_llm_context_tps_engine_mean/P'].pop('color')
    panels = report_panels([], {'engine_tps': {'mock/rtp_llm_context_tps':10}}, config)
    panel = next(p for p in panels if p['id'] == 'engine-tps')
    descriptor = next(p for p in config['charts']['panels'] if p['id'] == 'engine-tps')
    assert panel['caption'] == descriptor['empty_caption']
    assert panel['series'][0]['source_type'] == 'configuration'
    assert panel['series'][0]['color']


@pytest.mark.parametrize('declaration', [
    {'default.yaml':registry.ReportView(lambda *args: None)},
    {'bad/path.yaml':registry.ReportView(lambda *args: None)},
    {'extra.yaml':registry.ReportView('invalid')},
    {'extra.yaml':registry.ReportView(lambda *args: None, 'invalid')},
    [],
])
def test_registry_rejects_invalid_case_view_capabilities(declaration):
    module = SimpleNamespace(REPORT_VIEWS=declaration)
    with mock.patch.dict(registry.PROGRAMS, {'case':'registered.program'}, clear=True), \
         mock.patch('importlib.import_module', return_value=module):
        with pytest.raises(ValueError):
            registry.view_capabilities()


def test_two_registered_programs_cannot_assign_conflicting_view_capabilities():
    modules = [SimpleNamespace(REPORT_VIEWS={'extra.yaml':registry.ReportView(lambda *args: None)})
               for _ in range(2)]
    with mock.patch.dict(registry.PROGRAMS, {'a':'a.program','b':'b.program'}, clear=True), \
         mock.patch('importlib.import_module', side_effect=modules):
        with pytest.raises(ValueError, match='conflicting'):
            registry.view_capabilities()


def test_shared_capability_can_be_reused_by_registered_programs():
    module = SimpleNamespace(REPORT_VIEWS={'extra.yaml':registry.ReportView(lambda *args: None)})
    with mock.patch.dict(registry.PROGRAMS, {'a':'a.program','b':'b.program'}, clear=True), \
         mock.patch('importlib.import_module', return_value=module):
        assert registry.view_capabilities() == module.REPORT_VIEWS


def test_selected_presets_and_visibility_reach_the_html(tmp_path):
    from reporting import write_bundle

    config = presentation()
    config['charts']['curves']['b']['hidden'] = True
    config['charts']['curves']['b']['group'] = 'Other'
    config['charts']['panels'][0]['presets'] = {
        'Visible': {'visible': True}, 'Named': {'names': ['B']},
        'Group': {'groups': ['Other']}, 'Substring': {'contains': ['A']},
    }
    curves = [materialize(key, style, [[0, 0]])
              for key, style in config['charts']['curves'].items()]
    panels = project_panels(curves, config)
    assert panels[0]['presets'] == {
        'Visible': ['A'], 'Named': ['B'], 'Group': ['B'], 'Substring': ['A'],
    }
    assert [c['hidden'] for c in panels[0]['series']] == [True, False]
    bundle = write_bundle(tmp_path, 'run', 'selected-presets', {},
        dict(title='Test', timeAxis={'min': 0, 'max': 1}, panels=panels))
    html = (bundle / 'report.html').read_text()
    assert '"Visible": ["A"]' in html and '"Named": ["B"]' in html


@pytest.mark.parametrize('name', ['cache_scale_in', 'master_performance', 'master_ha_failover'])
def test_all_selected_views_require_complete_monitoring_classification(name):
    from monitoring.query_plan import load_plan
    from reporting.view_config import view
    from scenario.loader import ScenarioError

    data = view(name + '.yaml')
    plan = load_plan(name + '.yaml')
    plan['sources']['master']['new_query'] = {'promql': 'new_query'}
    with mock.patch('monitoring.query_plan.load_plan', return_value=plan):
        with pytest.raises(ScenarioError, match='lack presentation'):
            view(name + '.yaml')
    data['metrics'].pop('diagnostic_only')
    with mock.patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError, match='diagnostic_only'):
            view(name + '.yaml')


@pytest.mark.parametrize('name', ['cache_scale_in', 'master_performance', 'master_ha_failover'])
def test_frozen_archive_cannot_silently_discard_unclassified_queries(name, tmp_path):
    from monitoring.metric_store import export_metrics
    from monitoring.query_plan import load_plan
    from reporting.metric_binding import monitoring_audit
    from reporting.view_config import view

    config = view(name + '.yaml')
    store = export_metrics(tmp_path, load_plan(name + '.yaml'))
    audit = monitoring_audit(store, config)
    assert {row['metric_id'] for row in audit if row['classification'] == 'DIAGNOSTIC_ONLY'} == \
        set(config['metrics']['diagnostic_only'])
    store.document['definitions']['master/undeclared'] = {'promql': 'undeclared'}
    with pytest.raises(ValueError, match='unclassified monitoring metric master/undeclared'):
        monitoring_audit(store, config)


def test_ha_removes_queries_without_consumers_but_keeps_produced_measurements():
    from monitoring.query_plan import load_plan

    plan = load_plan('master_ha_failover.yaml')
    removed = {'arrivals_qps', 'dispatch_qps', 'schedule_responses_qps',
               'flexlb_app_flexlb_batcher_queue_size'}
    assert not removed & plan['sources']['master'].keys()
    assert 'schedule_p99_seconds' not in plan['sources']['client']
    assert len(plan['produced']) == 19


def test_unused_curve_declaration_cannot_masquerade_as_presentation():
    from reporting.view_config import view
    from scenario.loader import ScenarioError

    data = view('master_performance.yaml')
    data['charts']['curves']['unused'] = dict(next(iter(data['charts']['curves'].values())))
    with mock.patch('reporting.view_config.load_document', return_value=data):
        with pytest.raises(ScenarioError, match='referenced by a panel'):
            view('master_performance.yaml')


def test_performance_presets_include_configuration_lines_without_validating_missing_data():
    from cases.master_performance.panels import report_panels
    from reporting.view_config import view

    config = view('master_performance.yaml')
    descriptor = next(p for p in config['charts']['panels'] if p['id'] == 'engine-tps')
    descriptor['presets'] = {'Floors': {'groups': ['门禁']}}
    panel = next(p for p in report_panels([], {
        'engine_tps': {'mock/rtp_llm_context_tps': 10}}, config) if p['id'] == 'engine-tps')
    assert panel['presets']['Floors'] == [panel['series'][0]['name']]
    assert panel['caption'] == descriptor['empty_caption']


@pytest.mark.parametrize('name', ['cache_scale_in', 'master_performance', 'master_ha_failover'])
def test_unbound_produced_series_is_rejected_just_like_a_query(name):
    from monitoring.query_plan import load_plan
    from reporting.view_config import view
    from scenario.loader import ScenarioError

    plan = load_plan(name + '.yaml')
    definition = next(d for d in plan['produced'].values() if d['value_kind'] == 'gauge')
    plan['produced']['unbound/series'] = copy.deepcopy(definition)
    with mock.patch('monitoring.query_plan.load_plan', return_value=plan):
        with pytest.raises(ScenarioError, match='lack presentation'):
            view(name + '.yaml')
