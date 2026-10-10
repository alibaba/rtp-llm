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
