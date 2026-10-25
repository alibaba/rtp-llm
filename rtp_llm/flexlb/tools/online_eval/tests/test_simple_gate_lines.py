"""Reference overlays preserve observations and use one contract across cases."""
import copy

import pytest

from reporting.curves import materialize, project_panels
from reporting.gate_lines import Threshold, apply_gate_lines
from reporting.view_config import view
from reporting.view_model import ViewDefinition


def test_overlay_preserves_observations_clock_and_scales_threshold_once():
    data = view('cache_scale_in.yaml').to_dict()
    data['charts']['curves']['derived/survivor_hit_ratio'].update(
        scale=100, source_unit='ratio', unit='%')
    presentation = ViewDefinition.read(data, subtitle_fields={'expected_behavior', 'observed_behavior'})
    style = presentation.charts.curves['derived/survivor_hit_ratio']
    curve = materialize('derived/survivor_hit_ratio', style, [[10, .8], [15, None], [20, .9]])
    panels = project_panels([curve], presentation)
    original = copy.deepcopy(panels)
    result = apply_gate_lines(panels, presentation, {'survivor_hit_floor': Threshold(.6, 11, 19)})
    assert panels == original
    assert [p['id'] for p in result] == [p['id'] for p in original]
    panel = next(p for p in result if p['id'] == 'cache-hit')
    assert panel['series'][0] == curve
    line = panel['series'][-1]
    assert line['points'] == [{'x': 11, 'y': 60}, {'x': 19, 'y': 60}]
    assert line['axis'] == style.axis and line['unit'] == '%'
    assert line['role'] == 'gate_reference' and line['source_type'] == 'configuration'


def test_optional_reference_is_omitted_but_unknown_reference_is_an_error():
    presentation = view('cache_scale_in.yaml')
    panels = project_panels([], presentation)
    assert apply_gate_lines(panels, presentation, {'survivor_hit_floor': None}) == panels
    with pytest.raises(ValueError, match='unknown gate line'):
        apply_gate_lines(panels, presentation, {})


@pytest.mark.parametrize('value,start,end', [(float('nan'), 0, 1), (1, 2, 2), (1, 3, 2), (1, 0, float('inf')), (True, 0, 1)])
def test_invalid_window_or_value_is_rejected(value, start, end):
    with pytest.raises(ValueError):
        Threshold(value, start, end)


def test_binding_must_reuse_an_existing_panel_curve():
    raw = view('cache_scale_in.yaml').to_dict()
    raw['charts']['panels'][0]['gate_lines'][0]['curve_id'] = 'client/actual_send_qps'
    presentation = ViewDefinition.read(raw, subtitle_fields={'expected_behavior', 'observed_behavior'})
    with pytest.raises(ValueError, match='same panel'):
        presentation.charts.panels[0].validate_bindings(presentation.charts.curves)


def test_case_views_do_not_require_new_gate_panels_or_metrics():
    expected = {'master_performance': 3, 'cache_scale_in': 1, 'master_ha_failover': 0}
    for case, count in expected.items():
        presentation = view(case + '.yaml')
        assert sum(len(p.gate_lines) for p in presentation.charts.panels) == count
        assert not any(p.id.startswith('gate-') for p in presentation.charts.panels)
        assert not any(c.metric_id.startswith('cache_gate/') for c in presentation.charts.curves.values())


def test_observations_and_reference_use_the_same_frozen_display_clock():
    from reporting.timeline import apply, freeze
    presentation = view('cache_scale_in.yaml')
    style = presentation.charts.curves['derived/survivor_hit_ratio']
    panels = project_panels([materialize('derived/survivor_hit_ratio', style, [[5, .8]])], presentation)
    panels = apply_gate_lines(panels, presentation, {'survivor_hit_floor': Threshold(.6, 3, 9)})
    clock = freeze({'time_axis': {'origin': {'event': 'traffic_started'},
        'range': {'from': 'origin', 'until': {'event': 'end'}}}},
        events=[{'id': 'traffic_started', 'epoch_s': 100}, {'id': 'end', 'epoch_s': 130}], phases=[])
    spec = {'timeOriginEpochS': 110, 'timeAxis': {'min': 0, 'max': 10}, 'panels': panels}
    before = copy.deepcopy(spec)
    projected = apply(spec, {'report_timeline': clock, 'status': 'PASS'})
    assert spec == before
    assert projected['timeAxis'] == {'min': 0, 'max': 30}
    series = next(p for p in projected['panels'] if p['id'] == 'cache-hit')['series']
    assert series[0]['points'] == [{'x': 15, 'y': .8}]
    assert series[-1]['points'] == [{'x': 13, 'y': .6}, {'x': 19, 'y': .6}]
