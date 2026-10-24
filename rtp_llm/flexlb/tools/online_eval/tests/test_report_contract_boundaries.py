"""Measurement identity is independent of data and human-readable presentation."""
from unittest.mock import patch
from reporting.report_model import MeasurementIdentity
from reporting.comparison import _series_key
from reporting.workload_panels import build_panels
from reporting.view_config import view, DEFAULT_VIEW


def test_pairing_identity_does_not_iterate_point_arrays():
    class UnreadablePoints:
        def __iter__(self):
            raise AssertionError('identity computation must not visit point arrays')
    curve = dict(metric_id='mock/qps', unit='req/s', points=UnreadablePoints(), name='original')
    panel = dict(timeX=True, axes={'y': dict(title='req/s')})
    signature = _series_key(panel, curve, 'run-start')
    curve.update(name='new title', description='new description')
    assert _series_key(panel, curve, 'run-start') == signature


def test_default_metric_identity_retains_source_labels_and_aggregation():
    keys = ['1/mock/qps/{"engine_name":"p-0"}', '1/mock/qps/{"engine_name":"p-1"}']
    series = {key: [[0, 2], [1, None]] for key in keys}
    panel = build_panels(series, {key: dict(unit='req/s') for key in keys}, view(DEFAULT_VIEW))[0]
    mean, maximum, *details = panel['series']
    assert mean['metric_id'] == maximum['metric_id'] == 'mock/qps'
    assert MeasurementIdentity.read(mean).signature() != MeasurementIdentity.read(maximum).signature()
    assert all(c['measurement'] == {'epoch':'1','source':'mock'} for c in panel['series'])
    assert details[0]['labels'] == {'engine_name':'p-0'}
    assert set(panel['presets']['概览']) == {mean['name'], maximum['name']}
    assert mean['points'][1]['y'] is None


def test_bundle_publication_does_not_import_comparison_models(tmp_path):
    from reporting import write_bundle
    with patch.dict('sys.modules', {'reporting.comparison_model': None}):
        write_bundle(tmp_path, 'run', 'independent', {}, dict(panels=[], sections=[], kpis=[]))


def test_views_decode_once_and_projection_does_not_parse_again():
    from reporting.view_model import CurvePresentation, PanelPresentation, SectionPresentation
    from reporting.curves import materialize, project_panels
    from reporting.view_sections import view_details
    with patch.object(CurvePresentation, 'read', wraps=CurvePresentation.read) as curves, \
         patch.object(PanelPresentation, 'read', wraps=PanelPresentation.read) as panels, \
         patch.object(SectionPresentation, 'read', wraps=SectionPresentation.read) as sections:
        definition = view('master_ha_failover.yaml')
        assert curves.call_count == len(definition.charts.curves)
        assert panels.call_count == len(definition.charts.panels)
        assert sections.call_count == len(definition.sections)
        counts = (curves.call_count, panels.call_count, sections.call_count)
        measured = [materialize('ha/sent', definition.charts.curves['ha/sent'], [(1, 0)])]
        project_panels(measured, definition)
        view_details(definition, 'sources', {'status':'frozen'})
        assert counts == (curves.call_count, panels.call_count, sections.call_count)


def test_default_labels_are_configured_and_no_projection_reads_yaml():
    from reporting.view_model import ViewDefinition
    raw = view(DEFAULT_VIEW).to_dict()
    raw['charts']['detail_labels'] = ['device_id']
    with patch('reporting.view_config.load_document', return_value=raw):
        definition = view(DEFAULT_VIEW)
    keys = ['1/mock/qps/{"device_id":"a"}', '1/mock/qps/{"device_id":"b"}']
    with patch('reporting.view_config.view', side_effect=AssertionError('hidden YAML IO')):
        panel = build_panels({key:[[0,2]] for key in keys},
                             {key:dict(unit='req/s') for key in keys}, definition)[0]
    assert len(panel['series']) == 4
    raw['charts']['presets'] = {'mixed':['mean','detail']}
    mixed = build_panels({key:[[0,2]] for key in keys}, {key:dict(unit='req/s') for key in keys},
                         ViewDefinition.read(raw))[0]
    assert len(mixed['presets']['mixed']) == 3


def test_preset_identity_survives_display_name_changes():
    from reporting.curves import resolve_presets
    from reporting.view_model import PresetSelection
    selection = {'selected':(PresetSelection.read({'ids':['stable']}),)}
    curve = dict(curve_id='stable',name='old',group='G',hidden=False)
    assert resolve_presets([curve],selection) == {'selected':['old']}
    curve['name'] = 'new'
    assert resolve_presets([curve],selection) == {'selected':['new']}
