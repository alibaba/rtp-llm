"""Mixed frozen input, metric pairing and portable comparison contracts."""
import copy
import json
import shutil
import zipfile

import pytest

from artifacts.experiment import ArtifactRole, ExperimentWriter
from reporting import read_bundle, run_meta, write_bundle
from reporting.comparison import compare, main
from reporting.comparison_inputs import load_snapshot


def bundle(tmp_path, name, *, panel='latency', extra=False, points=None):
    curve = dict(name='Latency', metric_id='request/latency', unit='ms',
                 measurement={'cohort': 'completed'}, points=points or [{'x': 5, 'y': None}, {'x': 10, 'y': 9}],
                 statistics_points=[{'x': 5, 'y': None}, {'x': 10, 'y': 9}])
    spec = dict(title=name, timeOriginLabel='observation start', events=[dict(id='scale', name='Scale', t=5)],
        timeAxis={'min': 0, 'max': 20}, panels=[dict(id=panel, timeX=True,
        axes={'y': {'title': 'ms'}}, series=[curve])], kpis=[dict(label='Verdict', value='PASS')])
    if extra:
        spec['panels'][0]['series'].append(dict(name='Queue', metric_id='queue', unit='ms', points=[{'x': 5, 'y': 4}]))
    return write_bundle(tmp_path / name, 'run', name, {'verdict': 'PASS', 'criteria': {'limit': 20}}, spec,
        meta=run_meta({'id': name}, implementation={'commit': name}, configuration={'profile': name},
                      workload={'trace': 'same'}, environment={'cores': 32},
                      clock={'origin_kind': 'observation_start'}))


def experiment(tmp_path, source, *, name='experiment'):
    path = tmp_path / (name + '.zip')
    writer = ExperimentWriter(path)
    with writer.publication():
        writer.add_json('experiment.json', {'plan': {'reports': ['view.yaml']}}, ArtifactRole.DEFINITION)
        writer.add_json('execution.json', {'status': 'FAIL', 'cleanup': [{'status': 'PASS'}],
            'workload': {'runtime_validity': 'VALID'}}, ArtifactRole.EXECUTION)
        writer.add_json('completeness.json', {}, ArtifactRole.EXECUTION)
        writer.add_view('view.yaml', json.loads((source / 'report-spec.json').read_text()))
    return path


def report(output):
    return json.loads((output / 'reports/comparison/runs/report-spec.json').read_text())


def test_mixed_zip_and_bundle_named_reference_and_series_level_pairing(tmp_path):
    a = bundle(tmp_path, 'a', extra=True)
    b = bundle(tmp_path, 'b')
    archive = experiment(tmp_path, b)
    output = tmp_path / 'comparison'
    result = compare([a, archive], output, labels=['single', 'batch'], baseline='batch', alignment_event='scale')
    assert result['controls']['reference'] == 'batch'
    assert result['controls']['comparisons']['single']['differences']
    assert result['executions']['batch']['status'] == 'FAIL'
    assert 'verdict' not in result
    spec = report(output)
    overlay = next(p for p in spec['panels'] if p['id'].startswith('overlay:'))
    assert len(overlay['series']) == 2
    assert overlay['series'][0]['points'] == [{'x': 0, 'y': None}, {'x': 5, 'y': 9}]
    assert overlay['series'][0]['statistics_points'] == overlay['series'][0]['points']
    assert [s['metric_id'] for p in spec['panels'] if p['id'].startswith('run-') for s in p['series']] == ['queue']
    assert len(spec['summary_sections']) == 2
    portable = tmp_path / 'portable'
    shutil.copytree(output / 'reports/comparison/runs', portable)
    shutil.rmtree(output)
    shutil.rmtree(a.parents[2])
    archive.unlink()
    read_bundle(portable)
    assert (portable / 'original-run-1.html').is_file()
    assert (portable / 'original-run-2.html').is_file()


def test_original_panel_opt_in_and_unknown_measurements(tmp_path):
    a, b = bundle(tmp_path, 'a'), bundle(tmp_path, 'b')
    compare([a, b], tmp_path / 'out', include_original_panels=True)
    assert len(report(tmp_path / 'out')['panels']) == 3


def test_input_validation_precedes_publication(tmp_path):
    a, b = bundle(tmp_path, 'a'), bundle(tmp_path, 'b')
    for kwargs in [{'labels': ['same', 'same']}, {'labels': ['a']}, {'baseline': 'missing'}, {'views': ['view.yaml']}]:
        with pytest.raises(ValueError):
            compare([a, b], tmp_path / 'out', **kwargs)
    with pytest.raises(ValueError, match='duplicate'):
        compare([a, a], tmp_path / 'out')
    with pytest.raises(ValueError, match='contain'):
        compare([a, b], a / 'out')
    assert not (tmp_path / 'out').exists()


def test_zip_dataset_checksum_and_view_selection(tmp_path):
    a = bundle(tmp_path, 'a')
    path = experiment(tmp_path, a)
    with pytest.raises(ValueError, match='frozen view'):
        load_snapshot(path, view='missing')
    with pytest.raises(ValueError, match='only'):
        load_snapshot(a, view='view.yaml')
    with zipfile.ZipFile(path) as source:
        files = {name: source.read(name) for name in source.namelist()}
    manifest = json.loads(files['manifest.json'])
    dataset = next(value for name, value in manifest['files'].items() if '/datasets/' in name)
    files[dataset['blob']] = b'[]'
    bad = tmp_path / 'bad.zip'
    with zipfile.ZipFile(bad, 'w') as target:
        for name, data in files.items(): target.writestr(name, data)
    with pytest.raises(ValueError, match='checksum'):
        compare([a, bad], tmp_path / 'out')
    assert not (tmp_path / 'out').exists()


def test_duplicate_metric_identity_with_different_values_is_not_silently_dropped(tmp_path):
    a = bundle(tmp_path, 'a')
    spec = json.loads((a / 'report-spec.json').read_text())
    curve = copy.deepcopy(spec['panels'][0]['series'][0])
    curve.update(name='Different cohort', points=[{'x': 5, 'y': 10}])
    spec['panels'][0]['series'].append(curve)
    ambiguous = write_bundle(tmp_path / 'ambiguous', 'run', 'ambiguous', {}, spec, meta=spec['run_meta'])
    compare([ambiguous, bundle(tmp_path, 'b')], tmp_path / 'out')
    assert sum(len(p['series']) for p in report(tmp_path / 'out')['panels']) == 3
    assert not any(p['id'].startswith('overlay:') for p in report(tmp_path / 'out')['panels'])


def test_cli_mixed_inputs_and_labels(tmp_path):
    a = bundle(tmp_path, 'a')
    z = experiment(tmp_path, bundle(tmp_path, 'b'))
    assert main([str(a), str(z), '--label', 'single', '--label', 'batch', '--baseline', 'batch',
        '--view', 'auto', '--view', 'view.yaml', '--output', str(tmp_path / 'out')]) == 0


def test_alignment_shifts_shared_statistics_arrays_once():
    from reporting.pairing import shifted_panel
    points = [{'x': 10, 'y': None}]
    panel = {'timeX': True, 'series': [{'points': points, 'statistics_points': points}, {'points': points}]}
    shifted = shifted_panel(panel, 5)
    assert all(curve['points'] == [{'x': 5, 'y': None}] for curve in shifted['series'])
    assert shifted['series'][0]['statistics_points'] == [{'x': 5, 'y': None}]
    assert points == [{'x': 10, 'y': None}]


def test_attachment_validation_fails_before_publication(tmp_path):
    for filename in ('../escape.html', 'report.html', 'manifest.json'):
        with pytest.raises(ValueError, match='attachment'):
            write_bundle(tmp_path / 'invalid', 'run', 'invalid', {}, {'panels': []},
                         attachments={filename: 'payload'})
    assert not (tmp_path / 'invalid').exists()


def test_default_preserves_multi_metric_layout_visibility_and_display_range(tmp_path):
    a = bundle(tmp_path, 'a', extra=True, points=[{'x': -100, 'y': None}, {'x': 10, 'y': 9}])
    b = bundle(tmp_path, 'b', extra=True)
    result = compare([a, b], tmp_path / 'out')
    spec = report(tmp_path / 'out')
    assert spec['timeAxis'] == {'min': 0, 'max': 20}
    assert spec['timeOriginLabel'] == 'observation start'
    assert result['time_alignment']['status'] == 'NOT_REQUESTED'
    assert len(spec['panels']) == 1
    panel = spec['panels'][0]
    assert panel['id'] == 'overlay:latency'
    assert [curve['metric_id'] for curve in panel['series']] == ['request/latency', 'request/latency', 'queue', 'queue']
    assert panel['series'][0]['points'][0] == {'x': -100, 'y': None}
    assert panel['timeAxis'] == {'min': 0, 'max': 20}
    assert '指标 · Queue' in panel['presets']
    assert 'run · A' in panel['presets']


def test_different_panel_ids_do_not_implicitly_map_even_with_same_metric(tmp_path):
    compare([bundle(tmp_path, 'a'), bundle(tmp_path, 'b', panel='different')], tmp_path / 'out')
    assert len(report(tmp_path / 'out')['panels']) == 2
    assert all(p['id'].startswith('run-') for p in report(tmp_path / 'out')['panels'])


def test_clock_semantics_required_and_separate_ranges_preserved(tmp_path):
    a, b = bundle(tmp_path, 'a'), bundle(tmp_path, 'b')
    spec = json.loads((b / 'report-spec.json').read_text())
    spec['run_meta']['clock'] = None
    spec['timeAxis'] = {'min': 100, 'max': 150}
    unknown = write_bundle(tmp_path / 'unknown', 'run', 'unknown', {}, spec, meta=spec['run_meta'])
    compare([a, unknown], tmp_path / 'out')
    panels = report(tmp_path / 'out')['panels']
    assert len(panels) == 2
    assert panels[0]['timeAxis'] == {'min': 0, 'max': 20}
    assert panels[1]['timeAxis'] == {'min': 100, 'max': 150}


def test_selected_baseline_sets_layout_and_display_range(tmp_path):
    a, b = bundle(tmp_path, 'a'), bundle(tmp_path, 'b', extra=True)
    spec = json.loads((b / 'report-spec.json').read_text())
    spec['timeAxis'] = {'min': 3, 'max': 15}
    other = write_bundle(tmp_path / 'other', 'run', 'other', {}, spec, meta=spec['run_meta'])
    compare([a, other], tmp_path / 'out', labels=['a', 'b'], baseline='b')
    result = report(tmp_path / 'out')
    assert result['timeAxis'] == {'min': 3, 'max': 15}
    assert len(result['panels']) == 1
    assert len(result['panels'][0]['series']) == 3


def test_original_gate_is_not_inferred_from_execution_status():
    from types import SimpleNamespace
    from reporting.comparison import _original_result
    snapshot = SimpleNamespace(analysis={'status': 'PASS', 'outcome': {'gate': 'FAIL', 'validity': 'VALID'}}, execution={})
    assert _original_result(snapshot) == {'verdict': 'FAIL', 'validity': 'VALID', 'status': 'PASS'}
    snapshot = SimpleNamespace(analysis={}, execution={'status': 'PASS', 'outcome': {'gate': 'PASS', 'validity': 'VALID'}})
    assert _original_result(snapshot) == {'verdict': 'PASS', 'validity': 'VALID', 'status': 'PASS'}
    snapshot = SimpleNamespace(analysis={}, execution={'status': 'PASS'})
    assert _original_result(snapshot)['verdict'] is None
