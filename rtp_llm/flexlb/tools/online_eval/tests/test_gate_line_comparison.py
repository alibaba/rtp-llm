"""Compare frozen reference overlays without changing values or rerunning gates."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reporting import read_bundle, run_meta, write_bundle
from reporting.comparison import compare
from reporting.curves import materialize, project_panels
from reporting.gate_lines import Threshold, apply_gate_lines
from reporting.view_config import view


CURVE = 'mock/rtp_llm_context_tps_engine_mean/P'
REFERENCE = 'mock/rtp_llm_context_tps'


def demo_runs(root):
    """Three explicit fixture runs: different floors/windows, one optional floor off."""
    presentation = view('master_performance.yaml')
    paths = []
    for label, floor, start, event, actual, verdict in (
            ('A', 50, 2, 4, 60, 'PASS'), ('B', 80, 5, 8, 75, 'FAIL'),
            ('C', None, 1, 6, 90, 'PASS')):
        curve = materialize(CURVE, presentation.charts.curves[CURVE],
                            [[t, value] for t, value in ((0, actual), (5, None), (10, actual + 5), (20, actual))],
                            unit='tokens/s')
        thresholds = {REFERENCE: Threshold(floor, start, start + 10) if floor is not None else None,
                      'mock/rtp_llm_context_tps_with_cache': None, 'mock/mock_decode_wall_tps': None}
        panels = apply_gate_lines(project_panels([curve], presentation), presentation, thresholds)
        for panel in panels:
            panel['caption'] = '兼容性演示数据，不是实际压测。' + panel['caption']
        spec = dict(title='门禁线 compare 演示 ' + label, subtitle='演示数据，不是实际压测',
                    timeOriginLabel='演示 run 开始', timeAxis=dict(min=0, max=25),
                    events=[dict(id='checkpoint', name='演示对齐事件', t=event)], panels=panels)
        paths.append(write_bundle(Path(root) / label, 'run', 'fixture',
            dict(verdict=verdict, criteria=dict(floor=floor), purpose='COMPATIBILITY_DEMO'), spec,
            meta=run_meta(dict(id='demo-' + label), clock=dict(origin_kind='demo_start')),
            producer='comparison-fixture'))
    return paths


@pytest.mark.parametrize('alignment', [None, 'checkpoint'])
def test_three_runs_keep_each_floor_window_and_style_in_the_overlay(tmp_path, alignment):
    paths = demo_runs(tmp_path / 'inputs')
    originals = [(path / 'report-spec.json').read_bytes() for path in paths]
    with patch('cases.master_performance.analysis.analyze', side_effect=AssertionError('reanalysis')):
        result = compare(paths, tmp_path / 'output', labels=['A', 'B', 'C'], alignment_event=alignment)
    output = tmp_path / 'output/reports/comparison/runs'
    read_bundle(output)
    spec = json.loads((output / 'report-spec.json').read_text())
    assert [p['id'] for p in spec['panels']] == ['overlay:engine-tps']
    series = spec['panels'][0]['series']
    observations = [s for s in series if s.get('role') != 'gate_reference']
    references = [s for s in series if s.get('role') == 'gate_reference']
    assert len(observations) == 3 and len(references) == 2
    assert {s['group'] for s in references} == {'A', 'B'}
    shifts = {'A': 4, 'B': 8, 'C': 6} if alignment else dict.fromkeys('ABC', 0)
    for line, run, floor, start in zip(references, 'AB', [50, 80], [2, 5]):
        assert line['name'].startswith(run + ' · ')
        assert line['points'] == [{'x': start - shifts[run], 'y': floor},
                                  {'x': start + 10 - shifts[run], 'y': floor}]
        assert line['dash'] == [6, 4] and line['marker_count'] == 0
        assert line['source_type'] == 'configuration'
        assert line['measurement'] == {'kind': 'gate_reference', 'reference': REFERENCE}
    for curve, run in zip(observations, 'ABC'):
        assert curve['dash'] == [] and curve['marker_count'] == 12
        original = json.loads(originals['ABC'.index(run)])['panels'][0]['series'][0]
        assert curve['points'] == [dict(p, x=p['x'] - shifts[run]) for p in original['points']]
    assert len({s['color'] for s in references}) == 2
    assert {k: v['verdict'] for k, v in result['runs'].items()} == {'A': 'PASS', 'B': 'FAIL', 'C': 'PASS'}
    assert 'verdict' not in result
    assert originals == [(path / 'report-spec.json').read_bytes() for path in paths]
    html = (output / 'report.html').read_text()
    payload, _ = json.JSONDecoder().raw_decode(html.split('const SPEC = ', 1)[1])
    assert sum(s.get('role') == 'gate_reference' for p in payload['panels'] for s in p['series']) == 2
    assert all(s['dash'] == [6, 4] for p in payload['panels'] for s in p['series'] if s.get('role') == 'gate_reference')
