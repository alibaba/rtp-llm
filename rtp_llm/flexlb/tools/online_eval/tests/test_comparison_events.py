"""Event grouping affects display only and never changes frozen input clocks."""
import copy
import json

import pytest

from reporting.comparison import compare
from reporting.pairing import merge_nearby_events
from test_gate_line_comparison import demo_runs


def event(run, t, identity='checkpoint', name='门禁开始'):
    return dict(id=identity, event_name=name, name=run + ' · ' + name, run=run, t=t)


def test_same_type_nearby_runs_merge_with_exact_members_and_mean_position():
    events = [event('B', 10.8), event('A', 10)]
    original = copy.deepcopy(events)
    merged = merge_nearby_events(events)
    assert len(merged) == 1
    assert merged[0]['t'] == 10.4
    assert merged[0]['name'] == '门禁开始 ×2'
    assert merged[0]['members'] == [events[1], events[0]]
    assert events == original
    merged[0]['members'][0]['t'] = 99
    assert events == original


@pytest.mark.parametrize('events', [
    [event('A', 0), event('B', 1.01)],
    [event('A', 0), event('B', .2, identity='other')],
    [event('A', 0), event('B', .2, name='门禁开始 · FAIL')],
    [event('A', 0), event('A', .2)],
])
def test_distant_different_types_statuses_and_same_run_occurrences_remain_distinct(events):
    assert len(merge_nearby_events(events)) == 2


def test_cluster_bounds_prevent_chaining_nearby_pairs():
    merged = merge_nearby_events([event('A', 0), event('B', .8), event('C', 1.6)])
    assert len(merged) == 2 and len(merged[0]['members']) == 2
    assert merged[1]['run'] == 'C'
    assert len(merge_nearby_events([event('A', 0), event('B', 1)])) == 1


def test_zero_disables_merge_and_missing_identity_is_preserved():
    events = [event('A', 0), event('B', 0)]
    assert merge_nearby_events(events, 0) == events
    unknown = [dict(name='event', t=0), dict(name='event', t=0)]
    assert merge_nearby_events(unknown) == unknown


@pytest.mark.parametrize('value', [-1, float('nan'), float('inf'), True, '1'])
def test_invalid_tolerance_rejected(value):
    with pytest.raises(ValueError, match='tolerance'):
        merge_nearby_events([], value)


@pytest.mark.parametrize('tolerance, expected', [(1, 1), (0, 3)])
def test_compare_groups_only_after_alignment_and_preserves_source_bundles(tmp_path, tolerance, expected):
    paths = demo_runs(tmp_path / 'inputs')
    original = [(path / 'report-spec.json').read_bytes() for path in paths]
    result = compare(paths, tmp_path / 'out', alignment_event='checkpoint', event_merge_tolerance=tolerance)
    spec = json.loads((tmp_path / 'out/reports/comparison/runs/report-spec.json').read_text())
    events = spec['panels'][0]['events']
    assert len(events) == expected
    if tolerance:
        assert events[0]['t'] == 0 and len(events[0]['members']) == 3
        assert {m['run'] for m in events[0]['members']} == {'A', 'B', 'C'}
    assert result['time_alignment']['observed_times'] == {'A': 4, 'B': 8, 'C': 6}
    assert result['event_display']['merge_tolerance_s'] == tolerance
    assert original == [(path / 'report-spec.json').read_bytes() for path in paths]
    assert {k: v['verdict'] for k, v in result['runs'].items()} == {'A': 'PASS', 'B': 'FAIL', 'C': 'PASS'}
