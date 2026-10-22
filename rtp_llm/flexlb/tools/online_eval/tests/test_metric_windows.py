"""Raw evidence windows are bounded, immutable and independent of case vocabulary."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from monitoring.windows import capture_window, FrozenMetricWindow
from runtime.deadline import Deadline


FIELDS = {'temperature': dict(metric='sensor/temperature', labels={'rack': 'west'})}
LIMITS = dict(max_samples=10, max_bytes=100000)


def client(rows=None):
    if rows is None:
        rows = [dict(metric=dict(rack='west', host='a', incarnation='one'),
                     values=[[1, '10'], [2, '11']])]
    return SimpleNamespace(
        metric_definition=Mock(return_value=dict(unit='celsius', labels=['host', 'incarnation'])),
        metric_chunks=Mock(return_value=[rows]))


def capture(c, **kwargs):
    return capture_window(c, FIELDS, source='rack-probe', windows=[(0, 2)], epoch=7,
                          deadline=kwargs.pop('deadline', Deadline(float('inf'))),
                          limits=kwargs.pop('limits', LIMITS), **kwargs)


def test_freeze_and_replay_use_existing_metric_store_without_requery(tmp_path):
    c = client()
    frozen = capture(c, directory=tmp_path/'window')
    assert frozen.store().select('sensor/temperature', source='rack-probe', epoch=7,
        labels={'host': 'a'}, start=0, end=2, max_gap_s=1)[0]['points'] == [[1, 10], [2, 11]]
    calls = c.metric_chunks.call_count
    read = FrozenMetricWindow.read(tmp_path/'window')
    assert read.field_rows() == frozen.field_rows()
    assert c.metric_chunks.call_count == calls
    projected = frozen.field_rows()
    projected['temperature'][0]['values'].clear()
    assert len(frozen.field_rows()['temperature'][0]['values']) == 2
    args = c.metric_chunks.call_args.kwargs
    assert args['labels'] == {'rack': 'west'} and args['deadline'] is not None


@pytest.mark.parametrize('failure', ['query', 'cancel', 'samples', 'bytes', 'conflict', 'identity', 'nonfinite'])
def test_partial_capture_retains_completed_points_but_cannot_be_read_as_complete(tmp_path, failure):
    c = client()
    deadline = Deadline(float('inf'))
    limits = LIMITS
    def chunks(*args, **kwargs):
        yield [dict(metric=dict(rack='west', host='a', incarnation='one'), values=[[1, '10']])]
        if failure == 'query':
            raise RuntimeError('TSDB failed')
        if failure == 'cancel':
            deadline.cancelled.set()
        row = dict(metric=dict(rack='west', host='a', incarnation='one'), values=[[2, '11']])
        if failure == 'conflict':
            row['values'] = [[1, '99']]
        elif failure == 'identity':
            del row['metric']['incarnation']
        elif failure == 'nonfinite':
            row['values'] = [[2, 'NaN']]
        yield [row]
    c.metric_chunks.side_effect = chunks
    if failure == 'samples':
        limits = dict(LIMITS, max_samples=1)
    if failure == 'bytes':
        # Reject the request before retaining a response when metadata exceeds the budget.
        limits = dict(LIMITS, max_bytes=1)
    with pytest.raises((ValueError, RuntimeError, TimeoutError)):
        capture(c, deadline=deadline, limits=limits, directory=tmp_path/'window')
    raw = json.loads((tmp_path/'window/metrics.json').read_text())
    assert raw['capture']['status'] == 'INCOMPLETE' and raw['errors']
    if failure != 'bytes':
        assert raw['metrics']['sensor/temperature'][0]['points'] == [[1, 10]]
    with pytest.raises(ValueError, match='incomplete'):
        FrozenMetricWindow.read(tmp_path/'window')


def test_real_gaps_and_counter_resets_are_preserved_and_duplicate_chunks_are_deduplicated():
    c = client([dict(metric=dict(rack='west', host='a', incarnation='one'),
                     values=[[1, '10'], [8, '2'], [8, '2']])])
    frozen = capture_window(c, FIELDS, source='rack-probe', windows=[(0, 10), (0, 10)],
                            epoch=7, deadline=Deadline(float('inf')), limits=LIMITS)
    assert frozen.field_rows()['temperature'][0]['values'] == [[1, 10], [8, 2]]
    with pytest.raises(ValueError, match='coverage'):
        frozen.store().select('sensor/temperature', start=0, end=10, max_gap_s=2)


def test_manifest_corruption_and_overwriting_a_frozen_window_are_rejected(tmp_path):
    capture(client(), directory=tmp_path/'window')
    with pytest.raises(FileExistsError):
        capture(client(), directory=tmp_path/'window')
    path = tmp_path/'window/metrics.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='digest'):
        FrozenMetricWindow.read(tmp_path/'window')


def test_new_metric_and_new_target_require_only_declarations():
    fields = dict(FIELDS, voltage=dict(metric='sensor/voltage', labels={'rack': 'east'}))
    c = client()
    def chunks(metric_id, **kwargs):
        yield [dict(metric=dict(kwargs['labels'], host='b', incarnation='two'),
                    values=[[1.5, '5']])]
    c.metric_chunks.side_effect = chunks
    result = capture_window(c, fields, source='new-target', windows=[(1, 2)], epoch=8,
                            deadline=Deadline(float('inf')), limits=LIMITS)
    assert set(result.field_rows()) == {'temperature', 'voltage'}
    assert all(row['source'] == 'new-target' for rows in result.store().document['metrics'].values() for row in rows)


def test_missing_declared_series_remains_unavailable_even_after_successful_acquisition():
    fields = dict(FIELDS, voltage=dict(metric='sensor/voltage', labels={'rack': 'west'}))
    c = client()
    c.metric_chunks.side_effect = lambda identity, **kw: [[]] if identity.endswith('voltage') else client().metric_chunks()
    result = capture_window(c, fields, source='rack-probe', windows=[(0, 2)], epoch=7,
        deadline=Deadline(float('inf')), limits=LIMITS)
    assert result.field_rows()['voltage'] == []
    with pytest.raises(ValueError, match='no matching series'):
        result.store().select('sensor/voltage')


def test_existing_case_adds_a_registered_diagnostic_metric_in_one_binding():
    from pathlib import Path
    from document_io import load_document
    from scenario.program_compiler import configure_program
    path = Path(__file__).resolve().parents[1]/'config/scenarios/cache_scale_in.yaml'
    data = load_document(path)
    fields = data['parameters']['observation']['inputs']['engine_counters']['fields']
    fields['worker_status_qps'] = dict(metric='mock/worker_status_qps', labels={'role': 'prefill'})
    program = configure_program(data, str(path))
    assert all(v['stages'][2]['params']['gate_input']['fields']['worker_status_qps']['metric']
               == 'mock/worker_status_qps' for v in program['variants'])
