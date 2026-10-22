"""Foreground and joined background observations use the same absolute ticks."""
from types import SimpleNamespace
from unittest.mock import Mock
import io
import json

import pytest

from runtime.observation import FrozenSnapshot, SampleSchedule
from runtime.sample_worker import SampleWorker, SamplingError
from runtime.deadline import Deadline
from monitoring.client import PrometheusClient


def test_background_sampling_skips_overruns_without_drift_or_catchup():
    now = [0.0]
    def capture(deadline):
        deadline.check()
        stamp = now[0]
        now[0] += 1.3
        return FrozenSnapshot(json.dumps(stamp))
    worker = SampleWorker(capture, lambda: now[0], dict(duration_s=6.8, interval_s=1,
        max_samples=20, max_bytes=1000))
    def wait(seconds):
        now[0] += seconds
        return False
    worker.stop_event = SimpleNamespace(wait=wait, is_set=lambda: False, set=lambda: None)
    worker._run()
    assert [json.loads(v) for v in worker.samples] == [1, 3, 5]
    assert worker.worker_exit_mono == 6.8
    assert worker.done_event.is_set()


def test_background_failure_and_unjoined_exit_are_not_success():
    worker = SampleWorker(Mock(side_effect=RuntimeError('source failed')), lambda: 0,
        dict(duration_s=10, interval_s=.001, max_samples=10, max_bytes=100))
    worker._run()
    assert 'source failed' in worker.error
    assert worker.done_event.is_set()
    worker.thread = Mock()
    worker.thread.is_alive.return_value = True
    with pytest.raises(TimeoutError, match='still alive'):
        worker.stop_sampling(Deadline(float('inf')))
    worker.worker_exit_mono = None
    with pytest.raises(SamplingError, match='exit record'):
        worker.stop_sampling(Deadline(float('inf')))


def test_schedule_rejects_invalid_intervals():
    for value in (0, -1, float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            SampleSchedule(0, value)


def client(**kwargs):
    return PrometheusClient('http://unused', query_plan={}, target_kinds={'worker': 'mock'},
                            max_gap_s=5, **kwargs)


def test_health_read_does_not_fetch_metric_payload():
    c = client()
    c.query = Mock(return_value=[dict(values=[[1, '1']])])
    deadline = Deadline(float('inf'))
    assert c.scrape_time('worker', deadline=deadline) == 1
    assert c.query.call_count == 1
    assert c.query.call_args.args[0] == 'up{job="worker"}[5000ms]'
    assert c.query.call_args.kwargs['deadline'] is deadline


def test_query_response_budget_and_expired_deadline(monkeypatch):
    open_url = Mock(return_value=io.BytesIO(b'x'*101))
    monkeypatch.setattr('urllib.request.urlopen', open_url)
    with pytest.raises(ValueError, match='byte budget'):
        client(max_response_bytes=100).query('up')
    open_url.reset_mock()
    with pytest.raises(TimeoutError):
        client().query('up', deadline=Deadline(0))
    open_url.assert_not_called()
