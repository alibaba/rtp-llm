"""Source latency, cancellation and failures stay separate from control reads."""
import json
import threading
import time

import pytest

from runtime.deadline import Deadline
from adapters.background_source import BackgroundSource
from runtime.sample_worker import SamplingError


def source(tmp_path, read, **params):
    return BackgroundSource('custom', read, tmp_path/'samples.jsonl',
        identity=dict(env_epoch=7),
        clock=time.monotonic, wall_clock=time.time,
        params=dict(duration_s=30, interval_s=.01, max_samples=100, max_bytes=10000) | params)


def test_blocked_diagnostic_does_not_block_other_sources_and_unjoined_stop_is_visible(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def read(deadline):
        entered.set()
        release.wait(2)
        return {'value': 1}
    sampler = source(tmp_path, read)
    sampler.start(Deadline(time.monotonic()+1))
    try:
        assert entered.wait(1)
        assert sampler.latest() is None
        # No read or join is performed by a latest lookup.
        assert [sampler.latest() for _ in range(10)] == [None]*10
        with pytest.raises(TimeoutError):
            sampler.stop_sampling(Deadline(time.monotonic()+.01))
        assert json.loads((tmp_path/'samples.status.json').read_text())['joined'] is False
    finally:
        release.set()
        sampler.stop_sampling(Deadline(time.monotonic()+2))
    status = json.loads((tmp_path/'samples.status.json').read_text())
    assert status['joined'] is True


def test_new_source_only_supplies_adapter_name_and_budget_and_retains_read_times(tmp_path):
    ready = threading.Event()
    def read(deadline):
        ready.set()
        return dict(host='new-host', generation=3, new_metric=12)
    sampler = source(tmp_path, read)
    sampler.start(Deadline(time.monotonic()+1))
    assert ready.wait(1)
    deadline = Deadline(time.monotonic()+1)
    while sampler.latest() is None:
        deadline.sleep(.001)
    # stop joins an in-progress capture before the artifact is consumed.
    sampler.stop_sampling(Deadline(time.monotonic()+1))
    rows = [json.loads(line) for line in sampler.path.read_text().splitlines()]
    assert rows and rows[0]['data']['new_metric'] == 12
    assert rows[0]['started_mono'] <= rows[0]['finished_mono']
    assert rows[0]['started_epoch_s'] <= rows[0]['finished_epoch_s']
    latest = sampler.latest()
    latest['data'].clear()
    assert sampler.latest()['data']['generation'] == 3


def test_source_exception_and_budget_exhaustion_never_become_success(tmp_path):
    sampler = source(tmp_path, lambda d: {'payload': 'x'*1000}, max_bytes=10)
    sampler.start(Deadline(time.monotonic()+1))
    assert sampler.done_event.wait(1)
    sampler.stop_sampling(Deadline(time.monotonic()+1))
    with pytest.raises(SamplingError, match='budget'):
        sampler.latest()
    assert sampler.path.read_text() == ''


def test_source_read_failure_is_preserved_in_shutdown_artifact(tmp_path):
    def read(deadline):
        raise RuntimeError('adapter failed')
    sampler = source(tmp_path, read)
    sampler.start(Deadline(time.monotonic()+1))
    assert sampler.done_event.wait(1)
    sampler.stop_sampling(Deadline(time.monotonic()+1))
    with pytest.raises(SamplingError, match='adapter failed'):
        sampler.latest()
    assert 'adapter failed' in json.loads((tmp_path/'samples.status.json').read_text())['error']


def test_real_resource_scope_stops_and_joins_source_and_preserves_retry_ownership(tmp_path):
    from scenario.resources import ResourceScope
    entered = threading.Event()
    def read(deadline):
        entered.set()
        while True:
            deadline.check()
            time.sleep(.001)
    scope=ResourceScope(time.monotonic,time.sleep,False)
    sampler=source(tmp_path,read).attach(scope,Deadline(time.monotonic()+1))
    assert entered.wait(1)
    assert all(row['status']=='PASS' for row in scope.cleanup(1))
    assert not sampler.thread.is_alive()
    assert json.loads((tmp_path/'samples.status.json').read_text())['joined'] is True
    with pytest.raises(ValueError,match='owner'):
        sampler.attach(scope,Deadline(time.monotonic()+1))
