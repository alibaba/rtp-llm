"""Sampler faults must never prevent owned client cancellation and reaping."""

import subprocess
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from tests.ha_fixtures import client_resource
from runtime.cleanup import cleanup_all
from scenario.runtime import Deadline


def test_sampler_failure_still_reaps_a_real_client():
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    sampler = Mock()
    sampler.stop.side_effect = RuntimeError('sample budget exceeded')
    owned = client_resource(state_sampler=sampler, proc=NS(proc=child, alive=lambda: child.poll() is None))
    try:
        with pytest.raises(RuntimeError, match='sample budget exceeded'):
            owned.cleanup(Deadline(time.monotonic() + 5))
        assert child.poll() is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_client_failure_does_not_skip_sampler_and_both_errors_are_retained():
    process, sampler = Mock(), Mock()
    process.alive.return_value = True
    process.proc.terminate.side_effect = OSError('cannot terminate')
    sampler.stop.side_effect = RuntimeError('sampler failed')
    with pytest.raises(RuntimeError, match='cannot terminate.*sampler failed'):
        client_resource(proc=process, state_sampler=sampler).cleanup(NS(remaining=lambda: 5))
    sampler.stop.assert_called_once()


def test_unresponsive_client_is_killed_and_reaped():
    process, sampler = Mock(), Mock()
    process.alive.return_value = True
    process.proc.wait.side_effect = [subprocess.TimeoutExpired('client', 2), 0]
    client_resource(proc=process, state_sampler=sampler).cleanup(NS(remaining=lambda: 4))
    process.proc.terminate.assert_called_once()
    process.proc.kill.assert_called_once()
    assert process.proc.wait.call_count == 2
    sampler.stop.assert_called_once()


def test_expired_budget_still_cancels_client_and_attempts_sampler():
    process, sampler = Mock(), Mock()
    process.alive.return_value = True
    deadline = Mock()
    deadline.remaining.side_effect = TimeoutError('expired')
    with pytest.raises(TimeoutError, match='expired'):
        client_resource(proc=process, state_sampler=sampler).cleanup(deadline)
    process.proc.terminate.assert_called_once()
    process.proc.kill.assert_called_once()
    process.proc.wait.assert_called_once_with(timeout=0)
    sampler.stop.assert_called_once()


def test_independent_cleanup_retains_timeout_classification():
    later = Mock()
    earlier = Mock(side_effect=TimeoutError('join expired'))
    with pytest.raises(TimeoutError, match='join expired'):
        cleanup_all([('earlier', earlier), ('later', later)])
    later.assert_called_once()


def test_kill_reap_timeout_is_reported_as_timeout_and_sampler_is_still_stopped():
    process, sampler = Mock(), Mock()
    process.alive.return_value = True
    process.proc.wait.side_effect = subprocess.TimeoutExpired('client', 2)
    with pytest.raises(TimeoutError, match='TimeoutExpired'):
        client_resource(proc=process, state_sampler=sampler).cleanup(NS(remaining=lambda: 4))
    process.proc.kill.assert_called_once()
    sampler.stop.assert_called_once()
