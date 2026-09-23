"""CPU-only tests for the actual read-only GPU lease drain policy."""

import importlib.util
import subprocess
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

_SPEC = importlib.util.spec_from_file_location(
    "device_resource_under_test", Path(__file__).with_name("device_resource.py")
)
resource = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(resource)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _reply(pids="", rc=0):
    return types.SimpleNamespace(returncode=rc, stdout=pids, stderr="")


class ReadOnlyGpuDrainTest(unittest.TestCase):
    def setUp(self):
        self._patches = ExitStack()
        self.addCleanup(self._patches.close)
        self.clock = _Clock()
        self.owner = resource.DeviceResource.__new__(resource.DeviceResource)
        self.owner.gpu_ids = ["0", "1"]
        self.stack = self._patches.enter_context(
            mock.patch.dict(resource.os.environ, {"RTP_LLM_GPU_LOCK_NO_KILL": "1"})
        )
        self._patches.enter_context(
            mock.patch.object(resource.time, "monotonic", self.clock.monotonic)
        )
        self._patches.enter_context(
            mock.patch.object(resource.time, "sleep", self.clock.sleep)
        )
        self.kill = self._patches.enter_context(
            mock.patch.object(
                resource.os, "kill", side_effect=AssertionError("no signals")
            )
        )

    def test_already_empty_queries_every_leased_gpu(self):
        with mock.patch.object(
            resource.subprocess, "run", return_value=_reply()
        ) as query:
            self.assertTrue(self.owner._ensure_gpus_released(timeout=1))
        self.assertEqual(query.call_count, 2)
        self.assertEqual(self.clock.now, 0)
        self.kill.assert_not_called()

    def test_delayed_exit_is_polled_without_signals(self):
        with mock.patch.object(
            resource.subprocess,
            "run",
            side_effect=[_reply("606601\n"), _reply(), _reply(), _reply()],
        ) as query:
            self.assertTrue(self.owner._ensure_gpus_released(timeout=1))
        self.assertEqual(query.call_count, 4)
        self.assertEqual(self.clock.now, 0.25)
        self.kill.assert_not_called()

    def test_busy_second_gpu_is_not_ignored(self):
        with mock.patch.object(
            resource.subprocess,
            "run",
            side_effect=[_reply(), _reply("42\n"), _reply(), _reply()],
        ):
            self.assertTrue(self.owner._ensure_gpus_released(timeout=1))
        self.assertEqual(self.clock.now, 0.25)

    def test_persistent_client_times_out_and_fails(self):
        with mock.patch.object(resource.subprocess, "run", return_value=_reply("42\n")):
            self.assertFalse(self.owner._ensure_gpus_released(timeout=0.6))
        self.assertAlmostEqual(self.clock.now, 0.6)
        self.kill.assert_not_called()

    def test_zero_timeout_is_one_observation_not_a_waiver(self):
        with mock.patch.object(
            resource.subprocess, "run", return_value=_reply("42\n")
        ) as query:
            self.assertFalse(self.owner._ensure_gpus_released(timeout=0))
        self.assertEqual(query.call_count, 2)
        self.assertEqual(self.clock.now, 0)

    def test_query_error_fails_closed_even_with_empty_output(self):
        with mock.patch.object(
            resource.subprocess, "run", return_value=_reply(rc=1)
        ) as query:
            self.assertFalse(self.owner._ensure_gpus_released(timeout=1))
        self.assertEqual(query.call_count, 1)
        self.assertEqual(self.clock.now, 0)

    def test_query_exception_fails_closed(self):
        with mock.patch.object(
            resource.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 10),
        ):
            self.assertFalse(self.owner._ensure_gpus_released(timeout=1))
        self.kill.assert_not_called()

    def test_query_failure_after_busy_does_not_become_success(self):
        with mock.patch.object(
            resource.subprocess,
            "run",
            side_effect=[_reply("42\n"), _reply(), _reply(rc=1)],
        ):
            self.assertFalse(self.owner._ensure_gpus_released(timeout=1))

    def test_lease_remains_owned_until_drain_completes(self):
        events = []
        self.owner.gpu_locks = types.SimpleNamespace(
            close=lambda: events.append("unlock")
        )
        self.owner.global_lock_file = "not-opened-by-test"
        lock = mock.MagicMock()
        replies = iter([_reply("42\n"), _reply(), _reply(), _reply()])

        def query(*args, **kwargs):
            self.assertEqual(self.owner.gpu_ids, ["0", "1"])
            self.assertNotIn("unlock", events)
            events.append("query")
            return next(replies)

        with mock.patch.object(
            resource, "FileLock", return_value=lock
        ), mock.patch.object(resource.subprocess, "run", side_effect=query):
            self.owner.__exit__(None, None, None)
        self.assertEqual(events, ["query"] * 4 + ["unlock"])
        self.assertEqual(self.owner.gpu_ids, [])

    def test_release_failure_is_not_silenced(self):
        self.owner.gpu_locks = mock.Mock()
        self.owner.global_lock_file = "not-opened-by-test"
        with mock.patch.object(resource, "FileLock"), mock.patch.object(
            self.owner, "_ensure_gpus_released", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU release failed"):
                self.owner.__exit__(None, None, None)
        self.owner.gpu_locks.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
