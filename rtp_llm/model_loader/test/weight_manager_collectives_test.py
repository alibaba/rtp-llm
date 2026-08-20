"""Unit tests for WeightManager's NCCL sleep/wake seam.

GPU-free. These three methods are thin, but they are the *whole* Python side of
the C++ sleep-hook contract, so what is asserted here is the contract rather than
the forwarding:

  - suspend is gated on ``--sleep_release_collective_memory`` and resume is NOT
    (a mid-sleep config flip must not leave a communicator unmapped);
  - both propagate; sleep/wake is atomic, so swallowing here would let the
    instance wake onto memory that was never remapped;
  - the device and reason reach :mod:`rtp_llm.utils.nccl_memory` intact -- the
    device in particular, because ``getCommPtr()`` keys on the *current* device
    and silently enumerates nothing when it is wrong;
  - ``nccl_memory_status`` is a single-line, empty-unless-NCCL diagnostic.

The policy layer itself is covered by rtp_llm/utils/test/nccl_memory_test.py.
"""

import unittest
from typing import Any, List, Optional, Tuple
from unittest import mock

from rtp_llm.model_loader import weight_memory_saver as wms
from rtp_llm.model_loader.weight_manager import WeightManager
from rtp_llm.utils import nccl_memory

_DEVICE = "cuda:3"


class _Boom(RuntimeError):
    """Stands in for NcclMemoryError; the seam must not care which it is."""


def _manager(device: Any = _DEVICE) -> WeightManager:
    """A WeightManager with only the attribute these three methods read.

    ``__init__`` loads a model, and none of that is reachable from here. Bypassing
    it keeps the test CPU-only while still exercising the real methods rather than
    a copy of them.
    """

    manager = WeightManager.__new__(WeightManager)
    manager._device = device
    return manager


class WeightManagerCollectivesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.suspend_calls: List[Tuple[Any, Optional[str]]] = []
        self.resume_calls: List[Tuple[Any, Optional[str]]] = []

    def _patch(
        self,
        gate: bool,
        suspend_exc: Optional[BaseException] = None,
        resume_exc: Optional[BaseException] = None,
    ) -> None:
        def fake_suspend(device: Any, reason: str = "sleep") -> None:
            self.suspend_calls.append((device, reason))
            if suspend_exc is not None:
                raise suspend_exc

        def fake_resume(device: Any, reason: str = "wake") -> None:
            self.resume_calls.append((device, reason))
            if resume_exc is not None:
                raise resume_exc

        for target, attr, value in (
            (wms, "release_collective_memory", lambda: gate),
            (nccl_memory, "suspend_for_sleep", fake_suspend),
            (nccl_memory, "resume_after_wake", fake_resume),
        ):
            patcher = mock.patch.object(target, attr, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_suspend_is_a_no_op_while_the_switch_is_off(self) -> None:
        self._patch(gate=False)
        _manager().suspend_collectives_for_sleep()
        self.assertEqual(self.suspend_calls, [])

    def test_suspend_forwards_the_device_and_reason_once(self) -> None:
        self._patch(gate=True)
        _manager().suspend_collectives_for_sleep(reason="sleep-l2")
        self.assertEqual(self.suspend_calls, [(_DEVICE, "sleep-l2")])

    def test_suspend_defaults_the_reason_to_sleep(self) -> None:
        self._patch(gate=True)
        _manager().suspend_collectives_for_sleep()
        self.assertEqual(self.suspend_calls, [(_DEVICE, "sleep")])

    def test_suspend_propagates_so_the_transition_fails(self) -> None:
        # The C++ hook turns this into ERROR -> FAILED_PRECONDITION. Swallowing it
        # would report a successful sleep over communicators that are still mapped
        # on some ranks and not others.
        self._patch(gate=True, suspend_exc=_Boom("ncclCommSuspend failed"))
        with self.assertRaises(_Boom):
            _manager().suspend_collectives_for_sleep()
        self.assertEqual(len(self.suspend_calls), 1)

    def test_resume_runs_even_though_the_switch_is_off(self) -> None:
        # What must be undone is whatever was actually suspended, not whatever the
        # config currently says. resume_after_wake itself no-ops when nothing was.
        self._patch(gate=False)
        _manager().resume_collectives_for_wake()
        self.assertEqual(self.resume_calls, [(_DEVICE, "wake")])

    def test_resume_never_consults_the_switch(self) -> None:
        def exploding_gate() -> bool:
            raise AssertionError("resume must not read the config switch")

        self._patch(gate=True)
        with mock.patch.object(wms, "release_collective_memory", exploding_gate):
            _manager().resume_collectives_for_wake(reason="wake-l2")
        self.assertEqual(self.resume_calls, [(_DEVICE, "wake-l2")])

    def test_resume_propagates_so_the_wake_fails(self) -> None:
        self._patch(gate=True, resume_exc=_Boom("ncclCommResume failed"))
        with self.assertRaises(_Boom):
            _manager().resume_collectives_for_wake()
        self.assertEqual(len(self.resume_calls), 1)

    def test_a_none_device_is_forwarded_rather_than_rejected(self) -> None:
        # nccl_memory treats None as "do not touch the current device"; the seam
        # must not invent a device of its own.
        self._patch(gate=True)
        _manager(device=None).suspend_collectives_for_sleep()
        _manager(device=None).resume_collectives_for_wake()
        self.assertEqual(self.suspend_calls, [(None, "sleep")])
        self.assertEqual(self.resume_calls, [(None, "wake")])


class NcclMemoryStatusTest(unittest.TestCase):
    def tearDown(self) -> None:
        nccl_memory._reset_for_testing()

    def test_status_is_empty_when_nccl_is_not_the_cause(self) -> None:
        # Load-bearing: the C++ hook appends whatever this returns to
        # last_error for failures that are usually nothing to do with NCCL.
        nccl_memory._reset_for_testing()
        self.assertEqual(_manager().nccl_memory_status(), "")

    def test_status_reports_the_recorded_failure_on_one_line(self) -> None:
        nccl_memory._record_failure("[sleep] ncclCommSuspend failed\n  for tp(rc=3)")
        status = _manager().nccl_memory_status()
        self.assertIn("ncclCommSuspend failed", status)
        self.assertNotIn("\n", status)

    def test_status_survives_a_failure_recorded_from_another_thread(self) -> None:
        # status_text() is deliberately lock-free: _lock is held across the untimed
        # NCCL barrier, so a diagnostic that took it could block on exactly the hang
        # it exists to explain. Reading mid-transition must still yield a string.
        import threading

        def record() -> None:
            nccl_memory._record_failure("[wake] resume vote did not reach GO")

        thread = threading.Thread(target=record)
        thread.start()
        thread.join()
        self.assertIn("resume vote did not reach GO", _manager().nccl_memory_status())


if __name__ == "__main__":
    unittest.main()
