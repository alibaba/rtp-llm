"""Contract tests for the control-plane-only Epsilon integration.

These tests deliberately use a tiny in-process Epsilon double.  RTP-LLM must
only register state and arrive at the native snapshot barrier; check/dump/
restore are controller operations and are therefore not invoked here.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import threading
import unittest
from unittest import mock

from rtp_llm.utils import scr_template_utils as scr


class _EpsilonDouble:
    def __init__(self, result=0, error=None):
        self.result = result
        self.error = error
        self.arrivals = []
        self._lock = threading.Lock()

    def is_snapstart_enable(self):
        return True

    def snapstart_checkpoint(self, **kwargs):
        if self.error is not None:
            raise self.error
        with self._lock:
            self.arrivals.append(kwargs)
        return self.result


class _ControllerDouble:
    """Small controller-side quorum model used by contract tests."""

    def __init__(self, worker_num):
        self.worker_num = worker_num
        self.ids = []
        self.events = []

    def observe(self, arrivals):
        self.ids = [item["worker_id"] for item in arrivals]

    def checkpoint_ready(self):
        return len(self.ids) == self.worker_num and len(set(self.ids)) == self.worker_num

    def dump(self, path):
        self.events.append(("dump", path))
        return 0

    def wait_cr_done(self):
        self.events.append(("wait-cr-done",))
        return 0

    def restore(self, path):
        self.events.append(("restore", path))
        return 0


class ScrTemplateUtilsTest(unittest.TestCase):
    def setUp(self):
        scr._reset_for_test()

    def tearDown(self):
        scr._reset_for_test()

    def _enabled(self):
        return mock.patch.dict(
            os.environ, {"RTPLLM_ENABLE_SCR": "1"}, clear=True
        )

    def test_concurrent_rank_arrivals_use_one_scheduler_scope(self):
        epsilon = _EpsilonDouble()
        with self._enabled(), mock.patch.object(scr, "_load_epsilon", return_value=epsilon):
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(
                    executor.map(
                        lambda worker_id: scr.arrive_scr_checkpoint_barrier(
                            worker_id=worker_id, worker_num=4
                        ),
                        range(4),
                    )
                )

        self.assertEqual(results, [0, 0, 0, 0])
        # Every rank must have made exactly one arrival.
        self.assertEqual(
            sorted(item["worker_id"] for item in epsilon.arrivals), [0, 1, 2, 3]
        )
        self.assertTrue(all(item["worker_num"] == 4 for item in epsilon.arrivals))

    def test_timeout_is_fail_open_and_is_logged(self):
        epsilon = _EpsilonDouble(error=TimeoutError("snapshot barrier timeout"))
        with self._enabled(), mock.patch.object(
            scr, "_load_epsilon", return_value=epsilon
        ), mock.patch.object(scr.LOGGER, "exception") as log_exception:
            result = scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1)

        self.assertIsNone(result)
        log_exception.assert_called_once()
        self.assertIn("snapshot barrier arrival failed", log_exception.call_args.args[0])

    def test_successful_arrival_logs_snapstart_checkpoint_reached(self):
        epsilon = _EpsilonDouble()
        with self._enabled(), mock.patch.object(
            scr, "_load_epsilon", return_value=epsilon
        ), mock.patch.object(scr.LOGGER, "info") as log_info:
            self.assertEqual(
                scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1), 0
            )

        messages = [call.args[0] for call in log_info.call_args_list]
        self.assertTrue(any("sCR snapstart checkpoint reached" in message for message in messages))

    def test_fail_closed_timeout_raises_for_active_template_path(self):
        epsilon = _EpsilonDouble(error=TimeoutError("snapshot barrier timeout"))
        with self._enabled(), mock.patch.object(
            scr, "_load_epsilon", return_value=epsilon
        ):
            with self.assertRaisesRegex(scr.ScrArrivalError, "barrier arrival failed"):
                scr.arrive_scr_checkpoint_barrier(
                    worker_id=0, worker_num=1, fail_closed=True
                )

    def test_fail_closed_nonzero_result_raises(self):
        epsilon = _EpsilonDouble(result=17)
        with self._enabled(), mock.patch.object(
            scr, "_load_epsilon", return_value=epsilon
        ):
            with self.assertRaisesRegex(scr.ScrArrivalError, "barrier arrival failed"):
                scr.arrive_scr_checkpoint_barrier(
                    worker_id=0, worker_num=1, fail_closed=True
                )

    def test_fail_closed_late_success_is_treated_as_timeout(self):
        epsilon = _EpsilonDouble(result=0)
        with self._enabled(), mock.patch.object(
            scr, "_load_epsilon", return_value=epsilon
        ), mock.patch.object(scr.time, "monotonic", side_effect=[0.0, 2.0, 2.0]):
            with self.assertRaisesRegex(scr.ScrArrivalError, "exceeded timeout"):
                scr.arrive_scr_checkpoint_barrier(
                    worker_id=0,
                    worker_num=1,
                    timeout=1,
                    fail_closed=True,
                )

    def test_timeout_budget_accepts_controller_values_and_rejects_invalid_values(self):
        with mock.patch.dict(
            os.environ,
            {
                "RTPLLM_ENABLE_SCR": "1",
                "RTPLLM_SCR_CHECKPOINT_TIMEOUT_S": "37",
                "RTPLLM_SCR_INACTIVITY_TIMEOUT_S": "5",
            },
            clear=True,
        ):
            self.assertEqual(scr._scr_timeouts(), (37, 5))

        with mock.patch.dict(
            os.environ,
            {
                "RTPLLM_SCR_CHECKPOINT_TIMEOUT_S": "not-a-number",
                "RTPLLM_SCR_INACTIVITY_TIMEOUT_S": "0",
            },
            clear=True,
        ), mock.patch.object(scr.LOGGER, "error") as log_error:
            self.assertEqual(scr._scr_timeouts(), (scr.DEFAULT_SCR_TIMEOUT_S, scr.DEFAULT_SCR_INACTIVITY_TIMEOUT_S))
        self.assertEqual(log_error.call_count, 2)

    def test_arrival_passes_timeout_budget_and_restore_elapsed_is_observable(self):
        epsilon = _EpsilonDouble()
        restore_elapsed_ms = None
        generation = None
        with mock.patch.dict(
            os.environ,
            {
                "RTPLLM_ENABLE_SCR": "1",
                "RTPLLM_SCR_CHECKPOINT_TIMEOUT_S": "37",
                "RTPLLM_SCR_INACTIVITY_TIMEOUT_S": "5",
                scr.SCR_GENERATION_ENV: "generation-7",
                scr.SCR_PHASE_ENV: scr.SCR_PHASE_RESTORE,
                scr.SCR_RESTORE_START_TIME_ENV: "1000000",
            },
            clear=True,
        ), mock.patch.object(scr, "_load_epsilon", return_value=epsilon), mock.patch(
            "time.time", return_value=1002
        ):
            self.assertEqual(
                scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1), 0
            )
            restore_elapsed_ms = scr._restore_elapsed_ms()
            generation = scr._scr_generation()

        self.assertEqual(
            epsilon.arrivals,
            [
                {
                    "wait_mode": 1,
                    "worker_id": 0,
                    "worker_num": 1,
                    "timeout": 37,
                    "inactivity_timeout": 5,
                }
            ],
        )
        self.assertEqual(generation, "generation-7")
        self.assertEqual(restore_elapsed_ms, 2000.0)

    def test_nonzero_epsilon_result_is_preserved_for_controller_diagnostics(self):
        epsilon = _EpsilonDouble(result=17)
        with self._enabled(), mock.patch.object(scr, "_load_epsilon", return_value=epsilon):
            result = scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1)
        self.assertEqual(result, 17)

    def test_unclassifiable_epsilon_result_is_rejected(self):
        epsilon = _EpsilonDouble(result={"message": "success?"})
        with self._enabled(), mock.patch.object(
            scr, "_load_epsilon", return_value=epsilon
        ), mock.patch.object(scr.LOGGER, "exception") as log_exception:
            result = scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1)
        self.assertIsNone(result)
        log_exception.assert_called_once()

    def test_invalid_mapping_does_not_call_epsilon(self):
        epsilon = _EpsilonDouble()
        with self._enabled(), mock.patch.object(
            scr, "_load_epsilon", return_value=epsilon
        ):
            self.assertIsNone(scr.arrive_scr_checkpoint_barrier(worker_id=4, worker_num=4))
            self.assertIsNone(scr.arrive_scr_checkpoint_barrier(worker_id=-1, worker_num=4))
            self.assertIsNone(scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=0))
        self.assertEqual(epsilon.arrivals, [])

    def test_duplicate_and_missing_participants_are_visible_to_controller_double(self):
        """RTP-LLM emits IDs; the controller owns quorum validation.

        This intentionally does not hide duplicate IDs in RTP-LLM.  A
        controller can reject this generation before dump instead of allowing
        two processes to silently overwrite one Epsilon slot.
        """
        epsilon = _EpsilonDouble()
        controller = _ControllerDouble(worker_num=3)
        with self._enabled(), mock.patch.object(scr, "_load_epsilon", return_value=epsilon):
            for worker_id in (0, 1, 1):
                scr.arrive_scr_checkpoint_barrier(worker_id=worker_id, worker_num=3)

        controller.observe(epsilon.arrivals)
        self.assertFalse(controller.checkpoint_ready())
        self.assertEqual(controller.ids.count(1), 2)

        # A missing participant must remain non-ready even when fewer calls
        # happen to have been observed.  This is the control-plane decision,
        # not an RTP-LLM dump/restore operation.
        epsilon.arrivals.clear()
        for worker_id in (0, 1):
            scr.arrive_scr_checkpoint_barrier(worker_id=worker_id, worker_num=3)
        controller.observe(epsilon.arrivals)
        self.assertFalse(controller.checkpoint_ready())

    def test_mock_controller_lifecycle_is_external_to_rtp_llm(self):
        epsilon = _EpsilonDouble()
        controller = _ControllerDouble(worker_num=1)
        with self._enabled(), mock.patch.object(scr, "_load_epsilon", return_value=epsilon):
            self.assertEqual(
                scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1), 0
            )
        controller.observe(epsilon.arrivals)
        self.assertTrue(controller.checkpoint_ready())
        self.assertEqual(controller.dump("/tmp/mock-template"), 0)
        self.assertEqual(controller.wait_cr_done(), 0)
        self.assertEqual(controller.restore("/tmp/mock-template"), 0)
        self.assertEqual(
            controller.events,
            [
                ("dump", "/tmp/mock-template"),
                ("wait-cr-done",),
                ("restore", "/tmp/mock-template"),
            ],
        )
        # The runtime only made the Epsilon arrival; controller calls stayed in
        # the test-side double and are not reachable from production helpers.
        self.assertEqual(len(epsilon.arrivals), 1)

    def test_arrival_thread_returns_without_joining_controller_lifecycle(self):
        epsilon = _EpsilonDouble()
        with self._enabled(), mock.patch.object(scr, "_load_epsilon", return_value=epsilon):
            thread = scr.start_scr_checkpoint_arrival_thread(
                worker_id=0, worker_num=1, name="test-scr-arrival"
            )
            self.assertIsNotNone(thread)
            thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual([item["worker_id"] for item in epsilon.arrivals], [0])

    def test_scr_disabled_arrival_path_does_not_import_or_start_thread(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            scr, "_load_epsilon"
        ) as load_epsilon:
            self.assertIsNone(
                scr.start_scr_checkpoint_arrival_thread(worker_id=0, worker_num=1)
            )
        load_epsilon.assert_not_called()

    def test_template_phase_requires_feature_gate_and_external_phase(self):
        cases = [
            ({}, False),
            ({"RTPLLM_ENABLE_SCR": "1"}, False),
            ({"SCR_PHASE": "checkpoint"}, False),
            (
                {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "checkpoint"},
                True,
            ),
            ({"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "restore"}, True),
            ({"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "normal"}, False),
        ]
        for environment, expected in cases:
            with self.subTest(environment=environment), mock.patch.dict(
                os.environ, environment, clear=True
            ):
                self.assertEqual(scr.is_scr_template_phase_active(), expected)


if __name__ == "__main__":
    unittest.main()
