"""Contract tests for the control-plane-only Epsilon integration.

These tests deliberately use a tiny in-process Epsilon double.  RTP-LLM must
only register state and arrive at the native snapshot barrier; check/dump/
restore are controller operations and are therefore not invoked here.
"""

import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

from rtp_llm.utils.scr import template_utils as scr


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

    def test_provider_error_is_logged_for_optional_arrival(self):
        epsilon = _EpsilonDouble(error=TimeoutError("snapshot barrier timeout"))
        with self._enabled(), mock.patch.object(
            scr, "_load_epsilon", return_value=epsilon
        ), mock.patch.object(scr.LOGGER, "exception") as log_exception:
            result = scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1)

        self.assertIsNone(result)
        log_exception.assert_called_once()
        self.assertIn("snapshot barrier arrival failed", log_exception.call_args.args[0])

    def test_provider_cancellation_fails_closed_for_active_template(self):
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

    def test_success_records_a_fresh_barrier_return_time(self):
        # Startup timing begins when the provider releases the restored process;
        # RTP never uses this clock to decide whether the barrier timed out.
        epsilon = _EpsilonDouble(result=0)
        with self._enabled(), mock.patch.object(
            scr, "_load_epsilon", return_value=epsilon
        ), mock.patch.object(scr.time, "monotonic", return_value=100000.0):
            self.assertEqual(
                scr.arrive_scr_checkpoint_barrier(
                    worker_id=0,
                    worker_num=1,
                    fail_closed=True,
                ),
                0,
            )
        self.assertEqual(scr.get_scr_barrier_return_time(), 100000.0)

    def test_failed_arrival_clears_previous_return_time(self):
        with self._enabled(), mock.patch.object(
            scr, "_load_epsilon", return_value=_EpsilonDouble()
        ):
            scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1)
            self.assertIsNotNone(scr.get_scr_barrier_return_time())
        with self._enabled(), mock.patch.object(
            scr, "_load_epsilon", return_value=_EpsilonDouble(error=TimeoutError())
        ):
            with self.assertRaises(scr.ScrArrivalError):
                scr.arrive_scr_checkpoint_barrier(
                    worker_id=0, worker_num=1, fail_closed=True
                )
        self.assertIsNone(scr.get_scr_barrier_return_time())

    def test_arrival_supplies_only_blocking_mode_and_participant_identity(self):
        epsilon = _EpsilonDouble()
        with mock.patch.dict(
            os.environ,
            {
                "RTPLLM_ENABLE_SCR": "1",
                scr.SCR_PHASE_ENV: scr.SCR_PHASE_RESTORE,
            },
            clear=True,
        ), mock.patch.object(scr, "_load_epsilon", return_value=epsilon):
            self.assertEqual(
                scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1), 0
            )

        self.assertEqual(
            epsilon.arrivals,
            [
                {
                    "wait_mode": 1,
                    "worker_id": 0,
                    "worker_num": 1,
                }
            ],
        )

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

    def test_scr_disabled_arrival_path_does_not_import_epsilon(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            scr, "_load_epsilon"
        ) as load_epsilon:
            self.assertIsNone(
                scr.arrive_scr_checkpoint_barrier(worker_id=0, worker_num=1)
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
