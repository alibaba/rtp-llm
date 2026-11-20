"""In-process scheduler/controller contract test for the sCR integration."""

from __future__ import annotations

import os
import threading
import unittest
from unittest import mock
from concurrent.futures import ThreadPoolExecutor

from rtp_llm.utils.scr import template_utils as scr


class _MockScheduler:
    def __init__(self, worker_num: int) -> None:
        self.worker_num = worker_num
        self.arrivals = {}
        self.released = False
        self.result = 0
        self.condition = threading.Condition()

    def arrive(self, kwargs):
        worker_id = kwargs["worker_id"]
        with self.condition:
            if worker_id in self.arrivals:
                raise RuntimeError(f"duplicate worker_id={worker_id}")
            if kwargs["worker_num"] != self.worker_num:
                raise RuntimeError("worker_num mismatch")
            self.arrivals[worker_id] = dict(kwargs)
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.released)
            return self.result

    def wait_ready(self, timeout: float) -> bool:
        with self.condition:
            return self.condition.wait_for(
                lambda: len(self.arrivals) == self.worker_num,
                timeout=timeout,
            )

    def release(self, result=0) -> None:
        with self.condition:
            self.result = result
            self.released = True
            self.condition.notify_all()


class _MockEpsilon:
    def __init__(self, scheduler: _MockScheduler) -> None:
        self.scheduler = scheduler

    def is_snapstart_enable(self):
        return True

    def snapstart_checkpoint(
        self,
        *,
        wait_mode,
        worker_id,
        worker_num,
    ):
        return self.scheduler.arrive(
            {
                "wait_mode": wait_mode,
                "worker_id": worker_id,
                "worker_num": worker_num,
            }
        )


class ScrSchedulerE2ETest(unittest.TestCase):
    def test_participants_block_until_control_plane_release(self) -> None:
        scheduler = _MockScheduler(worker_num=4)
        with mock.patch.dict(
            os.environ,
            {
                scr.RTPLLM_ENABLE_SCR_ENV: "1",
                scr.SCR_PHASE_ENV: scr.SCR_PHASE_CHECKPOINT,
            },
            clear=True,
        ), mock.patch.object(
            scr, "_load_epsilon", return_value=_MockEpsilon(scheduler)
        ), ThreadPoolExecutor(
            max_workers=4
        ) as workers:
            arrivals = [
                workers.submit(
                    scr.arrive_scr_checkpoint_barrier,
                    worker_id=worker_id,
                    worker_num=4,
                )
                for worker_id in range(4)
            ]
            try:
                self.assertTrue(scheduler.wait_ready(timeout=2))
                self.assertEqual(sorted(scheduler.arrivals), [0, 1, 2, 3])
                self.assertTrue(all(not arrival.done() for arrival in arrivals))
            finally:
                scheduler.release()
            self.assertEqual(
                [arrival.result(timeout=3) for arrival in arrivals], [0] * 4
            )

    def test_control_plane_timeout_aborts_the_waiting_participant(self):
        scheduler = _MockScheduler(worker_num=2)
        with mock.patch.dict(
            os.environ, {scr.RTPLLM_ENABLE_SCR_ENV: "1"}, clear=True
        ), mock.patch.object(
            scr, "_load_epsilon", return_value=_MockEpsilon(scheduler)
        ), ThreadPoolExecutor(max_workers=1) as workers:
            arrival = workers.submit(
                scr.arrive_scr_checkpoint_barrier,
                worker_id=0,
                worker_num=2,
                fail_closed=True,
            )
            try:
                with scheduler.condition:
                    self.assertTrue(scheduler.condition.wait_for(
                        lambda: len(scheduler.arrivals) == 1, timeout=2
                    ))
                self.assertFalse(arrival.done())
            finally:
                # Model the controller's timeout result; RTP supplies no budget.
                scheduler.release(result=110)
            with self.assertRaises(scr.ScrArrivalError):
                arrival.result(timeout=3)


if __name__ == "__main__":
    unittest.main()
