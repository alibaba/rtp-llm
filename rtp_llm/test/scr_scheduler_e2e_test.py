"""In-process scheduler/controller contract test for the sCR integration."""

from __future__ import annotations

import os
import threading
import time
import unittest
from unittest import mock

from rtp_llm.utils import scr_template_utils as scr


class _MockScheduler:
    def __init__(self, worker_num: int) -> None:
        self.worker_num = worker_num
        self.arrivals = {}
        self.released = False
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
            deadline = time.monotonic() + kwargs["timeout"]
            while not self.released:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return 110
                self.condition.wait(timeout=remaining)
            return 0

    def wait_ready(self, timeout: float) -> bool:
        with self.condition:
            return self.condition.wait_for(
                lambda: len(self.arrivals) == self.worker_num,
                timeout=timeout,
            )

    def release(self) -> None:
        with self.condition:
            self.released = True
            self.condition.notify_all()


class _MockEpsilon:
    def __init__(self, scheduler: _MockScheduler) -> None:
        self.scheduler = scheduler
        self.API_VERSION = "mock-v2"

    def is_snapstart_enable(self):
        return True

    def snapstart_checkpoint(
        self,
        *,
        wait_mode,
        worker_id,
        worker_num,
        timeout,
        inactivity_timeout,
    ):
        return self.scheduler.arrive(
            {
                "wait_mode": wait_mode,
                "worker_id": worker_id,
                "worker_num": worker_num,
                "timeout": timeout,
                "inactivity_timeout": inactivity_timeout,
            }
        )


class _MockController:
    def __init__(self, scheduler: _MockScheduler) -> None:
        self.scheduler = scheduler
        self.events = []

    def run(self) -> None:
        if not self.scheduler.wait_ready(timeout=2):
            self.events.append(("abort", "missing-participant"))
            self.scheduler.release()
            return
        ids = sorted(self.scheduler.arrivals)
        if ids != list(range(self.scheduler.worker_num)):
            self.events.append(("abort", "invalid-manifest"))
            self.scheduler.release()
            return
        self.events.append(("check", 0))
        self.events.append(("checkpoint-ready", True))
        self.events.append(("dump", 0))
        self.events.append(("wait-cr-done", 0))
        self.events.append(("restore", 0))
        self.scheduler.release()


class ScrSchedulerE2ETest(unittest.TestCase):
    def test_arrival_check_dump_restore_release(self) -> None:
        scheduler = _MockScheduler(worker_num=4)
        epsilon = _MockEpsilon(scheduler)
        controller = _MockController(scheduler)
        with mock.patch.dict(
            os.environ,
            {
                scr.SCR_ENABLE_ENV: "1",
                scr.SCR_PHASE_ENV: scr.SCR_PHASE_CHECKPOINT,
                scr.SCR_GENERATION_ENV: "e2e-generation-1",
            },
            clear=True,
        ), mock.patch.object(scr, "_load_epsilon", return_value=epsilon):
            controller_thread = threading.Thread(target=controller.run, daemon=True)
            controller_thread.start()
            threads = [
                scr.start_scr_checkpoint_arrival_thread(
                    worker_id=worker_id,
                    worker_num=4,
                    generation="e2e-generation-1",
                    timeout=2,
                    inactivity_timeout=1,
                    name=f"e2e-arrival-{worker_id}",
                )
                for worker_id in range(4)
            ]
            for thread in threads:
                self.assertIsNotNone(thread)
                thread.join(timeout=3)
            controller_thread.join(timeout=3)

        self.assertFalse(controller_thread.is_alive())
        self.assertEqual(sorted(scheduler.arrivals), [0, 1, 2, 3])
        self.assertEqual(
            controller.events,
            [
                ("check", 0),
                ("checkpoint-ready", True),
                ("dump", 0),
                ("wait-cr-done", 0),
                ("restore", 0),
            ],
        )
        self.assertTrue(scheduler.released)

    def test_generation_mismatch_never_reaches_scheduler(self) -> None:
        scheduler = _MockScheduler(worker_num=1)
        epsilon = _MockEpsilon(scheduler)
        with mock.patch.dict(
            os.environ,
            {
                scr.SCR_ENABLE_ENV: "1",
                scr.SCR_PHASE_ENV: scr.SCR_PHASE_CHECKPOINT,
                scr.SCR_GENERATION_ENV: "current-generation",
            },
            clear=True,
        ), mock.patch.object(scr, "_load_epsilon", return_value=epsilon):
            result = scr.arrive_scr_checkpoint_barrier(
                worker_id=0,
                worker_num=1,
                generation="stale-generation",
            )
        self.assertIsNone(result)
        self.assertEqual(scheduler.arrivals, {})


if __name__ == "__main__":
    unittest.main()
