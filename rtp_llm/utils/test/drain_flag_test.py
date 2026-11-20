import multiprocessing
import os
import signal
import threading
import time
import unittest

from rtp_llm.utils.drain_flag import DrainFlag


def _wait_forever(flag: DrainFlag, waiting) -> None:
    waiting.set()
    while not flag.wait(0.1):
        pass


def _report_when_set(flag: DrainFlag, seen) -> None:
    seen.put(flag.wait(30))


def _relay(flag: DrainFlag, seen) -> None:
    # A backend manager hands the flag on to its ranks.
    child = multiprocessing.get_context("spawn").Process(
        target=_report_when_set, args=(flag, seen)
    )
    child.start()
    child.join()


class DrainFlagTest(unittest.TestCase):
    def setUp(self):
        self.ctx = multiprocessing.get_context("spawn")

    def test_set_returns_after_a_waiter_was_killed_inside_wait(self):
        # multiprocessing.Event.set() blocks forever here.
        flag = DrainFlag()
        waiting = (
            self.ctx.Event()
        )  # only the test's own handshake, never set() after a kill
        waiter = self.ctx.Process(target=_wait_forever, args=(flag, waiting))
        waiter.start()
        self.assertTrue(waiting.wait(60))
        time.sleep(0.3)  # inside wait()
        os.kill(waiter.pid, signal.SIGKILL)
        waiter.join()
        done = threading.Event()
        threading.Thread(target=lambda: (flag.set(), done.set()), daemon=True).start()
        self.assertTrue(done.wait(5), "set() blocked on a dead waiter")
        self.assertTrue(flag.is_set())

    def test_a_set_reaches_children_and_grandchildren(self):
        flag = DrainFlag()
        seen = self.ctx.Queue()
        child = self.ctx.Process(target=_report_when_set, args=(flag, seen))
        relay = self.ctx.Process(target=_relay, args=(flag, seen))
        child.start()
        relay.start()
        time.sleep(1.0)
        flag.set()
        self.assertEqual([seen.get(timeout=30), seen.get(timeout=30)], [True, True])
        child.join()
        relay.join()

    def test_wait_times_out_while_unset(self):
        flag = DrainFlag()
        start = time.monotonic()
        self.assertFalse(flag.wait(0.2))
        self.assertGreaterEqual(time.monotonic() - start, 0.2)


if __name__ == "__main__":
    unittest.main()
