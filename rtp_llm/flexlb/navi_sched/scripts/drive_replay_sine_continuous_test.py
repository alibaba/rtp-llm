import math
import signal
import sys
import unittest
from unittest.mock import patch

import drive_replay_sine_continuous as driver


class DriveReplaySineContinuousTest(unittest.TestCase):
    def test_target_qps(self):
        self.assertAlmostEqual(20.0, driver.target_qps(0, 20, 400, 180))
        self.assertAlmostEqual(400.0, driver.target_qps(90, 20, 400, 180))
        self.assertAlmostEqual(20.0, driver.target_qps(180, 20, 400, 180))
        self.assertTrue(math.isfinite(driver.target_qps(1_000_000, 20, 400, 180)))

    def test_exit_does_not_stop_replay(self):
        commands = []
        handlers = {}

        def record_signal(signum, handler):
            handlers[signum] = handler

        def fake_run(_tool, *arguments):
            commands.append(arguments)
            if arguments[0] == "query":
                return {"accepted": True, "action": "start", "mode": "fixed_qps"}
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            return {"accepted": True}

        argv = [
            "driver", "--min-qps", "20", "--max-qps", "400",
            "--period-seconds", "180",
        ]
        with patch.object(sys, "argv", argv), patch.object(driver, "run_command", fake_run), \
             patch.object(driver.signal, "signal", record_signal), \
             patch.object(driver.time, "monotonic", side_effect=[0.0, 0.0, 1.0]), \
             patch.object(driver.time, "sleep"):
            self.assertEqual(0, driver.main())

        self.assertEqual(("query",), commands[0])
        self.assertEqual("update", commands[1][0])
        self.assertEqual("20.115743", commands[1][2])
        self.assertEqual(2, len(commands))


if __name__ == "__main__":
    unittest.main()
