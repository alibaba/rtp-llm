import asyncio
import os
import sys
import unittest

from navi_sched.adapters.sglang.SGLangWorkerMonitor import SGLangWorkerMonitor


class SGLangWorkerMonitorTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.exited = asyncio.get_running_loop().create_future()
        self.monitor = SGLangWorkerMonitor(self.exited.set_result, 2)

    async def asyncTearDown(self):
        self.monitor.close()

    async def test_real_child_exit_removes_registration(self):
        child = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import signal; signal.pause()"
        )
        try:
            self.monitor.register("old-worker", child.pid)
            fd = self.monitor.entries["old-worker"][1]
            child.terminate()
            self.assertEqual(await asyncio.wait_for(self.exited, 2), "old-worker")
            self.assertFalse(self.monitor.entries)
            with self.assertRaises(OSError):
                os.fstat(fd)
        finally:
            if child.returncode is None:
                child.kill()
            await child.wait()

    async def test_stale_exit_cannot_remove_replacement(self):
        self.monitor.register("worker", os.getpid())
        generation = self.monitor.entries["worker"][3]
        self.monitor.remove("worker")
        self.monitor.register("worker", os.getpid())
        self.monitor._exited("worker", generation)
        self.assertIn("worker", self.monitor.entries)
        self.assertFalse(self.exited.done())

    async def test_duplicate_registration_and_close_release_fds(self):
        self.monitor.register("worker", os.getpid())
        entry = self.monitor.entries["worker"]
        self.monitor.register("worker", os.getpid())
        self.assertIs(self.monitor.entries["worker"], entry)
        self.monitor.close()
        self.monitor.close()
        with self.assertRaises(OSError):
            os.fstat(entry[1])

    async def test_registration_bound_and_invalid_pid(self):
        with self.assertRaises(ValueError):
            self.monitor.register("bad", True)
        self.monitor.register("a", os.getpid())
        self.monitor.register("b", os.getpid())
        with self.assertRaisesRegex(RuntimeError, "too many"):
            self.monitor.register("c", os.getpid())


if __name__ == "__main__":
    unittest.main()
