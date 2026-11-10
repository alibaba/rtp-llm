import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from filelock import FileLock

from rtp_llm.test.utils.device_resource import DeviceResource


class DeviceResourceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        with patch(
            "rtp_llm.test.utils.device_resource.get_gpu_ids", return_value=[0, 1]
        ):
            self.resource = DeviceResource(1)
        self.resource.gpu_status_root_path = self.directory.name
        self.resource.global_lock_file = str(Path(self.directory.name) / "global")
        self.addCleanup(self.resource.gpu_locks.close)
        self.child = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read()"],
            stdin=subprocess.PIPE,
        )
        self.addCleanup(self.stop_child)

    def stop_child(self):
        self.child.stdin.close()
        self.child.wait(timeout=10)

    def test_occupied_gpu_is_skipped_and_its_file_lock_is_released(self):
        with patch.object(
            self.resource,
            "_get_gpu_pids",
            side_effect=lambda device: [self.child.pid] if device == "0" else [],
        ):
            self.assertTrue(self.resource._lock_gpus())
        self.assertEqual(self.resource.gpu_ids, ["1"])
        with FileLock(str(Path(self.directory.name) / "0")).acquire(timeout=0):
            self.assertIsNone(self.child.poll())

    def test_release_timeout_preserves_another_tasks_process(self):
        self.resource.gpu_ids = ["0"]
        now = [0]
        with (
            patch.object(self.resource, "_get_gpu_pids", return_value=[self.child.pid]),
            patch(
                "rtp_llm.test.utils.device_resource.time.time",
                side_effect=lambda: now[0],
            ),
            patch(
                "rtp_llm.test.utils.device_resource.time.sleep",
                side_effect=lambda _: now.__setitem__(0, 31),
            ),
            patch("rtp_llm.test.utils.device_resource.os.kill") as send_signal,
        ):
            self.assertFalse(self.resource._ensure_gpus_released())
        send_signal.assert_not_called()
        self.assertIsNone(self.child.poll())

    def test_release_succeeds_without_external_contexts(self):
        self.resource.gpu_ids = ["0"]
        with patch.object(self.resource, "_get_gpu_pids", return_value=[os.getpid()]):
            self.assertTrue(self.resource._ensure_gpus_released())


if __name__ == "__main__":
    unittest.main()
