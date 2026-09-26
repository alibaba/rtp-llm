import os
import subprocess
import unittest
from unittest.mock import patch

from rtp_llm.test.utils.device_resource import DeviceResource


class DeviceResourceQueryTest(unittest.TestCase):
    def test_slow_gpu_query_succeeds_with_larger_budget(self):
        def query(command, **kwargs):
            if kwargs["timeout"] < 20:
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return subprocess.CompletedProcess(command, 0, stdout="1001\n", stderr="")

        with patch(
            "rtp_llm.test.utils.device_resource.get_gpu_ids", return_value=[4]
        ), patch("rtp_llm.test.utils.device_resource.subprocess.run", side_effect=query):
            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(DeviceResource(1)._get_gpu_pids("4"))
            with patch.dict(os.environ, {"RTP_GPU_QUERY_TIMEOUT": "30"}):
                self.assertEqual(DeviceResource(1)._get_gpu_pids("4"), [1001])

    def test_query_timeout_keeps_gpu_unusable(self):
        with patch.dict(os.environ, {"RTP_GPU_QUERY_TIMEOUT": "30"}), patch(
            "rtp_llm.test.utils.device_resource.get_gpu_ids", return_value=[4]
        ), patch(
            "rtp_llm.test.utils.device_resource.subprocess.run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 30),
        ), patch("rtp_llm.test.utils.device_resource._nvidia_smi", return_value="nvidia-smi"):
            resource = DeviceResource(1)
            self.assertTrue(resource._has_zombie_gpu_contexts("4"))


if __name__ == "__main__":
    unittest.main()
