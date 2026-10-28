import unittest
from unittest.mock import patch

from rtp_llm.test.utils.device_resource import DeviceResource


class DeviceResourceSafetyTest(unittest.TestCase):
    def test_waiting_for_foreign_process_never_signals_it(self):
        resource = DeviceResource.__new__(DeviceResource)
        resource.gpu_ids = ["0"]
        with (
            patch.object(resource, "_get_gpu_pids", return_value=[987654]),
            patch.object(resource, "_pid_alive", return_value=True),
            patch("rtp_llm.test.utils.device_resource.os.getpid", return_value=123),
            patch("rtp_llm.test.utils.device_resource.os.kill") as kill,
            patch("rtp_llm.test.utils.device_resource.time") as clock,
        ):
            clock.time.side_effect = [0, 0, 2]
            self.assertFalse(resource._ensure_gpus_released(timeout=1))
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
