import unittest
from unittest.mock import Mock, patch

from rtp_llm.test.utils.device_resource import DeviceResource


class DeviceResourceTest(unittest.TestCase):
    def test_gpu_availability_never_signals_existing_processes(self):
        with patch("rtp_llm.test.utils.device_resource.get_gpu_ids", return_value=[3]):
            resource = DeviceResource(1)
        resource.gpu_ids = ["3"]
        for pids, alive, expected in (
            ([], True, True),
            ([123456789], True, False),
            ([123456789], False, False),
            (None, True, False),
        ):
            with self.subTest(pids=pids, alive=alive):
                resource._get_gpu_pids = Mock(return_value=pids)
                resource._pid_alive = Mock(return_value=alive)
                with (
                    patch("rtp_llm.test.utils.device_resource.os.kill") as send_signal,
                    patch("rtp_llm.test.utils.device_resource.time") as clock,
                    patch(
                        "rtp_llm.test.utils.device_resource._nvidia_smi",
                        return_value="/fake/nvidia-smi",
                    ),
                ):
                    clock.time.side_effect = [0, 0, 31]
                    self.assertEqual(resource._ensure_gpus_released(30), expected)
                    send_signal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
