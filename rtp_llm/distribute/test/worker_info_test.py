import os
import unittest
from unittest.mock import patch

from rtp_llm.distribute.worker_info import ParallelInfo


class TestParallelInfo(unittest.TestCase):
    def setUp(self):
        self.env_vars = {
            "TP_SIZE": "1",
            "EP_SIZE": "1",
            "PP_SIZE": "1",
            "DP_SIZE": "1",
            "WORLD_SIZE": "1",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "FFN_SP_SIZE": "1",
        }

    def test_reload(self):
        # Initial setup
        with patch.dict(os.environ, self.env_vars):
            info = ParallelInfo.from_env(worker_info_port_num=1234)
            self.assertEqual(info.tp_size, 1)
            self.assertEqual(info.worker_info_port_num, 1234)

            # Change env vars
            new_env_vars = self.env_vars.copy()
            new_env_vars["TP_SIZE"] = "2"
            new_env_vars["WORLD_SIZE"] = "2"
            new_env_vars["WORLD_RANK"] = "1"
            with patch.dict(os.environ, new_env_vars):
                info.reload(worker_info_port_num=5678)

                expected_info = ParallelInfo.from_env(worker_info_port_num=5678)
                self.assertEqual(info, expected_info)


class TestWorkerInfo(unittest.TestCase):
    def setUp(self):
        # Mock g_parallel_info as it is used by WorkerInfo.from_env
        self.parallel_info_patcher = patch(
            "rtp_llm.distribute.worker_info.g_parallel_info"
        )
        self.mock_parallel_info = self.parallel_info_patcher.start()
        self.mock_parallel_info.worker_info_port_num = 10
        self.mock_parallel_info.local_rank = 0
        self.mock_parallel_info.world_rank = 0

    def tearDown(self):
        self.parallel_info_patcher.stop()

    def test_reload(self):
        from rtp_llm.distribute.worker_info import WorkerInfo

        # Initial WorkerInfo
        info = WorkerInfo.from_env(start_port=1000, remote_server_port=2000)
        self.assertEqual(info.server_port, 1000)  # 1000 + 0 * 10

        # Change parallel info to simulate environment change effect on calculation
        self.mock_parallel_info.local_rank = 1

        # Reload with new ports
        info.reload(start_port=3000, remote_server_port=4000)

        # Verify updates
        expected_info = WorkerInfo.from_env(start_port=3000, remote_server_port=4000)
        self.assertEqual(info, expected_info)


class TestUpdateWorkerInfo(unittest.TestCase):
    @patch("rtp_llm.distribute.worker_info.g_worker_info")
    @patch("rtp_llm.distribute.worker_info.g_parallel_info")
    def test_update_worker_info(self, mock_parallel_info, mock_worker_info):
        from rtp_llm.distribute.worker_info import update_worker_info

        start_port = 1000
        worker_info_port_num = 20
        remote_server_port = 3000

        update_worker_info(start_port, worker_info_port_num, remote_server_port)

        mock_parallel_info.reload.assert_called_once_with(worker_info_port_num)
        mock_worker_info.reload.assert_called_once_with(start_port, remote_server_port)


if __name__ == "__main__":
    unittest.main()
