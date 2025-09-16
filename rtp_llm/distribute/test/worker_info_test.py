import unittest
from unittest.mock import patch

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM
from rtp_llm.distribute.worker_info import (
    ParallelInfo,
    WorkerInfo,
    g_parallel_info,
    g_worker_info,
    update_worker_info,
)


class ParallelInfoTest(unittest.TestCase):
    @patch("rtp_llm.distribute.worker_info.torch.cuda.is_available", return_value=False)
    def test_from_params_valid(self, _mock_cuda_available):
        params = {
            "TP_SIZE": "2",
            "EP_SIZE": "2",
            "PP_SIZE": "1",
            "DP_SIZE": "2",
            "WORLD_SIZE": "4",
            "WORLD_RANK": "1",
            "LOCAL_WORLD_SIZE": "2",
        }
        info = ParallelInfo.from_params(params, MIN_WORKER_INFO_PORT_NUM)

        self.assertEqual(info.world_size, 4)
        self.assertEqual(info.world_rank, 1)
        self.assertEqual(info.tp_size, 2)
        self.assertEqual(info.dp_size, 2)
        self.assertEqual(info.pp_size, 1)
        self.assertEqual(info.ep_size, 2)
        self.assertEqual(info.worker_info_port_num, MIN_WORKER_INFO_PORT_NUM)
        self.assertEqual(info.tp_rank, 1)
        self.assertEqual(info.dp_rank, 0)
        self.assertEqual(info.local_rank, 1)

    @patch("rtp_llm.distribute.worker_info.torch.cuda.is_available", return_value=False)
    def test_from_params_invalid_world_size_raises(self, _mock_cuda_available):
        params = {
            "TP_SIZE": "2",
            "PP_SIZE": "1",
            "DP_SIZE": "1",
            "WORLD_SIZE": "3",  # mismatch: 2*1*1 != 3
            "WORLD_RANK": "0",
        }
        with self.assertRaises(Exception):
            ParallelInfo.from_params(params, MIN_WORKER_INFO_PORT_NUM)


class WorkerInfoOffsetTest(unittest.TestCase):
    def test_port_offsets(self):
        port_step = max(MIN_WORKER_INFO_PORT_NUM, 7)
        start_port = 20000
        local_rank = 1

        base = WorkerInfo.server_port_offset(local_rank, start_port, port_step)
        self.assertEqual(base, start_port + local_rank * port_step)
        self.assertEqual(
            WorkerInfo.rpc_server_port_offset(local_rank, start_port, port_step),
            base + 1,
        )
        self.assertEqual(
            WorkerInfo.cache_store_listen_port_offset(
                local_rank, start_port, port_step
            ),
            base + 2,
        )
        self.assertEqual(
            WorkerInfo.gang_hb_port_offset(local_rank, start_port, port_step), base + 3
        )
        self.assertEqual(
            WorkerInfo.cache_store_rdma_listen_port_offset(
                local_rank, start_port, port_step
            ),
            base + 4,
        )
        self.assertEqual(
            WorkerInfo.http_port_offset(local_rank, start_port, port_step), base + 5
        )
        self.assertEqual(
            WorkerInfo.backend_server_port_offset(local_rank, start_port, port_step),
            base + 6,
        )
        self.assertEqual(
            WorkerInfo.embedding_rpc_server_port_offset(
                local_rank, start_port, port_step
            ),
            base + 7,
        )
        self.assertEqual(
            WorkerInfo.vit_http_server_port_offset(local_rank, start_port, port_step),
            base + 8,
        )
        self.assertEqual(
            WorkerInfo.vit_grpc_server_port_offset(local_rank, start_port, port_step),
            base + 9,
        )


class UpdateWorkerInfoTest(unittest.TestCase):
    @patch("rtp_llm.distribute.worker_info.torch.cuda.is_available", return_value=False)
    def test_update_worker_info_reload(self, _mock_cuda_available):
        # prepare globals to a known state
        g_parallel_info.tp_size = 1
        g_parallel_info.pp_size = 1
        g_parallel_info.dp_size = 1
        g_parallel_info.world_size = 1
        g_parallel_info.world_rank = 0
        g_parallel_info.local_world_size = 1

        start_port = 30000
        worker_info_port_num = MIN_WORKER_INFO_PORT_NUM
        remote_server_port = 40000

        update_worker_info(start_port, worker_info_port_num, remote_server_port)

        # g_parallel_info should be reloaded with new worker_info_port_num
        self.assertEqual(g_parallel_info.worker_info_port_num, worker_info_port_num)
        # g_worker_info should be reloaded with derived ports
        base = WorkerInfo.server_port_offset(
            g_parallel_info.local_rank, start_port, worker_info_port_num
        )
        self.assertEqual(g_worker_info.server_port, base)
        self.assertEqual(g_worker_info.rpc_server_port, base + 1)
        self.assertEqual(g_worker_info.cache_store_listen_port, base + 2)
        self.assertEqual(g_worker_info.gang_hb_port, base + 3)
        self.assertEqual(g_worker_info.cache_store_rdma_listen_port, base + 4)
        self.assertEqual(g_worker_info.http_port, base + 5)
        self.assertEqual(g_worker_info.backend_server_port, base + 6)
        self.assertEqual(g_worker_info.embedding_rpc_server_port, base + 7)
        self.assertEqual(g_worker_info.remote_rpc_server_port, remote_server_port + 1)
        self.assertEqual(g_worker_info.cache_store_connect_port, remote_server_port + 2)
        self.assertEqual(
            g_worker_info.cache_store_rdma_connect_port, remote_server_port + 4
        )
        self.assertEqual(g_worker_info.vit_http_server_port, base + 8)
        self.assertEqual(g_worker_info.vit_grpc_server_port, base + 9)


if __name__ == "__main__":
    unittest.main()
