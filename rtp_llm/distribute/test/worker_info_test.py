import unittest
from unittest.mock import Mock

from rtp_llm.config.py_config_modules import MASTER_INFO_PORT_NUM
from rtp_llm.distribute.worker_info import WorkerInfo


class MockParallelismConfig:
    """Mock ParallelismConfig for testing."""

    def __init__(
        self,
        local_rank=0,
        world_rank=0,
        local_world_size=1,
        dp_rank=0,
        ffn_sp_size=1,
        tp_size=1,
    ):
        self.local_rank = local_rank
        self.world_rank = world_rank
        self.local_world_size = local_world_size
        self.dp_rank = dp_rank
        self.ffn_sp_size = ffn_sp_size
        self.tp_size = tp_size


class WorkerInfoTest(unittest.TestCase):
    """Unit tests for WorkerInfo class."""

    def test_init_basic(self):
        """Test basic initialization."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
            local_world_size=1,
        )
        self.assertEqual(worker_info.ip, "127.0.0.1")
        self.assertEqual(worker_info.local_rank, 0)
        self.assertEqual(worker_info.world_rank, 0)
        self.assertEqual(worker_info.local_world_size, 1)
        self.assertEqual(worker_info.master_ip, "")

    def test_init_with_ports(self):
        """Test initialization with port configuration."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=1,
            world_rank=2,
            local_world_size=4,
            start_port=20000,
            remote_server_port=30000,
            worker_info_port_num=8,
            master_ip="10.0.0.1",
        )
        self.assertEqual(worker_info.ip, "127.0.0.1")
        self.assertEqual(worker_info.local_rank, 1)
        self.assertEqual(worker_info.world_rank, 2)
        self.assertEqual(worker_info.local_world_size, 4)
        self.assertEqual(worker_info.master_ip, "10.0.0.1")

    def test_configure_ports(self):
        """Test configure_ports method."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
        )
        worker_info.configure_ports(
            local_rank=2,
            world_rank=5,
            start_port=20000,
            remote_server_port=30000,
            worker_info_port_num=8,
            local_world_size=4,
        )
        self.assertEqual(worker_info.local_rank, 2)
        self.assertEqual(worker_info.world_rank, 5)
        self.assertEqual(worker_info.local_world_size, 4)

    def test_port_calculation_with_worker_info_port_num(self):
        """Test port calculation when worker_info_port_num is set."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=2,
            world_rank=5,
            start_port=20000,
            worker_info_port_num=8,
        )
        # base_port = 20000 + 2 * 8 = 20016
        self.assertEqual(worker_info.server_port, 20016)
        self.assertEqual(worker_info.rpc_server_port, 20017)  # base_port + 1
        self.assertEqual(worker_info.backend_server_port, 20022)  # base_port + 6
        self.assertEqual(worker_info.http_port, 20021)  # base_port + 5

    def test_port_calculation_without_worker_info_port_num(self):
        """Test port calculation when worker_info_port_num is 0 or None."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=2,
            world_rank=5,
            start_port=20016,  # Already calculated base_port
            worker_info_port_num=0,
        )
        # When worker_info_port_num is 0, start_port is treated as base_port
        self.assertEqual(worker_info.server_port, 20016)
        self.assertEqual(worker_info.rpc_server_port, 20017)
        self.assertEqual(worker_info.backend_server_port, 20022)

    def test_port_calculation_error_when_not_configured(self):
        """Test that accessing ports raises error when not configured."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
            # start_port is None
        )
        with self.assertRaises(ValueError) as context:
            _ = worker_info.server_port
        self.assertIn("Port configuration not set", str(context.exception))

    def test_remote_ports(self):
        """Test remote port calculation."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=1,
            world_rank=3,
            remote_server_port=30000,
            worker_info_port_num=8,
        )
        # remote_base_port = 30000 + 1 * 8 = 30008
        self.assertEqual(
            worker_info.remote_rpc_server_port, 30009
        )  # remote_base_port + 1
        self.assertEqual(
            worker_info.cache_store_connect_port, 30010
        )  # remote_base_port + 2

    def test_remote_ports_without_remote_server_port(self):
        """Test remote ports when remote_server_port is None."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
            remote_server_port=None,
        )
        # When remote_server_port is None, remote_base_port returns 0
        self.assertEqual(worker_info.remote_rpc_server_port, 1)  # 0 + 1
        self.assertEqual(worker_info.cache_store_connect_port, 2)  # 0 + 2

    def test_adjust_ports_by_rank_id_with_start_port(self):
        """Test adjust_ports_by_rank_id when _start_port is set."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
            start_port=20000,
            worker_info_port_num=8,
        )
        original_server_port = worker_info.server_port
        # Adjust by rank_id=2
        worker_info.adjust_ports_by_rank_id(rank_id=2, worker_info_port_num=8)
        # New start_port = 20000 + 2 * 8 = 20016
        # New server_port = 20016 + 0 * 8 = 20016
        self.assertEqual(worker_info.server_port, 20016)
        self.assertEqual(worker_info.server_port, original_server_port + 16)

    def test_adjust_ports_by_rank_id_without_start_port(self):
        """Test adjust_ports_by_rank_id when _start_port is None."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=1,
            world_rank=3,
            start_port=20000,
            worker_info_port_num=8,
        )
        # Current server_port = 20000 + 1 * 8 = 20008
        original_server_port = worker_info.server_port
        self.assertEqual(original_server_port, 20008)

        # Clear start_port to simulate unconfigured state
        worker_info._start_port = None
        # Now adjust by rank_id=2
        worker_info.adjust_ports_by_rank_id(rank_id=2, worker_info_port_num=8)
        # Should reconfigure with new start_port
        # base_start_port = 20008 - 1 * 8 = 20000
        # new_start_port = 20000 + 2 * 8 = 20016
        # new server_port = 20016 + 1 * 8 = 20024
        self.assertEqual(worker_info.server_port, 20024)

    def test_update_master_info(self):
        """Test update_master_info method."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
        )
        parallelism_config = MockParallelismConfig(dp_rank=1, ffn_sp_size=2, tp_size=4)
        worker_info.update_master_info(
            ip="10.0.0.1", base_port=50000, parallelism_config=parallelism_config
        )
        self.assertEqual(worker_info.master_ip, "10.0.0.1")
        self.assertEqual(worker_info.ip, "127.0.0.1")  # Worker IP should not change
        self.assertEqual(worker_info._master_base_port, 50000)
        self.assertEqual(worker_info._master_dp_rank, 1)
        self.assertEqual(worker_info._master_ffn_sp_size, 2)
        self.assertEqual(worker_info._master_tp_size, 4)

    def test_master_nccl_ports(self):
        """Test master NCCL port calculation."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
        )
        parallelism_config = MockParallelismConfig(dp_rank=0, ffn_sp_size=1, tp_size=1)
        base_port = 50000
        worker_info.update_master_info(
            ip="10.0.0.1", base_port=base_port, parallelism_config=parallelism_config
        )

        # Test direct master ports
        self.assertEqual(worker_info.dp_tp_nccl_port, base_port - 10)
        self.assertEqual(worker_info.th_nccl_port, base_port - 11)

        # Test calculated master ports
        # adjusted_base_port = 50000 - 0 * 11 = 50000
        self.assertEqual(worker_info.tp_nccl_port, 50000 - 2)
        self.assertEqual(worker_info.nccl_op_port, 50000 - 3)
        self.assertEqual(worker_info.sp_gpt_nccl_port, 50000 - 4)

        # ffn_base_port = 50000 (since ffn_sp_size == tp_size)
        self.assertEqual(worker_info.ffn_tp_nccl_port, 50000 - 5)

    def test_master_nccl_ports_with_dp_rank(self):
        """Test master NCCL ports with dp_rank offset."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
        )
        parallelism_config = MockParallelismConfig(dp_rank=2, ffn_sp_size=1, tp_size=1)
        base_port = 50000
        worker_info.update_master_info(
            ip="10.0.0.1", base_port=base_port, parallelism_config=parallelism_config
        )

        # adjusted_base_port = 50000 - 2 * 11 = 49978
        self.assertEqual(worker_info.tp_nccl_port, 49978 - 2)
        self.assertEqual(worker_info.nccl_op_port, 49978 - 3)
        self.assertEqual(worker_info.sp_gpt_nccl_port, 49978 - 4)

    def test_master_nccl_ports_with_ffn_sp_size(self):
        """Test master NCCL ports with ffn_sp_size adjustment."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
        )
        parallelism_config = MockParallelismConfig(dp_rank=0, ffn_sp_size=2, tp_size=1)
        base_port = 50000
        worker_info.update_master_info(
            ip="10.0.0.1", base_port=base_port, parallelism_config=parallelism_config
        )

        # ffn_base_port = 50000 - 2 = 49998 (since ffn_sp_size != tp_size)
        self.assertEqual(worker_info.ffn_tp_nccl_port, 49998 - 5)

    def test_master_nccl_ports_without_master_info(self):
        """Test master NCCL ports return 0 when master info is not set."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
        )
        # _master_base_port is None
        self.assertEqual(worker_info.dp_tp_nccl_port, 0)
        self.assertEqual(worker_info.th_nccl_port, 0)
        self.assertEqual(worker_info.tp_nccl_port, 0)
        self.assertEqual(worker_info.nccl_op_port, 0)
        self.assertEqual(worker_info.sp_gpt_nccl_port, 0)
        self.assertEqual(worker_info.ffn_tp_nccl_port, 0)

    def test_from_parallelism_config(self):
        """Test from_parallelism_config static method."""
        parallelism_config = MockParallelismConfig(
            local_rank=1, world_rank=3, local_world_size=4
        )
        worker_info = WorkerInfo.from_parallelism_config(
            parallelism_config=parallelism_config,
            start_port=20000,
            remote_server_port=30000,
            worker_info_port_num=8,
        )
        self.assertEqual(worker_info.local_rank, 1)
        self.assertEqual(worker_info.world_rank, 3)
        self.assertEqual(worker_info.local_world_size, 4)
        # server_port = 20000 + 1 * 8 = 20008
        self.assertEqual(worker_info.server_port, 20008)

    def test_static_port_offset_methods(self):
        """Test static port offset calculation methods."""
        base_port = 20000
        local_rank = 2
        worker_info_port_num = 8

        # Test server_port_offset
        result = WorkerInfo.server_port_offset(
            local_rank, base_port, worker_info_port_num
        )
        self.assertEqual(result, 20016)  # 20000 + 2 * 8

        # Test rpc_server_port_offset
        result = WorkerInfo.rpc_server_port_offset(
            local_rank, base_port, worker_info_port_num
        )
        self.assertEqual(result, 20017)  # 20016 + 1

        # Test backend_server_port_offset
        result = WorkerInfo.backend_server_port_offset(
            local_rank, base_port, worker_info_port_num
        )
        self.assertEqual(result, 20022)  # 20016 + 6

        # Test http_port_offset
        result = WorkerInfo.http_port_offset(
            local_rank, base_port, worker_info_port_num
        )
        self.assertEqual(result, 20021)  # 20016 + 5

    def test_equals_method(self):
        """Test equals method (simplified comparison)."""
        worker_info1 = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
            start_port=20000,
            worker_info_port_num=8,
        )
        worker_info2 = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
            start_port=20000,
            worker_info_port_num=8,
        )
        worker_info3 = WorkerInfo(
            ip="127.0.0.2",
            local_rank=0,
            world_rank=0,
            start_port=20000,
            worker_info_port_num=8,
        )
        self.assertTrue(worker_info1.equals(worker_info2))
        self.assertFalse(worker_info1.equals(worker_info3))

    def test_eq_method(self):
        """Test __eq__ method (full comparison)."""
        worker_info1 = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
            start_port=20000,
            remote_server_port=30000,
            worker_info_port_num=8,
            local_world_size=1,
        )
        worker_info2 = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
            start_port=20000,
            remote_server_port=30000,
            worker_info_port_num=8,
            local_world_size=1,
        )
        worker_info3 = WorkerInfo(
            ip="127.0.0.1",
            local_rank=1,
            world_rank=0,
            start_port=20000,
            remote_server_port=30000,
            worker_info_port_num=8,
            local_world_size=1,
        )
        self.assertEqual(worker_info1, worker_info2)
        self.assertNotEqual(worker_info1, worker_info3)
        self.assertNotEqual(worker_info1, "not a WorkerInfo")

    def test_str_method(self):
        """Test __str__ method."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
            start_port=20000,
            worker_info_port_num=8,
        )
        str_repr = str(worker_info)
        self.assertIn("127.0.0.1", str_repr)
        self.assertIn("server_port=20000", str_repr)
        self.assertIn("local_rank=0", str_repr)

    def test_all_worker_ports(self):
        """Test all worker port properties."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
            start_port=20000,
            worker_info_port_num=8,
        )
        base_port = 20000

        self.assertEqual(worker_info.server_port, base_port + 0)
        self.assertEqual(worker_info.rpc_server_port, base_port + 1)
        self.assertEqual(worker_info.cache_store_listen_port, base_port + 2)
        self.assertEqual(worker_info.gang_hb_port, base_port + 3)
        self.assertEqual(worker_info.cache_store_rdma_listen_port, base_port + 4)
        self.assertEqual(worker_info.http_port, base_port + 5)
        self.assertEqual(worker_info.backend_server_port, base_port + 6)
        self.assertEqual(worker_info.embedding_rpc_server_port, base_port + 7)

    def test_all_remote_ports(self):
        """Test all remote port properties."""
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=0,
            world_rank=0,
            remote_server_port=30000,
            worker_info_port_num=8,
        )
        remote_base_port = 30000

        self.assertEqual(worker_info.remote_rpc_server_port, remote_base_port + 1)
        self.assertEqual(worker_info.cache_store_connect_port, remote_base_port + 2)
        self.assertEqual(
            worker_info.cache_store_rdma_connect_port, remote_base_port + 4
        )

    def test_port_calculation_edge_cases(self):
        """Test edge cases in port calculation."""
        # Test with local_rank > 0 and worker_info_port_num > 0
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=3,
            world_rank=5,
            start_port=20000,
            worker_info_port_num=8,
        )
        # base_port = 20000 + 3 * 8 = 20024
        self.assertEqual(worker_info.server_port, 20024)

        # Test with worker_info_port_num = 0
        worker_info2 = WorkerInfo(
            ip="127.0.0.1",
            local_rank=3,
            world_rank=5,
            start_port=20024,  # Already calculated
            worker_info_port_num=0,
        )
        # base_port = 20024 (no offset applied)
        self.assertEqual(worker_info2.server_port, 20024)

    def test_adjust_ports_by_rank_id_edge_cases(self):
        """Test adjust_ports_by_rank_id edge cases."""
        # Test with rank_id = 0 (no change)
        worker_info = WorkerInfo(
            ip="127.0.0.1",
            local_rank=1,
            world_rank=3,
            start_port=20000,
            worker_info_port_num=8,
        )
        original_port = worker_info.server_port  # 20008
        worker_info.adjust_ports_by_rank_id(rank_id=0, worker_info_port_num=8)
        self.assertEqual(worker_info.server_port, original_port)

        # Test with negative rank_id
        worker_info.adjust_ports_by_rank_id(rank_id=-1, worker_info_port_num=8)
        # start_port = 20000 - 1 * 8 = 19992
        # server_port = 19992 + 1 * 8 = 20000
        self.assertEqual(worker_info.server_port, 20000)


if __name__ == "__main__":
    unittest.main()
