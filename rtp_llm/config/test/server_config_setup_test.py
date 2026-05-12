import unittest
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.engine_config import update_dp_peer_addrs
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import (
    set_parallelism_config,
    setup_and_configure_server,
)
from rtp_llm.distribute.distributed_server import WorldInfo
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.ops import ParallelismConfig
from rtp_llm.server.server_args.server_args import setup_args


class GenerateConfigTest(TestCase):

    # EnvArgumentParser in setup_args() reads these env vars (START_PORT, TP_SIZE, etc.)
    # and binds them to py_env_configs; server_port = start_port + rank_id * worker_info_port_num (rank_id=0 here).
    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "4",
            "PP_SIZE": "1",
            "WORLD_SIZE": "4",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "CONCURRENCY_LIMIT": "32",
            "START_PORT": "20000",
            "MODEL_TYPE": "fake_model",
            "USE_ALL_GATHER": "0",
        },
        clear=True,
    )
    def test_simple(self):
        from rtp_llm.config.server_config_setup import (
            fetch_model_files_to_local,
            setup_default_args,
        )

        py_env_configs: PyEnvConfigs = setup_args()
        setup_and_configure_server(py_env_configs)
        pc = py_env_configs.parallelism_config
        self.assertEqual(pc.tp_size, 4)
        self.assertEqual(pc.world_size, 4)
        self.assertEqual(pc.local_world_size, 2)
        self.assertEqual(py_env_configs.server_config.server_port, 20000)

        self.assertEqual(py_env_configs.moe_config.use_deepep_moe, True)
        self.assertEqual(py_env_configs.moe_config.use_deepep_low_latency, False)
        self.assertEqual(py_env_configs.moe_config.use_deepep_internode, True)
        self.assertEqual(py_env_configs.moe_config.ll_num_max_token, 32)

    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "2",
            "PP_SIZE": "1",
            "WORLD_SIZE": "2",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "CONCURRENCY_LIMIT": "32",
            "START_PORT": "20000",
            "MODEL_TYPE": "fake_model",
            "SP_TYPE": "eagle",
            "SP_MODEL_TYPE": "qwen_2-mtp",
            "GEN_NUM_PER_CIRCLE": "4",
            "ROLE_TYPE": "DECODE",
            "USE_ALL_GATHER": "0",
        },
        clear=True,
    )
    def test_sp_deepep_low_latency(self):
        py_env_configs: PyEnvConfigs = setup_args()
        setup_and_configure_server(py_env_configs)

        self.assertEqual(py_env_configs.moe_config.use_deepep_moe, True)
        self.assertEqual(py_env_configs.moe_config.use_deepep_low_latency, True)
        self.assertEqual(py_env_configs.moe_config.use_deepep_internode, False)
        self.assertEqual(py_env_configs.moe_config.ll_num_max_token, 160)

    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "4",
            "PP_SIZE": "1",
            "WORLD_SIZE": "4",
            "WORLD_RANK": "4",
            "LOCAL_WORLD_SIZE": "2",
            "CONCURRENCY_LIMIT": "32",
            "START_PORT": "20000",
            "MODEL_TYPE": "fake_model",
            "USE_ALL_GATHER": "0",
        },
        clear=True,
    )
    def test_world_rank_consistent_with_env_after_setup_args(self):
        """After setup_args(), set_parallelism_config(parallelism_config) keeps world_rank from env and it is not None."""
        py_env_configs: PyEnvConfigs = setup_args()
        set_parallelism_config(py_env_configs.parallelism_config)
        pc = py_env_configs.parallelism_config
        self.assertIsNotNone(pc.world_rank)
        self.assertEqual(pc.world_rank, 4)

    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "4",
            "DP_SIZE": "2",
            "PP_SIZE": "1",
            "WORLD_SIZE": "8",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "CONCURRENCY_LIMIT": "32",
            "START_PORT": "20000",
            "MODEL_TYPE": "fake_model",
            "USE_ALL_GATHER": "0",
        },
        clear=True,
    )
    def test_set_parallelism_config_after_setup_and_configure_server_world_rank_not_none(
        self,
    ):
        """After setup_and_configure_server(), set_parallelism_config(..., world_rank=5) assigns world_rank and derived ranks correctly."""
        py_env_configs: PyEnvConfigs = setup_args()
        setup_and_configure_server(py_env_configs)
        set_parallelism_config(py_env_configs.parallelism_config, world_rank=5)
        pc = py_env_configs.parallelism_config
        self.assertEqual(pc.world_rank, 5)
        self.assertEqual(pc.local_rank, 5 % pc.local_world_size)
        self.assertEqual(pc.tp_rank, 5 % pc.tp_size)
        self.assertEqual(pc.dp_rank, 5 // pc.tp_size)
        self.assertEqual(pc.ep_rank, 5 % pc.ep_size)
        self.assertEqual(pc.ffn_tp_rank, pc.tp_rank % pc.ffn_tp_size)
        self.assertEqual(pc.tp_rank, 1)
        self.assertEqual(pc.dp_rank, 1)
        self.assertEqual(pc.local_rank, 1)
        self.assertEqual(pc.ep_size, 8)
        self.assertEqual(pc.ep_rank, 5)
        self.assertEqual(pc.ffn_tp_rank, 1)


def _make_worker(ip: str, local_rank: int, world_rank: int, server_port: int = 20000, port_num: int = 10) -> WorkerInfo:
    return WorkerInfo(
        ip=ip,
        local_rank=local_rank,
        world_rank=world_rank,
        name=f"worker_{world_rank}",
        server_port=server_port,
        worker_info_port_num=port_num,
    )


def _make_world_info(members: list) -> WorldInfo:
    return WorldInfo(
        members=members,
        master=members[0] if members else None,
        self=members[0] if members else None,
        num_nodes=1,
        initialized=True,
    )


class UpdateDpPeerAddrsTest(TestCase):

    def test_disabled_when_dp_controller_managed_false(self):
        pc = ParallelismConfig()
        pc.dp_controller_managed = False
        pc.dp_size = 2
        pc.tp_size = 2
        members = [_make_worker("10.0.0.1", 0, 0), _make_worker("10.0.0.1", 1, 1),
                    _make_worker("10.0.0.2", 0, 2), _make_worker("10.0.0.2", 1, 3)]
        world_info = _make_world_info(members)
        update_dp_peer_addrs(pc, world_info)
        self.assertEqual(len(pc.dp_peer_addrs), 0)

    def test_disabled_when_dp_size_1(self):
        pc = ParallelismConfig()
        pc.dp_controller_managed = True
        pc.dp_size = 1
        pc.tp_size = 2
        members = [_make_worker("10.0.0.1", 0, 0), _make_worker("10.0.0.1", 1, 1)]
        world_info = _make_world_info(members)
        update_dp_peer_addrs(pc, world_info)
        self.assertEqual(len(pc.dp_peer_addrs), 0)

    def test_noop_when_world_info_none(self):
        pc = ParallelismConfig()
        pc.dp_controller_managed = True
        pc.dp_size = 2
        pc.tp_size = 2
        update_dp_peer_addrs(pc, None)
        self.assertEqual(len(pc.dp_peer_addrs), 0)

    def test_tp2_dp2_four_workers(self):
        """4 workers: tp=2, dp=2. TP-leaders are world_rank 0 (dp0) and 2 (dp1)."""
        pc = ParallelismConfig()
        pc.dp_controller_managed = True
        pc.dp_size = 2
        pc.tp_size = 2
        members = [
            _make_worker("10.0.0.1", 0, 0, server_port=20000, port_num=10),
            _make_worker("10.0.0.1", 1, 1, server_port=20000, port_num=10),
            _make_worker("10.0.0.2", 0, 2, server_port=20000, port_num=10),
            _make_worker("10.0.0.2", 1, 3, server_port=20000, port_num=10),
        ]
        world_info = _make_world_info(members)
        update_dp_peer_addrs(pc, world_info)

        self.assertEqual(len(pc.dp_peer_addrs), 2)
        # world_rank 0: local_rank=0, base=20000+0*10=20000, rpc_server_port=20001
        self.assertEqual(pc.dp_peer_addrs[0], "10.0.0.1:20001")
        # world_rank 2: local_rank=0, base=20000+0*10=20000, rpc_server_port=20001
        self.assertEqual(pc.dp_peer_addrs[1], "10.0.0.2:20001")

    def test_tp1_dp3_three_workers(self):
        """3 workers: tp=1, dp=3. Every worker is a TP-leader."""
        pc = ParallelismConfig()
        pc.dp_controller_managed = True
        pc.dp_size = 3
        pc.tp_size = 1
        members = [
            _make_worker("10.0.0.1", 0, 0, server_port=30000, port_num=5),
            _make_worker("10.0.0.2", 0, 1, server_port=30000, port_num=5),
            _make_worker("10.0.0.3", 0, 2, server_port=30000, port_num=5),
        ]
        world_info = _make_world_info(members)
        update_dp_peer_addrs(pc, world_info)

        self.assertEqual(len(pc.dp_peer_addrs), 3)
        # local_rank=0, base=30000, rpc=30001
        self.assertEqual(pc.dp_peer_addrs[0], "10.0.0.1:30001")
        self.assertEqual(pc.dp_peer_addrs[1], "10.0.0.2:30001")
        self.assertEqual(pc.dp_peer_addrs[2], "10.0.0.3:30001")


if __name__ == "__main__":
    unittest.main()
