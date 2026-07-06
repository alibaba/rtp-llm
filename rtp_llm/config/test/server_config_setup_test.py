import unittest
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.engine_config import EngineConfig, setup_pd_sep_config
from rtp_llm.config.py_config_modules import PyEnvConfigs, ServerConfig
from rtp_llm.config.server_config_setup import (
    set_parallelism_config,
    setup_and_configure_server,
)
from rtp_llm.ops import RoleType
from rtp_llm.server.server_args.server_args import setup_args


class ServerConfigPortLayoutTest(TestCase):
    def test_dash_sc_rejects_legacy_stride_eight(self):
        config = ServerConfig()
        config.worker_info_port_num = 8

        with self.assertRaisesRegex(ValueError, "must be at least 9"):
            config.validate_port_layout(dash_sc_enabled=True)

    def test_dash_sc_accepts_stride_nine_without_cross_rank_overlap(self):
        config = ServerConfig()
        config.worker_info_port_num = 9
        config.validate_port_layout(dash_sc_enabled=True)

        config.rank_id = 0
        rank_zero_dash_sc_port = config.dash_sc_grpc_server_port
        config.rank_id = 1
        rank_one_server_port = config.server_port

        self.assertEqual(rank_zero_dash_sc_port, config.start_port + 8)
        self.assertEqual(rank_one_server_port, config.start_port + 9)
        self.assertNotEqual(rank_zero_dash_sc_port, rank_one_server_port)

    def test_vit_without_dash_sc_allows_legacy_stride(self):
        config = ServerConfig()
        config.worker_info_port_num = 8

        config.validate_port_layout(dash_sc_enabled=False)


class GenerateConfigTest(TestCase):

    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "1",
            "PP_SIZE": "1",
            "WORLD_SIZE": "1",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "START_PORT": "20000",
            "MODEL_TYPE": "fake_model",
            "ENABLE_MEMORY_CACHE_DISK": "1",
            "MEMORY_CACHE_DISK_PATHS": "/tmp/cache-a,/tmp/cache-b",
            "MEMORY_CACHE_DISK_SIZE_MB": "4096",
            "MEMORY_CACHE_DISK_BUFFERED_IO": "0",
            "MEMORY_CACHE_DISK_SYNC_TIMEOUT_MS": "12345",
            "ENABLE_GPU_PREFIX_TREE": "1",
            "ENABLE_PREFIX_TREE_MEMORY_CACHE": "1",
            "ENABLE_LEGACY_MEMORY_CONNECTOR_FALLBACK": "0",
            "PREFIX_TREE_MEMORY_STATE_SWA_POOL_RATIO": "25",
            "ENABLE_INDEPENDENT_GROUP_EVICTION": "1",
        },
        clear=True,
    )
    def test_kv_cache_strategy_args_propagate_from_env(self):
        py_env_configs: PyEnvConfigs = setup_args()
        config = py_env_configs.kv_cache_config

        self.assertTrue(config.enable_memory_cache_disk)
        self.assertEqual(config.memory_cache_disk_paths, "/tmp/cache-a,/tmp/cache-b")
        self.assertEqual(config.memory_cache_disk_size_mb, 4096)
        self.assertFalse(config.memory_cache_disk_buffered_io)
        self.assertEqual(config.memory_cache_disk_sync_timeout_ms, 12345)
        self.assertTrue(config.enable_gpu_prefix_tree)
        self.assertTrue(config.enable_prefix_tree_memory_cache)
        self.assertFalse(config.enable_legacy_memory_connector_fallback)
        self.assertEqual(config.prefix_tree_memory_state_swa_pool_ratio, 25)
        self.assertTrue(config.enable_independent_group_eviction)

    def test_kv_cache_strategy_defaults_are_rollback_safe(self):
        config = PyEnvConfigs().kv_cache_config

        self.assertFalse(config.enable_gpu_prefix_tree)
        self.assertFalse(config.enable_prefix_tree_memory_cache)
        self.assertTrue(config.enable_legacy_memory_connector_fallback)

    def test_engine_config_propagates_role_to_parallelism_config(self):
        py_env_configs = PyEnvConfigs()
        py_env_configs.role_config.role_type = RoleType.PREFILL

        engine_config = EngineConfig.create(py_env_configs)

        self.assertEqual(engine_config.pd_sep_config.role_type, RoleType.PREFILL)
        self.assertEqual(engine_config.parallelism_config.role_type, RoleType.PREFILL)

    def test_set_parallelism_config_propagates_prefill_cp_cache_fields(self):
        py_env_configs = PyEnvConfigs()
        py_env_configs.prefill_cp_config.kv_cache_sharded = True
        py_env_configs.prefill_cp_config.prefill_cp_size = 4

        set_parallelism_config(
            py_env_configs.parallelism_config,
            py_prefill_cp_config=py_env_configs.prefill_cp_config,
        )

        self.assertTrue(
            py_env_configs.parallelism_config.prefill_cp_config.kv_cache_sharded
        )
        self.assertEqual(
            py_env_configs.parallelism_config.prefill_cp_config.prefill_cp_size, 4
        )

    @patch.dict(
        "os.environ",
        {
            "START_PORT": "20000",
            "REMOTE_SERVER_PORT": "30000",
            "WORKER_INFO_PORT_NUM": "13",
        },
        clear=True,
    )
    def test_custom_stride_reaches_local_remote_and_pd_port_consumers(self):
        py_env_configs = setup_args()
        server_config = py_env_configs.server_config
        distribute_config = py_env_configs.distribute_config
        server_config.rank_id = 2
        distribute_config.rank_id = 2
        py_env_configs.pd_separation_config.role_type = RoleType.DECODE

        setup_pd_sep_config(
            py_env_configs.pd_separation_config,
            py_env_configs.cache_store_config,
            server_config,
            distribute_config,
        )

        self.assertEqual(server_config.server_port, 20026)
        self.assertEqual(distribute_config.remote_rpc_server_port, 30027)
        self.assertEqual(py_env_configs.pd_separation_config.worker_port_offset, 13)
        self.assertEqual(
            py_env_configs.pd_separation_config.remote_rpc_server_port, 30027
        )
        self.assertEqual(
            py_env_configs.pd_separation_config.cache_store_connect_port, 30028
        )
        self.assertEqual(
            py_env_configs.pd_separation_config.cache_store_rdma_connect_port,
            30030,
        )

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
            "PREFILL_CP_KV_CACHE_SHARDED": "1",
            "PREFILL_CP_SIZE": "4",
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
        self.assertTrue(pc.prefill_cp_config.kv_cache_sharded)
        self.assertEqual(pc.prefill_cp_config.prefill_cp_size, 4)
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


if __name__ == "__main__":
    unittest.main()
