import contextlib
import io
import os
import sys
import unittest
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.engine_config import EngineConfig, setup_pd_sep_config
from rtp_llm.config.py_config_modules import PyEnvConfigs, ServerConfig
from rtp_llm.config.server_config_setup import (
    set_parallelism_config,
    setup_and_configure_server,
)
from rtp_llm.ops import CPRotateMethod, NcclCommConfig, RoleType
from rtp_llm.server.server_args.server_args import setup_args

# clear=True must preserve gpu_lock isolation across Torch lazy initialization.
_PINNED_DEVICES = {
    name: os.environ[name]
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
    if name in os.environ
}


def _jit_env(**values):
    return {**_PINNED_DEVICES, "MODEL_TYPE": "fake_model", **values}


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
            **_PINNED_DEVICES,
            "TP_SIZE": "1",
            "PP_SIZE": "1",
            "WORLD_SIZE": "1",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "START_PORT": "20000",
            "MODEL_TYPE": "fake_model",
            "ENABLE_DISK_CACHE": "1",
            "DISK_CACHE_PATHS": "/tmp/cache-a,/tmp/cache-b",
            "DISK_CACHE_SIZE_MB": "4096",
            "DISK_CACHE_BUFFERED_IO": "0",
            "DISK_CACHE_SYNC_TIMEOUT_MS": "12345",
            "DISK_CACHE_STAGING_BLOCK_COUNT": "8",
            "ENABLE_HOST_CACHE": "1",
            "ENABLE_HOST_CACHE_PINNED": "0",
            "HOST_CACHE_SIZE_MB": "2048",
            "HOST_CACHE_SYNC_TIMEOUT_MS": "6789",
            "BLOCK_TREE_FULL_PREFIX_SCAN_INTERVAL_MS": "30000",
        },
        clear=True,
    )
    def test_kv_cache_strategy_args_propagate_from_env(self):
        py_env_configs: PyEnvConfigs = setup_args()
        config = py_env_configs.kv_cache_config

        self.assertTrue(config.enable_disk_cache)
        self.assertEqual(config.disk_cache_paths, "/tmp/cache-a,/tmp/cache-b")
        self.assertEqual(config.disk_cache_size_mb, 4096)
        self.assertFalse(config.disk_cache_buffered_io)
        self.assertEqual(config.disk_cache_sync_timeout_ms, 12345)
        self.assertEqual(config.disk_cache_staging_block_count, 8)
        self.assertTrue(config.enable_host_cache)
        self.assertFalse(config.enable_host_cache_pinned)
        self.assertEqual(config.host_cache_size_mb, 2048)
        self.assertEqual(config.host_cache_sync_timeout_ms, 6789)
        self.assertEqual(config.block_tree_full_prefix_scan_interval_ms, 30000)

    def test_kv_cache_strategy_defaults_are_rollback_safe(self):
        config = PyEnvConfigs().kv_cache_config

        self.assertFalse(config.enable_host_cache)
        self.assertTrue(config.enable_host_cache_pinned)
        self.assertEqual(config.disk_cache_staging_block_count, 4)
        self.assertEqual(config.device_eviction_policy, "lru")
        self.assertEqual(config.host_eviction_policy, "lru")
        self.assertEqual(config.disk_eviction_policy, "fifo")
        self.assertEqual(config.device_cache_min_free_blocks, 0)
        self.assertEqual(config.block_tree_full_prefix_scan_interval_ms, 0)

    def test_legacy_kv_cache_cli_aliases(self):
        config = setup_args(
            [
                "--enable_memory_cache",
                "1",
                "--memory_cache_size_mb",
                "2048",
                "--enable_memory_cache_disk",
                "1",
                "--memory_cache_disk_paths",
                "/tmp/legacy-cache",
                "--memory_cache_disk_size_mb",
                "4096",
                "--write_cache_sync",
                "1",
            ]
        ).kv_cache_config

        self.assertTrue(config.enable_host_cache)
        self.assertEqual(config.host_cache_size_mb, 2048)
        self.assertTrue(config.enable_disk_cache)
        self.assertEqual(config.disk_cache_paths, "/tmp/legacy-cache")
        self.assertEqual(config.disk_cache_size_mb, 4096)

    @patch.dict(
        "os.environ",
        {
            **_PINNED_DEVICES,
            "MODEL_TYPE": "fake_model",
            "ENABLE_MEMORY_CACHE": "1",
            "MEMORY_CACHE_SIZE_MB": "1024",
            "ENABLE_MEMORY_CACHE_DISK": "1",
            "MEMORY_CACHE_DISK_PATHS": "/tmp/legacy-disk",
            "MEMORY_CACHE_DISK_SIZE_MB": "2048",
        },
        clear=True,
    )
    def test_legacy_kv_cache_env_aliases(self):
        config = setup_args().kv_cache_config

        self.assertTrue(config.enable_host_cache)
        self.assertEqual(config.host_cache_size_mb, 1024)
        self.assertTrue(config.enable_disk_cache)
        self.assertEqual(config.disk_cache_paths, "/tmp/legacy-disk")
        self.assertEqual(config.disk_cache_size_mb, 2048)

    @patch.dict(
        "os.environ",
        {
            **_PINNED_DEVICES,
            "MODEL_TYPE": "fake_model",
            "ENABLE_HOST_CACHE": "0",
            "ENABLE_MEMORY_CACHE": "1",
            "HOST_CACHE_SIZE_MB": "512",
            "MEMORY_CACHE_SIZE_MB": "1024",
        },
        clear=True,
    )
    def test_canonical_kv_cache_env_wins_over_legacy_alias(self):
        config = setup_args().kv_cache_config

        self.assertFalse(config.enable_host_cache)
        self.assertEqual(config.host_cache_size_mb, 512)

    def test_kv_cache_config_pickle_round_trip_includes_eviction_fields(self):
        import pickle

        from rtp_llm.ops import KVCacheConfig

        config = KVCacheConfig()
        config.disk_cache_staging_block_count = 8
        config.enable_disk_cache = False
        config.enable_host_cache_pinned = False
        config.device_eviction_policy = "fifo"
        config.host_eviction_policy = "lfu"
        config.disk_eviction_policy = "lru"
        config.device_cache_min_free_blocks = 123
        config.dsv4_fixed_pool_blocks = 512
        config.dsv4_hca_state_pool_blocks = 256
        config.dsv4_fixed_pool_use_memory = True
        config.memory_cache_max_descriptors_per_transfer_batch = 17
        config.block_tree_full_prefix_scan_interval_ms = 5000

        state = config.__getstate__()
        self.assertEqual(len(state), 57)
        self.assertEqual(state[:2], ("KVCacheConfig", 1))

        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.disk_cache_staging_block_count, 8)
        self.assertFalse(restored.enable_disk_cache)
        self.assertFalse(restored.enable_host_cache_pinned)
        self.assertEqual(restored.device_eviction_policy, "fifo")
        self.assertEqual(restored.host_eviction_policy, "lfu")
        self.assertEqual(restored.disk_eviction_policy, "lru")
        self.assertEqual(restored.device_cache_min_free_blocks, 123)
        self.assertEqual(restored.dsv4_fixed_pool_blocks, 512)
        self.assertEqual(restored.dsv4_hca_state_pool_blocks, 256)
        self.assertTrue(restored.dsv4_fixed_pool_use_memory)
        self.assertEqual(restored.memory_cache_max_descriptors_per_transfer_batch, 17)
        self.assertEqual(restored.block_tree_full_prefix_scan_interval_ms, 5000)

        config.enable_disk_cache = True
        restored_enabled = pickle.loads(pickle.dumps(config))
        self.assertTrue(restored_enabled.enable_disk_cache)

    def test_kv_cache_config_pickle_rejects_incompatible_states(self):
        from rtp_llm.ops import KVCacheConfig

        def restore(state):
            value = KVCacheConfig.__new__(KVCacheConfig)
            value.__setstate__(state)
            return value

        state = KVCacheConfig().__getstate__()
        for invalid_state in (
            state[2:],
            ("OtherConfig", *state[1:]),
            (state[0], 99, *state[2:]),
            state[:-1],
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid KVCacheConfig state"):
                restore(invalid_state)

    def test_jit_config(self):
        valid = (
            ([], {}, ("", 180)),
            (
                [],
                {"REMOTE_JIT_DIR": "/remote/jit", "JIT_CACHE_SETUP_TIMEOUT_S": "60"},
                ("/remote/jit", 60),
            ),
            (["--jit_cache_setup_timeout_s", "5"], {}, ("", 5)),
            (["--jit_cache_setup_timeout_s", "-1"], {}, ("", -1)),
            ([], {"JIT_CACHE_SETUP_TIMEOUT_S": "-1"}, ("", -1)),
            # CLI wins over env even for -1, which the provided_args scanner sees
            # as an option rather than as a value.
            (
                ["--jit_cache_setup_timeout_s", "-1"],
                {"JIT_CACHE_SETUP_TIMEOUT_S": "60"},
                ("", -1),
            ),
        )
        for args, env, expected in valid:
            with self.subTest(args=args, env=env), patch.dict(
                os.environ, _jit_env(**env), clear=True
            ):
                config = setup_args(args).jit_config
                self.assertEqual(
                    (config.remote_jit_dir, config.jit_cache_setup_timeout_s), expected
                )

        for args, env, expected in (
            ([], {}, True),
            (["--manage_jit_cache", "0"], {}, False),
            ([], {"MANAGE_JIT_CACHE": "0"}, False),
        ):
            with self.subTest(args=args, env=env), patch.dict(
                os.environ, _jit_env(**env), clear=True
            ):
                self.assertIs(setup_args(args).jit_config.manage_jit_cache, expected)

        timeout_error, bool_error = "positive integer or -1", "Boolean value expected"
        for args, env, message in (
            (["--jit_cache_setup_timeout_s", "0"], {}, timeout_error),
            (["--jit_cache_setup_timeout_s", "-2"], {}, timeout_error),
            ([], {"JIT_CACHE_SETUP_TIMEOUT_S": "0"}, timeout_error),
            ([], {"JIT_CACHE_SETUP_TIMEOUT_S": "invalid"}, timeout_error),
            ([], {"MANAGE_JIT_CACHE": "maybe"}, bool_error),
            # An empty value fails like every other int/str2bool arg does.
            ([], {"JIT_CACHE_SETUP_TIMEOUT_S": ""}, timeout_error),
            ([], {"MANAGE_JIT_CACHE": ""}, bool_error),
        ):
            stderr = io.StringIO()
            with self.subTest(args=args, env=env), patch.dict(
                os.environ, _jit_env(**env), clear=True
            ), contextlib.redirect_stderr(stderr), self.assertRaises(
                SystemExit
            ) as context:
                setup_args(args)
            self.assertEqual(context.exception.code, 2)
            # Fail *because of this validator*, not from an unrelated parse error.
            self.assertIn(message, stderr.getvalue())

    def test_jit_config_pure_env_path_without_cli_args(self):
        """setup_args() with no argv reaches the env-only branch of the scanner.

        This is the deployment path: every value arrives as a raw env string.
        """
        for env, expected in (
            ({"JIT_CACHE_SETUP_TIMEOUT_S": "45"}, (45, True)),
            ({"MANAGE_JIT_CACHE": "0"}, (180, False)),
        ):
            with self.subTest(env=env), patch.object(sys, "argv", ["prog"]), patch.dict(
                os.environ, _jit_env(**env), clear=True
            ):
                config = setup_args().jit_config
                self.assertEqual(
                    (config.jit_cache_setup_timeout_s, config.manage_jit_cache),
                    expected,
                )

    def test_engine_config_propagates_role_to_parallelism_config(self):
        py_env_configs = PyEnvConfigs()
        py_env_configs.role_config.role_type = RoleType.PREFILL

        engine_config = EngineConfig.create(py_env_configs)

        self.assertEqual(engine_config.pd_sep_config.role_type, RoleType.PREFILL)
        self.assertEqual(engine_config.parallelism_config.role_type, RoleType.PREFILL)

    def test_engine_config_preserves_role_when_context_parallel_is_enabled(self):
        for role_type in (
            RoleType.PDFUSION,
            RoleType.PREFILL,
            RoleType.DECODE,
        ):
            with self.subTest(role_type=role_type):
                py_env_configs = PyEnvConfigs()
                py_env_configs.role_config.role_type = role_type
                py_env_configs.parallelism_config.prefill_cp_config.method = (
                    CPRotateMethod.ALL_GATHER
                )

                engine_config = EngineConfig.create(py_env_configs)
                self.assertEqual(engine_config.pd_sep_config.role_type, role_type)
                self.assertEqual(engine_config.parallelism_config.role_type, role_type)

    def test_engine_config_minimal_dataclass_construction_from_py_env_configs(self):
        py_env_configs = PyEnvConfigs()

        engine_config = EngineConfig(
            parallelism_config=py_env_configs.parallelism_config,
            runtime_config=py_env_configs.runtime_config,
            nccl_comm_config=NcclCommConfig(
                nccl_ip="",
                tp_nccl_port=0,
                dp_tp_nccl_port=0,
                ffn_tp_nccl_port=0,
            ),
            server_config=py_env_configs.server_config,
            pd_sep_config=py_env_configs.pd_separation_config,
            concurrency_config=py_env_configs.concurrency_config,
            fmha_config=py_env_configs.fmha_config,
            kv_cache_config=py_env_configs.kv_cache_config,
            profiling_debug_logging_config=(
                py_env_configs.profiling_debug_logging_config
            ),
            hw_kernel_config=py_env_configs.py_hw_kernel_config,
            device_resource_config=py_env_configs.device_resource_config,
            moe_config=py_env_configs.moe_config,
            model_specific_config=py_env_configs.model_specific_config,
            sp_config=py_env_configs.sp_config,
            cache_store_config=py_env_configs.cache_store_config,
            misc_config=py_env_configs.misc_config.misc_config,
            arpc_config=py_env_configs.arpc_config,
            grpc_config=py_env_configs.grpc_config,
            dash_sc_grpc_config=py_env_configs.dash_sc_grpc_config,
            grammar_config=py_env_configs.grammar_config,
            load_config=py_env_configs.load_config,
        )

        self.assertIs(engine_config.grammar_config, py_env_configs.grammar_config)

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
            **_PINNED_DEVICES,
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

    def setUp(self):
        # clear=True below removes CUDA_VISIBLE_DEVICES after torch is imported;
        # these config-only tests must not trigger real CUDA lazy initialization.
        cuda_available = patch(
            "rtp_llm.config.server_config_setup.torch.cuda.is_available",
            return_value=False,
        )
        cuda_available.start()
        self.addCleanup(cuda_available.stop)

    # EnvArgumentParser in setup_args() reads these env vars (START_PORT, TP_SIZE, etc.)
    # and binds them to py_env_configs; server_port = start_port + rank_id * worker_info_port_num (rank_id=0 here).
    @patch.dict(
        "os.environ",
        {
            **_PINNED_DEVICES,
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
            **_PINNED_DEVICES,
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
            **_PINNED_DEVICES,
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
            **_PINNED_DEVICES,
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
