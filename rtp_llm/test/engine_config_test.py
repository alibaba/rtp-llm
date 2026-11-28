import pickle
from types import SimpleNamespace
from unittest import TestCase, main

from rtp_llm.config.engine_config import setup_pd_sep_config, update_worker_addrs
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import (
    CacheStoreConfig,
    ParallelismConfig,
    PDSepConfig,
    RoleType,
    RuntimeConfig,
)
from rtp_llm.server.server_args.cache_store_group_args import init_cache_store_group_args
from rtp_llm.server.server_args.server_args import EnvArgumentParser


class EngineConfigTest(TestCase):
    def _parallelism_config(self) -> ParallelismConfig:
        config = ParallelismConfig()
        config.local_rank = 0
        config.tp_size = 2
        config.dp_size = 1
        config.dp_rank = 0
        return config

    def _world_info(self):
        members = [
            SimpleNamespace(
                world_rank=0,
                ip="127.0.0.1",
                cache_store_listen_port=12000,
                cache_store_rdma_listen_port=12100,
                rpc_server_port=13000,
            ),
            SimpleNamespace(
                world_rank=1,
                ip="127.0.0.2",
                cache_store_listen_port=12010,
                cache_store_rdma_listen_port=12110,
                rpc_server_port=13010,
            ),
        ]
        return SimpleNamespace(members=members)

    def test_update_worker_addrs_formats_by_role_and_orders_by_rank(self):
        legacy = ["127.0.0.1:12000:12100", "127.0.0.2:12010:12110"]
        p2p = ["127.0.0.1:12001:13000", "127.0.0.2:12011:13010"]
        for role, expected in (
            (RoleType.PDFUSION, legacy),
            (RoleType.PREFILL, p2p),
            (RoleType.DECODE, p2p),
        ):
            with self.subTest(role=role):
                runtime = RuntimeConfig()
                world = self._world_info()
                world.members.reverse()
                update_worker_addrs(runtime, self._parallelism_config(), world, role_type=role)
                self.assertEqual(runtime.worker_addrs, expected)
                self.assertEqual(
                    runtime.worker_grpc_addrs, ["127.0.0.1:13000", "127.0.0.2:13010"]
                )

    def test_update_worker_addrs_rejects_p2p_port_outside_worker_block(self):
        runtime_config = RuntimeConfig()
        world_info = SimpleNamespace(
            members=[
                SimpleNamespace(
                    world_rank=0,
                    ip="127.0.0.1",
                    server_port=12000,
                    _worker_info_port_num=3,
                    cache_store_listen_port=12002,
                    cache_store_rdma_listen_port=12004,
                    rpc_server_port=12001,
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "reserved port block"):
            update_worker_addrs(
                runtime_config,
                self._parallelism_config(),
                world_info,
                role_type=RoleType.PREFILL,
            )

    def test_update_worker_addrs_rejects_p2p_port_conflict(self):
        runtime_config = RuntimeConfig()
        world_info = SimpleNamespace(
            members=[
                SimpleNamespace(
                    world_rank=0,
                    ip="127.0.0.1",
                    cache_store_listen_port=12000,
                    cache_store_rdma_listen_port=12001,
                    rpc_server_port=13000,
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "cache_store_rdma_listen_port"):
            update_worker_addrs(
                runtime_config,
                self._parallelism_config(),
                world_info,
                role_type=RoleType.PREFILL,
            )

    def test_update_worker_addrs_rejects_duplicate_p2p_port_on_same_ip(self):
        runtime_config = RuntimeConfig()
        world_info = SimpleNamespace(
            members=[
                SimpleNamespace(
                    world_rank=0,
                    ip="127.0.0.1",
                    cache_store_listen_port=12000,
                    cache_store_rdma_listen_port=12100,
                    rpc_server_port=13000,
                ),
                SimpleNamespace(
                    world_rank=1,
                    ip="127.0.0.1",
                    cache_store_listen_port=12000,
                    cache_store_rdma_listen_port=12110,
                    rpc_server_port=13010,
                ),
            ]
        )

        with self.assertRaisesRegex(ValueError, "duplicate p2p_transfer_port"):
            update_worker_addrs(
                runtime_config,
                self._parallelism_config(),
                world_info,
                role_type=RoleType.PREFILL,
            )

    def test_pd_sep_config_roundtrip_preserves_active_fields(self):
        config = PDSepConfig()
        config.role_type = RoleType.DECODE
        config.cache_store_rdma_mode = False
        config.load_cache_timeout_ms = 12345
        config.max_rpc_timeout_ms = 67890
        config.prefill_retry_times = 3
        config.prefill_retry_timeout_ms = 75
        config.decode_linear_tail_only = True
        state = config.__getstate__()
        self.assertEqual(len(state), 13)
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.__getstate__(), state)
        self.assertEqual(restored.role_type, RoleType.DECODE)
        self.assertEqual(restored.load_cache_timeout_ms, 12345)
        self.assertEqual(restored.prefill_retry_times, 3)
        self.assertEqual(restored.prefill_retry_timeout_ms, 75)
        legacy = PDSepConfig.__new__(PDSepConfig)
        legacy_state = state[:9] + (600000, 1, 30, 0) + state[9:12] + (True, 2000, 0)
        legacy.__setstate__(legacy_state + (True,))
        self.assertEqual(legacy.__getstate__(), state)
        for previous_state in (state[:12], legacy_state):
            previous = PDSepConfig.__new__(PDSepConfig)
            previous.__setstate__(previous_state)
            self.assertEqual(previous.__getstate__(), state[:12] + (False,))

    def test_cache_store_config_roundtrip_preserves_active_fields(self):
        config = CacheStoreConfig()
        self.assertEqual(config.p2p_prefill_sender_thread_count, 4)
        self.assertEqual(config.p2p_prefill_sender_queue_size, 10000)
        config.p2p_prefill_sender_thread_count = 8
        config.p2p_prefill_sender_queue_size = 20000
        config.p2p_kv_force_release_timeout_ms = 120000
        state = config.__getstate__()
        self.assertEqual(len(state), 23)
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.__getstate__(), state)
        self.assertEqual(restored.p2p_prefill_sender_thread_count, 8)
        self.assertEqual(restored.p2p_prefill_sender_queue_size, 20000)
        previous = CacheStoreConfig.__new__(CacheStoreConfig)
        previous.__setstate__(state[:22] + (180000, 250, 100, 100, 100000, 300000))
        self.assertEqual(previous.__getstate__(), state[:22] + (300000,))
        legacy = CacheStoreConfig.__new__(CacheStoreConfig)
        legacy.__setstate__(state[:22] + (40, 45, 180000, 250, 100, 100, 100000))
        self.assertEqual(legacy.__getstate__(), state[:22] + (300000,))

    def test_cache_store_config_preserves_rdma_h2d_staging_total_bytes(self):
        config = CacheStoreConfig()
        config.p2p_rdma_enable_h2d_copy = True
        config.p2p_rdma_staging_total_bytes = 32 * 1024 * 1024

        restored = pickle.loads(pickle.dumps(config))

        self.assertTrue(restored.p2p_rdma_enable_h2d_copy)
        self.assertEqual(restored.p2p_rdma_staging_total_bytes, 32 * 1024 * 1024)


class SetupPdSepConfigTest(TestCase):
    def _make_configs(self, role_type, cache_rdma_mode: bool):
        pd_sep_config = PDSepConfig()
        pd_sep_config.role_type = role_type
        cache_store_config = CacheStoreConfig()
        cache_store_config.cache_store_rdma_mode = cache_rdma_mode
        server_config = SimpleNamespace(
            cache_store_listen_port=0,
            cache_store_rdma_listen_port=0,
            worker_info_port_num=0,
        )
        distribute_config = SimpleNamespace(
            cache_store_connect_port=0,
            cache_store_rdma_connect_port=0,
            remote_rpc_server_port=0,
        )
        return pd_sep_config, cache_store_config, server_config, distribute_config

    def test_sync_rdma_mode_for_pd_roles(self):
        for role in (RoleType.PREFILL, RoleType.DECODE):
            for enabled in (False, True):
                with self.subTest(role=role, enabled=enabled):
                    pd, cache, server, distribute = self._make_configs(role, enabled)
                    pd.cache_store_rdma_mode = not enabled
                    setup_pd_sep_config(pd, cache, server, distribute)
                    self.assertEqual(pd.cache_store_rdma_mode, enabled)


class CacheStoreGroupArgsBindingTest(TestCase):
    def test_p2p_safety_ttl_arg_binds_to_cache_store_config(self):
        py_env_configs = PyEnvConfigs()
        parser = EnvArgumentParser(add_help=False)
        parser.set_root_config(py_env_configs)
        init_cache_store_group_args(parser, py_env_configs.cache_store_config)

        parser.parse_args(
            [
                "--p2p_cancelled_keys_ttl_ms",
                "9012",
            ]
        )

        config = py_env_configs.cache_store_config
        self.assertEqual(config.p2p_cancelled_keys_ttl_ms, 9012)


if __name__ == "__main__":
    main()
