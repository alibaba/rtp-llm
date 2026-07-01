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

    def test_update_worker_addrs_keeps_legacy_cache_store_format(self):
        runtime_config = RuntimeConfig()

        update_worker_addrs(
            runtime_config,
            self._parallelism_config(),
            self._world_info(),
            decode_entrance=False,
        )

        self.assertEqual(
            runtime_config.worker_addrs,
            [
                "127.0.0.1:12000:12100",
                "127.0.0.2:12010:12110",
            ],
        )
        self.assertEqual(
            runtime_config.worker_grpc_addrs,
            [
                "127.0.0.1:13000",
                "127.0.0.2:13010",
            ],
        )
        self.assertEqual(runtime_config.p2p_worker_addrs, [])

    def test_update_worker_addrs_uses_p2p_format_for_decode_entrance(self):
        runtime_config = RuntimeConfig()

        update_worker_addrs(
            runtime_config,
            self._parallelism_config(),
            self._world_info(),
            decode_entrance=True,
        )

        self.assertEqual(
            runtime_config.worker_addrs,
            [
                "127.0.0.1:12001:13000",
                "127.0.0.2:12011:13010",
            ],
        )
        self.assertEqual(
            runtime_config.worker_grpc_addrs,
            [
                "127.0.0.1:13000",
                "127.0.0.2:13010",
            ],
        )
        self.assertEqual(
            runtime_config.p2p_worker_addrs,
            [
                "127.0.0.1:12001:13000",
                "127.0.0.2:12011:13010",
            ],
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
                decode_entrance=True,
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
                decode_entrance=True,
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
                decode_entrance=True,
            )

    def test_cache_store_config_accepts_legacy_20_field_pickle_state(self):
        config = CacheStoreConfig()
        state = config.__getstate__()
        self.assertEqual(len(state), 26)

        # Legacy state before the three p2p deadline fields and three worker
        # queue fields were added: first 18 fields + TCP anet thread/queue.
        legacy_state = state[:18] + state[21:23]
        self.assertEqual(len(legacy_state), 20)

        restored = CacheStoreConfig()
        maybe_restored = restored.__setstate__(legacy_state)
        if maybe_restored is not None:
            restored = maybe_restored

        defaults = CacheStoreConfig()
        self.assertEqual(
            restored.cache_store_tcp_anet_rpc_thread_num,
            config.cache_store_tcp_anet_rpc_thread_num,
        )
        self.assertEqual(
            restored.cache_store_tcp_anet_rpc_queue_num,
            config.cache_store_tcp_anet_rpc_queue_num,
        )
        self.assertEqual(
            restored.p2p_prefill_resource_hold_ms,
            defaults.p2p_prefill_resource_hold_ms,
        )
        self.assertEqual(
            restored.rdma_transfer_worker_queue_size,
            defaults.rdma_transfer_worker_queue_size,
        )


class SetupPdSepConfigTest(TestCase):
    """Regression tests for setup_pd_sep_config: ensure CACHE_STORE_RDMA_MODE
    (which argparse binds to CacheStoreConfig.cache_store_rdma_mode) is always
    propagated into PDSepConfig.cache_store_rdma_mode, regardless of role_type.

    Background: the C++ defaults for the two fields disagree
    (CacheStoreConfig=false, PDSepConfig=true), so a silent split caused the
    5/22 .202 incident — CacheStoreConfig stayed false so P2PConnectorWorker
    picked the kTcp transfer backend (synchronous GPU->CPU copy in
    TcpKVCacheSender), while PDSepConfig retained the C++ default true
    elsewhere. The sync used to be gated on role_type ∈ {PREFILL, DECODE};
    we now sync unconditionally.
    """

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

    def test_sync_rdma_mode_true_in_prefill_role(self):
        pd_sep_config, cache_store_config, server_config, distribute_config = (
            self._make_configs(RoleType.PREFILL, cache_rdma_mode=True)
        )
        setup_pd_sep_config(
            pd_sep_config, cache_store_config, server_config, distribute_config
        )
        self.assertTrue(pd_sep_config.cache_store_rdma_mode)

    def test_sync_rdma_mode_false_in_prefill_role(self):
        pd_sep_config, cache_store_config, server_config, distribute_config = (
            self._make_configs(RoleType.PREFILL, cache_rdma_mode=False)
        )
        # PDSepConfig C++ default is true; ensure it gets overridden to false.
        pd_sep_config.cache_store_rdma_mode = True
        setup_pd_sep_config(
            pd_sep_config, cache_store_config, server_config, distribute_config
        )
        self.assertFalse(pd_sep_config.cache_store_rdma_mode)

    def test_sync_rdma_mode_true_in_decode_role(self):
        pd_sep_config, cache_store_config, server_config, distribute_config = (
            self._make_configs(RoleType.DECODE, cache_rdma_mode=True)
        )
        pd_sep_config.cache_store_rdma_mode = False
        setup_pd_sep_config(
            pd_sep_config, cache_store_config, server_config, distribute_config
        )
        self.assertTrue(pd_sep_config.cache_store_rdma_mode)

    def test_sync_rdma_mode_true_in_pdfusion_role(self):
        """Regression: before the fix, PDFUSION skipped the sync entirely, so
        PDSepConfig.cache_store_rdma_mode kept its C++ default (true), even
        when the user set CACHE_STORE_RDMA_MODE=0. After the fix the sync runs
        unconditionally."""
        pd_sep_config, cache_store_config, server_config, distribute_config = (
            self._make_configs(RoleType.PDFUSION, cache_rdma_mode=False)
        )
        pd_sep_config.cache_store_rdma_mode = True  # C++ default
        setup_pd_sep_config(
            pd_sep_config, cache_store_config, server_config, distribute_config
        )
        # Must follow cache_store_config now, not retain the C++ default.
        self.assertFalse(pd_sep_config.cache_store_rdma_mode)

    def test_sync_rdma_mode_false_in_pdfusion_role(self):
        pd_sep_config, cache_store_config, server_config, distribute_config = (
            self._make_configs(RoleType.PDFUSION, cache_rdma_mode=True)
        )
        pd_sep_config.cache_store_rdma_mode = False
        setup_pd_sep_config(
            pd_sep_config, cache_store_config, server_config, distribute_config
        )
        self.assertTrue(pd_sep_config.cache_store_rdma_mode)


class CacheStoreGroupArgsBindingTest(TestCase):
    def test_new_p2p_deadline_args_bind_to_cache_store_config(self):
        py_env_configs = PyEnvConfigs()
        parser = EnvArgumentParser(add_help=False)
        parser.set_root_config(py_env_configs)
        init_cache_store_group_args(parser, py_env_configs.cache_store_config)

        parser.parse_args(
            [
                "--p2p_prefill_resource_hold_ms",
                "1234",
                "--p2p_max_transfer_deadline_ms",
                "5678",
                "--p2p_cancelled_keys_ttl_ms",
                "9012",
            ]
        )

        config = py_env_configs.cache_store_config
        self.assertEqual(config.p2p_prefill_resource_hold_ms, 1234)
        self.assertEqual(config.p2p_max_transfer_deadline_ms, 5678)
        self.assertEqual(config.p2p_cancelled_keys_ttl_ms, 9012)


if __name__ == "__main__":
    main()
