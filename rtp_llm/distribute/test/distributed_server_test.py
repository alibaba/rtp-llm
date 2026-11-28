import json
import multiprocessing
import os
import socket
import threading
import time
import unittest
from multiprocessing import Process
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import set_parallelism_config

torch.cuda.set_device = lambda x: None

from typing import Any, Dict

import rtp_llm.distribute.distributed_server as ds
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.distribute.test.fake_model import FakeModel
from rtp_llm.frontend.frontend_server import FrontendWorker
from rtp_llm.model_factory_register import register_model
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.server.server_args.server_args import setup_args


def fake_init(self, *args, **kwargs):
    self.model_config = None
    self.tokenizer = None
    self.model_cls = None
    self.pipeline = None
    self.backend_rpc_server_visitor = None


FrontendWorker.__init__ = fake_init
OpenaiEndpoint.__init__ = fake_init


class FakeStore:
    def __init__(self):
        self._data: Dict[str, bytes] = {}

    def set(self, key: str, value: str):
        self._data[key] = value.encode("utf-8")

    def get(self, key: str) -> bytes:
        if key not in self._data:
            raise RuntimeError(f"Key {key} not found")
        return self._data[key]


def test_get_ip_with_ip():
    assert ds.get_ip("127.0.0.1") == "127.0.0.1"


def init_server(
    py_env_configs: PyEnvConfigs,
    rank: int,
    world_size: int,
    stop_event: threading.Event,
):

    py_env_configs.server_config.ip = socket.gethostbyname(socket.gethostname())
    set_parallelism_config(py_env_configs.parallelism_config, rank)
    py_env_configs.server_config.set_local_rank(
        py_env_configs.parallelism_config.local_rank
    )
    py_env_configs.distribute_config.set_local_rank(
        py_env_configs.parallelism_config.local_rank
    )
    ds.DistributedServer(py_env_configs, rank, world_size)
    while not stop_event.is_set():
        time.sleep(1)


def regist_server(
    py_env_configs: PyEnvConfigs,
    rank: int,
    world_size: int,
    stop_event: threading.Event,
):
    py_env_configs.server_config.ip = socket.gethostbyname(socket.gethostname())
    set_parallelism_config(py_env_configs.parallelism_config, rank)
    py_env_configs.server_config.set_local_rank(
        py_env_configs.parallelism_config.local_rank
    )
    py_env_configs.distribute_config.set_local_rank(
        py_env_configs.parallelism_config.local_rank
    )
    server = ds.DistributedServer(py_env_configs, rank, world_size)
    server.regist()
    while not stop_event.is_set():
        time.sleep(1)


def start_server(
    py_env_configs: PyEnvConfigs,
    rank: int,
    world_size: int,
):
    py_env_configs.server_config.ip = socket.gethostbyname(socket.gethostname())
    set_parallelism_config(py_env_configs.parallelism_config, rank)
    py_env_configs.server_config.set_local_rank(
        py_env_configs.parallelism_config.local_rank
    )
    py_env_configs.distribute_config.set_local_rank(
        py_env_configs.parallelism_config.local_rank
    )
    server = ds.DistributedServer(py_env_configs, rank, world_size)
    server.start(py_env_configs)
    while True:
        time.sleep(1)


class TestGetWorldInfo(TestCase):

    @patch.dict(
        "os.environ",
        {
            "TP_SIZE": "2",
            "PP_SIZE": "1",
            "WORLD_SIZE": "2",
            "WORLD_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "START_PORT": "20000",
            "WORKER_INFO_PORT_NUM": "8",
            "MODEL_TYPE": "fake_model",
        },
        clear=True,
    )
    def test_single_node(self):
        py_env_configs: PyEnvConfigs = setup_args()
        from rtp_llm.config.server_config_setup import (
            fetch_model_files_to_local,
            setup_default_args,
        )

        setup_default_args(py_env_configs)
        fetch_model_files_to_local(py_env_configs)
        setup_and_configure_server(py_env_configs)
        py_env_configs.server_config.ip = socket.gethostbyname(socket.gethostname())

        world_info = get_world_info(
            py_env_configs.server_config,
            py_env_configs.distribute_config,
            py_env_configs.parallelism_config,
        )
        self.assertTrue(world_info.initialized)
        self.assertEqual(len(world_info.members), 2)
        self.assertEqual(world_info.members[0].server_port, 20000)
        self.assertEqual(world_info.members[1].server_port, 20008)


class PPPartitionConsistencyTest(unittest.TestCase):
    @staticmethod
    def _config(world_rank=0, world_size=4, pp_size=2):
        return SimpleNamespace(
            pp_size=pp_size,
            world_rank=world_rank,
            world_size=world_size,
            pp_stage_layer_counts=[2, 3] if pp_size > 1 else [],
        )

    @staticmethod
    def _server():
        server = ds.DistributedServer.__new__(ds.DistributedServer)
        server.store = FakeStore()
        return server

    def test_rank_publishes_and_validates_all_snapshots(self):
        server = self._server()
        for rank in (0, 1, 3):
            server.store.set(
                server.PP_PARTITION_SNAPSHOT_KEY + str(rank),
                json.dumps([2, 3]),
            )

        server.validate_pp_partition_consistency(self._config(world_rank=2))

        snapshot = json.loads(
            server.store.get(server.PP_PARTITION_SNAPSHOT_KEY + "2").decode("utf-8")
        )
        self.assertEqual(snapshot, [2, 3])

    def test_missing_rank_snapshot_becomes_bounded_store_failure(self):
        server = self._server()
        with self.assertRaisesRegex(RuntimeError, "world_rank=1"):
            server.validate_pp_partition_consistency(self._config(world_size=2))

    def test_single_stage_does_not_touch_store(self):
        server = self._server()
        server.validate_pp_partition_consistency(self._config(world_size=1, pp_size=1))
        self.assertEqual(server.store._data, {})


class MoeEpConsistencyTest(unittest.TestCase):
    @staticmethod
    def _snap(
        rank,
        *,
        role_type=0,
        pp_size=1,
        dp_size=4,
        tp_size=1,
        ep_size=4,
        cap=(12, 0),
        moe_strategy="sm120_a2a_decode",
        use_deepep_moe=False,
        use_deepep_low_latency=False,
        use_deepep_internode=False,
        use_mori_ep=False,
        use_all_gather=False,
        deepep_supported=False,
    ):
        return ds.MoeEpRuntimeSnapshot(
            world_rank=rank,
            role_type=role_type,
            pp_size=pp_size,
            dp_size=dp_size,
            tp_size=tp_size,
            ep_size=ep_size,
            cuda_capability=cap,
            is_sm120=(cap == (12, 0)),
            moe_strategy=moe_strategy,
            use_deepep_moe=use_deepep_moe,
            use_deepep_low_latency=use_deepep_low_latency,
            use_deepep_internode=use_deepep_internode,
            use_mori_ep=use_mori_ep,
            use_all_gather=use_all_gather,
            deepep_supported=deepep_supported,
        )

    def test_homogeneous_sm120_and_sm100_pass(self):
        ds.validate_moe_ep_snapshots([self._snap(r) for r in range(4)])
        ds.validate_moe_ep_snapshots(
            [
                self._snap(
                    r,
                    cap=(10, 0),
                    moe_strategy="auto",
                    use_deepep_moe=True,
                    deepep_supported=True,
                )
                for r in range(8)
            ]
        )

    def test_mixed_sm100_and_sm120_ranks_rejected(self):
        for deepep_flag in (False, True):
            with self.subTest(use_deepep_moe=deepep_flag):
                snaps = [
                    self._snap(
                        0,
                        cap=(10, 0),
                        moe_strategy="auto",
                        use_deepep_moe=deepep_flag,
                        deepep_supported=True,
                    ),
                    self._snap(
                        1,
                        cap=(10, 0),
                        moe_strategy="auto",
                        use_deepep_moe=deepep_flag,
                        deepep_supported=True,
                    ),
                    self._snap(
                        2,
                        cap=(12, 0),
                        moe_strategy="sm120_a2a_decode",
                        use_deepep_moe=deepep_flag,
                        deepep_supported=False,
                    ),
                    self._snap(
                        3,
                        cap=(12, 0),
                        moe_strategy="sm120_a2a_decode",
                        use_deepep_moe=deepep_flag,
                        deepep_supported=False,
                    ),
                ]
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    ds.validate_moe_ep_snapshots(snaps)

    def test_explicit_deepep_on_unsupported_sm120_rejected(self):
        snaps = [
            self._snap(
                r,
                cap=(12, 0),
                moe_strategy="sm120_a2a_decode",
                use_deepep_moe=True,
                deepep_supported=False,
            )
            for r in range(4)
        ]
        with self.assertRaisesRegex(ValueError, "unsupported ranks"):
            ds.validate_moe_ep_snapshots(snaps)

    def test_mixed_deepep_supported_allowed_when_deepep_not_selected_and_rejected_when_selected(
        self,
    ):
        for use_all_gather, use_deepep_moe in (
            (True, False),
            (True, True),
            (False, False),
        ):
            with self.subTest(
                use_all_gather=use_all_gather, use_deepep_moe=use_deepep_moe
            ):
                snaps = [
                    self._snap(
                        r,
                        cap=(9, 0),
                        moe_strategy="auto",
                        use_all_gather=use_all_gather,
                        use_deepep_moe=use_deepep_moe,
                        deepep_supported=(r == 0),
                    )
                    for r in range(4)
                ]
                ds.validate_moe_ep_snapshots(snaps)

        selected_snaps = [
            self._snap(
                r,
                cap=(9, 0),
                moe_strategy="auto",
                use_all_gather=False,
                use_deepep_moe=True,
                deepep_supported=(r == 0),
            )
            for r in range(4)
        ]
        with self.assertRaisesRegex(ValueError, "unsupported ranks"):
            ds.validate_moe_ep_snapshots(selected_snaps)

    def test_pp_stage_asymmetry_and_pure_pp_preserved(self):
        # Pure PP (ep_size=1) across heterogeneous stages is allowed.
        pure_pp = [
            self._snap(
                0,
                pp_size=2,
                dp_size=1,
                tp_size=1,
                ep_size=1,
                cap=(10, 0),
                moe_strategy="auto",
                deepep_supported=True,
            ),
            self._snap(
                1,
                pp_size=2,
                dp_size=1,
                tp_size=1,
                ep_size=1,
                cap=(12, 0),
                moe_strategy="auto",
                deepep_supported=False,
            ),
        ]
        ds.validate_moe_ep_snapshots(pure_pp)

        # Multi-stage PP with ep_size=4 validates per Group.STAGE so stage 0
        # internode=False and stage 1 internode=True stay valid.
        multi_stage = [
            self._snap(
                r,
                pp_size=2,
                dp_size=1,
                tp_size=4,
                ep_size=4,
                cap=(10, 0),
                moe_strategy="auto",
                use_deepep_moe=True,
                use_deepep_internode=(r >= 4),
                deepep_supported=True,
            )
            for r in range(8)
        ]
        ds.validate_moe_ep_snapshots(multi_stage)

    def test_distributed_server_store_handles_dense_noop_and_ep_skew(self):
        server = PPPartitionConsistencyTest._server()
        cfg = SimpleNamespace(
            parallelism_config=SimpleNamespace(
                world_rank=0,
                world_size=2,
                role_type=0,
                pp_size=1,
                dp_size=2,
                tp_size=1,
                ep_size=2,
            ),
            moe_config=SimpleNamespace(
                moe_strategy="sm120_a2a_decode",
                use_deepep_moe=False,
                use_deepep_low_latency=False,
                use_deepep_internode=False,
                use_mori_ep=False,
                use_all_gather=False,
            ),
        )
        # Dense model (expert_num == 0): no TCPStore exchange.
        server.validate_moe_ep_consistency(cfg, expert_num=0)
        self.assertEqual(server.store._data, {})

        # Rank 1 published ep_size=1 (skewed against rank 0's ep_size=2).
        skewed_r1 = self._snap(1, dp_size=1, ep_size=1, moe_strategy="auto")
        server.store.set(
            server.MOE_EP_SNAPSHOT_KEY + "1",
            json.dumps(skewed_r1.to_dict()),
        )
        with (
            patch(
                "rtp_llm.device.cuda_capability.device_cuda_capability",
                return_value=(12, 0),
            ),
            patch(
                "rtp_llm.models_py.distributed.deepep_wrapper.DeepEPWrapper.supported",
                return_value=False,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "WORLD topology mismatch"):
                server.validate_moe_ep_consistency(cfg, expert_num=256)

    def test_validate_moe_ep_consistency_skips_deepep_import_when_not_required(self):
        for use_deepep, use_all_gather in ((False, False), (True, True)):
            with self.subTest(use_deepep=use_deepep, use_all_gather=use_all_gather):
                server = PPPartitionConsistencyTest._server()
                cfg = SimpleNamespace(
                    parallelism_config=SimpleNamespace(
                        world_rank=0,
                        world_size=2,
                        role_type=0,
                        pp_size=1,
                        dp_size=2,
                        tp_size=1,
                        ep_size=2,
                    ),
                    moe_config=SimpleNamespace(
                        moe_strategy="sm120_a2a_decode",
                        use_deepep_moe=use_deepep,
                        use_deepep_low_latency=False,
                        use_deepep_internode=False,
                        use_mori_ep=False,
                        use_all_gather=use_all_gather,
                    ),
                )
                r1 = self._snap(
                    1,
                    dp_size=2,
                    ep_size=2,
                    cap=(12, 0),
                    moe_strategy="sm120_a2a_decode",
                    use_deepep_moe=use_deepep,
                    use_all_gather=use_all_gather,
                    deepep_supported=False,
                )
                server.store.set(
                    server.MOE_EP_SNAPSHOT_KEY + "1",
                    json.dumps(r1.to_dict()),
                )
                with (
                    patch.dict(
                        "sys.modules",
                        {"rtp_llm.models_py.distributed.deepep_wrapper": None},
                    ),
                    patch(
                        "rtp_llm.device.cuda_capability.device_cuda_capability",
                        return_value=(12, 0),
                    ),
                ):
                    server.validate_moe_ep_consistency(cfg, expert_num=256)
                published = ds.MoeEpRuntimeSnapshot.from_dict(
                    json.loads(server.store.get(server.MOE_EP_SNAPSHOT_KEY + "0"))
                )
                self.assertFalse(published.deepep_supported)


class DistributedServerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.maxDiff = None
        super().__init__(*args, **kwargs)

        def test_get_master_from_json(self):
            gang_info_json: Dict[str, Any] = {
                "worker_part0": {"ip": "10.0.0.1"},
                "worker_part1": {"ip": "10.0.0.2"},
            }
            ip, port = ds.get_master_from_json(gang_info_json)
            assert ip == "10.0.0.1"
            assert port == ""

        def test_get_master_from_json_no_part0(self):
            gang_info_json: Dict[str, Any] = {
                "worker_part1": {"ip": "10.0.0.2"},
            }
            ip, port = ds.get_master_from_json(gang_info_json)
            assert ip == ""
            assert port == ""

        def test_get_master_from_test_env(self):
            env_str = "name:worker_part1,ip:10.0.0.2;name:worker_part0,ip:10.0.0.1"
            ip, port = ds.get_master_from_test_env(env_str)
            assert ip == "10.0.0.1"
            assert port == ""

        def test_get_master_from_test_env_not_found(self):
            env_str = "name:worker_part1,ip:10.0.0.2"
            ip, port = ds.get_master_from_test_env(env_str)
            assert ip == ""
            assert port == ""

        @patch.dict(
            "os.environ",
            {
                "DISTRIBUTE_CONFIG_FILE": "rtp_llm/distribute/test/testdata/parallel.json",
                "TP_SIZE": "2",
                "PP_SIZE": "1",
                "WORLD_SIZE": "2",
                "WORLD_RANK": "0",
                "LOCAL_WORLD_SIZE": "1",
                "MODEL_TYPE": "fake_model",
            },
            clear=True,
        )
        def test_get_master_use_distribute_config_file(self):
            py_env_configs: PyEnvConfigs = setup_args()
            setup_and_configure_server(py_env_configs)

            ip, port = ds.get_master(
                py_env_configs.distribute_config,
                py_env_configs.parallelism_config,
            )
            assert ip == "11.161.48.116"
            assert port == "10000"

        @patch.dict(
            "os.environ",
            {
                "GANG_CONFIG_STRING": "name:worker_part0,ip:10.0.0.123",
                "TP_SIZE": "2",
                "PP_SIZE": "1",
                "WORLD_SIZE": "2",
                "WORLD_RANK": "0",
                "LOCAL_WORLD_SIZE": "1",
                "MODEL_TYPE": "fake_model",
            },
            clear=True,
        )
        def test_get_master_use_gang_config_string(self):
            py_env_configs: PyEnvConfigs = setup_args()
            setup_and_configure_server(py_env_configs)
            ip, port = ds.get_master(
                py_env_configs.distribute_config,
                py_env_configs.parallelism_config,
            )
            assert ip == "10.0.0.123"
            assert port == ""

        @patch.dict(
            "os.environ",
            {
                "LEADER_ADDRESS": "10.0.0.5",
                "TP_SIZE": "2",
                "PP_SIZE": "1",
                "WORLD_SIZE": "2",
                "WORLD_RANK": "0",
                "LOCAL_WORLD_SIZE": "1",
                "MODEL_TYPE": "fake_model",
            },
            clear=True,
        )
        def test_get_master_use_leader_address(self):
            py_env_configs: PyEnvConfigs = setup_args()
            setup_and_configure_server(py_env_configs)

            ip, port = ds.get_master(
                py_env_configs.distribute_config,
                py_env_configs.parallelism_config,
            )
            assert ip == "10.0.0.5"
            assert port == ""

        @patch.dict(
            "os.environ",
            {
                "GANG_ANNOCATION_PATH": "rtp_llm/distribute/test/testdata/annocation",
                "TP_SIZE": "2",
                "PP_SIZE": "1",
                "WORLD_SIZE": "2",
                "WORLD_RANK": "0",
                "LOCAL_WORLD_SIZE": "1",
                "MODEL_TYPE": "fake_model",
            },
            clear=True,
        )
        def test_get_master_use_c2_file(self):
            py_env_configs: PyEnvConfigs = setup_args()
            setup_and_configure_server(py_env_configs)
            ip, port = ds.get_master(
                py_env_configs.distribute_config,
                py_env_configs.parallelism_config,
            )
            # 具体 IP 取决于 annocation 文件的内容，这里只检查非空
            assert isinstance(ip, str)
            assert ip == "33.115.125.211"
            assert port == ""

        @patch.dict(
            "os.environ",
            {
                "WORLD_SIZE": "1",
                "WORLD_RANK": "0",
                "LOCAL_WORLD_SIZE": "1",
                "MODEL_TYPE": "fake_model",
            },
            clear=True,
        )
        def test_get_master_single_machine(self):
            py_env_configs: PyEnvConfigs = setup_args()
            setup_and_configure_server(py_env_configs)

            ip, port = ds.get_master(
                py_env_configs.distribute_config,
                py_env_configs.parallelism_config,
            )
            assert ip == socket.gethostbyname(socket.gethostname())
            assert port == ""

        @patch.dict(
            "os.environ",
            {
                "DISTRIBUTE_CONFIG_FILE": "rtp_llm/distribute/test/testdata/parallel.json",
                "MODEL_TYPE": "fake_model",
            },
            clear=True,
        )
        def test_get_master_from_file(self):
            py_env_configs: PyEnvConfigs = setup_args()
            setup_and_configure_server(py_env_configs)
            ip, port = ds.get_master_from_file()
            assert ip == "11.161.48.116"
            assert port == "10000"

        @patch.dict(
            "os.environ",
            {
                "GANG_ANNOCATION_PATH": "rtp_llm/distribute/test/testdata/annocation",
                "MODEL_TYPE": "fake_model",
            },
            clear=True,
        )
        def test_get_master_from_c2(self):
            py_env_configs: PyEnvConfigs = setup_args()
            setup_and_configure_server(py_env_configs)
            ip, port = ds.get_master_from_c2()
            assert ip == "33.115.125.211"
            assert port == ""

        @patch.dict(
            "os.environ",
            {
                "TP_SIZE": "2",
                "PP_SIZE": "1",
                "WORLD_SIZE": "2",
                "WORLD_RANK": "0",
                "LOCAL_WORLD_SIZE": "2",
                "WORKER_INFO_PORT_NUM": "7",
                "START_PORT": "20000",
                "MODEL_TYPE": "fake_model",
            },
            clear=True,
        )
        def test_distributed_server_safe_store_set_get(self):
            py_env_configs: PyEnvConfigs = setup_args()
            setup_and_configure_server(py_env_configs)
            stop_event = threading.Event()

            t = threading.Thread(
                target=init_server, args=(py_env_configs, 1, 2, stop_event)
            )
            t.start()
            py_env_configs.server_config.ip = socket.gethostbyname(socket.gethostname())
            set_parallelism_config(
                py_env_configs.parallelism_config,
                0,
            )
            py_env_configs.server_config.set_local_rank(
                py_env_configs.parallelism_config.local_rank
            )
            py_env_configs.distribute_config.set_local_rank(
                py_env_configs.parallelism_config.local_rank
            )
            server = ds.DistributedServer(
                py_env_configs,
                rank=0,
                world_size=2,
            )

            server.safe_store_set("foo", "bar")
            assert server.safe_store_get("foo") == "bar"
            stop_event.set()

        def test_split_ip_port_valid(self):
            ip, port = ds.split_ip_port("1.2.3.4:1234")
            assert ip == "1.2.3.4"
            assert port == 1234

        def test_split_ip_port_invalid_no_colon(self):
            ip, port = ds.split_ip_port("1.2.3.4")
            assert ip == ""
            assert port == 0

        def test_split_ip_port_invalid_port_not_digit(self):
            ip, port = ds.split_ip_port("1.2.3.4:abc")
            assert ip == ""
            assert port == 0

        def test_split_ip_port_empty_ip(self):
            ip, port = ds.split_ip_port(":1234")
            assert ip == ""
            assert port == 0

        @patch.dict(
            "os.environ",
            {
                "TP_SIZE": "2",
                "PP_SIZE": "1",
                "WORLD_SIZE": "2",
                "WORLD_RANK": "0",
                "LOCAL_WORLD_SIZE": "2",
                "WORKER_INFO_PORT_NUM": "7",
                "START_PORT": "20000",
                "GANG_TIMEOUT_MIN": "1",
                "GANG_SLEEP_TIME": "0",
                "MODEL_TYPE": "fake_model",
            },
            clear=True,
        )
        def test_distributed_server_regist_and_bootstrap(self):
            py_env_configs: PyEnvConfigs = setup_args()
            setup_and_configure_server(py_env_configs)
            stop_event = threading.Event()

            # rank1
            stop_event = threading.Event()
            t = threading.Thread(
                target=regist_server, args=(py_env_configs, 1, 2, stop_event)
            )
            t.start()

            # rank0
            py_env_configs_0: PyEnvConfigs = setup_args()
            setup_and_configure_server(py_env_configs_0)
            stop_event = threading.Event()
            py_env_configs_0.server_config.ip = socket.gethostbyname(
                socket.gethostname()
            )
            set_parallelism_config(
                py_env_configs_0.parallelism_config,
                0,
            )
            py_env_configs_0.server_config.set_local_rank(
                py_env_configs_0.parallelism_config.local_rank
            )
            py_env_configs_0.distribute_config.set_local_rank(
                py_env_configs_0.parallelism_config.local_rank
            )
            server0 = ds.DistributedServer(
                py_env_configs_0,
                rank=0,
                world_size=2,
            )
            server0.bootstrap()

            assert len(server0._world_info.members) == 2
            assert server0._world_info.master is not None
            stop_event.set()


if __name__ == "__main__":
    unittest.main()
