import json
import multiprocessing
import os
import threading
import time
import unittest
from multiprocessing import Process
from unittest import TestCase
from unittest.mock import patch

import torch

from rtp_llm.config.py_config_modules import (
    MIN_WORKER_INFO_PORT_NUM,
    DistributeConfig,
    PyEnvConfigs,
)

torch.cuda.set_device = lambda x: None

from typing import Any, Dict

import rtp_llm.distribute.distributed_server as ds
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.distribute.test.fake_model import FakeModel
from rtp_llm.distribute.worker_info import (
    WorkerInfo,
    g_master_info,
    g_parallel_info,
    g_worker_info,
    update_worker_info,
)
from rtp_llm.frontend.frontend_server import FrontendWorker
from rtp_llm.model_factory_register import register_model
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.start_backend_server import main


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
    ds.DistributedServer(py_env_configs, rank, world_size)
    while not stop_event.is_set():
        time.sleep(1)


def regist_server(
    py_env_configs: PyEnvConfigs,
    rank: int,
    world_size: int,
    stop_event: threading.Event,
):
    server = ds.DistributedServer(py_env_configs, rank, world_size)
    g_parallel_info.world_rank = rank
    g_worker_info.world_rank = rank
    g_worker_info.local_rank = rank
    server.regist()
    while not stop_event.is_set():
        time.sleep(1)


def start_server(
    py_env_configs: PyEnvConfigs,
    rank: int,
    world_size: int,
):
    server = ds.DistributedServer(py_env_configs, rank, world_size)
    g_parallel_info.world_rank = rank
    g_worker_info.world_rank = rank
    g_worker_info.local_rank = rank
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
            "MODEL_TYPE": "fake_model",
        },
        clear=True,
    )
    def test_single_node(self):
        py_env_configs: PyEnvConfigs = setup_args()
        setup_and_configure_server(py_env_configs)

        world_info = get_world_info(
            py_env_configs.server_config, py_env_configs.distribute_config
        )
        self.assertTrue(world_info.initialized)
        self.assertEqual(len(world_info.members), 2)
        self.assertEqual(world_info.members[0].server_port, 20000)
        self.assertEqual(world_info.members[1].server_port, 20008)


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
            # update_worker_info(
            #    py_env_configs.server_config.start_port,
            #    py_env_configs.server_config.worker_info_port_num,
            #    py_env_configs.distribute_config.remote_server_port,
            # )

            ip, port = ds.get_master()
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
            ip, port = ds.get_master()
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

            ip, port = ds.get_master()
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
            ip, port = ds.get_master()
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

            ip, port = ds.get_master()
            assert ip == g_worker_info.ip
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
            server = ds.DistributedServer(
                py_env_configs=py_env_configs, rank=0, world_size=2
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
            py_env_configs: PyEnvConfigs = setup_args()
            setup_and_configure_server(py_env_configs)
            stop_event = threading.Event()
            server0 = ds.DistributedServer(
                py_env_configs=py_env_configs, rank=0, world_size=2
            )
            server0.bootstrap()

            assert len(ds._g_world_info.members) == 2
            assert ds._g_world_info.master is not None
            stop_event.set()


#    @patch("torch.cuda.device_count")
#    @patch.dict(
#        "os.environ",
#        {
#            "TP_SIZE": "2",
#            "PP_SIZE": "1",
#            "WORLD_SIZE": "2",
#            "WORLD_RANK": "0",
#            "LOCAL_WORLD_SIZE": "2",
#            "WORKER_INFO_PORT_NUM": "8",
#            "START_PORT": "20000",
#            "dist_comm_timeout": "3",
#            "GANG_SLEEP_TIME": "1",
#            "FAKE_GANG_ENV": "1",
#            "MODEL_TYPE": "fake_model",
#            "TOKENIZER_PATH": os.path.join(
#                os.getcwd(), "rtp_llm/distribute/test/testdata/tokenizer"
#            ),
#            "CHECKPOINT_PATH": os.path.join(
#                os.getcwd(), "rtp_llm/distribute/test/testdata/cpt"
#            ),
#            "dist_comm_timeout": "10",
#            "CUDA_VISIBLE_DEVICES": "0,1",
#        },
#    )
#    def test_distributed_server_start(self, torch_device_count):
#        StaticConfig.update_from_env()
#        try:
#            multiprocessing.set_start_method("spawn")
#        except RuntimeError as e:
#            logging.warn(str(e))
#
#        torch_device_count.return_value = 2
#        g_parallel_info.reload(MIN_WORKER_INFO_PORT_NUM)
#        g_worker_info.reload()
#        procs: List[Process] = list()
#        StaticConfig.update_from_env()
#        procs = main()
#        time.sleep(100)
# p = Process(target=start_server, args=(1, 2))
# p.start()
# time.sleep(2)


#
# StaticConfig.distribute_config = DistributeConfig()
# StaticConfig.update_from_env()
# py_env = StaticConfig
# g_parallel_info.reload(MIN_WORKER_INFO_PORT_NUM)
# g_worker_info.reload()
# g_parallel_info.world_rank = 0
# g_worker_info.world_rank = 0
# g_worker_info.local_rank = 0
# server0 = ds.DistributedServer(py_env_configs=py_env, rank=0, world_size=2)
# server0.start(py_env_configs=py_env)
#
# assert len(ds._g_world_info.members) == 2
# assert ds._g_world_info.master is not None


if __name__ == "__main__":
    unittest.main()
