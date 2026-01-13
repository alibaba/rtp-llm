
import socket
import threading
import time
import unittest
from unittest import TestCase
from typing import Any, Dict

import torch

import rtp_llm.distribute.distributed_server as ds
import rtp_llm.distribute.worker_info as wi
from rtp_llm.config.py_config_modules import (
    PyEnvConfigs,
    ServerConfig,
    DistributeConfig,
)
from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.distribute.worker_info import (
    ParallelInfo,
    WorkerInfo,
    MIN_WORKER_INFO_PORT_NUM,
)
from pytest import mark

_ORIG_TORCH_CUDA_SET_DEVICE = torch.cuda.set_device


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


def create_py_env_configs(
    tp_size: int = 1,
    pp_size: int = 1,
    world_size: int = 1,
    world_rank: int = 0,
    local_world_size: int = 1,
    start_port: int = 8088,
    worker_info_port_num: int = MIN_WORKER_INFO_PORT_NUM,
    remote_server_port: int = 0,
    distribute_config_file: str = "",
    gang_config_string: str = "",
    leader_address: str = "",
    gang_annocation_path: str = "",
) -> PyEnvConfigs:
    """Create PyEnvConfigs with specified parameters."""
    py_env_configs = PyEnvConfigs()
    
    # Setup ServerConfig
    py_env_configs.server_config.start_port = start_port
    py_env_configs.server_config.worker_info_port_num = worker_info_port_num
    
    # Setup DistributeConfig
    py_env_configs.distribute_config.distribute_config_file = distribute_config_file
    py_env_configs.distribute_config.gang_config_string = gang_config_string if gang_config_string else None
    py_env_configs.distribute_config.leader_address = leader_address if leader_address else None
    py_env_configs.distribute_config.gang_annocation_path = gang_annocation_path
    py_env_configs.distribute_config.remote_server_port = remote_server_port
    
    return py_env_configs


def setup_parallel_info(
    tp_size: int = 1,
    pp_size: int = 1,
    world_size: int = 1,
    world_rank: int = 0,
    local_world_size: int = 1,
    worker_info_port_num: int = MIN_WORKER_INFO_PORT_NUM,
):
    """Setup g_parallel_info with specified parameters."""
    wi.g_parallel_info = ParallelInfo(
        tp_size=tp_size,
        ep_size=world_size,  # Default ep_size to world_size
        pp_size=pp_size,
        dp_size=world_size // (tp_size * pp_size) if tp_size * pp_size > 0 else 1,
        ffn_sp_size=1,
        world_size=world_size,
        world_rank=world_rank,
        local_world_size=local_world_size,
        worker_info_port_num=worker_info_port_num,
    )
    # Keep distributed_server module's cached reference consistent.
    ds.g_parallel_info = wi.g_parallel_info


def setup_worker_info(
    start_port: int = 8088,
    remote_server_port: int = 0,
    local_rank: int = 0,
    world_rank: int = 0,
):
    """Setup g_worker_info with specified parameters."""
    worker_info_port_num = wi.g_parallel_info.worker_info_port_num
    
    wi.g_worker_info = WorkerInfo(
        ip=socket.gethostbyname(socket.gethostname()),
        server_port=WorkerInfo.server_port_offset(
            local_rank, start_port, worker_info_port_num
        ),
        gang_hb_port=WorkerInfo.gang_hb_port_offset(
            local_rank, start_port, worker_info_port_num
        ),
        http_port=WorkerInfo.http_port_offset(
            local_rank, start_port, worker_info_port_num
        ),
        rpc_server_port=WorkerInfo.rpc_server_port_offset(
            local_rank, start_port, worker_info_port_num
        ),
        embedding_rpc_server_port=WorkerInfo.embedding_rpc_server_port_offset(
            local_rank, start_port, worker_info_port_num
        ),
        remote_rpc_server_port=WorkerInfo.rpc_server_port_offset(
            local_rank, remote_server_port, worker_info_port_num
        ),
        cache_store_listen_port=WorkerInfo.cache_store_listen_port_offset(
            local_rank, start_port, worker_info_port_num
        ),
        cache_store_connect_port=WorkerInfo.cache_store_listen_port_offset(
            local_rank, remote_server_port, worker_info_port_num
        ),
        cache_store_rdma_listen_port=WorkerInfo.cache_store_rdma_listen_port_offset(
            local_rank, start_port, worker_info_port_num
        ),
        cache_store_rdma_connect_port=WorkerInfo.cache_store_rdma_listen_port_offset(
            local_rank, remote_server_port, worker_info_port_num
        ),
        backend_server_port=WorkerInfo.backend_server_port_offset(
            local_rank, start_port, worker_info_port_num
        ),
        local_rank=local_rank,
        world_rank=world_rank,
        name="",
        info=None,
    )
    # Keep distributed_server module's cached reference consistent.
    ds.g_worker_info = wi.g_worker_info


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
    wi.g_parallel_info.world_rank = rank
    wi.g_worker_info.world_rank = rank
    wi.g_worker_info.local_rank = rank
    server.regist()
    while not stop_event.is_set():
        time.sleep(1)


def start_server(
    py_env_configs: PyEnvConfigs,
    rank: int,
    world_size: int,
):
    server = ds.DistributedServer(py_env_configs, rank, world_size)
    wi.g_parallel_info.world_rank = rank
    wi.g_worker_info.world_rank = rank
    wi.g_worker_info.local_rank = rank
    server.start(py_env_configs)
    while True:
        time.sleep(1)


def _reset_world_info() -> None:
    """Reset distributed_server module global cache between tests."""
    try:
        ds._g_world_info.members.clear()
        ds._g_world_info.master = None
        ds._g_world_info.self = None
        ds._g_world_info.num_nodes = -1
        ds._g_world_info.initialized = False
        if hasattr(ds._g_world_info, "bootstrap"):
            delattr(ds._g_world_info, "bootstrap")
    except Exception:
        pass


class TestGetWorldInfo(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Preserve global singletons to avoid impacting other test modules.
        orig_wi_parallel = wi.g_parallel_info
        orig_wi_worker = wi.g_worker_info
        orig_ds_parallel = ds.g_parallel_info
        orig_ds_worker = ds.g_worker_info
        orig_world_state = {
            "members": list(ds._g_world_info.members),
            "master": ds._g_world_info.master,
            "self": ds._g_world_info.self,
            "num_nodes": ds._g_world_info.num_nodes,
            "initialized": ds._g_world_info.initialized,
            "has_bootstrap": hasattr(ds._g_world_info, "bootstrap"),
            "bootstrap": getattr(ds._g_world_info, "bootstrap", None),
        }

        def _restore_globals() -> None:
            wi.g_parallel_info = orig_wi_parallel
            wi.g_worker_info = orig_wi_worker
            ds.g_parallel_info = orig_ds_parallel
            ds.g_worker_info = orig_ds_worker
            ds._g_world_info.members = orig_world_state["members"]
            ds._g_world_info.master = orig_world_state["master"]
            ds._g_world_info.self = orig_world_state["self"]
            ds._g_world_info.num_nodes = orig_world_state["num_nodes"]
            ds._g_world_info.initialized = orig_world_state["initialized"]
            if orig_world_state["has_bootstrap"]:
                setattr(ds._g_world_info, "bootstrap", orig_world_state["bootstrap"])
            elif hasattr(ds._g_world_info, "bootstrap"):
                delattr(ds._g_world_info, "bootstrap")

        self.addCleanup(_restore_globals)

        torch.cuda.set_device = lambda x: None
        self.addCleanup(
            lambda: setattr(torch.cuda, "set_device", _ORIG_TORCH_CUDA_SET_DEVICE)
        )
        _reset_world_info()

    def test_single_node(self):
        # Setup parallel info
        setup_parallel_info(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=2,
            worker_info_port_num=MIN_WORKER_INFO_PORT_NUM,
        )
        
        # Setup worker info
        setup_worker_info(
            start_port=20000,
            remote_server_port=0,
            local_rank=0,
            world_rank=0,
        )
        
        # Create configs
        py_env_configs = create_py_env_configs(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=2,
            start_port=20000,
        )

        world_info = get_world_info(
            py_env_configs.server_config, py_env_configs.distribute_config
        )
        self.assertTrue(world_info.initialized)
        self.assertEqual(len(world_info.members), 2)
        self.assertEqual(world_info.members[0].server_port, 20000)
        self.assertEqual(world_info.members[1].server_port, 20008)


@mark.H20
@mark.gpu(count=2)
@mark.cuda
class DistributedServerTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        self.maxDiff = None
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        super().setUp()
        # Preserve global singletons to avoid impacting other test modules.
        orig_wi_parallel = wi.g_parallel_info
        orig_wi_worker = wi.g_worker_info
        orig_ds_parallel = ds.g_parallel_info
        orig_ds_worker = ds.g_worker_info
        orig_world_state = {
            "members": list(ds._g_world_info.members),
            "master": ds._g_world_info.master,
            "self": ds._g_world_info.self,
            "num_nodes": ds._g_world_info.num_nodes,
            "initialized": ds._g_world_info.initialized,
            "has_bootstrap": hasattr(ds._g_world_info, "bootstrap"),
            "bootstrap": getattr(ds._g_world_info, "bootstrap", None),
        }

        def _restore_globals() -> None:
            wi.g_parallel_info = orig_wi_parallel
            wi.g_worker_info = orig_wi_worker
            ds.g_parallel_info = orig_ds_parallel
            ds.g_worker_info = orig_ds_worker
            ds._g_world_info.members = orig_world_state["members"]
            ds._g_world_info.master = orig_world_state["master"]
            ds._g_world_info.self = orig_world_state["self"]
            ds._g_world_info.num_nodes = orig_world_state["num_nodes"]
            ds._g_world_info.initialized = orig_world_state["initialized"]
            if orig_world_state["has_bootstrap"]:
                setattr(ds._g_world_info, "bootstrap", orig_world_state["bootstrap"])
            elif hasattr(ds._g_world_info, "bootstrap"):
                delattr(ds._g_world_info, "bootstrap")

        self.addCleanup(_restore_globals)

        torch.cuda.set_device = lambda x: None
        self.addCleanup(
            lambda: setattr(torch.cuda, "set_device", _ORIG_TORCH_CUDA_SET_DEVICE)
        )
        _reset_world_info()

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

    def test_get_master_use_distribute_config_file(self):
        setup_parallel_info(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
        )
        setup_worker_info(
            start_port=8088,
            remote_server_port=0,
            local_rank=0,
            world_rank=0,
        )
        
        py_env_configs = create_py_env_configs(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
            distribute_config_file="rtp_llm/distribute/test/testdata/parallel.json",
        )

        ip, port = ds.get_master(py_env_configs.distribute_config)
        assert ip == "11.161.48.116"
        assert port == "10000"

    def test_get_master_use_gang_config_string(self):
        setup_parallel_info(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
        )
        setup_worker_info(
            start_port=8088,
            remote_server_port=0,
            local_rank=0,
            world_rank=0,
        )
        
        py_env_configs = create_py_env_configs(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
            gang_config_string="name:worker_part0,ip:10.0.0.123",
        )
        
        ip, port = ds.get_master(py_env_configs.distribute_config)
        assert ip == "10.0.0.123"
        assert port == ""

    def test_get_master_use_leader_address(self):
        setup_parallel_info(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
        )
        setup_worker_info(
            start_port=8088,
            remote_server_port=0,
            local_rank=0,
            world_rank=0,
        )
        
        py_env_configs = create_py_env_configs(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
            leader_address="10.0.0.5",
        )

        ip, port = ds.get_master(py_env_configs.distribute_config)
        assert ip == "10.0.0.5"
        assert port == ""

    def test_get_master_use_c2_file(self):
        setup_parallel_info(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
        )
        setup_worker_info(
            start_port=8088,
            remote_server_port=0,
            local_rank=0,
            world_rank=0,
        )
        
        py_env_configs = create_py_env_configs(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
            gang_annocation_path="rtp_llm/distribute/test/testdata/annocation",
        )
        
        ip, port = ds.get_master(py_env_configs.distribute_config)
        # 具体 IP 取决于 annocation 文件的内容，这里只检查非空
        assert isinstance(ip, str)
        assert ip == "33.115.125.211"
        assert port == ""

    def test_get_master_single_machine(self):
        setup_parallel_info(
            tp_size=1,
            pp_size=1,
            world_size=1,
            world_rank=0,
            local_world_size=1,
        )
        setup_worker_info(
            start_port=8088,
            remote_server_port=0,
            local_rank=0,
            world_rank=0,
        )
        
        py_env_configs = create_py_env_configs(
            tp_size=1,
            pp_size=1,
            world_size=1,
            world_rank=0,
            local_world_size=1,
        )

        ip, port = ds.get_master(py_env_configs.distribute_config)
        assert ip == wi.g_worker_info.ip
        assert port == ""

    def test_get_master_from_file(self):
        # Setup parallel info for multi-node scenario (local_world_size < world_size)
        setup_parallel_info(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
        )
        setup_worker_info(
            start_port=8088,
            remote_server_port=0,
            local_rank=0,
            world_rank=0,
        )
        
        py_env_configs = create_py_env_configs(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
            distribute_config_file="rtp_llm/distribute/test/testdata/parallel.json",
        )
        
        ip, port = ds.get_master_from_file(py_env_configs.distribute_config)
        assert ip == "11.161.48.116"
        assert port == "10000"

    def test_get_master_from_c2(self):
        # Setup parallel info for multi-node scenario (local_world_size < world_size)
        setup_parallel_info(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
        )
        setup_worker_info(
            start_port=8088,
            remote_server_port=0,
            local_rank=0,
            world_rank=0,
        )
        
        py_env_configs = create_py_env_configs(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=1,
            gang_annocation_path="rtp_llm/distribute/test/testdata/annocation",
        )
        
        ip, port = ds.get_master_from_c2(py_env_configs.distribute_config)
        assert ip == "33.115.125.211"
        assert port == ""

    def test_distributed_server_safe_store_set_get(self):
        setup_parallel_info(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=2,
            worker_info_port_num=MIN_WORKER_INFO_PORT_NUM,
        )
        setup_worker_info(
            start_port=20000,
            remote_server_port=0,
            local_rank=0,
            world_rank=0,
        )
        
        py_env_configs = create_py_env_configs(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=2,
            start_port=20000,
            worker_info_port_num=MIN_WORKER_INFO_PORT_NUM,
        )
        
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

    def test_distributed_server_regist_and_bootstrap(self):
        setup_parallel_info(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=2,
            worker_info_port_num=MIN_WORKER_INFO_PORT_NUM,
        )
        setup_worker_info(
            start_port=20000,
            remote_server_port=0,
            local_rank=0,
            world_rank=0,
        )
        
        py_env_configs = create_py_env_configs(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=2,
            start_port=20000,
            worker_info_port_num=MIN_WORKER_INFO_PORT_NUM,
        )
        py_env_configs.distribute_config.gang_timeout_min = 1
        py_env_configs.distribute_config.gang_sleep_time = 0

        # rank1
        stop_event = threading.Event()
        t = threading.Thread(
            target=regist_server, args=(py_env_configs, 1, 2, stop_event)
        )
        t.start()

        # rank0
        setup_parallel_info(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=2,
            worker_info_port_num=MIN_WORKER_INFO_PORT_NUM,
        )
        setup_worker_info(
            start_port=20000,
            remote_server_port=0,
            local_rank=0,
            world_rank=0,
        )
        
        py_env_configs = create_py_env_configs(
            tp_size=2,
            pp_size=1,
            world_size=2,
            world_rank=0,
            local_world_size=2,
            start_port=20000,
            worker_info_port_num=MIN_WORKER_INFO_PORT_NUM,
        )
        py_env_configs.distribute_config.gang_timeout_min = 1
        py_env_configs.distribute_config.gang_sleep_time = 0
        
        server0 = ds.DistributedServer(
            py_env_configs=py_env_configs, rank=0, world_size=2
        )
        server0.bootstrap()

        assert len(ds._g_world_info.members) == 2
        assert ds._g_world_info.master is not None
        stop_event.set()


if __name__ == "__main__":
    unittest.main()
