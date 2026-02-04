import datetime
import ipaddress
import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, NamedTuple, Optional

from torch.distributed import TCPStore

from rtp_llm.config.py_config_modules import (
    DistributeConfig,
    PyEnvConfigs,
    ServerConfig,
)
from rtp_llm.distribute.worker_info import (
    MasterInfo,
    WorkerInfo,
    g_parallel_info,
    g_worker_info,
)

@dataclass
class WorldInfo:
    members: List[WorkerInfo]
    master: WorkerInfo
    self: WorkerInfo
    num_nodes: int
    initialized: bool

    def workers(self) -> List[WorkerInfo]:
        return [member for member in self.members if not member.equals(self.master)]


# 全局 WorldInfo 缓存，可选：也可以只提供 get_world_info()
_g_world_info = WorldInfo(
    members=[],
    master=None,
    self=None,
    num_nodes=-1,
    initialized=False,
)

_registry_rank_address_key = "registry_rank_address_"


def get_world_info(
    server_config: ServerConfig,
    distribute_config: DistributeConfig,
) -> WorldInfo:
    global _g_world_info
    if _g_world_info is None:
        raise RuntimeError(
            "WorldInfo has not been initialized yet. "
            "Call start() after all ranks are ready."
        )

    if g_parallel_info.world_size == 1:
        return WorldInfo(
            members=[g_worker_info],
            self=g_worker_info,
            master=g_worker_info,
            num_nodes=1,
            initialized=True,
        )

    # frontend 获取本机信息
    if len(_g_world_info.members) == 0 and not _g_world_info.initialized:
        return get_local_world_info(server_config, distribute_config)

    return _g_world_info


def get_local_world_info(
    server_config: ServerConfig,
    distribute_config: DistributeConfig,
) -> WorldInfo:
    num_nodes = (
        g_parallel_info.world_size + g_parallel_info.local_world_size - 1
    ) // g_parallel_info.local_world_size
    all_members: List[WorkerInfo] = []
    for local_rank in range(g_parallel_info.local_world_size):
        logging.info(
            f"get_local_world_info local_world_size: {g_parallel_info.local_world_size} local_rank: {local_rank}"
        )
        rank = (
            g_parallel_info.world_rank
            // g_parallel_info.local_world_size
            * g_parallel_info.local_world_size
            + local_rank
        )
        new_member = WorkerInfo(
            ip=g_worker_info.ip,
            server_port=WorkerInfo.server_port_offset(
                local_rank, server_config.start_port, server_config.worker_info_port_num
            ),
            gang_hb_port=WorkerInfo.gang_hb_port_offset(
                local_rank, server_config.start_port, server_config.worker_info_port_num
            ),
            http_port=WorkerInfo.http_port_offset(
                local_rank, server_config.start_port, server_config.worker_info_port_num
            ),
            rpc_server_port=WorkerInfo.rpc_server_port_offset(
                local_rank, server_config.start_port, server_config.worker_info_port_num
            ),
            backend_server_port=WorkerInfo.backend_server_port_offset(
                local_rank, server_config.start_port, server_config.worker_info_port_num
            ),
            cache_store_listen_port=WorkerInfo.cache_store_listen_port_offset(
                local_rank, server_config.start_port, server_config.worker_info_port_num
            ),
            embedding_rpc_server_port=WorkerInfo.embedding_rpc_server_port_offset(
                local_rank, server_config.start_port, server_config.worker_info_port_num
            ),
            cache_store_rdma_listen_port=WorkerInfo.cache_store_rdma_listen_port_offset(
                local_rank, server_config.start_port, server_config.worker_info_port_num
            ),
            remote_rpc_server_port=WorkerInfo.rpc_server_port_offset(
                local_rank, distribute_config.remote_server_port
            ),
            cache_store_connect_port=WorkerInfo.cache_store_listen_port_offset(
                local_rank, distribute_config.remote_server_port
            ),
            cache_store_rdma_connect_port=WorkerInfo.cache_store_rdma_listen_port_offset(
                local_rank, distribute_config.remote_server_port
            ),
            info=None,
            local_rank=local_rank,
            name=f"{distribute_config.zone_name}_rank_{rank}_{local_rank}",
            world_rank=rank,
        )
        all_members.append(new_member)

    return WorldInfo(
        members=all_members,
        self=g_worker_info,
        master=None,
        num_nodes=num_nodes,
        initialized=True,
    )


class DistributedServer(object):
    def __init__(
        self,
        py_env_configs: PyEnvConfigs,
        rank: int = -1,
        world_size: int = -1,
        wait_for_workers=True,
    ):
        logging.info(
            f"init DistributedServer, rank: {g_parallel_info.world_rank},  size: {g_parallel_info.world_size}"
        )
        global _g_world_info
        if _g_world_info is not None:
            _g_world_info.self = g_worker_info
            _g_world_info.num_nodes = (
                g_parallel_info.world_size + g_parallel_info.local_world_size - 1
            ) // g_parallel_info.local_world_size

        if g_parallel_info.world_size == 1:
            logging.info("world_size == 1, do not start distributed_server")
            self.master_info = MasterInfo(
                ip=g_worker_info.ip,
                base_port=py_env_configs.server_config.start_port,
                dp_rank=g_parallel_info.dp_rank,
            )
            return

        if rank == -1:
            rank = g_parallel_info.world_rank
        if world_size == -1:
            world_size = g_parallel_info.world_size
        self._initialized = True
        self.py_env_configs = py_env_configs
        self.rank = rank
        self.world_size = world_size

        self.master_ip, master_server_port = get_master(
            self.py_env_configs.distribute_config
        )
        if master_server_port == "":
            self.master_server_port = WorkerInfo.server_port_offset(
                local_rank=0, server_port=py_env_configs.server_config.start_port
            )
        else:
            self.master_server_port = int(master_server_port)

        self.master_info = MasterInfo(
            ip=self.master_ip,
            base_port=self.master_server_port,
            dp_rank=g_parallel_info.dp_rank,
        )

        logging.info(
            f"{g_parallel_info} init tcpstore "
            f"{self.master_ip}:{self.master_server_port - 1}"
        )

        init_process_timeout = py_env_configs.distribute_config.dist_comm_timeout
        if init_process_timeout is not None:
            init_process_timeout = timedelta(seconds=init_process_timeout)
        store = TCPStore(
            host_name=self.master_info.ip,
            port=self.master_server_port - 1,
            world_size=world_size,
            is_master=(rank == 0),
            wait_for_workers=wait_for_workers,
            timeout=init_process_timeout,
        )
        logging.info(f"{g_parallel_info} init tcpstore done")
        self.store = store

    def get_master_info(self) -> MasterInfo:
        """Return the master NCCL/connection info (ip and base_port-derived ports)."""
        return self.master_info

    def safe_store_set(self, key: str, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("Value must be a string for safe serialization")

        try:
            self.store.set(key, value)
            logging.debug(f"Set key '{key}' successfully")
        except RuntimeError as e:
            logging.error(f"Failed to set key '{key}': {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error when setting key '{key}': {e}")
            raise

    def safe_store_get(self, key: str, encoding: str = "utf-8") -> str:
        try:
            value_bytes = self.store.get(key)  # 阻塞直到 key 出现或超时
            try:
                value_str = value_bytes.decode(encoding)
                return value_str
            except UnicodeDecodeError as e:
                logging.error(f"Failed to decode value for key '{key}': {e}")
                return ""
        except RuntimeError as e:
            # 通常是 timeout 或 connection error
            logging.error(f"Failed to get key '{key}': {e}")
            return ""
        except Exception as e:
            logging.error(f"Unexpected error when getting key '{key}': {e}")
            return ""

    def regist(self) -> None:
        key = _registry_rank_address_key + str(self.rank)
        self.safe_store_set(key, f"{g_worker_info.ip}:{g_worker_info.server_port}")

    def bootstrap(self) -> None:
        timeout_minutes = self.py_env_configs.distribute_config.gang_timeout_min
        sleep_time = self.py_env_configs.distribute_config.gang_sleep_time

        start_time = datetime.datetime.now()
        retry_time = 0
        global _g_world_info
        while True:
            self.regist()
            members_address: Dict[int, str] = {}

            for i in range(self.world_size):
                key = _registry_rank_address_key + str(i)
                address = self.safe_store_get(key)
                if not address:
                    logging.info(
                        "get regist members failed, address is empty, "
                        f"key {key}, retry_time {retry_time}, "
                        f"start_time {start_time}"
                    )
                    continue
                members_address[i] = address
                logging.info(f"get rank {i} address: {address}")

            if len(members_address) == self.world_size:
                for i in range(self.world_size):
                    rank = i
                    ip, server_port = split_ip_port(members_address[i])
                    if ip == "":
                        raise Exception(
                            f"rank {rank} error address: {members_address[i]}"
                        )
                    local_rank = rank % g_parallel_info.local_world_size
                    new_member = WorkerInfo(
                        ip=ip,
                        server_port=WorkerInfo.server_port_offset(
                            server_port=server_port
                        ),
                        gang_hb_port=WorkerInfo.gang_hb_port_offset(
                            server_port=server_port
                        ),
                        http_port=WorkerInfo.http_port_offset(server_port=server_port),
                        rpc_server_port=WorkerInfo.rpc_server_port_offset(
                            server_port=server_port
                        ),
                        embedding_rpc_server_port=WorkerInfo.embedding_rpc_server_port_offset(
                            server_port=server_port
                        ),
                        backend_server_port=WorkerInfo.backend_server_port_offset(
                            server_port=server_port
                        ),
                        cache_store_listen_port=WorkerInfo.cache_store_listen_port_offset(
                            server_port=server_port
                        ),
                        cache_store_rdma_listen_port=WorkerInfo.cache_store_rdma_listen_port_offset(
                            server_port=server_port
                        ),
                        remote_rpc_server_port=WorkerInfo.rpc_server_port_offset(
                            self.py_env_configs.distribute_config.remote_server_port
                        ),
                        cache_store_connect_port=WorkerInfo.cache_store_listen_port_offset(
                            self.py_env_configs.distribute_config.remote_server_port
                        ),
                        cache_store_rdma_connect_port=WorkerInfo.cache_store_rdma_listen_port_offset(
                            self.py_env_configs.distribute_config.remote_server_port
                        ),
                        info=None,
                        local_rank=local_rank,
                        name=f"{self.py_env_configs.distribute_config.zone_name}_rank_{rank}_{local_rank}",
                        world_rank=rank,
                    )
                    _g_world_info.members.append(new_member)
                    if rank == 0:
                        _g_world_info.master = new_member
                _g_world_info.bootstrap = True
                return

            cur_time = datetime.datetime.now()
            if cur_time - start_time > datetime.timedelta(minutes=timeout_minutes):
                raise Exception(
                    "regist members failed, timeout "
                    f"{timeout_minutes} minutes, retry_time {retry_time}, "
                    f"start_time {start_time}"
                )

            retry_time += 1
            time.sleep(sleep_time)

    def start(self, py_env_configs: PyEnvConfigs) -> None:
        logging.info(
            f"DistributedServer start, rank: {g_parallel_info.world_rank},  size: {g_parallel_info.world_size}"
        )
        if g_parallel_info.world_size == 1:
            return
        self.bootstrap()

        master_url = f"tcp://{self.master_info.ip}:{self.master_server_port - 1}"
        logging.info(
            f"DistributedServer bootstrap done, rank: {g_parallel_info.world_rank},  size: {g_parallel_info.world_size}, master {master_url}"
        )
        logging.info(
            f"DistributedServer started, rank: {g_parallel_info.world_rank},  size: {g_parallel_info.world_size}, master {master_url}"
        )


def get_ip(leader_address: str) -> str:
    try:
        ipaddress.ip_address(leader_address)
        return leader_address
    except Exception:
        return socket.gethostbyname(leader_address)


def split_ip_port(addr: str):
    addr = addr.strip()
    if not addr:
        return "", 0

    if ":" not in addr:
        logging.warning(f"error address: {addr}")
        return "", 0

    ip, port_str = addr.rsplit(":", 1)
    if not ip:
        logging.warning(f"error address: {addr}")
        return "", 0

    if not port_str.isdigit():
        logging.warning(f"error address: {addr}")
        return "", 0

    port = int(port_str)
    return ip, port


def get_master(distribute_config) -> (str, str):
    port = ""
    if g_parallel_info.local_world_size < g_parallel_info.world_size:
        # from config file
        if distribute_config.distribute_config_file:
            address, port = get_master_from_file(distribute_config)
            logging.info(
                f"get master address from distribute_config_file {address} {distribute_config.distribute_config_file}"
            )
        # for distributed test
        elif distribute_config.gang_config_string:
            address, port = get_master_from_test_env(
                distribute_config.gang_config_string
            )
            logging.info(
                f"get master address from GANG_CONFIG_STRING {address} {distribute_config.gang_config_string}"
            )
        # for lws
        elif distribute_config.leader_address:
            address = distribute_config.leader_address
            logging.info(f"get master address from LEADER_ADDRESS {address}")
        # from c2 annotation
        else:
            address, port = get_master_from_c2(distribute_config)
    else:
        # 单机/特殊分布式场景，这里原先逻辑没有 else 分支，保持空值或自行约定
        address = g_worker_info.ip
        logging.info(f"no other workers, leader is self: {address}")

    return get_ip(address), port


def get_master_from_json(gang_info_json: Dict[str, Any]) -> (str, str):
    # here is only the fake ip
    for name, info in gang_info_json.items():
        if name.endswith("part0"):
            port = info.get("port", 0)
            port_str = str(port) if port else ""
            return info["ip"], port_str
    return "", ""


def get_master_from_test_env(env_str: str) -> (str, str):
    for member_str in env_str.split(";"):
        member_info: Dict[str, str] = {}
        for item in member_str.split(","):
            key, value = item.split(":")
            member_info[key] = value

        if member_info.get("name", "").endswith("part0"):
            return member_info["ip"], member_info.get("port", "")
    return "", ""


def get_master_from_file(distribute_config) -> (str, str):
    file = distribute_config.distribute_config_file
    with open(file, "r") as reader:
        config_json = json.loads(reader.read())
    return get_master_from_json(config_json)


def get_master_from_c2(distribute_config) -> (str, str):
    file_name = distribute_config.gang_annocation_path
    if not os.path.exists(file_name):
        raise Exception(f"not found file: {file_name}")
    with open(file_name, "r") as reader:
        content = reader.read()
    infos = [x for x in content.split("\n") if "app.c2.io/biz-detail-ganginfo" in x]
    if len(infos) != 1:
        raise Exception("ganginfo length is not equal to 1, " f"actual: {infos}")
    gang_info = infos[0].replace("\\", "")
    logging.info(f"gang info: {gang_info[gang_info.index('=') + 2: -1]}")
    gang_info_json = json.loads(gang_info[gang_info.index("=") + 2 : -1])
    logging.info(f"gang info json: {gang_info_json}")
    return get_master_from_json(gang_info_json)


"""
test env example:
name:smoke_part0,ip:127.0.0.1,port:13045;name:smoke_part1,ip:127.0.0.1,port:12053
"""


def members_from_test_env(env_str: str) -> List[WorkerInfo]:
    members: List[WorkerInfo] = []
    for member_str in env_str.split(";"):
        member_info = {}
        for item in member_str.split(","):
            key, value = item.split(":")
            member_info[key] = value
        members.append(
            WorkerInfo(
                server_port=int(member_info["port"]),
                gang_hb_port=-1,
                http_port=-1,
                rpc_server_port=-1,
                backend_server_port=-1,
                remote_rpc_server_port=-1,
                cache_store_listen_port=-1,
                cache_store_connect_port=-1,
                cache_store_rdma_connect_port=-1,
                cache_store_rdma_listen_port=-1,
                embedding_rpc_server_port=-1,
                local_rank=0,
                world_rank=0,
                name=member_info["name"],
                ip=member_info["ip"],
                info=member_info,
            )
        )
    masters = [member for member in members if member.name.endswith("part0")]
    if len(masters) != 1:
        raise Exception(f"gang master should contains 1 but got {len(masters)}")
    sorted_members = sorted(members, key=lambda x: x.name)
    if masters[0].name != sorted_members[0].name:
        raise Exception(
            f"gang master should be the first one but got {sorted_members[0].name}"
        )
    return sorted_members
