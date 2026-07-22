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
    COORDINATOR_INFO_PORT_NUM,
    DistributeConfig,
    PyEnvConfigs,
    ServerConfig,
)
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.ops import NcclCommConfig, ParallelismConfig



@dataclass
class WorldInfo:
    members: List[WorkerInfo]
    master: Optional[WorkerInfo]  # None until bootstrap completes
    self: Optional[WorkerInfo]  # None in default-constructed state
    num_nodes: int
    initialized: bool

    def workers(self) -> List[WorkerInfo]:
        if self.master is None:
            return []
        return [member for member in self.members if not member.equals(self.master)]


def _build_nccl_comm_config(ip: str, base_port: int, dp_rank: int) -> NcclCommConfig:
    """Build NcclCommConfig from ip, base_port, dp_rank (same port layout as former NodeCommInfo)."""
    rank_base_port = base_port - dp_rank * COORDINATOR_INFO_PORT_NUM
    return NcclCommConfig(
        nccl_ip=ip,
        tp_nccl_port=rank_base_port - 2,
        dp_tp_nccl_port=base_port - 10,
        ffn_tp_nccl_port=rank_base_port - 5,
    )


def get_world_info(
    server_config: ServerConfig,
    distribute_config: DistributeConfig,
    parallelism_config: ParallelismConfig,
    distributed_server: Optional["DistributedServer"] = None,
) -> WorldInfo:
    """Get world info. When distributed_server is provided (e.g. from backend), returns
    its member world_info; otherwise returns local/single-rank world info."""
    if distributed_server is not None:
        return distributed_server.get_world_info(
            server_config, distribute_config, parallelism_config
        )

    if parallelism_config.world_size == 1:
        ip = server_config.ip or socket.gethostbyname(socket.gethostname())
        self_info = WorkerInfo(
            ip=ip,
            local_rank=parallelism_config.local_rank,
            world_rank=parallelism_config.world_rank,
            name="",
            server_port=server_config.start_port,
            worker_info_port_num=server_config.worker_info_port_num,
            remote_server_port=distribute_config.remote_server_port,
        )
        return WorldInfo(
            members=[self_info],
            self=self_info,
            master=self_info,
            num_nodes=1,
            initialized=True,
        )

    return get_local_world_info(
        server_config,
        distribute_config,
        parallelism_config,
    )


def get_local_world_info(
    server_config: ServerConfig,
    distribute_config: DistributeConfig,
    parallelism_config: ParallelismConfig,
) -> WorldInfo:
    ip = server_config.ip or socket.gethostbyname(socket.gethostname())
    self_info = WorkerInfo(
        ip=ip,
        local_rank=parallelism_config.local_rank,
        world_rank=parallelism_config.world_rank,
        name="",
        server_port=server_config.start_port,
        worker_info_port_num=server_config.worker_info_port_num,
        remote_server_port=distribute_config.remote_server_port,
    )
    num_nodes = (
        parallelism_config.world_size + parallelism_config.local_world_size - 1
    ) // parallelism_config.local_world_size
    all_members: List[WorkerInfo] = []
    logging.info(
        f"get_local_world_info world_size: {parallelism_config.world_size}, local_world_size: {parallelism_config.local_world_size}"
    )
    for local_rank in range(parallelism_config.local_world_size):
        logging.info(
            f"get_local_world_info local_world_size: {parallelism_config.local_world_size} local_rank: {local_rank}"
        )
        rank = (
            parallelism_config.world_rank
            // parallelism_config.local_world_size
            * parallelism_config.local_world_size
            + local_rank
        )
        new_member = WorkerInfo(
            ip=socket.gethostbyname(socket.gethostname()),
            local_rank=local_rank,
            world_rank=rank,
            name=f"{distribute_config.zone_name}_rank_{rank}_{local_rank}",
            server_port=server_config.start_port,
            worker_info_port_num=server_config.worker_info_port_num,
            remote_server_port=distribute_config.remote_server_port,
        )
        all_members.append(new_member)

    return WorldInfo(
        members=all_members,
        self=self_info,
        master=None,
        num_nodes=num_nodes,
        initialized=True,
    )


class DistributedServer(object):
    """Registry key prefix for rank address in TCPStore."""

    REGISTRY_RANK_ADDRESS_KEY = "registry_rank_address_"

    def __init__(
        self,
        py_env_configs: PyEnvConfigs,
        rank: int = -1,
        world_size: int = -1,
        wait_for_workers=True,
    ):
        server_config = py_env_configs.server_config
        distribute_config = py_env_configs.distribute_config
        pc = py_env_configs.parallelism_config
        ip = server_config.ip or socket.gethostbyname(socket.gethostname())
        self.worker_info = WorkerInfo(
            ip=ip,
            local_rank=pc.local_rank,
            world_rank=pc.world_rank,
            name="",
            server_port=server_config.start_port,
            worker_info_port_num=server_config.worker_info_port_num,
            remote_server_port=distribute_config.remote_server_port,
        )
        logging.info(
            f"init DistributedServer, rank: {pc.world_rank},  size: {pc.world_size}"
        )
        self._world_info = WorldInfo(
            members=[],
            master=None,
            self=self.worker_info,
            num_nodes=(
                (pc.world_size + pc.local_world_size - 1) // pc.local_world_size
                if pc.world_size > 1
                else -1
            ),
            initialized=False,
        )

        if pc.world_size == 1:
            logging.info("world_size == 1, do not start distributed_server")
            self.master_server_port = server_config.start_port
            self._nccl_comm_config = _build_nccl_comm_config(
                self.worker_info.ip, server_config.start_port, pc.dp_rank
            )
            return

        if rank == -1:
            rank = pc.world_rank
        if world_size == -1:
            world_size = pc.world_size
        self._initialized = True
        self.py_env_configs = py_env_configs
        self.rank = rank
        self.world_size = world_size

        self.master_ip, master_server_port = get_master(
            self.py_env_configs.distribute_config,
            self.py_env_configs.parallelism_config,
        )
        if master_server_port == "":
            self.master_server_port = py_env_configs.server_config.start_port
        else:
            self.master_server_port = int(master_server_port)

        self._nccl_comm_config = _build_nccl_comm_config(
            self.master_ip, self.master_server_port, pc.dp_rank
        )

        logging.info(
            f"{pc} init tcpstore " f"{self.master_ip}:{self.master_server_port - 1}"
        )

        init_process_timeout = py_env_configs.distribute_config.dist_comm_timeout
        if init_process_timeout is not None:
            init_process_timeout = timedelta(seconds=init_process_timeout)
        store = TCPStore(
            host_name=self._nccl_comm_config.nccl_ip,
            port=self.master_server_port - 1,
            world_size=world_size,
            is_master=(rank == 0),
            wait_for_workers=wait_for_workers,
            timeout=init_process_timeout,
        )
        logging.info(f"{pc} init tcpstore done")
        self.store = store

    def get_world_info(
        self,
        server_config: ServerConfig,
        distribute_config: DistributeConfig,
        parallelism_config: ParallelismConfig,
    ) -> WorldInfo:
        """Return this server's world_info (filled after start/bootstrap)."""
        if parallelism_config.world_size == 1:
            return WorldInfo(
                members=[self.worker_info],
                self=self.worker_info,
                master=self.worker_info,
                num_nodes=1,
                initialized=True,
            )
        return self._world_info

    def get_nccl_comm_config(self) -> NcclCommConfig:
        """Return the NCCL communication config (ip and ports)."""
        return self._nccl_comm_config

    def get_nccl_init_port(self) -> int:
        """Return the port used for torch.distributed init (base_port - 11)."""
        return self.master_server_port - 11

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
        key = self.REGISTRY_RANK_ADDRESS_KEY + str(self.rank)
        self.safe_store_set(
            key, f"{self.worker_info.ip}:{self.worker_info.server_port}"
        )

    def bootstrap(self) -> None:
        timeout_minutes = self.py_env_configs.distribute_config.gang_timeout_min
        sleep_time = self.py_env_configs.distribute_config.gang_sleep_time

        start_time = datetime.datetime.now()
        retry_time = 0
        while True:
            self.regist()
            members_address: Dict[int, str] = {}

            for i in range(self.world_size):
                key = self.REGISTRY_RANK_ADDRESS_KEY + str(i)
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
                    local_rank = (
                        rank % self.py_env_configs.parallelism_config.local_world_size
                    )
                    # server_port from address is already the base port for this worker
                    new_member = WorkerInfo(
                        ip=ip,
                        local_rank=local_rank,
                        world_rank=rank,
                        name=f"{self.py_env_configs.distribute_config.zone_name}_rank_{rank}_{local_rank}",
                        server_port=server_port,
                        worker_info_port_num=0,
                        remote_server_port=self.py_env_configs.distribute_config.remote_server_port,
                    )
                    self._world_info.members.append(new_member)
                    if rank == 0:
                        self._world_info.master = new_member
                self._world_info.initialized = True
                setattr(self._world_info, "bootstrap", True)
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
        pc = py_env_configs.parallelism_config
        logging.info(
            f"DistributedServer start, rank: {pc.world_rank},  size: {pc.world_size}"
        )
        if pc.world_size == 1:
            return
        self.bootstrap()

        master_url = (
            f"tcp://{self._nccl_comm_config.nccl_ip}:{self.master_server_port - 1}"
        )
        logging.info(
            f"DistributedServer bootstrap done, rank: {pc.world_rank},  size: {pc.world_size}, master {master_url}"
        )
        logging.info(
            f"DistributedServer started, rank: {pc.world_rank},  size: {pc.world_size}, master {master_url}"
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


def get_master(
    distribute_config: DistributeConfig,
    parallelism_config: ParallelismConfig,
) -> (str, str):
    port = ""
    if parallelism_config.local_world_size < parallelism_config.world_size:
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
        address = socket.gethostbyname(socket.gethostname())
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
        port = int(member_info["port"])
        members.append(
            WorkerInfo(
                ip=member_info["ip"],
                local_rank=0,
                world_rank=0,
                name=member_info["name"],
                server_port=port,
                worker_info_port_num=0,
                remote_server_port=port,
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
