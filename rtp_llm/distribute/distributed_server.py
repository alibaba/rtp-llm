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

from rtp_llm.config.py_config_modules import DistributeConfig, ServerConfig
from rtp_llm.distribute.worker_info import WorkerInfo


@dataclass
class WorldInfo:
    members: List[WorkerInfo]
    master: WorkerInfo
    self: WorkerInfo
    num_nodes: int
    initialized: bool

    def workers(self) -> List[WorkerInfo]:
        """Get all worker members (excluding master)."""
        return [
            member
            for member in self.members
            if member.ip != self.master.ip
            or member.server_port != self.master.server_port
        ]


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
    parallelism_config,
    worker_info: Optional[WorkerInfo] = None,
) -> WorldInfo:
    """Get WorldInfo for the current process.

    Args:
        server_config: Server configuration
        distribute_config: Distribution configuration
        parallelism_config: Parallelism configuration
        worker_info: Optional current worker info. If not provided, will be inferred
                     from parallelism_config and server_config.

    Returns:
        WorldInfo containing all members and current worker info.
    """
    global _g_world_info
    if _g_world_info is None:
        raise RuntimeError(
            "WorldInfo has not been initialized yet. "
            "Call start() after all ranks are ready."
        )
    if parallelism_config.world_size == 1:
        # For single worker, create worker_info if not provided
        if worker_info is None:
            worker_info = WorkerInfo.from_parallelism_config(
                parallelism_config,
                server_config.start_port,
                distribute_config.remote_server_port,
                server_config.worker_info_port_num,
            )
        return WorldInfo(
            members=[worker_info],
            self=worker_info,
            master=worker_info,
            num_nodes=1,
            initialized=True,
        )

    # frontend 获取本机信息
    if len(_g_world_info.members) == 0 and not _g_world_info.initialized:
        return get_local_world_info(
            server_config, distribute_config, parallelism_config, worker_info
        )

    return _g_world_info


def get_local_world_info(
    server_config: ServerConfig,
    distribute_config: DistributeConfig,
    parallelism_config,
    worker_info: Optional[WorkerInfo] = None,
) -> WorldInfo:
    """Get local world info by creating WorkerInfo for all local ranks.

    Args:
        server_config: Server configuration
        distribute_config: Distribution configuration
        parallelism_config: Parallelism configuration
        worker_info: Optional current worker info. If provided, will be used to
                    determine the current worker from created members. If not provided,
                    will use parallelism_config.local_rank to find the current worker.

    Returns:
        WorldInfo containing all local members and current worker info.
    """
    if parallelism_config is None:
        raise ValueError("parallelism_config must be provided to get_local_world_info")

    # Get IP address - use worker_info.ip if provided, otherwise get from hostname
    if worker_info is not None:
        local_ip = worker_info.ip
    else:
        local_ip = socket.gethostbyname(socket.gethostname())

    num_nodes = (
        parallelism_config.world_size + parallelism_config.local_world_size - 1
    ) // parallelism_config.local_world_size
    all_members: List[WorkerInfo] = []
    current_worker: Optional[WorkerInfo] = None

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
            ip=local_ip,
            local_rank=local_rank,
            world_rank=rank,
            local_world_size=parallelism_config.local_world_size,
            start_port=server_config.start_port,
            remote_server_port=distribute_config.remote_server_port,
            worker_info_port_num=server_config.worker_info_port_num,
        )
        all_members.append(new_member)

        # Determine current worker: match by local_rank or world_rank
        if worker_info is not None:
            if worker_info.local_rank == local_rank or worker_info.world_rank == rank:
                current_worker = new_member
        elif local_rank == parallelism_config.local_rank:
            current_worker = new_member

    # Fallback: if no match found, use the first member
    # This ensures current_worker is always in all_members
    if current_worker is None:
        if all_members:
            current_worker = all_members[0]
            logging.warning(
                f"Could not match worker_info to any member, using first member "
                f"(local_rank={current_worker.local_rank}, world_rank={current_worker.world_rank})"
            )
        else:
            raise RuntimeError("Failed to determine current worker: no members created")

    return WorldInfo(
        members=all_members,
        self=current_worker,
        master=None,
        num_nodes=num_nodes,
        initialized=True,
    )


class DistributedServer(object):
    def __init__(
        self,
        parallelism_config,
        distribute_config: DistributeConfig,
        start_port: int,
        worker_info: WorkerInfo,
        rank: int = -1,
        world_size: int = -1,
        wait_for_workers=True,
    ):
        self.worker_info = worker_info
        self.parallelism_config = parallelism_config
        self.distribute_config = distribute_config
        self.start_port = start_port
        logging.info(
            f"init DistributedServer, rank: {parallelism_config.world_rank},  size: {parallelism_config.world_size}"
        )
        global _g_world_info
        if _g_world_info is not None:
            _g_world_info.self = worker_info
            _g_world_info.num_nodes = (
                parallelism_config.world_size + parallelism_config.local_world_size - 1
            ) // parallelism_config.local_world_size

        if parallelism_config.world_size == 1:
            logging.info("world_size == 1, do not start distributed_server")
            # For world_size == 1, master is the same as worker
            worker_info.update_master_info(
                worker_info.ip,
                start_port,
                parallelism_config,
            )
            return

        if rank == -1:
            rank = parallelism_config.world_rank
        if world_size == -1:
            world_size = parallelism_config.world_size
        self._initialized = True
        self.rank = rank
        self.world_size = world_size

        self.master_ip, master_server_port = get_master(
            distribute_config, parallelism_config=parallelism_config
        )
        if master_server_port == "":
            self.master_server_port = WorkerInfo.server_port_offset(
                local_rank=0, server_port=start_port
            )
        else:
            self.master_server_port = int(master_server_port)
        logging.info(
            f"distributed_server init, before update_master_info: {worker_info}"
        )
        worker_info.update_master_info(
            self.master_ip, self.master_server_port, parallelism_config
        )
        logging.info(
            f"distributed_server init, after update_master_info: {worker_info}"
        )
        _g_world_info.master = worker_info

        logging.info(
            f"{parallelism_config} init tcpstore "
            f"{self.master_ip}:{self.master_server_port - 1}"
        )

        init_process_timeout = distribute_config.dist_comm_timeout
        if init_process_timeout is not None:
            init_process_timeout = timedelta(seconds=init_process_timeout)
        # Use self.master_ip (not worker_info.ip) for TCPStore, as TCPStore connects to master
        store = TCPStore(
            host_name=self.master_ip,
            port=self.master_server_port - 1,
            world_size=world_size,
            is_master=(rank == 0),
            wait_for_workers=wait_for_workers,
            timeout=init_process_timeout,
        )
        logging.info(f"{parallelism_config} init tcpstore done")
        self.store = store

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
        self.safe_store_set(
            key, f"{self.worker_info.ip}:{self.worker_info.server_port}"
        )

    def bootstrap(self) -> None:
        timeout_minutes = self.distribute_config.gang_timeout_min
        sleep_time = self.distribute_config.gang_sleep_time

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
                    parallelism_config = self.parallelism_config
                    local_rank = rank % parallelism_config.local_world_size
                    # server_port from registration is already the base_port (start_port + local_rank * worker_info_port_num)
                    # Since we don't know start_port and worker_info_port_num, we pass server_port as start_port
                    # and set worker_info_port_num=0, so base_port = start_port + local_rank * 0 = start_port
                    new_member = WorkerInfo(
                        ip=ip,
                        local_rank=local_rank,
                        world_rank=rank,
                        local_world_size=parallelism_config.local_world_size,
                        start_port=server_port,
                        remote_server_port=self.distribute_config.remote_server_port,
                        worker_info_port_num=0,  # Treat start_port as already-calculated base_port
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

    def start(self) -> None:
        parallelism_config = self.parallelism_config
        logging.info(
            f"DistributedServer start, rank: {parallelism_config.world_rank},  size: {parallelism_config.world_size}"
        )
        if parallelism_config.world_size == 1:
            return
        self.bootstrap()

        master_url = f"tcp://{self.master_ip}:{self.master_server_port - 1}"
        logging.info(
            f"DistributedServer bootstrap done, rank: {parallelism_config.world_rank},  size: {parallelism_config.world_size}, master {master_url}"
        )
        logging.info(
            f"DistributedServer started, rank: {parallelism_config.world_rank},  size: {parallelism_config.world_size}, master {master_url}"
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


def get_master(distribute_config, parallelism_config) -> (str, str):
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
        # Note: This function may need worker_info parameter in the future
        # For now, we'll get it from environment or use a default
        import socket

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
    member_info_list = []
    for member_str in env_str.split(";"):
        member_info = {}
        for item in member_str.split(","):
            key, value = item.split(":")
            member_info[key] = value
        member_info_list.append(member_info)
        members.append(
            WorkerInfo(
                ip=member_info["ip"],
                local_rank=0,
                world_rank=0,
                start_port=int(member_info["port"]),
                worker_info_port_num=0,  # Treat start_port as already-calculated base_port
            )
        )
    # Find master by name ending with "part0"
    masters = [
        (member, info)
        for member, info in zip(members, member_info_list)
        if info.get("name", "").endswith("part0")
    ]
    if len(masters) != 1:
        raise Exception(f"gang master should contains 1 but got {len(masters)}")
    # Sort by name
    sorted_pairs = sorted(
        zip(members, member_info_list), key=lambda x: x[1].get("name", "")
    )
    sorted_members = [member for member, _ in sorted_pairs]
    master_member, master_info = masters[0]
    if master_info.get("name", "") != sorted_pairs[0][1].get("name", ""):
        raise Exception(
            f"gang master should be the first one but got {sorted_pairs[0][1].get('name', '')}"
        )
    return sorted_members
