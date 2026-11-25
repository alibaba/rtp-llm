import ipaddress
import json
import logging
import os
import socket
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, Iterator, List, Optional

from torch.distributed import TCPStore
from torch.distributed.elastic.utils.store import get_all

from rtp_llm.config.py_config_modules import DistributeConfig, PyEnvConfigs
from rtp_llm.distribute.worker_info import WorkerInfo, WorldInfo, workers_on_host
from rtp_llm.models_py.distributed.rank_layout import Coord, RankLayout
from rtp_llm.ops import ParallelismConfig
from rtp_llm.utils.util import split_ip_port


def get_local_workers(configs: PyEnvConfigs) -> list[WorkerInfo]:
    """Return only the current host's workers, with their global rank numbers."""
    pc = configs.parallelism_config
    sc = configs.server_config
    host_index = pc.world_rank // pc.local_world_size
    world = configs.distribute_config.world_info
    if world is not None:
        ip, start_port = split_ip_port(world.host_addresses[host_index])
    else:
        # Standalone frontends route multi-rank requests to their local hostname.
        ip = (
            sc.ip
            if pc.world_size == 1 and sc.ip
            else socket.gethostbyname(socket.gethostname())
        )
        start_port = sc.start_port
    return workers_on_host(configs, ip, start_port, host_index * pc.local_world_size)


def get_dp_rpc_addresses(py_env_configs: PyEnvConfigs) -> list[str]:
    """Select local serving RPC addresses at TP=0 and PP=0."""
    pc = py_env_configs.parallelism_config
    layout = RankLayout.from_parallelism_config(pc)
    addresses = [
        f"{member.ip}:{member.rpc_server_port}"
        for member in get_local_workers(py_env_configs)
        if (coord := layout.coord_of(member.world_rank)).tp == 0 and coord.pp == 0
    ]
    logging.info(
        "[world_rank: %d] local serving RPC addresses: %s", pc.world_rank, addresses
    )
    return addresses


def get_eplb_stage_root_addresses(py_env_configs: PyEnvConfigs) -> list[str]:
    """Resolve DP=0, TP=0 roots for PP EPLB control, in stage order."""
    pc = py_env_configs.parallelism_config
    layout = RankLayout.from_parallelism_config(pc)
    root_ranks = [
        layout.world_rank_of(Coord(pp=stage)) for stage in range(layout.pp_size)
    ]
    world = py_env_configs.distribute_config.world_info
    if world is not None:
        members = world.members(py_env_configs)
        return [
            f"{members[rank].ip}:{members[rank].rpc_server_port}" for rank in root_ranks
        ]

    addresses = {
        member.world_rank: f"{member.ip}:{member.rpc_server_port}"
        for member in get_local_workers(py_env_configs)
        if member.world_rank in root_ranks
    }
    missing = [rank for rank in root_ranks if rank not in addresses]
    if missing:
        master_ip, master_port = get_master(py_env_configs.distribute_config, pc)
        master_port = int(master_port or py_env_configs.server_config.start_port)
        store = TCPStore(
            host_name=master_ip,
            port=master_port - 1,
            is_master=False,
            wait_for_workers=False,
            timeout=timedelta(seconds=3),
        )
        for rank in missing:
            address = store.get(
                DistributedServer.REGISTRY_RANK_ADDRESS_KEY + str(rank)
            ).decode()
            ip, server_port = split_ip_port(address)
            # The registered server port already includes the local-rank offset;
            # its RPC port is the next port in WorkerInfo's layout.
            addresses[rank] = f"{ip}:{server_port + 1}"
    return [addresses[rank] for rank in root_ranks]


class DistributedServer:
    """Own the TCPStore rendezvous in the host launcher, before workers exist."""

    REGISTRY_RANK_ADDRESS_KEY = "registry_rank_address_"

    def __init__(self, py_env_configs: PyEnvConfigs):
        self.py_env_configs = py_env_configs
        pc = py_env_configs.parallelism_config
        self.rank = pc.world_rank
        self.world_size = pc.world_size
        self.local_world_size = pc.local_world_size
        if (
            self.local_world_size <= 0
            or self.world_size <= 0
            or (self.world_size > 1 and self.world_size % self.local_world_size)
        ):
            raise ValueError(
                f"Invalid host layout: WORLD_SIZE={self.world_size}, LOCAL_WORLD_SIZE={self.local_world_size}"
            )
        if not 0 <= self.rank < self.world_size or self.rank % self.local_world_size:
            raise ValueError(
                f"WORLD_RANK={self.rank} must identify the first rank on this host"
            )
        self.host_index = self.rank // self.local_world_size

        sc = py_env_configs.server_config
        local_ip = sc.ip or socket.gethostbyname(socket.gethostname())
        self.local_address = f"{local_ip}:{sc.start_port}"
        # Indexed by the original host_index before selection.
        self.host_addresses: list[str] = []
        self.master_ip = local_ip
        self.master_base_port = sc.start_port
        self.store: Optional[TCPStore] = None

    @contextmanager
    def rendezvous(self) -> Iterator["DistributedServer"]:
        """Keep the startup store alive through the last host exchange."""
        try:
            self.start()
            yield self
        finally:
            self.store = None

    @property
    def world_info(self) -> WorldInfo:
        return WorldInfo(
            tuple(self.host_addresses), (self.master_ip, self.master_base_port)
        )

    def start(self) -> None:
        dc = self.py_env_configs.distribute_config
        if self.world_size > 1:
            self.master_ip, port = get_master(
                dc, self.py_env_configs.parallelism_config
            )
            if port:
                self.master_base_port = int(port)
        if self.world_size == 1 or self.world_size == self.local_world_size:
            self.host_addresses = [self.local_address]
            return
        self.store = TCPStore(
            self.master_ip,
            self.master_base_port - 1,
            world_size=self.world_size // self.local_world_size,
            is_master=self.rank == 0,
            wait_for_workers=True,
            timeout=timedelta(minutes=dc.gang_timeout_min),
        )
        self.bootstrap()

    def regist(self) -> None:
        """Register this host once, using the master's original gang dimensions."""
        layout = [self.world_size, self.local_world_size]
        if (
            self.host_index
            and layout != json.loads(self.store.get("startup_host/0"))["layout"]
        ):
            raise ValueError(
                f"Host rank {self.rank} layout {layout} disagrees with the original master"
            )
        self.store.set(
            f"startup_host/{self.host_index}",
            json.dumps({"layout": layout, "address": self.local_address}),
        )

    def bootstrap(self) -> None:
        self.regist()
        host_count = self.world_size // self.local_world_size
        records = get_all(self.store, self.host_index, "startup_host/", host_count)
        self.host_addresses = [json.loads(record)["address"] for record in records]
        logging.info(
            "DistributedServer bootstrap complete: host=%d hosts=%s",
            self.host_index,
            self.host_addresses,
        )


def get_ip(leader_address: str) -> str:
    try:
        ipaddress.ip_address(leader_address)
        return leader_address
    except Exception:
        return socket.gethostbyname(leader_address)


def get_master(
    distribute_config: DistributeConfig,
    parallelism_config: ParallelismConfig,
) -> tuple[str, str]:
    """Resolve the communication endpoint used for distributed initialization."""
    if distribute_config.world_info is not None:
        ip, base_port = distribute_config.world_info.collective_master
        return ip, str(base_port)

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
