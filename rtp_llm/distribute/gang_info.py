import ipaddress
import json
import logging
import os
import socket
from typing import Any, Dict, List, NamedTuple, Optional

from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info, g_worker_info

CONFIG_FILE_ENV = "DISTRIBUTE_CONFIG_FILE"
JSON_GANG_PARTS_ENV = "JSON_GANG_PARTS"
from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.distribute.worker_info import WorkerInfo, g_parallel_info, g_worker_info


def members_from_json(gang_info_json: Dict[str, Any]) -> List[WorkerInfo]:
    members: List[WorkerInfo] = []
    # here is only the fake ip
    for name, info in gang_info_json.items():
        server_port = info["port"] if "port" in info else -1
        members.append(
            WorkerInfo(
                server_port=server_port,
                gang_hb_port=-1,
                http_port=-1,
                rpc_server_port=-1,
                embedding_rpc_server_port=-1,
                backend_server_port=-1,
                remote_rpc_server_port=-1,
                cache_store_listen_port=-1,
                cache_store_connect_port=-1,
                cache_store_rdma_connect_port=-1,
                cache_store_rdma_listen_port=-1,
                local_rank=0,
                world_rank=0,
                name=info["name"],
                ip=info["ip"],
                info=info,
            )
        )
    zone_name = StaticConfig.gang_config.zone_name
    if zone_name:
        members = [
            member for member in members if member.name.split("_")[-2] == zone_name
        ]
    masters = [member for member in members if member.name.endswith("part0")]
    if len(masters) != 1:
        raise Exception(f"gang master should contains 1 but got {len(masters)}")
    return sorted(members, key=lambda x: x.name)


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
                embedding_rpc_server_port=-1,
                remote_rpc_server_port=-1,
                cache_store_listen_port=-1,
                cache_store_connect_port=-1,
                cache_store_rdma_connect_port=-1,
                cache_store_rdma_listen_port=-1,
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


"""
raw gang info example:
app.c2.io/biz-detail-ganginfo="{\"llama13B_2A10_PCIE_1_inference_part0\":{\"name\":\"llama13B_2A10_PCIE_1_inference_part0\",\"ip\":\"33.76.194.173\"},\"llama13B_2A10_PCIE_1_inference_part1\":{\"name\":\"llama13B_2A10_PCIE_1_inference_part1\",\"ip\":\"33.76.194.182\"}}"
"""


def get_c2_members():
    file_name = StaticConfig.gang_config.gang_annocation_path
    if not os.path.exists(file_name):
        raise Exception(f"not found file: {file_name}")

    with open(file_name, "r") as reader:
        content = reader.read()

    infos = [x for x in content.split("\n") if "app.c2.io/biz-detail-ganginfo" in x]
    if len(infos) != 1:
        raise Exception(f"ganginfo length is not equal to 1, actual: {infos}")

    gang_info = infos[0].replace("\\", "")
    logging.info(f"gang info: {gang_info[gang_info.index('=') + 2: -1]}")
    gang_info_json = json.loads(gang_info[gang_info.index("=") + 2 : -1])
    logging.info(f"gang info json: {gang_info_json}")
    return members_from_json(gang_info_json)


def get_leader_ip(leader_address: str) -> str:
    try:
        ipaddress.ip_address(leader_address)
        return leader_address
    except:
        return socket.gethostbyname(leader_address)


def get_leader_members(env_str: str) -> List[WorkerInfo]:
    ip_str = get_leader_ip(env_str)
    members: List[WorkerInfo] = []
    zone_name = StaticConfig.gang_config.zone_name
    member_info = {}
    member_info["name"] = "part0"
    if zone_name:
        member_info["name"] = zone_name + "_part0"
    member_info["ip"] = ip_str
    members.append(
        WorkerInfo(
            server_port=-1,
            gang_hb_port=-1,
            http_port=-1,
            rpc_server_port=-1,
            backend_server_port=-1,
            embedding_rpc_server_port=-1,
            remote_rpc_server_port=-1,
            cache_store_listen_port=-1,
            cache_store_connect_port=-1,
            cache_store_rdma_connect_port=-1,
            cache_store_rdma_listen_port=-1,
            local_rank=0,
            world_rank=0,
            name=member_info["name"],
            ip=member_info["ip"],
            info=member_info,
        )
    )

    if os.environ.get("WORLD_INDEX") and ip_str != g_worker_info.ip:
        self_member_info = {}
        self_member_info["name"] = "part" + os.environ.get("WORLD_INDEX")
        if zone_name:
            self_member_info["name"] = (
                zone_name + "_part" + os.environ.get("WORLD_INDEX")
            )
        self_member_info["ip"] = g_worker_info.ip
        members.append(
            WorkerInfo(
                server_port=-1,
                gang_hb_port=-1,
                http_port=-1,
                rpc_server_port=-1,
                backend_server_port=-1,
                embedding_rpc_server_port=-1,
                remote_rpc_server_port=-1,
                cache_store_listen_port=-1,
                cache_store_connect_port=-1,
                cache_store_rdma_connect_port=-1,
                cache_store_rdma_listen_port=-1,
                local_rank=0,
                world_rank=0,
                name=self_member_info["name"],
                ip=self_member_info["ip"],
                info=self_member_info,
            )
        )

    return members


def get_members_from_file():
    file = StaticConfig.gang_config.distribute_config_file
    with open(file, "r") as reader:
        config_json = json.loads(reader.read())
    return members_from_json(config_json)


def get_members_from_json_env(env_str: str) -> List[WorkerInfo]:
    return members_from_json(json.loads(env_str))


class GangInfo(NamedTuple):
    members: List[WorkerInfo]
    master: WorkerInfo
    self: WorkerInfo
    num_nodes: int
    only_leader: bool

    def workers(self) -> List[WorkerInfo]:
        return [member for member in self.members if not member.equals(self.master)]


def get_gang_info() -> GangInfo:
    only_leader = False
    if g_parallel_info.local_world_size < g_parallel_info.world_size:
        # from config file
        if StaticConfig.gang_config.distribute_config_file:
            members = get_members_from_file()
        # for distributed test
        elif StaticConfig.gang_config.gang_config_string:
            logging.info(
                f"use GANG_CONFIG_STRING: {StaticConfig.gang_config.gang_config_string}"
            )
            members = members_from_test_env(StaticConfig.gang_config.gang_config_string)
        # from env json
        elif StaticConfig.gang_config.json_gang_parts:
            logging.info(
                f"use JSON_GANG_PARTS_ENV: {StaticConfig.gang_config.json_gang_parts}"
            )
            members = get_members_from_json_env(
                StaticConfig.gang_config.json_gang_parts
            )
        # for lws
        elif StaticConfig.gang_config.leader_address:
            logging.info(
                f"use LEADER_ADDRESS: {StaticConfig.gang_config.leader_address}"
            )
            members = get_leader_members(StaticConfig.gang_config.leader_address)
        # from c2 annotation
        else:
            members = get_c2_members()
    else:
        members = [
            WorkerInfo(
                socket.gethostbyname(socket.gethostname()),
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                0,
                0,
                "local",
                None,
            )
        ]
    if StaticConfig.gang_config.leader_address:
        only_leader = True

    # 假设 GPU 均匀分布，可以整除
    # member 是按 part 排序的
    self: Optional[WorkerInfo] = None
    master: Optional[WorkerInfo] = None
    all_members: List[WorkerInfo] = []
    for part_rank, member in enumerate(members):
        for local_rank in range(g_parallel_info.local_world_size):
            new_member = WorkerInfo(
                ip=member.ip,
                server_port=WorkerInfo.server_port_offset(
                    local_rank, member.server_port
                ),
                gang_hb_port=WorkerInfo.gang_hb_port_offset(
                    local_rank, member.server_port
                ),
                http_port=WorkerInfo.http_port_offset(local_rank, member.server_port),
                rpc_server_port=WorkerInfo.rpc_server_port_offset(
                    local_rank, member.server_port
                ),
                embedding_rpc_server_port=WorkerInfo.embedding_rpc_server_port_offset(
                    local_rank, member.server_port
                ),
                backend_server_port=WorkerInfo.backend_server_port_offset(
                    local_rank, member.server_port
                ),
                cache_store_listen_port=WorkerInfo.cache_store_listen_port_offset(
                    local_rank, member.server_port
                ),
                cache_store_rdma_listen_port=WorkerInfo.cache_store_rdma_listen_port_offset(
                    local_rank, member.server_port
                ),
                remote_rpc_server_port=WorkerInfo.rpc_server_port_offset(
                    local_rank, int(os.environ.get("REMOTE_SERVER_PORT", 0))
                ),
                cache_store_connect_port=WorkerInfo.cache_store_listen_port_offset(
                    local_rank, int(os.environ.get("REMOTE_SERVER_PORT", 0))
                ),
                cache_store_rdma_connect_port=WorkerInfo.cache_store_rdma_listen_port_offset(
                    local_rank, int(os.environ.get("REMOTE_SERVER_PORT", 0))
                ),
                local_rank=local_rank,
                world_rank=part_rank * g_parallel_info.local_world_size + local_rank,
                name=member.name + "_" + str(local_rank),
                info=member.info,
            )
            all_members.append(new_member)
            logging.info(
                f"local rank {local_rank} vs {g_parallel_info.local_rank}, \
                         new_member: {new_member.ip} vs {g_worker_info.ip}, \
                         server port {new_member.server_port} vs {g_worker_info.server_port}"
            )
            if (
                local_rank == g_parallel_info.local_rank
                and new_member.ip == g_worker_info.ip
                and new_member.server_port == g_worker_info.server_port
            ):
                self = new_member
            if part_rank == 0 and local_rank == 0:
                master = new_member
    # not check master and self empty here for ut
    return GangInfo(all_members, master, self, len(members), only_leader)
