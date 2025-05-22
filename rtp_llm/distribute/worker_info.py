from __future__ import annotations
import os
import json
import socket
import torch
import logging
from typing import Any, Dict
from dataclasses import dataclass


DEFAULT_START_PORT = 8088
MASTER_INFO_PORT_NUM = 11
MIN_WORKER_INFO_PORT_NUM = 7
WORKER_INFO_PORT_NUM = MIN_WORKER_INFO_PORT_NUM

def get_worker_port_num():
    global WORKER_INFO_PORT_NUM
    global MIN_WORKER_INFO_PORT_NUM
    WORKER_INFO_PORT_NUM = int(os.environ.get('WORKER_INFO_PORT_NUM', MIN_WORKER_INFO_PORT_NUM))
    logging.info(f'env WORKER_INFO_PORT_NUM: {WORKER_INFO_PORT_NUM}')
    if WORKER_INFO_PORT_NUM < MIN_WORKER_INFO_PORT_NUM:
        raise Exception(f"env worker info port num {WORKER_INFO_PORT_NUM} " \
            f"is small than min worker info port num {MIN_WORKER_INFO_PORT_NUM}")

get_worker_port_num()

class FrontendServerInfo(object):
    def __init__(self, frontend_server_id: int):
        self.frontend_server_id = frontend_server_id

    @staticmethod
    def from_env() -> FrontendServerInfo:
        return FrontendServerInfo.from_params(dict(os.environ))

    @staticmethod
    def from_params(params: Dict[str, str]) -> FrontendServerInfo:
        info = FrontendServerInfo(
                frontend_server_id=int(params.get('FRONTEND_SERVER_ID', '0')))
        return info

    def reload(self):
        new_info = self.from_env()
        self.frontend_server_id = new_info.frontend_server_id

    def __str__(self):
        return f"FrontendServerInfo:[ frontend_server_id={self.frontend_server_id} ]"

g_frontend_server_info = FrontendServerInfo.from_env()

class ParallelInfo(object):
    # EP从TP里分
    def __init__(
            self, tp_size: int, ep_size: int,
            pp_size: int, dp_size: int, ffn_sp_size: int,
            world_size: int, world_rank: int,
            local_world_size: int
    ):
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.pp_size = pp_size
        self.dp_size = dp_size
        self.ffn_sp_size = ffn_sp_size
        self.ffn_tp_size = self.tp_size // self.ffn_sp_size
        self.world_size = world_size
        self.world_rank = world_rank
        self.local_world_size = local_world_size
        logging.info(f"ParallelInfo:[ tp_size={self.tp_size} ep_size={self.ep_size} pp_size={self.pp_size} world_size={self.world_size} world_rank={self.world_rank} local_world_size={self.local_world_size} ffn_sp_size={self.ffn_sp_size} ffn_tp_size={self.ffn_tp_size}]")
        assert ep_size <= world_size and world_size % ep_size == 0
        assert self.world_size == self.tp_size * self.dp_size * self.pp_size
        if torch.cuda.is_available():
            self.device = 'cuda:' + str(self.world_rank % self.local_world_size)
        else:
            self.device = 'cpu'

    @property
    def tp_rank(self) -> int:
        return self.world_rank % self.tp_size

    @property
    def dp_rank(self) -> int:
        return self.world_rank // self.tp_size

    # ep_rank只在MOE plugin生效
    @property
    def ep_rank(self) -> int:
        return self.world_rank % self.ep_size

    @property
    def ffn_tp_rank(self) -> int:
        return self.tp_rank % self.ffn_tp_size

    @property
    def local_rank(self) -> int:
        return self.world_rank % self.local_world_size

    @property
    def is_master(self):
        return self.world_rank == 0

    @staticmethod
    def from_env() -> ParallelInfo:
        return ParallelInfo.from_params(dict(os.environ))

    @staticmethod
    def from_params(params: Dict[str, str]) -> ParallelInfo:
        world_size = int(params.get('WORLD_SIZE', '1'))
        if 'LOCAL_WORLD_SIZE' in params:
            local_world_size = int(params['LOCAL_WORLD_SIZE'])
        else:
            local_world_size = min(torch.cuda.device_count(), world_size)
            local_world_size = max(local_world_size, 1) # make sure local_world_size >= 1
        info = ParallelInfo(
                tp_size=int(params.get('TP_SIZE', '1')),
                ep_size=int(params.get('EP_SIZE', params.get('WORLD_SIZE', '1'))),
                pp_size=int(params.get('PP_SIZE', '1')),
                dp_size=int(params.get('DP_SIZE', 1)),
                ffn_sp_size = int(params.get('FFN_SP_SIZE', '1')),
                world_size=world_size,
                world_rank=int(params.get('WORLD_RANK', '0')),
                local_world_size=local_world_size)
        if (torch.cuda.is_available() and (info.local_world_size > torch.cuda.device_count())):
            raise Exception(f'local_world_size:{info.local_world_size} > cuda device count:{torch.cuda.device_count()}')
        if (info.tp_size * info.pp_size * info.dp_size != info.world_size or
            info.world_rank >= info.world_size or (info.tp_size % info.ffn_sp_size != 0)):
            raise Exception(f'tp_size:{info.tp_size}, ep_size:{info.ep_size}, pp_size:{info.pp_size}, world_size:{info.world_size}, world_rank:{info.world_rank} ffn_sp_size: {info.ffn_sp_size} invalid world config')
        # 假设 GPU 均匀分布，可以整除
        if info.world_size % info.local_world_size != 0:
            raise Exception(f"not support info.world_size:[{info.world_size}] mod info.local_world_size:[{info.local_world_size}] != 0")

        if torch.cuda.is_available():
            torch.cuda.set_device(info.local_rank)

        if os.environ.get("ACCL_SELECT_PATH") == "1":
            select_port = str(info.local_rank % 2)
            os.environ["ACCL_SELECT_PORT"] = select_port
            logging.info(f"local rank {info.local_rank} set accl select port to {select_port} ")

        if os.environ.get("ACCL_USE_NICS") == None and os.environ.get("ACCL_NIC_GPU_AFFINITY") != None:
            content = os.environ.get("ACCL_NIC_GPU_AFFINITY")
            try:
                gpu_nic_affinity = json.loads(content)  # 验证内容是否为合法 JSON
                if str(info.local_rank) in gpu_nic_affinity:
                    affinity_nic = gpu_nic_affinity[str(info.local_rank)]
                    os.environ["ACCL_USE_NICS"] = affinity_nic
                    logging.info(f"local rank {info.local_rank} use cuda device {info.local_rank} set ACCL_USE_NICS to {affinity_nic}")
                else:
                    logging.info(f"local rank {info.local_rank} use cuda device {info.local_rank} get affinity nic failed, content is {content}")
            except json.JSONDecodeError:
                logging.info(f"try decode ACCL_NIC_GPU_AFFINITY failed, content is {content}")

        return info

    # used for ut
    def reload(self):
        new_info = self.from_env()
        self.tp_size=new_info.tp_size
        self.pp_size=new_info.pp_size
        self.world_size=new_info.world_size
        self.world_rank=new_info.world_rank
        self.local_world_size=new_info.local_world_size

    def __str__(self):
        return f"ParallelInfo:[ tp_size={self.tp_size} pp_size={self.pp_size} world_size={self.world_size} world_rank={self.world_rank} local_world_size={self.local_world_size} tp_rank={self.tp_rank} dp_rank={self.dp_rank} ep_size={self.ep_size} dp_size={self.dp_size} ep_rank={self.ep_rank} local_rank={self.local_rank} ffn_sp_size={self.ffn_sp_size} ]"

g_parallel_info = ParallelInfo.from_env()

class WorkerInfo(object):
    def __init__(self, ip: str, server_port: int, gang_hb_port: int,
                 http_port: int, rpc_server_port: int, remote_rpc_server_port: int,
                 cache_store_listen_port: int, cache_store_connect_port: int, cache_store_rdma_listen_port: int,
                 cache_store_rdma_connect_port: int, backend_server_port: int,
                 local_rank: int, world_rank: int, name: str, info: Any):
        self.ip = ip
        self.server_port = server_port
        self.gang_hb_port = gang_hb_port
        self.http_port = http_port
        self.rpc_server_port= rpc_server_port
        self.remote_rpc_server_port = remote_rpc_server_port
        self.cache_store_listen_port = cache_store_listen_port
        self.cache_store_connect_port = cache_store_connect_port
        self.cache_store_rdma_listen_port = cache_store_rdma_listen_port
        self.cache_store_rdma_connect_port = cache_store_rdma_connect_port
        self.backend_server_port = backend_server_port
        self.local_rank: int = local_rank
        self.world_rank: int = world_rank
        self.name = name
        self.info = info

    def equals(self, other: 'WorkerInfo') -> bool:
        return self.ip == other.ip and self.server_port == other.server_port

    @staticmethod
    def from_env():
        info = WorkerInfo(
            ip=socket.gethostbyname(socket.gethostname()),
            server_port=WorkerInfo.server_port_offset(g_parallel_info.local_rank),
            gang_hb_port=WorkerInfo.gang_hb_port_offset(g_parallel_info.local_rank),
            http_port=WorkerInfo.http_port_offset(g_parallel_info.local_rank),
            rpc_server_port=WorkerInfo.rpc_server_port_offset(g_parallel_info.local_rank),
            remote_rpc_server_port=WorkerInfo.rpc_server_port_offset(g_parallel_info.local_rank, int(os.environ.get("REMOTE_SERVER_PORT", 0))),
            cache_store_listen_port=WorkerInfo.cache_store_listen_port_offset(g_parallel_info.local_rank),
            cache_store_connect_port=WorkerInfo.cache_store_listen_port_offset(g_parallel_info.local_rank, int(os.environ.get("REMOTE_SERVER_PORT", 0))),
            cache_store_rdma_listen_port=WorkerInfo.cache_store_rdma_listen_port_offset(g_parallel_info.local_rank),
            cache_store_rdma_connect_port=WorkerInfo.cache_store_rdma_listen_port_offset(g_parallel_info.local_rank, int(os.environ.get("REMOTE_SERVER_PORT", 0))),
            backend_server_port=WorkerInfo.backend_server_port_offset(g_parallel_info.local_rank),
            local_rank=g_parallel_info.local_rank,
            world_rank=g_parallel_info.world_rank,
            name='', info=None)
        return info

    @staticmethod
    def self_server_port():
        return int(os.environ.get('START_PORT', DEFAULT_START_PORT))

    @staticmethod
    def server_port_offset(local_rank: int, server_port: int = -1) -> int:
        if server_port != -1:
            base_port = server_port
        else:
            base_port = WorkerInfo.self_server_port()
        return base_port + local_rank * WORKER_INFO_PORT_NUM

    @staticmethod
    def rpc_server_port_offset(local_rank: int, server_port: int = -1) -> int:
        return WorkerInfo.server_port_offset(local_rank, server_port) + 1

    @staticmethod
    def cache_store_listen_port_offset(local_rank: int, server_port: int = -1) -> int:
        return WorkerInfo.server_port_offset(local_rank, server_port) + 2

    @staticmethod
    def gang_hb_port_offset(local_rank: int, server_port: int = -1) -> int:
        return WorkerInfo.server_port_offset(local_rank, server_port) + 3

    @staticmethod
    def cache_store_rdma_listen_port_offset(local_rank: int, server_port: int = -1) -> int:
        return WorkerInfo.server_port_offset(local_rank, server_port) + 4

    @staticmethod
    def http_port_offset(local_rank: int, server_port: int = -1) -> int:
        return WorkerInfo.server_port_offset(local_rank, server_port) + 5

    @staticmethod
    def backend_server_port_offset(local_rank: int, server_port: int = -1) -> int:
        return WorkerInfo.server_port_offset(local_rank, server_port) + 6

    # used for ut
    def reload(self):
        new_info = self.from_env()
        self.ip = new_info.ip
        self.server_port = new_info.server_port
        self.gang_hb_port = new_info.gang_hb_port
        self.http_port = new_info.http_port
        self.remote_rpc_server_port = new_info.remote_rpc_server_port
        self.cache_store_listen_port = new_info.cache_store_listen_port
        self.cache_store_connect_port = new_info.cache_store_connect_port
        self.rpc_server_port = new_info.rpc_server_port
        self.backend_server_port = new_info.backend_server_port
        self.local_rank = new_info.local_rank
        self.world_rank = new_info.world_rank
        self.name = new_info.name
        self.info = new_info.info

    def __str__(self):
        return f"""
        WorkerInfo: [ip={self.ip} server_port={self.server_port} gang_hb_port={self.gang_hb_port}
        http_port={self.http_port} rpc_port={self.rpc_server_port} backend_server_port={self.backend_server_port}
        cache_store_listen_port={self.cache_store_listen_port} cache_store_connect_port={self.cache_store_connect_port} remote_rpc_server_port={self.remote_rpc_server_port}
        local_rank={self.local_rank} world_rank={self.world_rank} name={self.name} info={self.info} ]
        """

g_worker_info = WorkerInfo.from_env()

@dataclass
class MasterInfo:
    ip: str
    th_nccl_port: int
    tp_nccl_port: int
    nccl_op_port: int
    sp_gpt_nccl_port: int
    dp_tp_nccl_port: int
    ffn_tp_nccl_port: int

g_master_info = MasterInfo(
    ip='',
    th_nccl_port=0,
    tp_nccl_port = 0,
    nccl_op_port=0,
    sp_gpt_nccl_port=0,
    dp_tp_nccl_port=0,
    ffn_tp_nccl_port=0,
)

def update_master_info(ip: str, base_port: int):
    g_master_info.ip = ip
    g_master_info.dp_tp_nccl_port = base_port - 10
    base_port -= g_parallel_info.dp_rank * MASTER_INFO_PORT_NUM
    g_master_info.th_nccl_port = base_port - 1
    g_master_info.tp_nccl_port = base_port - 2
    g_master_info.nccl_op_port = base_port - 3
    g_master_info.sp_gpt_nccl_port = base_port - 4
    # note: reserve 4 ports for ffn_tp_nccl_port
    g_master_info.ffn_tp_nccl_port = base_port - 5
    if g_parallel_info.ffn_sp_size != g_parallel_info.tp_size:
        base_port -= g_parallel_info.ffn_sp_size

def total_need_port_num() -> int:
    return MASTER_INFO_PORT_NUM * g_parallel_info.dp_size + WORKER_INFO_PORT_NUM * g_parallel_info.tp_size
