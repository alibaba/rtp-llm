
from __future__ import annotations
import os
import socket
import torch
from typing import Any
from dataclasses import dataclass


DEFAULT_START_PORT = 8088

class ParallelInfo(object):
    def __init__(
            self, tp_size: int, pp_size: int,
            world_size: int, world_rank: int,
            local_world_size: int
    ):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.world_size = world_size
        self.world_rank = world_rank
        self.local_world_size = local_world_size

        if torch.cuda.is_available():
            self.device = 'cuda:' + str(self.world_rank % self.local_world_size)
        else:
            self.device = 'cpu'

    @property
    def is_pp_first(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_pp_last(self) -> bool:
        return self.pp_rank == self.pp_size - 1

    @property
    def pp_rank(self) -> int:
        return self.world_rank // self.tp_size

    @property
    def tp_rank(self) -> int:
        return self.world_rank % self.tp_size

    @property
    def local_rank(self) -> int:
        return self.world_rank % self.local_world_size

    @property
    def is_master(self):
        return self.world_rank == 0

    @staticmethod
    def from_env() -> ParallelInfo:
        info = ParallelInfo(
                tp_size=int(os.environ.get('TP_SIZE', '1')),
                pp_size=int(os.environ.get('PP_SIZE', '1')),
                world_size=int(os.environ.get('WORLD_SIZE', '1')),
                world_rank=int(os.environ.get('WORLD_RANK', '0')),
                local_world_size=int(os.environ.get('LOCAL_WORLD_SIZE', '1')))
        if (info.tp_size * info.pp_size != info.world_size or
            info.world_rank >= info.world_size):
            raise Exception(f'tp_size:{info.tp_size}, pp_size:{info.pp_size}, world_size:{info.world_size}, world_rank:{info.world_rank} invalid world config')
        # 假设 GPU 均匀分布，可以整除
        if info.world_size % info.local_world_size != 0:
            raise Exception("not support info.world_size mod info.local_world_size != 0")
        
        if torch.cuda.is_available():
            torch.cuda.set_device(info.local_rank)
     
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
        return f"ParallelInfo:[ tp_size={self.tp_size} pp_size={self.pp_size} world_size={self.world_size} world_rank={self.world_rank} local_world_size={self.local_world_size} ]"

g_parallel_info = ParallelInfo.from_env()

class WorkerInfo(object):
    def __init__(self, ip: str, server_port: int, gang_hb_port: int, name: str, info: Any):
        self.ip = ip
        self.server_port = server_port
        self.gang_hb_port = gang_hb_port
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
        return base_port + local_rank * 4

    @staticmethod
    def gang_hb_port_offset(local_rank: int, server_port: int = -1) -> int:
        if server_port != -1:
            base_port = server_port
        else:
            base_port = WorkerInfo.self_server_port()
        return base_port + local_rank * 4 + 3

    # used for ut
    def reload(self):
        new_info = self.from_env()
        self.ip = new_info.ip
        self.server_port = new_info.server_port
        self.gang_hb_port = new_info.gang_hb_port
        self.name = new_info.name
        self.info = new_info.info
        
    def __str__(self):
        return f"WorkerInfo: [ip={self.ip} server_port={self.server_port} gang_hb_port={self.gang_hb_port} name={self.name} info={self.info} ]"

g_worker_info = WorkerInfo.from_env()

@dataclass
class MasterInfo:
    ip: str
    th_nccl_port: int
    gpt_nccl_port: int
    dynamic_decoder_nccl_port: int
    nccl_op_port: int
    sp_gpt_nccl_port: int
    http_port: int
    model_rpc_port: int

g_master_info = MasterInfo(
    ip='',
    th_nccl_port=0,    
    gpt_nccl_port = 0,
    dynamic_decoder_nccl_port=0,
    nccl_op_port=0,
    sp_gpt_nccl_port=0,
    http_port=0,
    model_rpc_port=0)

def update_master_info(ip: str, base_port: int):
    g_master_info.ip = ip
    g_master_info.http_port = base_port + 2
    g_master_info.model_rpc_port = base_port + 1    
    g_master_info.th_nccl_port = base_port - 1
    g_master_info.gpt_nccl_port = base_port - 2
    g_master_info.dynamic_decoder_nccl_port = base_port - 3
    g_master_info.nccl_op_port = base_port - 4
    g_master_info.sp_gpt_nccl_port = base_port - 5
