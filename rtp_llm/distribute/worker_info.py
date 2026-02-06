import os
import socket
from dataclasses import dataclass

import torch

from rtp_llm.config.py_config_modules import (
    COORDINATOR_INFO_PORT_NUM,
    MIN_WORKER_INFO_PORT_NUM,
)


def _local_rank_and_world_rank_from_env():
    """Read local_rank and world_rank directly from environment variables."""
    params = dict(os.environ)
    world_size = int(params.get("WORLD_SIZE", "1"))
    if "LOCAL_WORLD_SIZE" in params:
        local_world_size = int(params["LOCAL_WORLD_SIZE"])
    else:
        local_world_size = (
            min(torch.cuda.device_count(), world_size)
            if torch.cuda.is_available()
            else world_size
        )
    local_world_size = max(local_world_size, 1)
    world_rank = int(params.get("WORLD_RANK", "0"))
    if ("WORLD_INDEX" in params) and ("WORLD_RANK" not in params):
        world_index = int(params["WORLD_INDEX"])
        world_rank = world_index * local_world_size
    local_rank = world_rank % local_world_size
    return local_rank, world_rank


class WorkerInfo(object):
    """Port layout: base = server_port + local_rank * worker_info_port_num, then +0..+7."""

    def __init__(
        self,
        ip: str,
        local_rank: int,
        world_rank: int,
        name: str,
        server_port: int,
        worker_info_port_num: int,
        remote_server_port: int = None,
    ):
        self.ip = ip
        self._local_rank = local_rank
        self._world_rank = world_rank
        self.name = name
        self._server_port = server_port
        self._worker_info_port_num = worker_info_port_num
        self._remote_server_port = (
            remote_server_port if remote_server_port is not None else server_port
        )

    def adjust_local_rank(self, local_rank: int, world_rank: int = None):
        """Update local_rank (and optionally world_rank) in place; port properties reflect new values."""
        self._local_rank = local_rank
        if world_rank is not None:
            self._world_rank = world_rank

    @staticmethod
    def from_env(start_port, remote_server_port, worker_info_port_num: int):
        """Create WorkerInfo from environment (reads local_rank, world_rank from env)."""
        local_rank, world_rank = _local_rank_and_world_rank_from_env()
        return WorkerInfo(
            ip=socket.gethostbyname(socket.gethostname()),
            local_rank=local_rank,
            world_rank=world_rank,
            name="",
            server_port=start_port,
            worker_info_port_num=worker_info_port_num,
            remote_server_port=remote_server_port,
        )

    @property
    def _base(self) -> int:
        return self._server_port + self._local_rank * self._worker_info_port_num

    @property
    def _base_remote(self) -> int:
        return self._remote_server_port + self._local_rank * self._worker_info_port_num

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def world_rank(self) -> int:
        return self._world_rank

    @property
    def server_port(self) -> int:
        return self._base + 0

    @property
    def rpc_server_port(self) -> int:
        return self._base + 1

    @property
    def cache_store_listen_port(self) -> int:
        return self._base + 2

    @property
    def gang_hb_port(self) -> int:
        return self._base + 3

    @property
    def cache_store_rdma_listen_port(self) -> int:
        return self._base + 4

    @property
    def http_port(self) -> int:
        return self._base + 5

    @property
    def backend_server_port(self) -> int:
        return self._base + 6

    @property
    def embedding_rpc_server_port(self) -> int:
        return self._base + 7

    @property
    def remote_rpc_server_port(self) -> int:
        return self._base_remote + 1

    @property
    def cache_store_connect_port(self) -> int:
        return self._base_remote + 2

    @property
    def cache_store_rdma_connect_port(self) -> int:
        return self._base_remote + 4

    def __str__(self):
        return f"""
        WorkerInfo: [ip={self.ip}
        server_port={self.server_port} (offset 0)
        rpc_server_port={self.rpc_server_port} (offset 1)
        cache_store_listen_port={self.cache_store_listen_port} (offset 2)
        gang_hb_port={self.gang_hb_port} (offset 3)
        cache_store_rdma_listen_port={self.cache_store_rdma_listen_port} (offset 4)
        http_port={self.http_port} (offset 5)
        backend_server_port={self.backend_server_port} (offset 6)
        embedding_rpc_server_port={self.embedding_rpc_server_port} (offset 7)
        remote_rpc_server_port={self.remote_rpc_server_port}
        cache_store_connect_port={self.cache_store_connect_port}
        cache_store_rdma_connect_port={self.cache_store_rdma_connect_port}
        local_rank={self.local_rank} world_rank={self.world_rank} name={self.name} ]
        """


@dataclass
class CoordinatorInfo:
    """Coordinator NCCL/connection info. Ports are derived from base_port via properties."""

    ip: str
    base_port: int
    dp_rank: int = 0

    def _rank_base_port(self) -> int:
        return self.base_port - self.dp_rank * COORDINATOR_INFO_PORT_NUM

    @property
    def dp_tp_nccl_port(self) -> int:
        return self.base_port - 10

    @property
    def th_nccl_port(self) -> int:
        return self.base_port - 11

    @property
    def tp_nccl_port(self) -> int:
        return self._rank_base_port() - 2

    @property
    def nccl_op_port(self) -> int:
        return self._rank_base_port() - 3

    @property
    def sp_gpt_nccl_port(self) -> int:
        return self._rank_base_port() - 4

    @property
    def ffn_tp_nccl_port(self) -> int:
        # note: reserve 4 ports for ffn_tp_nccl_port
        return self._rank_base_port() - 5

    def __str__(self) -> str:
        return (
            f"CoordinatorInfo(ip={self.ip}, base_port={self.base_port}, "
            f"dp_tp_nccl_port={self.dp_tp_nccl_port}, th_nccl_port={self.th_nccl_port}, "
            f"tp_nccl_port={self.tp_nccl_port}, nccl_op_port={self.nccl_op_port}, "
            f"sp_gpt_nccl_port={self.sp_gpt_nccl_port}, ffn_tp_nccl_port={self.ffn_tp_nccl_port})"
        )
