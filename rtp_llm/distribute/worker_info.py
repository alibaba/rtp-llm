import json
import logging
import os
import socket
from dataclasses import dataclass
from typing import Any, Dict

import torch

from rtp_llm.config.py_config_modules import (
    MASTER_INFO_PORT_NUM,
    MIN_WORKER_INFO_PORT_NUM,
)


class FrontendServerInfo(object):
    def __init__(self, frontend_server_id: int):
        self.frontend_server_id = frontend_server_id

    def __str__(self):
        return f"FrontendServerInfo:[ frontend_server_id={self.frontend_server_id} ]"


class WorkerInfo(object):
    def __init__(
        self,
        ip: str,
        server_port: int,
        gang_hb_port: int,
        http_port: int,
        rpc_server_port: int,
        embedding_rpc_server_port: int,
        remote_rpc_server_port: int,
        cache_store_listen_port: int,
        cache_store_connect_port: int,
        cache_store_rdma_listen_port: int,
        cache_store_rdma_connect_port: int,
        backend_server_port: int,
        local_rank: int,
        world_rank: int,
        name: str,
        info: Any,
        local_world_size: int = 1,
        # Master info fields (merged from MasterInfo)
        th_nccl_port: int = 0,
        tp_nccl_port: int = 0,
        nccl_op_port: int = 0,
        sp_gpt_nccl_port: int = 0,
        dp_tp_nccl_port: int = 0,
        ffn_tp_nccl_port: int = 0,
        master_ip: str = "",
    ):
        self.ip = ip  # Worker IP (current worker's IP)
        self.server_port = server_port
        self.gang_hb_port = gang_hb_port
        self.http_port = http_port
        self.rpc_server_port = rpc_server_port
        self.embedding_rpc_server_port = embedding_rpc_server_port
        self.remote_rpc_server_port = remote_rpc_server_port
        self.cache_store_listen_port = cache_store_listen_port
        self.cache_store_connect_port = cache_store_connect_port
        self.cache_store_rdma_listen_port = cache_store_rdma_listen_port
        self.cache_store_rdma_connect_port = cache_store_rdma_connect_port
        self.backend_server_port = backend_server_port
        self.local_rank: int = local_rank
        self.world_rank: int = world_rank
        self.local_world_size: int = local_world_size
        self.name = name
        self.info = info
        # Master info fields
        self.master_ip = master_ip  # Master IP (for NCCL communication)
        self.th_nccl_port = th_nccl_port
        self.tp_nccl_port = tp_nccl_port
        self.nccl_op_port = nccl_op_port
        self.sp_gpt_nccl_port = sp_gpt_nccl_port
        self.dp_tp_nccl_port = dp_tp_nccl_port
        self.ffn_tp_nccl_port = ffn_tp_nccl_port

    def equals(self, other: "WorkerInfo") -> bool:
        return self.ip == other.ip and self.server_port == other.server_port

    def __eq__(self, other):
        if not isinstance(other, WorkerInfo):
            return False
        return (
            self.ip == other.ip
            and self.server_port == other.server_port
            and self.gang_hb_port == other.gang_hb_port
            and self.http_port == other.http_port
            and self.rpc_server_port == other.rpc_server_port
            and self.embedding_rpc_server_port == other.embedding_rpc_server_port
            and self.remote_rpc_server_port == other.remote_rpc_server_port
            and self.cache_store_listen_port == other.cache_store_listen_port
            and self.cache_store_connect_port == other.cache_store_connect_port
            and self.cache_store_rdma_listen_port == other.cache_store_rdma_listen_port
            and self.cache_store_rdma_connect_port
            == other.cache_store_rdma_connect_port
            and self.backend_server_port == other.backend_server_port
            and self.local_rank == other.local_rank
            and self.world_rank == other.world_rank
            and self.local_world_size == other.local_world_size
            and self.name == other.name
        )

    @staticmethod
    def from_parallelism_config(
        parallelism_config, start_port, remote_server_port, worker_info_port_num
    ):
        """Create WorkerInfo from ParallelismConfig (doesn't depend on global variables)"""
        local_rank = parallelism_config.local_rank
        world_rank = parallelism_config.world_rank
        local_world_size = parallelism_config.local_world_size

        info = WorkerInfo(
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
            local_world_size=local_world_size,
            name="",
            info=None,
        )
        logging.info(
            f"WorkerInfo from_parallelism_config: {info}, worker_info_port_num: {worker_info_port_num}, local_rank: {local_rank}"
        )

        return info

    @staticmethod
    def server_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        base_port = server_port
        return base_port + local_rank * worker_info_port_num

    @staticmethod
    def rpc_server_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 1
        )

    @staticmethod
    def cache_store_listen_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 2
        )

    @staticmethod
    def gang_hb_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 3
        )

    @staticmethod
    def cache_store_rdma_listen_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 4
        )

    @staticmethod
    def http_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 5
        )

    @staticmethod
    def backend_server_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 6
        )

    @staticmethod
    def embedding_rpc_server_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 7
        )

    @staticmethod
    def update_ports(
        worker_info: "WorkerInfo",
        local_rank: int,
        world_rank: int,
        start_port: int,
        remote_server_port: int,
        worker_info_port_num: int,
        local_world_size: int,
    ) -> None:
        """
        Update worker_info ports based on local_rank and world_rank.
        This replaces the reload() method with explicit parameter-based calculation.

        Port calculation formula:
        - base_port = start_port + local_rank * worker_info_port_num
        - Each port type has a fixed offset from base_port (0, 1, 2, 3, 4, 5, 6, 7)

        Args:
            worker_info: WorkerInfo instance to update
            local_rank: Local rank for this process
            world_rank: World rank for this process
            start_port: Base start port
            remote_server_port: Base remote server port
            worker_info_port_num: Number of ports per worker
            local_world_size: Local world size for this process
        """
        # Calculate base port for this local_rank
        base_port = start_port + local_rank * worker_info_port_num
        remote_base_port = remote_server_port + local_rank * worker_info_port_num

        # Update all ports based on fixed offsets
        worker_info.server_port = base_port + 0
        worker_info.rpc_server_port = base_port + 1
        worker_info.cache_store_listen_port = base_port + 2
        worker_info.gang_hb_port = base_port + 3
        worker_info.cache_store_rdma_listen_port = base_port + 4
        worker_info.http_port = base_port + 5
        worker_info.backend_server_port = base_port + 6
        worker_info.embedding_rpc_server_port = base_port + 7

        # Update remote ports
        worker_info.remote_rpc_server_port = remote_base_port + 1
        worker_info.cache_store_connect_port = remote_base_port + 2
        worker_info.cache_store_rdma_connect_port = remote_base_port + 4

        # Update rank information
        worker_info.local_rank = local_rank
        worker_info.world_rank = world_rank
        worker_info.local_world_size = local_world_size

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
        local_rank={self.local_rank} world_rank={self.world_rank} local_world_size={self.local_world_size} name={self.name} info={self.info}
        master_info: th_nccl_port={self.th_nccl_port} tp_nccl_port={self.tp_nccl_port} nccl_op_port={self.nccl_op_port}
        sp_gpt_nccl_port={self.sp_gpt_nccl_port} dp_tp_nccl_port={self.dp_tp_nccl_port} ffn_tp_nccl_port={self.ffn_tp_nccl_port} ]
        """


# MasterInfo has been merged into WorkerInfo
# The master info fields (th_nccl_port, tp_nccl_port, etc.) are now part of WorkerInfo


def update_master_info(
    worker_info: WorkerInfo, ip: str, base_port: int, parallelism_config
):
    """Update master info fields in WorkerInfo based on base_port and parallelism config

    Note: This function updates master-related port fields and master_ip in worker_info,
    but does NOT modify worker_info.ip. The worker_info.ip should remain as the current
    worker's IP, not the master IP.
    """
    # Store master IP separately from worker IP
    worker_info.master_ip = ip
    worker_info.dp_tp_nccl_port = base_port - 10
    worker_info.th_nccl_port = base_port - 11
    base_port -= parallelism_config.dp_rank * MASTER_INFO_PORT_NUM
    worker_info.tp_nccl_port = base_port - 2
    worker_info.nccl_op_port = base_port - 3
    worker_info.sp_gpt_nccl_port = base_port - 4
    # note: reserve 4 ports for ffn_tp_nccl_port
    worker_info.ffn_tp_nccl_port = base_port - 5
    if parallelism_config.ffn_sp_size != parallelism_config.tp_size:
        base_port -= parallelism_config.ffn_sp_size
    logging.info(
        f"Updated master info in WorkerInfo: master_ip={worker_info.master_ip}, tp_nccl_port={worker_info.tp_nccl_port}, dp_tp_nccl_port={worker_info.dp_tp_nccl_port}"
    )


def total_need_port_num(parallelism_config, worker_info_port_num: int) -> int:
    return (
        MASTER_INFO_PORT_NUM * parallelism_config.dp_size
        + worker_info_port_num * parallelism_config.tp_size
    )
