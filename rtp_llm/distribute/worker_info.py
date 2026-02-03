import logging
import socket
from typing import Optional

from rtp_llm.config.py_config_modules import MASTER_INFO_PORT_NUM


class FrontendServerInfo(object):
    def __init__(self, frontend_server_id: int):
        self.frontend_server_id = frontend_server_id

    def __str__(self):
        return f"FrontendServerInfo:[ frontend_server_id={self.frontend_server_id} ]"


class WorkerInfo(object):
    """Worker information including IP, rank, and port configuration.

    Ports are calculated automatically based on start_port, local_rank, and worker_info_port_num.
    Master-related ports are calculated based on master_base_port and parallelism configuration.
    """

    # ============================================================================
    # Initialization
    # ============================================================================

    def __init__(
        self,
        ip: str,
        local_rank: int,
        world_rank: int,
        local_world_size: int = 1,
        start_port: Optional[int] = None,
        remote_server_port: Optional[int] = None,
        worker_info_port_num: Optional[int] = None,
        master_ip: str = "",
    ):
        self.ip = ip  # Worker IP (current worker's IP)
        self.local_rank: int = local_rank
        self.world_rank: int = world_rank
        self.local_world_size: int = local_world_size

        # Port configuration for property-based calculation
        self._start_port: Optional[int] = start_port
        self._remote_server_port: Optional[int] = remote_server_port
        self._worker_info_port_num: Optional[int] = worker_info_port_num

        # Master info fields
        self.master_ip = master_ip  # Master IP (for NCCL communication)
        # Master port configuration for property-based calculation
        self._master_base_port: Optional[int] = None
        self._master_dp_rank: int = 0
        self._master_ffn_sp_size: int = 1
        self._master_tp_size: int = 1

    # ============================================================================
    # Public Instance Methods
    # ============================================================================

    def configure_ports(
        self,
        local_rank: int,
        world_rank: int,
        start_port: int,
        remote_server_port: int,
        worker_info_port_num: int,
        local_world_size: int,
    ) -> None:
        """Configure port calculation parameters. After calling this, ports will be calculated automatically.

        Args:
            local_rank: Local rank for this process
            world_rank: World rank for this process
            start_port: Base start port
            remote_server_port: Base remote server port
            worker_info_port_num: Number of ports per worker
            local_world_size: Local world size for this process
        """
        self._start_port = start_port
        self._remote_server_port = remote_server_port
        self._worker_info_port_num = worker_info_port_num
        self.local_rank = local_rank
        self.world_rank = world_rank
        self.local_world_size = local_world_size

    def adjust_ports_by_rank_id(self, rank_id: int, worker_info_port_num: int) -> None:
        """Adjust port configuration based on rank_id offset.

        This method adjusts the internal _start_port by rank_id offset, which will
        automatically update all calculated ports (server_port, backend_server_port, etc.).

        Args:
            rank_id: Rank ID offset to apply
            worker_info_port_num: Number of ports per worker
        """
        if self._start_port is not None:
            # Adjust start_port by rank_id offset
            self._start_port = WorkerInfo.server_port_offset(
                rank_id,
                self._start_port,
                worker_info_port_num,
            )
        else:
            # If start_port is not set, get current server_port and calculate new start_port
            # Reverse the calculation: start_port = server_port - local_rank * worker_info_port_num
            current_server_port = self.server_port
            if worker_info_port_num > 0 and self.local_rank > 0:
                base_start_port = (
                    current_server_port - self.local_rank * worker_info_port_num
                )
            else:
                base_start_port = current_server_port
            # Now apply rank_id offset to get new start_port
            new_start_port = WorkerInfo.server_port_offset(
                rank_id,
                base_start_port,
                worker_info_port_num,
            )
            # Reconfigure with new start_port
            self.configure_ports(
                local_rank=self.local_rank,
                world_rank=self.world_rank,
                start_port=new_start_port,
                remote_server_port=self._remote_server_port or 0,
                worker_info_port_num=worker_info_port_num,
                local_world_size=self.local_world_size,
            )

    def update_master_info(self, ip: str, base_port: int, parallelism_config):
        """Update master info configuration. Ports will be calculated automatically via properties.

        Note: This method stores master configuration and master_ip in this WorkerInfo,
        but does NOT modify self.ip. The self.ip should remain as the current
        worker's IP, not the master IP.

        Args:
            ip: Master IP address
            base_port: Master base port (master_server_port)
            parallelism_config: ParallelismConfig containing dp_rank, ffn_sp_size, tp_size
        """
        # Store master IP separately from worker IP
        self.master_ip = ip
        # Store configuration for property-based calculation
        self._master_base_port = base_port
        self._master_dp_rank = parallelism_config.dp_rank
        self._master_ffn_sp_size = parallelism_config.ffn_sp_size
        self._master_tp_size = parallelism_config.tp_size
        logging.info(
            f"Updated master info in WorkerInfo: master_ip={self.master_ip}, "
            f"base_port={base_port}, dp_rank={self._master_dp_rank}, "
            f"ffn_sp_size={self._master_ffn_sp_size}, tp_size={self._master_tp_size}"
        )

    # ============================================================================
    # Static Factory Methods
    # ============================================================================

    @staticmethod
    def from_parallelism_config(
        parallelism_config, start_port, remote_server_port, worker_info_port_num
    ):
        """Create WorkerInfo from ParallelismConfig (doesn't depend on global variables)

        Ports will be automatically calculated based on start_port, remote_server_port,
        worker_info_port_num, and local_rank.
        """
        local_rank = parallelism_config.local_rank
        world_rank = parallelism_config.world_rank
        local_world_size = parallelism_config.local_world_size

        worker_info = WorkerInfo(
            ip=socket.gethostbyname(socket.gethostname()),
            local_rank=local_rank,
            world_rank=world_rank,
            local_world_size=local_world_size,
            start_port=start_port,
            remote_server_port=remote_server_port,
            worker_info_port_num=worker_info_port_num,
        )
        logging.info(
            f"WorkerInfo from_parallelism_config: {worker_info}, worker_info_port_num: {worker_info_port_num}, local_rank: {local_rank}"
        )

        return worker_info

    # ============================================================================
    # Static Utility Methods (Port Offset Calculations)
    # ============================================================================

    @staticmethod
    def server_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        """Calculate server port with rank offset."""
        return server_port + local_rank * worker_info_port_num

    @staticmethod
    def rpc_server_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        """Calculate RPC server port with rank offset."""
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 1
        )

    @staticmethod
    def cache_store_listen_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        """Calculate cache store listen port with rank offset."""
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 2
        )

    @staticmethod
    def gang_hb_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        """Calculate gang heartbeat port with rank offset."""
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 3
        )

    @staticmethod
    def cache_store_rdma_listen_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        """Calculate cache store RDMA listen port with rank offset."""
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 4
        )

    @staticmethod
    def http_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        """Calculate HTTP port with rank offset."""
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 5
        )

    @staticmethod
    def backend_server_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        """Calculate backend server port with rank offset."""
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 6
        )

    @staticmethod
    def embedding_rpc_server_port_offset(
        local_rank: int = 0, server_port: int = 0, worker_info_port_num: int = 0
    ) -> int:
        """Calculate embedding RPC server port with rank offset."""
        return (
            WorkerInfo.server_port_offset(local_rank, server_port, worker_info_port_num)
            + 7
        )

    # ============================================================================
    # Property Accessors - Worker Ports (based on base_port)
    # ============================================================================

    @property
    def server_port(self) -> int:
        """Server port (offset 0 from base_port)."""
        return self._get_base_port()

    @property
    def rpc_server_port(self) -> int:
        """RPC server port (offset 1 from base_port)."""
        return self._get_base_port() + 1

    @property
    def cache_store_listen_port(self) -> int:
        """Cache store listen port (offset 2 from base_port)."""
        return self._get_base_port() + 2

    @property
    def gang_hb_port(self) -> int:
        """Gang heartbeat port (offset 3 from base_port)."""
        return self._get_base_port() + 3

    @property
    def cache_store_rdma_listen_port(self) -> int:
        """Cache store RDMA listen port (offset 4 from base_port)."""
        return self._get_base_port() + 4

    @property
    def http_port(self) -> int:
        """HTTP port (offset 5 from base_port)."""
        return self._get_base_port() + 5

    @property
    def backend_server_port(self) -> int:
        """Backend server port (offset 6 from base_port)."""
        return self._get_base_port() + 6

    @property
    def embedding_rpc_server_port(self) -> int:
        """Embedding RPC server port (offset 7 from base_port)."""
        return self._get_base_port() + 7

    # ============================================================================
    # Property Accessors - Remote Ports (based on remote_base_port)
    # ============================================================================

    @property
    def remote_rpc_server_port(self) -> int:
        """Remote RPC server port (offset 1 from remote_base_port)."""
        return self._get_remote_base_port() + 1

    @property
    def cache_store_connect_port(self) -> int:
        """Cache store connect port (offset 2 from remote_base_port)."""
        return self._get_remote_base_port() + 2

    @property
    def cache_store_rdma_connect_port(self) -> int:
        """Cache store RDMA connect port (offset 4 from remote_base_port)."""
        return self._get_remote_base_port() + 4

    # ============================================================================
    # Property Accessors - Master NCCL Ports
    # ============================================================================

    @property
    def dp_tp_nccl_port(self) -> int:
        """DP TP NCCL port (base_port - 10)."""
        if self._master_base_port is None:
            return 0
        return self._master_base_port - 10

    @property
    def th_nccl_port(self) -> int:
        """TH NCCL port (base_port - 11)."""
        if self._master_base_port is None:
            return 0
        return self._master_base_port - 11

    @property
    def tp_nccl_port(self) -> int:
        """TP NCCL port (adjusted_base_port - 2). Per-rank: base_port - dp_rank * MASTER_INFO_PORT_NUM - 2."""
        if self._master_base_port is None:
            return 0
        return self._get_master_calculated_base_port() - 2

    @property
    def nccl_op_port(self) -> int:
        """NCCL OP port (adjusted_base_port - 3)."""
        if self._master_base_port is None:
            return 0
        return self._get_master_calculated_base_port() - 3

    @property
    def sp_gpt_nccl_port(self) -> int:
        """SP GPT NCCL port (adjusted_base_port - 4)."""
        if self._master_base_port is None:
            return 0
        return self._get_master_calculated_base_port() - 4

    @property
    def ffn_tp_nccl_port(self) -> int:
        """FFN TP NCCL port (adjusted_base_port - 5). Same formula as pre-refactor: base_port -= dp_rank*MASTER_INFO_PORT_NUM; ffn_tp_nccl_port = base_port - 5 (no ffn_sp_size subtract)."""
        if self._master_base_port is None:
            return 0
        return self._get_master_calculated_base_port() - 5

    # ============================================================================
    # Private Helper Methods
    # ============================================================================

    def _get_base_port(self) -> int:
        """Calculate base port based on local_rank and configuration.

        Formula: base_port = start_port + local_rank * worker_info_port_num

        If worker_info_port_num is None or 0, start_port is treated as the already-calculated base_port.
        """
        if self._start_port is None:
            raise ValueError(
                "Port configuration not set. Either provide start_port/worker_info_port_num "
                "in __init__ or call configure_ports()"
            )
        if self._worker_info_port_num is None or self._worker_info_port_num == 0:
            # If worker_info_port_num is not set or 0, treat start_port as already-calculated base_port
            return self._start_port
        return self._start_port + self.local_rank * self._worker_info_port_num

    def _get_remote_base_port(self) -> int:
        """Calculate remote base port based on local_rank and configuration.

        Formula: remote_base_port = remote_server_port + local_rank * worker_info_port_num

        If worker_info_port_num is None or 0, remote_server_port is treated as the already-calculated remote_base_port.
        """
        if self._remote_server_port is None:
            return 0
        if self._worker_info_port_num is None or self._worker_info_port_num == 0:
            # If worker_info_port_num is not set or 0, treat remote_server_port as already-calculated remote_base_port
            return self._remote_server_port
        return self._remote_server_port + self.local_rank * self._worker_info_port_num

    def _get_master_calculated_base_port(self) -> int:
        """Calculate the adjusted base_port after dp_rank offset.

        Formula: adjusted_base_port = base_port - dp_rank * MASTER_INFO_PORT_NUM
        """
        if self._master_base_port is None:
            return 0
        return self._master_base_port - self._master_dp_rank * MASTER_INFO_PORT_NUM

    def _get_master_ffn_base_port(self) -> int:
        """Calculate the base port for FFN ports (after ffn_sp_size adjustment if needed).

        Formula: ffn_base_port = adjusted_base_port - ffn_sp_size (if ffn_sp_size != tp_size)
        """
        base_port = self._get_master_calculated_base_port()
        if self._master_ffn_sp_size != self._master_tp_size:
            base_port -= self._master_ffn_sp_size
        return base_port

    def _get_master_ffn_base_port_unified(self) -> int:
        """Master's FFN base port, same for all ranks (for C++ parallelism_config consistency).

        Formula: master_base_port - ffn_sp_size (if ffn_sp_size != tp_size), else master_base_port.
        """
        if self._master_base_port is None:
            return 0
        base_port = self._master_base_port
        if self._master_ffn_sp_size != self._master_tp_size:
            base_port -= self._master_ffn_sp_size
        return base_port

    def __str__(self):
        """String representation of WorkerInfo."""
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
        local_rank={self.local_rank} world_rank={self.world_rank} local_world_size={self.local_world_size}
        master_info: th_nccl_port={self.th_nccl_port} tp_nccl_port={self.tp_nccl_port} nccl_op_port={self.nccl_op_port}
        sp_gpt_nccl_port={self.sp_gpt_nccl_port} dp_tp_nccl_port={self.dp_tp_nccl_port} ffn_tp_nccl_port={self.ffn_tp_nccl_port} ]
        """
