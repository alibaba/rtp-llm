import socket


class WorkerInfo(object):
    """Minimal worker identity and port layout for WorldInfo members.

    Port layout: base = server_port + local_rank * worker_info_port_num, then
    server_port = base+0, arpc_server_port = base+7, cache_store_* = base+2/4.
    Only these and (ip, world_rank, name) are read by callers; other ports removed.
    """

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
        self.local_rank = local_rank
        self.world_rank = world_rank
        self.name = name
        self._server_port = server_port
        self._worker_info_port_num = worker_info_port_num
        self._remote_server_port = (
            remote_server_port if remote_server_port is not None else server_port
        )

    @property
    def _base(self) -> int:
        return self._server_port + self.local_rank * self._worker_info_port_num

    @property
    def server_port(self) -> int:
        return self._base + 0

    @property
    def grpc_server_port(self) -> int:
        return self._base + 7

    @property
    def cache_store_listen_port(self) -> int:
        return self._base + 2

    @property
    def cache_store_rdma_listen_port(self) -> int:
        return self._base + 4

    def equals(self, other) -> bool:
        """True if other is the same worker (same ip and world_rank)."""
        if other is None or not isinstance(other, WorkerInfo):
            return False
        return self.ip == other.ip and self.world_rank == other.world_rank

    def __str__(self) -> str:
        return (
            f"WorkerInfo(ip={self.ip} local_rank={self.local_rank} world_rank={self.world_rank} "
            f"name={self.name} server_port={self.server_port} grpc_server_port={self.grpc_server_port} "
            f"cache_store_listen_port={self.cache_store_listen_port} "
            f"cache_store_rdma_listen_port={self.cache_store_rdma_listen_port})"
        )
