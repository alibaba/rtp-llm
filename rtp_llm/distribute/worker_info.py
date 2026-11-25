import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rtp_llm.utils.util import split_ip_port

if TYPE_CHECKING:
    from rtp_llm.config.py_config_modules import PyEnvConfigs, ServerConfig


class WorkerInfo(object):
    """Worker identity and port layout for one rank.

    Port layout: base = server_port + local_rank * worker_info_port_num, then
    server_port = base+0, rpc_server_port = base+1, cache_store_* = base+2/4.
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
    def rpc_server_port(self) -> int:
        return self._base + 1

    @property
    def cache_store_listen_port(self) -> int:
        return self._base + 2

    @property
    def cache_store_rdma_listen_port(self) -> int:
        return self._base + 4

    def __str__(self) -> str:
        return (
            f"WorkerInfo(ip={self.ip} local_rank={self.local_rank} world_rank={self.world_rank} "
            f"name={self.name} server_port={self.server_port} rpc_server_port={self.rpc_server_port} "
            f"cache_store_listen_port={self.cache_store_listen_port} "
            f"cache_store_rdma_listen_port={self.cache_store_rdma_listen_port})"
        )


def workers_on_host(
    configs: "PyEnvConfigs", ip: str, start_port: int, first_rank: int
) -> list[WorkerInfo]:
    pc = configs.parallelism_config
    dc = configs.distribute_config
    return [
        WorkerInfo(
            ip=ip,
            local_rank=local_rank,
            world_rank=first_rank + local_rank,
            name=f"{dc.zone_name}_rank_{first_rank + local_rank}_{local_rank}",
            server_port=start_port,
            worker_info_port_num=configs.server_config.worker_info_port_num,
            remote_server_port=dc.remote_server_port,
        )
        for local_rank in range(min(pc.local_world_size, pc.world_size))
    ]


@dataclass(frozen=True)
class WorldInfo:
    """Serving addresses in final host order and the endpoint for collectives."""

    host_addresses: tuple[str, ...]  # One ip:start_port per host.
    collective_master: tuple[str, int]  # IP and base port for collectives.

    @classmethod
    def for_single_rank(cls, server_config: "ServerConfig") -> "WorldInfo":
        ip = server_config.ip or socket.gethostbyname(socket.gethostname())
        port = server_config.start_port
        return cls(host_addresses=(f"{ip}:{port}",), collective_master=(ip, port))

    @property
    def master(self) -> tuple[str, int]:
        """Serving IP and port of world rank zero."""
        return split_ip_port(self.host_addresses[0])

    def members(self, configs: "PyEnvConfigs") -> list[WorkerInfo]:
        members = []
        for address in self.host_addresses:
            ip, start_port = split_ip_port(address)
            members.extend(workers_on_host(configs, ip, start_port, len(members)))
        return members
