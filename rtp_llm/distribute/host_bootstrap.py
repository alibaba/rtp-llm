"""Apply optional whole-host selection after the original ranks rendezvous."""

import os

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import DistributedServer
from rtp_llm.distribute.startup_rank_selection import (
    exchange_host_health,
    select_startup_hosts,
)
from rtp_llm.distribute.worker_info import WorldInfo
from rtp_llm.utils.util import split_ip_port


def prepare_host(configs: PyEnvConfigs) -> bool:
    """Finish the host rendezvous and return whether to start local workers."""
    if not configs.runs_backend:
        return True

    dc = configs.distribute_config
    with DistributedServer(configs).rendezvous() as server:
        world = server.world_info
        if not dc.enable_startup_rank_selection:
            dc.world_info = world
            return True
        host_index = server.host_index
        health = exchange_host_health(configs, server)

    pc = configs.parallelism_config
    active_hosts = select_startup_hosts(health, pc, world.host_addresses)
    if host_index not in active_hosts:
        return False

    if len(active_hosts) < len(world.host_addresses):
        host_addresses = tuple(world.host_addresses[i] for i in active_hosts)
        collective_master = (
            world.collective_master
            if active_hosts[0] == 0
            else split_ip_port(host_addresses[0])
        )
        world = WorldInfo(host_addresses, collective_master)
        pc.world_size = len(active_hosts) * pc.local_world_size
        pc.world_rank = active_hosts.index(host_index) * pc.local_world_size
        pc.dp_size = pc.world_size // pc.tp_size
        pc.ep_size = pc.world_size
        os.environ.update(
            WORLD_SIZE=str(pc.world_size),
            WORLD_RANK=str(pc.world_rank),
            DP_SIZE=str(pc.dp_size),
            EP_SIZE=str(pc.ep_size),
        )
    dc.world_info = world
    return True
