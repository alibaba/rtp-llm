"""Stable, controller-owned intra-gang addresses for multi-node SCR.

Opt in with RTP_LLM_SCR_VIP_INTERFACE. The controller must project the complete
C2 gang map into RTP_LLM_SCR_GANG_INFO (default /etc/c2/ganginfo). No address
range is assigned by RTP. External service advertisements still use Pod IPs.
"""

import ipaddress
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path


def enabled(pc) -> bool:
    from rtp_llm.utils.scr_template_lifecycle import template_phase_active

    return (
        bool(os.environ.get("RTP_LLM_SCR_VIP_INTERFACE"))
        and template_phase_active()
        and pc.world_size > pc.local_world_size
    )


def read_topology(
    world_size: int, local_world_size: int, *, external=False
) -> dict[int, str]:
    if local_world_size <= 0 or world_size % local_world_size:
        raise ValueError("SCR VIP requires equally sized nodes")
    raw = Path(os.environ.get("RTP_LLM_SCR_GANG_INFO", "/etc/c2/ganginfo")).read_text()
    rows = json.loads(raw)
    if not isinstance(rows, dict):
        raise ValueError("SCR gang info must be an object")
    nodes = {}
    external_nodes = {}
    for name, info in rows.items():
        match = re.search(r"(?:^|-)rank-(\d+)$", name)
        if match is None:
            raise ValueError(f"SCR gang member has no node rank: {name}")
        rank = int(match.group(1))
        vip = str(ipaddress.IPv4Address(info["ip"]))
        real = str(ipaddress.IPv4Address(info["real_ip"]))
        for address in (vip, real):
            parsed = ipaddress.IPv4Address(address)
            if parsed.is_loopback or parsed.is_unspecified or parsed.is_multicast:
                raise ValueError("SCR gang addresses must be unicast")
        if vip == real or rank in nodes or vip in nodes.values():
            raise ValueError(
                "SCR gang info requires distinct VIPs and unique node ranks"
            )
        nodes[rank] = vip
        external_nodes[rank] = real
    if set(nodes) != set(range(world_size // local_world_size)):
        raise ValueError("SCR gang info does not match the complete node topology")
    return external_nodes if external else nodes


def validate_device(address: str) -> None:
    device = os.environ["RTP_LLM_SCR_VIP_INTERFACE"]
    rows = json.loads(
        subprocess.check_output(
            ["ip", "-j", "-4", "address", "show", "dev", device],
            text=True,
            timeout=5,
        )
    )
    if not any(
        item.get("local") == address
        for row in rows
        for item in row.get("addr_info", [])
    ):
        raise RuntimeError(f"SCR interface {device} does not own {address}")
    ready_socket = Path(
        os.environ.get("RTP_LLM_SCR_NETWORK_READY_SOCKET", "/scr-share/snm/daemon.sock")
    )
    if not ready_socket.is_socket():
        raise RuntimeError(
            f"SCR network manager has not published readiness socket {ready_socket}"
        )
    # Binding checks the current namespace, including after CRIU restore.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((address, 0))


def topology(pc, *, wait: bool = False) -> dict[int, str]:
    if not 0 <= pc.world_rank < pc.world_size:
        raise ValueError("SCR world rank is outside the topology")
    deadline = time.monotonic() + (120 if wait else 0)
    while True:
        try:
            nodes = read_topology(pc.world_size, pc.local_world_size)
            validate_device(nodes[pc.world_rank // pc.local_world_size])
            return nodes
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            subprocess.SubprocessError,
        ):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)


def internal_ip(pc, fallback: str) -> str:
    if not enabled(pc):
        return fallback
    return topology(pc, wait=True)[pc.world_rank // pc.local_world_size]


def world_info(current, pc):
    """Validate stable endpoints after SCR's communicator restore barrier.

    This path keeps checkpointed TCP endpoints, rather than claiming an endpoint
    manifest rebuilt transports. A moved VIP or rank/port layout is an error.
    The network manager and SCR interposer own underlay and NCCL/RDMA recovery.
    """
    from rtp_llm.distribute.distributed_server import WorldInfo
    from rtp_llm.distribute.worker_info import WorkerInfo

    nodes = topology(pc)
    if current.self is None:
        raise ValueError("SCR VIP world requires a local worker")
    for member in current.members:
        if (
            member.ip != nodes[member.world_rank // pc.local_world_size]
            or member.local_rank != member.world_rank % pc.local_world_size
        ):
            raise RuntimeError(
                "SCR restore cannot change the checkpointed VIP/rank topology"
            )
    base = current.self
    members = [
        WorkerInfo(
            ip=nodes[rank // pc.local_world_size],
            local_rank=rank % pc.local_world_size,
            world_rank=rank,
            name=f"rank_{rank}",
            server_port=base._server_port,
            worker_info_port_num=base._worker_info_port_num,
            remote_server_port=base._remote_server_port,
        )
        for rank in range(pc.world_size)
    ]
    return WorldInfo(
        members=members,
        self=members[pc.world_rank],
        master=members[0],
        num_nodes=len(nodes),
        initialized=True,
    )
