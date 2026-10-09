"""Stable, controller-owned intra-gang addresses for multi-node SCR.

RTPLLM_ENABLE_SCR selects this path for multi-node checkpoint/restore phases.
C2 gang membership comes from the existing Pod annotations projection.
The platform provides scr_vxlan0 and the network-manager readiness socket.
Missing or invalid platform state fails startup; it never falls back to Pod IPs. RTP does not assign an address range. External service
advertisements still use Pod IPs.
"""

import ipaddress
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

from rtp_llm.utils.gang_info import GangInfoReader

VIP_INTERFACE = "scr_vxlan0"
NETWORK_READY_SOCKET = Path("/scr-share/snm/daemon.sock")


def enabled(pc) -> bool:
    from rtp_llm.utils.scr_template_lifecycle import template_phase_active

    return template_phase_active() and pc.world_size > pc.local_world_size


def _read_topology(world_size: int, local_world_size: int, gang_info: GangInfoReader):
    if local_world_size <= 0 or world_size % local_world_size:
        raise ValueError("SCR VIP requires equally sized nodes")
    rows = gang_info.read()
    if not isinstance(rows, dict):
        raise ValueError("SCR gang info must be an object")
    nodes = {}
    real_ip_by_vip = {}
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
        real_ip_by_vip[vip] = real
    if set(nodes) != set(range(world_size // local_world_size)):
        raise ValueError("SCR gang info does not match the complete node topology")
    return nodes, real_ip_by_vip


def read_topology(
    world_size: int, local_world_size: int, gang_info: GangInfoReader
) -> dict[int, str]:
    return _read_topology(world_size, local_world_size, gang_info)[0]


def real_ip_by_vip(pc, gang_info: GangInfoReader) -> dict[str, str] | None:
    """Resolve the current underlay only when multi-node SCR uses VIPs."""
    if not enabled(pc):
        return None
    return _read_topology(pc.world_size, pc.local_world_size, gang_info)[1]


def validate_device(address: str) -> None:
    device = VIP_INTERFACE
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
    ready_socket = NETWORK_READY_SOCKET
    if not ready_socket.is_socket():
        raise RuntimeError(
            f"SCR network manager has not published readiness socket {ready_socket}"
        )
    # Binding checks the current namespace, including after CRIU restore.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((address, 0))


def topology(pc, gang_info: GangInfoReader, *, wait: bool = False) -> dict[int, str]:
    if not 0 <= pc.world_rank < pc.world_size:
        raise ValueError("SCR world rank is outside the topology")
    if pc.local_world_size <= 0 or pc.world_size % pc.local_world_size:
        raise ValueError("SCR VIP requires equally sized nodes")
    # The runc interposer reads node topology separately from RTP's GPU ranks.
    # Missing RANK_SIZE silently selects its single-node IPC registry.
    expected = {
        "RANK_SIZE": pc.world_size // pc.local_world_size,
        "RANK_ID": pc.world_rank // pc.local_world_size,
    }
    for name, value in expected.items():
        if os.environ.get(name) != str(value):
            raise ValueError(
                f"SCR VIP requires {name}={value} in the worker environment"
            )
    deadline = time.monotonic() + (120 if wait else 0)
    while True:
        try:
            nodes = read_topology(pc.world_size, pc.local_world_size, gang_info)
            validate_device(nodes[int(os.environ["RANK_ID"])])
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


def internal_ip(pc, fallback: str, gang_info: GangInfoReader) -> str:
    if not enabled(pc):
        return fallback
    return topology(pc, gang_info, wait=True)[int(os.environ["RANK_ID"])]


def configure_network(pc) -> None:
    """Select the platform interface before constructing collective transports."""
    if not enabled(pc):
        return
    for name in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"):
        configured = os.environ.get(name)
        if configured not in (None, VIP_INTERFACE):
            raise ValueError(
                f"SCR VIP requires {name}={VIP_INTERFACE}, got {configured!r}"
            )
    os.environ["NCCL_SOCKET_IFNAME"] = VIP_INTERFACE
    os.environ["GLOO_SOCKET_IFNAME"] = VIP_INTERFACE


def validate_world_info(current, pc, gang_info: GangInfoReader):
    """Validate checkpointed endpoints without changing membership or port layout.

    The network manager owns underlay and transport recovery. RTP verifies its
    existing VIP topology after the restore barrier, including frontend subsets.
    """
    nodes = topology(pc, gang_info, wait=True)
    if current.self is None:
        raise ValueError("SCR VIP world requires a local worker")
    if current.self.world_rank != pc.world_rank:
        raise ValueError("SCR VIP local worker does not match world rank")
    for member in [
        *current.members,
        current.self,
        *([current.master] if current.master else []),
    ]:
        if (
            not 0 <= member.world_rank < pc.world_size
            or member.ip != nodes[member.world_rank // pc.local_world_size]
            or member.local_rank != member.world_rank % pc.local_world_size
        ):
            raise RuntimeError(
                "SCR restore cannot change the checkpointed VIP/rank topology"
            )
    return current
