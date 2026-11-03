"""Peer addressing for SCR: VIP topology plus single-Pod loopback.

Both shapes answer the same question: where does this rank advertise and how
does it reach its peers?

* multi-node: the platform provides ``scr_vxlan0`` and a network-manager
  readiness socket; gang membership comes from the existing Pod annotation
  projection.  Missing or invalid platform state fails startup instead of
  falling back to Pod IPs.
* single-Pod: every rank shares one network namespace, so control addresses
  stay on loopback while external advertisements keep the Pod IP.

``RTPLLM_ENABLE_SCR`` selects either path; without it every helper is inert.
"""

import ipaddress
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from rtp_llm.utils.gang_info import GangInfoReader, get_gang_member

VIP_INTERFACE = "scr_vxlan0"
NETWORK_READY_SOCKET = Path("/scr-share/snm/daemon.sock")


def enabled(pc) -> bool:
    from rtp_llm.utils.scr.lifecycle import template_phase_active

    return template_phase_active() and pc.world_size > pc.local_world_size


def _read_topology(world_size: int, local_world_size: int, gang_info: GangInfoReader):
    if local_world_size <= 0 or world_size % local_world_size:
        raise ValueError("SCR VIP requires equally sized nodes")
    rows = gang_info.read()
    if not isinstance(rows, dict):
        raise ValueError("SCR gang info must be an object")
    nodes = {}
    real_ip_by_vip = {}
    node_count = world_size // local_world_size
    if len(rows) != node_count:
        raise ValueError("SCR gang info does not match the complete node topology")
    for rank in range(node_count):
        info = get_gang_member(rows, rank)
        if info is None:
            raise ValueError(f"SCR gang info is missing node {rank}")
        vip = str(ipaddress.IPv4Address(info["ip"]))
        real = str(ipaddress.IPv4Address(info["real_ip"]))
        for address in (vip, real):
            parsed = ipaddress.IPv4Address(address)
            if parsed.is_loopback or parsed.is_unspecified or parsed.is_multicast:
                raise ValueError("SCR gang addresses must be unicast")
        if vip == real or vip in nodes.values():
            raise ValueError(
                "SCR gang info requires distinct VIPs and unique node ranks"
            )
        nodes[rank] = vip
        real_ip_by_vip[vip] = real
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
    node_count = pc.world_size // pc.local_world_size
    if os.environ.get("RANK_SIZE") != str(node_count):
        raise ValueError(
            f"SCR VIP requires RANK_SIZE={node_count} in the worker environment"
        )
    deadline = time.monotonic() + (120 if wait else 0)
    while True:
        try:
            nodes = read_topology(pc.world_size, pc.local_world_size, gang_info)
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


def internal_ip(pc, fallback: str, gang_info: GangInfoReader) -> str:
    if not enabled(pc):
        return fallback
    return topology(pc, gang_info, wait=True)[pc.world_rank // pc.local_world_size]


def configure_network(pc) -> None:
    """Select the platform interface before constructing collective transports."""
    if not enabled(pc):
        return
    # Launchers may default to an underlay interface (maga_start.sh uses eth0).
    # Once the SCR network is validated, its transports must use the VIP device.
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


def local_comm_enabled(
    world_size: int, local_world_size: int, num_nodes: int = 1
) -> bool:
    from rtp_llm.utils.scr.lifecycle import template_phase_active

    return (
        template_phase_active()
        and world_size > 0
        and world_size == local_world_size
        and num_nodes == 1
    )


def current_pod_ip():
    """Resolve the current network namespace, never a checkpoint-cached IP."""
    address = socket.gethostbyname(socket.gethostname())
    parsed = ipaddress.IPv4Address(address)
    if parsed.is_loopback or parsed.is_unspecified or parsed.is_multicast:
        raise ValueError("SCR external endpoints require a non-loopback Pod IP")
    return address


def cache_store_advertise_ip(world_info, parallelism_config):
    """Keep local control addresses separate from cross-Pod KV advertisements.

    Return an override only for the single-Pod SCR loopback topology. Explicit
    routable manifest addresses retain their authority. Resolve on every fixup
    because both the seed environment and server configuration survive CRIU.
    """
    if not local_comm_enabled(
        parallelism_config.world_size,
        parallelism_config.local_world_size,
        world_info.num_nodes,
    ):
        return None
    validate_local_members(world_info, parallelism_config)
    if world_info.members[0].ip != "127.0.0.1":
        return None
    from rtp_llm.utils.scr.runtime_fixup import get_restore_runtime_identity

    identity = get_restore_runtime_identity()
    if identity is not None:
        return identity.pod_ip
    return current_pod_ip()


def validate_local_world(world_size, local_world_size, num_nodes):
    if world_size < 1 or local_world_size != world_size or num_nodes != 1:
        raise ValueError("SCR loopback communication requires all ranks in one Pod")


def validate_local_members(world_info, parallelism_config):
    pc = parallelism_config
    validate_local_world(pc.world_size, pc.local_world_size, world_info.num_nodes)
    members = world_info.members
    # self.ip can retain the seed address while members resolve the restored Pod.
    # Validate the current topology, never equality with the checkpoint's self.ip.
    if (
        len(members) != pc.world_size
        or len({member.ip for member in members}) != 1
        or sorted(member.world_rank for member in members) != list(range(pc.world_size))
        or sorted(member.local_rank for member in members)
        != list(range(pc.local_world_size))
    ):
        raise ValueError(
            "SCR loopback communication requires a complete local rank topology"
        )
