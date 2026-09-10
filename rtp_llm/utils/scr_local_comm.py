"""Opt-in communication for SCR ranks sharing one Pod network namespace."""

import ipaddress
import os
import socket


def local_comm_enabled():
    return os.environ.get("RTP_LLM_SCR_LOCAL_COMM") == "1"


def current_pod_ip():
    """Resolve the current network namespace, never a checkpoint-cached IP."""
    address = socket.gethostbyname(socket.gethostname())
    parsed = ipaddress.IPv4Address(address)
    if parsed.is_loopback or parsed.is_unspecified or parsed.is_multicast:
        raise ValueError("SCR external endpoints require a non-loopback Pod IP")
    return address


def cache_store_advertise_ip(world_info, parallelism_config):
    """Keep local control addresses separate from cross-Pod KV advertisements.

    Return an override only for the opt-in single-Pod loopback topology. Explicit
    routable manifest addresses retain their authority. Resolve on every fixup
    because both the seed environment and server configuration survive CRIU.
    """
    if not local_comm_enabled():
        return None
    validate_local_members(world_info, parallelism_config)
    if world_info.members[0].ip != "127.0.0.1":
        return None
    from rtp_llm.utils.scr_runtime_fixup import get_restore_runtime_identity

    identity = get_restore_runtime_identity()
    if identity is not None:
        return identity.pod_ip
    return current_pod_ip()


def validate_local_world(world_size, local_world_size, num_nodes):
    if world_size < 1 or local_world_size != world_size or num_nodes != 1:
        raise ValueError("RTP_LLM_SCR_LOCAL_COMM requires all ranks in one Pod")


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
            "RTP_LLM_SCR_LOCAL_COMM requires a complete local rank topology"
        )
