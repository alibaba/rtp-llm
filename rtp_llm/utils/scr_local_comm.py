"""Opt-in communication for SCR ranks sharing one Pod network namespace."""

import os


def local_comm_enabled():
    return os.environ.get("RTP_LLM_SCR_LOCAL_COMM") == "1"


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
