"""Fresh, process-local inputs shared by all fixups after one SCR barrier."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, replace

from rtp_llm.utils.gang_info import GangInfoReader


@dataclass(frozen=True)
class RestoreContext:
    pod_ip: str

    def resolve_world_info(
        self, current, parallelism_config, gang_info: GangInfoReader
    ):
        """Keep checkpointed VIP routes or restore single-Pod control to loopback."""
        from rtp_llm.utils.scr import network

        pc = parallelism_config
        if network.enabled(pc):
            return network.validate_world_info(current, pc, gang_info)
        if not network.local_comm_enabled(
            pc.world_size, pc.local_world_size, current.num_nodes
        ):
            return current
        if sorted(member.local_rank for member in current.members) != list(
            range(pc.local_world_size)
        ):
            return current
        members = [copy(member) for member in current.members]
        for member in members:
            member.ip = "127.0.0.1"
        by_rank = {member.world_rank: member for member in members}
        return replace(
            current,
            members=members,
            self=by_rank[current.self.world_rank] if current.self else None,
            master=by_rank.get(current.master.world_rank) if current.master else None,
        )
