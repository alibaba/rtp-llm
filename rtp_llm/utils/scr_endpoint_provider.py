"""Restore-time endpoint manifest for host-bound distributed services."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rtp_llm.distribute.distributed_server import WorldInfo
from rtp_llm.distribute.worker_info import WorkerInfo


ENDPOINT_MANIFEST_ENV = "RTP_LLM_SCR_ENDPOINT_MANIFEST"


def _read_manifest() -> dict[str, Any] | None:
    value = os.environ.get(ENDPOINT_MANIFEST_ENV, "").strip()
    if not value:
        return None
    path = Path(value)
    raw = path.read_text() if path.exists() else value
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("SCR endpoint manifest must be a JSON object")
    return manifest


def resolve_world_info(
    current: WorldInfo,
    *,
    generation: str,
    require_manifest: bool = False,
    require_transport: bool = False,
) -> WorldInfo:
    """Replace transport endpoints while preserving logical rank identity.

    The control plane may provide a file path or inline JSON through
    ``RTP_LLM_SCR_ENDPOINT_MANIFEST``.  A missing manifest is tolerated during
    the initial checkpoint, but never during a restore of a multi-node world.
    """
    manifest = _read_manifest()
    if manifest is None:
        if require_manifest and current.num_nodes > 1:
            raise RuntimeError(
                "SCR restore requires a current endpoint manifest for a multi-node world"
            )
        if (
            os.environ.get("RTP_LLM_SCR_LOCAL_COMM") == "1"
            and current.num_nodes == 1
            and sorted(member.local_rank for member in current.members)
            == list(range(len(current.members)))
        ):
            members = [
                WorkerInfo(
                    ip="127.0.0.1",
                    local_rank=member.local_rank,
                    world_rank=member.world_rank,
                    name=member.name,
                    server_port=member._server_port,
                    worker_info_port_num=member._worker_info_port_num,
                    remote_server_port=member._remote_server_port,
                )
                for member in current.members
            ]
            by_rank = {member.world_rank: member for member in members}
            return WorldInfo(
                members=members,
                self=by_rank[current.self.world_rank] if current.self else None,
                master=by_rank.get(current.master.world_rank) if current.master else None,
                num_nodes=current.num_nodes,
                initialized=current.initialized,
            )
        return current
    actual_generation = str(manifest.get("generation", ""))
    if require_manifest and not actual_generation:
        raise ValueError("SCR endpoint manifest requires a non-empty generation")
    if require_transport:
        transport = manifest.get("transport")
        if not isinstance(transport, dict) or transport.get("ready") is not True:
            raise RuntimeError(
                "SCR restore requires transport.ready=true after NCCL/TCPStore/RDMA fixup"
            )
    if generation and actual_generation and actual_generation != generation:
        raise RuntimeError(
            f"SCR endpoint manifest generation mismatch expected={generation} actual={actual_generation}"
        )
    rows = manifest.get("members")
    if not isinstance(rows, list) or not rows:
        raise ValueError("SCR endpoint manifest members must be a non-empty list")
    by_rank: dict[int, WorkerInfo] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("SCR endpoint manifest member must be an object")
        rank = int(row["world_rank"])
        local_rank = int(row["local_rank"])
        host = str(row.get("ip", row.get("host", ""))).strip()
        if not host:
            raise ValueError(f"SCR endpoint manifest rank {rank} has no host")
        base_port = int(row["server_port"])
        if rank in by_rank:
            raise ValueError(f"duplicate SCR endpoint manifest rank {rank}")
        by_rank[rank] = WorkerInfo(
            ip=host,
            local_rank=local_rank,
            world_rank=rank,
            name=str(row.get("name", f"rank_{rank}")),
            server_port=base_port,
            worker_info_port_num=int(
                row.get("worker_info_port_num", current.self._worker_info_port_num if current.self else 9)
            ),
            remote_server_port=int(row.get("remote_server_port", base_port)),
        )
    expected = set(range(len(current.members)))
    if set(by_rank) != expected:
        raise ValueError(
            f"SCR endpoint manifest ranks {sorted(by_rank)} do not match world {sorted(expected)}"
        )
    manifest_nodes = int(manifest.get("num_nodes", current.num_nodes))
    if manifest_nodes <= 0:
        raise ValueError("SCR endpoint manifest num_nodes must be positive")
    members = [by_rank[rank] for rank in sorted(by_rank)]
    self_rank = current.self.world_rank if current.self is not None else 0
    master_rank = current.master.world_rank if current.master is not None else 0
    return WorldInfo(
        members=members,
        self=by_rank[self_rank],
        master=by_rank.get(master_rank),
        num_nodes=manifest_nodes,
        initialized=True,
    )
