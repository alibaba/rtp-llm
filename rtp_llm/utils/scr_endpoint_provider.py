"""Restore-time endpoint manifest for host-bound distributed services."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from rtp_llm.distribute.distributed_server import WorldInfo
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.utils.scr_local_comm import local_comm_enabled

ENDPOINT_MANIFEST_ENV = "RTP_LLM_SCR_ENDPOINT_MANIFEST"
_MANIFEST_UNSET = object()


def _read_manifest() -> dict[str, Any] | None:
    value = os.environ.get(ENDPOINT_MANIFEST_ENV, "").strip()
    if not value:
        return None
    # Inline manifests routinely exceed NAME_MAX.  Never stat JSON as a path.
    raw = value if value.startswith("{") else Path(value).read_text()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("SCR endpoint manifest must be a JSON object")
    return manifest


def read_restore_manifest(generation: str) -> dict[str, Any] | None:
    """Read the controller-owned file after the barrier, not snapshotted env.

    Controllers may atomically replace the file with phase=restore.  CRIU does
    not update Python's os.environ when restoring a process into a new Pod.
    """
    manifest = _read_manifest()
    if manifest is not None:
        actual = str(manifest.get("generation", ""))
        if generation and actual and actual != generation:
            raise RuntimeError(
                f"SCR endpoint manifest generation mismatch expected={generation} actual={actual}"
            )
        if "phase" in manifest and manifest["phase"] not in {"checkpoint", "restore"}:
            raise ValueError(
                "SCR endpoint manifest phase must be checkpoint or restore"
            )
    return manifest


def is_restore_phase(manifest: dict[str, Any] | None) -> bool:
    # A restored process may retain SCR_PHASE=checkpoint from the snapshot.
    # A controller-updated manifest can promote it to restore, but must never
    # weaken a restore requirement already declared by the environment.
    return (
        os.environ.get("SCR_PHASE", "").strip().lower() == "restore"
        or (manifest or {}).get("phase") == "restore"
    )


def resolve_world_info(
    current: WorldInfo,
    *,
    generation: str,
    require_manifest: bool = False,
    require_transport: bool = False,
    expected_world_size: int | None = None,
    manifest: Any = _MANIFEST_UNSET,
) -> WorldInfo:
    """Replace transport endpoints while preserving logical rank identity.

    The control plane may provide a file path or inline JSON through
    ``RTP_LLM_SCR_ENDPOINT_MANIFEST``.  A missing manifest is tolerated during
    the initial checkpoint, but never during a restore of a multi-node world.
    """
    if manifest is _MANIFEST_UNSET:
        manifest = read_restore_manifest(generation)
    if manifest is None:
        if require_manifest or require_transport:
            raise RuntimeError(
                "SCR restore requires a current endpoint manifest for distributed serving"
            )
        if local_comm_enabled(
            (
                expected_world_size
                if expected_world_size is not None
                else len(current.members)
            ),
            len(current.members),
            current.num_nodes,
        ) and sorted(member.local_rank for member in current.members) == list(
            range(len(current.members))
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
        if rank < 0 or local_rank < 0:
            raise ValueError("SCR endpoint manifest ranks must be non-negative")
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
        member = by_rank[rank]
        if any(
            not 0 < port <= 65535
            for port in (
                member.server_port,
                member.rpc_server_port,
                member.cache_store_listen_port,
                member.cache_store_rdma_listen_port,
            )
        ):
            raise ValueError(
                f"SCR endpoint manifest rank {rank} has an invalid port layout"
            )
    expected = (
        set(range(expected_world_size))
        if expected_world_size is not None
        else {member.world_rank for member in current.members}
    )
    if set(by_rank) != expected:
        raise ValueError(
            f"SCR endpoint manifest ranks {sorted(by_rank)} do not match world {sorted(expected)}"
        )
    manifest_nodes = int(manifest.get("num_nodes", current.num_nodes))
    if manifest_nodes <= 0:
        raise ValueError("SCR endpoint manifest num_nodes must be positive")
    members = [by_rank[rank] for rank in sorted(by_rank)]
    for previous in current.members:
        if by_rank[previous.world_rank].local_rank != previous.local_rank:
            raise ValueError("SCR restore cannot change logical/local rank identity")
    self_rank = current.self.world_rank if current.self is not None else 0
    master_rank = current.master.world_rank if current.master is not None else 0
    return WorldInfo(
        members=members,
        self=by_rank[self_rank],
        master=by_rank.get(master_rank),
        num_nodes=manifest_nodes,
        initialized=True,
    )
