# SPDX-License-Identifier: Apache-2.0
"""Startup-only, same-node immutable GPU scale shards using Torch symmetric memory.

Every WORLD rank must call select_scale_group once in the same startup order,
including ranks with the feature disabled. Members of a selected group must
then call create_sharded_scales in the same table order. Neither function may
run during capture. Forward uses only descriptor metadata and fixed GPU pointers.

Torch's CUDA symmetric-memory backend requires matching rendezvous histories
among members (its exchange sequence is process-wide in Torch 2.11). These are
fixed startup groups, not a facility for regrouping ranks after asymmetric
symmetric-memory initialization on other groups.

Published generations are intentionally retained for the process lifetime:
local destruction cannot prove that another rank has drained its CUDA graphs.
Reloading a different table therefore retains the previous allocation. There is
no destructor collective or local close operation. CUDA symmetric-memory setup
errors are startup errors, never permission for a rank-local runtime fallback.
"""

import hashlib
import json
import logging
import socket
import threading
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

_AGREEMENT_TIMEOUT_SECONDS = 120
_EPOCHS = {}
_EPOCH_LOCK = threading.Lock()
_GENERATIONS = {}
_NODE_GROUPS = {}
_FAILED_RENDEZVOUS_RESOURCES = []


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _members(group):
    return tuple(dist.get_process_group_ranks(group))


def _group_key(group):
    return str(group.group_name), _members(group)


def _exchange(group, payload, phase):
    """Bounded CPU/store exchange, independent of GPU collective health."""
    name, members = _group_key(group)
    rank = dist.get_rank(group)
    if rank < 0:
        raise ValueError("Nonmember cannot enter scale-shard agreement")
    identity = hashlib.sha256(_json([name, members]).encode()).hexdigest()
    with _EPOCH_LOCK:
        key = (name, members, phase)
        epoch = _EPOCHS.get(key, 0)
        _EPOCHS[key] = epoch + 1
    prefix = f"rtp_llm/engram_scale_shards/{identity}/{phase}/{epoch}/"
    store = dist.distributed_c10d._get_default_store()
    store.set(prefix + str(rank), _json(payload).encode())
    keys = [prefix + str(index) for index in range(len(members))]
    store.wait(keys, timedelta(seconds=_AGREEMENT_TIMEOUT_SECONDS))
    return [json.loads(store.get(key)) for key in keys]


def _device(device):
    if isinstance(device, int):
        device = torch.device("cuda", device)
    else:
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _local_metadata(device, enabled):
    result = {
        "enabled": bool(enabled),
        "eligible": False,
        "error": None,
        "hostname": socket.gethostname(),
        "boot_id": None,
        "uuid": None,
        "capability": None,
        "backend": None,
        "device": str(device),
    }
    if not enabled or device.type != "cuda":
        return result
    try:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Scale-shard initialization must run outside capture")
        import torch.distributed._symmetric_memory as symm

        properties = torch.cuda.get_device_properties(device)
        result.update(
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            uuid=str(properties.uuid),
            capability=[properties.major, properties.minor],
            num_sms=properties.multi_processor_count,
            backend=symm.get_backend(device),
        )
        result["eligible"] = (
            bool(result["boot_id"])
            and bool(result["uuid"])
            and result["capability"] in ([10, 0], [10, 3])
            and result["backend"] in (None, "CUDA")
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def _eligible_members(members, metadata):
    if len(members) <= 1 or len(set(members)) != len(members):
        return False
    try:
        peers = [metadata[rank] for rank in members]
    except (IndexError, KeyError):
        return False
    return (
        all(peer["enabled"] and peer["eligible"] for peer in peers)
        and len({(peer["hostname"], peer["boot_id"]) for peer in peers}) == 1
        and len({peer["uuid"] for peer in peers}) == len(peers)
        and len({peer["signature"] for peer in peers}) == 1
    )


def _can_access_peers(device, peer_uuids):
    """Conservative direct-load admission using physical UUIDs, not rank ordinals."""
    try:
        visible = {}
        for index in range(torch.cuda.device_count()):
            try:
                visible[str(torch.cuda.get_device_properties(index).uuid)] = index
            except Exception:
                # An unrelated visible device need not be a healthy group peer.
                continue
        for uuid in peer_uuids:
            peer = visible.get(uuid)
            if peer is None:
                return False
            if peer != device.index and not torch.cuda.can_device_access_peer(
                device.index, peer
            ):
                return False
        return True
    except Exception:
        return False


def _node_scale_groups(peers):
    """Plan disjoint physical-node groups, independently of TP/DP/EP ranks."""
    if (
        not peers
        or not all(peer["enabled"] for peer in peers)
        or len({peer["signature"] for peer in peers}) != 1
    ):
        return ()
    nodes = {}
    for rank, peer in enumerate(peers):
        node = (peer["hostname"], peer["boot_id"])
        nodes.setdefault(node, []).append(rank)
    return tuple(
        tuple(members)
        for members in nodes.values()
        if _eligible_members(members, peers)
    )


def _new_node_scale_group(world, groups):
    """All WORLD ranks create the same groups in the same startup order.

    Use the runtime's NCCL group-creation protocol, including nonmember splits
    when WORLD has a bound CUDA device. Only initialization uses this group;
    forward has no collective. Avoid locally hashed group names, which can
    differ when ranks already belong to different numbers of other subgroups.
    Cache the complete plan even on singleton/fallback ranks, which must still
    participate in group creation for other nodes.
    """
    key = (world, groups)
    cached = key in _NODE_GROUPS
    votes = _exchange(world, {"cached": cached}, "node_group_cache")
    if any(vote["cached"] for vote in votes):
        if not all(vote["cached"] for vote in votes):
            raise RuntimeError("Engram scale node-group caches differ across ranks")
        return _NODE_GROUPS[key]
    selected = None
    for members in groups:
        group = dist.new_group(
            ranks=list(members),
            backend="nccl",
            timeout=timedelta(seconds=_AGREEMENT_TIMEOUT_SECONDS),
            use_local_synchronization=False,
        )
        if dist.get_rank(world) in members:
            selected = group
    # Do not let a fast node enter table setup while another is still building
    # groups. Any failure here aborts startup instead of falling back rank-locally.
    _exchange(world, {"ready": True}, "node_groups_ready")
    _NODE_GROUPS[key] = selected
    return selected


def select_scale_group(device, signature, enabled=True):
    """Select a physical-node scale group, independent of global EP size.

    Like DeepEP's NVLink/LSA scale domain, each node stores a complete set of
    shards locally. An EP16/EP24 job on eight-GPU nodes therefore uses P=8 per
    node. Group membership comes from host identity, never global-rank modulo.
    Disabled ranks still participate so mixed feature settings cannot hang peers.
    """
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() <= 1
    ):
        return None
    device = (
        _device(device)
        if enabled
        else (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
    )
    world = dist.group.WORLD
    metadata = _local_metadata(device, enabled)
    try:
        metadata["signature"] = _json(signature)
    except (TypeError, ValueError) as error:
        metadata.update(
            signature=None, eligible=False, error=f"Invalid startup signature: {error}"
        )
    peers = _exchange(world, metadata, "select_metadata")
    groups = _node_scale_groups(peers)
    rank = dist.get_rank(world)
    candidate = next((members for members in groups if rank in members), None)
    accessible = candidate is not None and _can_access_peers(
        device, [peers[member]["uuid"] for member in candidate]
    )
    choices = _exchange(
        world,
        {"members": list(candidate) if accessible else None},
        "select_choice",
    )
    groups = tuple(
        members
        for members in groups
        if all(choices[member]["members"] == list(members) for member in members)
    )
    if not groups:
        if enabled:
            logging.info(
                "Engram scales: full local replica; no eligible node-local peer group"
            )
        return None
    selected = (
        world if groups == (_members(world),) else _new_node_scale_group(world, groups)
    )
    if selected is not None:
        logging.info(
            "Engram scales: node-local shards=%d, global members=%s, world size=%d",
            dist.get_world_size(selected),
            _members(selected),
            len(peers),
        )
    elif enabled:
        logging.info(
            "Engram scales: full local replica on this node; other nodes use shards"
        )
    return selected


@dataclass(frozen=True, eq=False)
class _ScaleShardOwner:
    local_scales: torch.Tensor
    peer_ptrs: torch.Tensor
    handle: object
    group: object
    source_identity: str
    table_rows: int
    group_size: int
    rank: int
    rows_per_shard: int
    members: tuple


@dataclass(frozen=True, eq=False)
class ShardedScaleDescriptor:
    local_scales: torch.Tensor
    peer_ptrs: torch.Tensor
    table_rows: int
    group_size: int
    rank: int
    rows_per_shard: int
    owner: object

    @property
    def device(self):
        return self.local_scales.device

    def record_stream(self, stream):
        # Generation retention also covers reads by other ranks and graph replay.
        self.local_scales.record_stream(stream)
        self.peer_ptrs.record_stream(stream)

    def validate_metadata(self, device, table_rows):
        """Metadata-only guard: no tensor reads, GPU synchronization or collective."""
        device = torch.device(device)
        expected_rows = (
            (self.table_rows + self.group_size - 1) // self.group_size
            if self.group_size > 0
            else 0
        )
        owner = self.owner
        if not (
            type(table_rows) is int
            and table_rows > 0
            and self.table_rows == table_rows
            and type(self.group_size) is int
            and self.group_size > 1
            and type(self.rank) is int
            and 0 <= self.rank < self.group_size
            and self.rows_per_shard == expected_rows
            and isinstance(self.local_scales, torch.Tensor)
            and self.local_scales.is_cuda
            and self.local_scales.dtype == torch.uint8
            and tuple(self.local_scales.shape) == (expected_rows, 8)
            and self.local_scales.is_contiguous()
            and self.local_scales.data_ptr() % 8 == 0
            and self.local_scales.device == device
            and isinstance(self.peer_ptrs, torch.Tensor)
            and self.peer_ptrs.is_cuda
            and self.peer_ptrs.dtype == torch.uint64
            and tuple(self.peer_ptrs.shape) == (self.group_size,)
            and self.peer_ptrs.is_contiguous()
            and self.peer_ptrs.device == device
            and isinstance(owner, _ScaleShardOwner)
            and owner.local_scales is self.local_scales
            and owner.peer_ptrs is self.peer_ptrs
            and owner.table_rows == self.table_rows
            and owner.group_size == self.group_size
            and owner.rank == self.rank
            and owner.rows_per_shard == self.rows_per_shard
        ):
            raise ValueError("Invalid or stale Engram sharded-scale descriptor")
        return True


def _scale_input_valid(scales):
    return (
        isinstance(scales, torch.Tensor)
        and scales.device.type == "cpu"
        and scales.dtype == torch.uint8
        and scales.ndim == 2
        and scales.shape[0] > 0
        and scales.shape[1] == 8
        and scales.is_contiguous()
        and not scales.requires_grad
        and not scales.is_conj()
        and not scales.is_neg()
    )


def _require_votes(votes, phase):
    failed = [
        (rank, vote.get("error"))
        for rank, vote in enumerate(votes)
        if not vote.get("ok")
    ]
    if failed:
        raise RuntimeError(
            f"Engram scale-shard {phase} failed on group rank(s): {failed}"
        )


def create_sharded_scales(scales_cpu, group, device, source_identity):
    """Copy only this rank's immutable scale rows into final symmetric storage.

    None/singleton groups retain the original full local GPU-scale behavior.
    Multi-rank setup is collective, including cache-hit agreement. All successful
    descriptors are process-lifetime objects; different identities retain their
    old generations rather than freeing storage still reachable by peer graphs.
    """
    device = _device(device)
    if group is None or (dist.is_initialized() and dist.get_world_size(group) == 1):
        if not _scale_input_valid(scales_cpu):
            raise ValueError("Expected contiguous CPU uint8 scales[table_rows,8]")
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Engram scale allocation is startup-only")
        return scales_cpu.to(device)
    if not dist.is_initialized() or dist.get_rank(group) < 0:
        raise ValueError("Scale-shard group must be initialized and contain this rank")
    import torch.distributed._symmetric_memory as symm

    members = _members(group)
    group_size, rank = len(members), dist.get_rank(group)
    info = _local_metadata(device, True)
    valid_input = _scale_input_valid(scales_cpu)
    try:
        if source_identity is None or source_identity == "":
            raise ValueError("A nonempty immutable table identity is required")
        identity = _json(source_identity)
    except (TypeError, ValueError) as error:
        identity = None
        info.update(eligible=False, error=f"Invalid table identity: {error}")
    table_rows = scales_cpu.shape[0] if valid_input else 0
    rows_per_shard = (table_rows + group_size - 1) // group_size
    info.update(
        signature=identity,
        table_rows=table_rows,
        rows_per_shard=rows_per_shard,
        valid_input=valid_input,
    )
    cache_key = (_group_key(group), identity, table_rows, str(device), info["uuid"])
    cached = _GENERATIONS.get(cache_key)
    info["cache_hit"] = cached is not None
    if cached is not None:
        try:
            cached.validate_metadata(device, table_rows)
        except ValueError as error:
            info.update(eligible=False, error=str(error))
    peers = _exchange(group, info, "table_metadata")
    if (
        not _eligible_members(tuple(range(group_size)), peers)
        or not all(peer["valid_input"] for peer in peers)
        or len({(peer["table_rows"], peer["rows_per_shard"]) for peer in peers}) != 1
    ):
        raise RuntimeError(
            f"Engram scale-shard startup metadata disagree or are unsupported: {peers}"
        )
    access = _can_access_peers(device, [peer["uuid"] for peer in peers])
    votes = _exchange(
        group,
        {
            "ok": access,
            "error": (
                None if access else "Peer GPU is invisible or not directly accessible"
            ),
        },
        "peer_access",
    )
    _require_votes(votes, "peer-access admission")
    hits = [peer["cache_hit"] for peer in peers]
    if any(hits):
        if not all(hits):
            raise RuntimeError(
                "Engram scale-shard cache generations differ across ranks"
            )
        return cached

    local = None
    error = None
    try:
        with torch.cuda.device(device):
            local = symm.empty((rows_per_shard, 8), dtype=torch.uint8, device=device)
            local.fill_(127)
            begin = rank * rows_per_shard
            count = max(0, min(rows_per_shard, table_rows - begin))
            if count:
                local[:count].copy_(scales_cpu[begin : begin + count])
            torch.cuda.current_stream(device).synchronize()
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    votes = _exchange(group, {"ok": error is None, "error": error}, "allocation")
    _require_votes(votes, "allocation/fill")
    # No rank-local recovery after peers may have imported allocation mappings.
    try:
        with torch.cuda.device(device):
            handle = symm.rendezvous(local, group=group)
    except Exception as failure:
        _FAILED_RENDEZVOUS_RESOURCES.append(local)
        raise RuntimeError(
            "Engram symmetric-memory rendezvous failed; abort coordinated startup"
        ) from failure

    descriptor = None
    peer_ptrs = None
    error = None
    try:
        pointers = handle.buffer_ptrs
        if (
            handle.rank != rank
            or handle.world_size != group_size
            or len(pointers) != group_size
            or any(
                type(pointer) is not int or pointer <= 0 or pointer % 8
                for pointer in pointers
            )
            or pointers[rank] != local.data_ptr()
            or handle.buffer_size < local.numel()
        ):
            raise RuntimeError("Symmetric-memory handle has inconsistent peer mappings")
        peer_ptrs = torch.tensor(pointers, dtype=torch.uint64, device=device)
        owner = _ScaleShardOwner(
            local,
            peer_ptrs,
            handle,
            group,
            identity,
            table_rows,
            group_size,
            rank,
            rows_per_shard,
            members,
        )
        descriptor = ShardedScaleDescriptor(
            local, peer_ptrs, table_rows, group_size, rank, rows_per_shard, owner
        )
        descriptor.validate_metadata(device, table_rows)
        torch.cuda.current_stream(device).synchronize()
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    try:
        votes = _exchange(group, {"ok": error is None, "error": error}, "ready")
        _require_votes(votes, "publication")
    except BaseException:
        _FAILED_RENDEZVOUS_RESOURCES.append((local, handle, peer_ptrs, descriptor))
        raise
    _GENERATIONS[cache_key] = descriptor
    return descriptor
