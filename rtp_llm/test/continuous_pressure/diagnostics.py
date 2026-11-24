"""FULL cache evidence, schema projection and conservative window assessment."""

import json
import math
from dataclasses import dataclass, fields, replace
from typing import Any, Callable

from .config import (
    PressureConfig,
    _integer,
    _number,
    mock_provider_enabled,
    mock_target_ppm,
)


@dataclass(frozen=True)
class PoolSample:
    pool_id: str
    capacity: int
    used: int
    pending_release: int
    low_ratio: float
    high_ratio: float
    host_resident: int = 0
    pool_name: str = ""
    tier: int = 0
    block_size_bytes: int = 0
    free: int = 0
    available: int = 0
    mock_source_ref_blocks: int = 0
    mock_source_ref_bytes: int = 0
    disk_read_bytes: int = 0
    disk_write_bytes: int = 0


@dataclass(frozen=True)
class GroupSample:
    group_id: int
    group_type: str
    device_pool_ids: tuple[str, ...]
    host_pool_id: str
    triggers: int
    gpu_host_published: int
    group_tags: tuple[str, ...] = ()
    disk_pool_id: str = ""
    geometry: tuple[int, int, int, bool] = (0, 0, 0, False)


# Version 1 counters change at transfer terminal notification, before deferred
# tree/context settlement. Version 0 servers cannot certify event timing.
MOCK_MEMORY_TERMINAL_OUTCOME_VERSION = 1


@dataclass(frozen=True)
class MockMemorySample:
    eligible_blocks: int
    planned_host_blocks: int
    successful_h2d_blocks: int
    failed_h2d_blocks: int
    d2h_submitted_blocks: int
    d2h_completed_blocks: int
    d2h_failed_blocks: int
    terminal_outcome_version: int = 0


@dataclass(frozen=True)
class MockCacheSample:
    eligible_blocks: int = 0
    planned_memory_blocks: int = 0
    planned_disk_blocks: int = 0
    successful_memory_blocks: int = 0
    successful_disk_blocks: int = 0
    failed_blocks: int = 0
    disk_write_submitted_records: int = 0
    disk_write_completed_records: int = 0
    disk_write_failed_records: int = 0
    terminal_outcome_version: int = 0
    planned_remote_blocks: int = 0
    successful_remote_blocks: int = 0


@dataclass(frozen=True)
class MockRemoteWriteSample:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    pending_requests: int = 0
    pending_intents: int = 0
    retained_descriptors: int = 0
    faulted: bool = False
    quarantined_blocks: int = 0


@dataclass(frozen=True)
class LocalMockTargets:
    memory_enabled: bool
    disk_enabled: bool
    memory_ratio: float
    disk_ratio: float
    memory_ppm: int
    disk_ppm: int


@dataclass(frozen=True)
class LocalStoreSample:
    released_requests: int = 0
    pending_requests: int = 0
    abandoned_requests: int = 0
    rejected_host_stores: int = 0
    rejected_disk_stores: int = 0


@dataclass(frozen=True)
class PressureOwnership:
    mode: int
    owner_rank: int
    participants: tuple[int, ...]
    publication_scope: int


@dataclass(frozen=True)
class RankSample:
    rank: int
    instance: str
    sequence: int
    server_s: float
    client_send_s: float
    client_receive_s: float
    pools: tuple[PoolSample, ...]
    groups: tuple[GroupSample, ...]
    offset_bounds: tuple[float, float] | None = None
    counters: tuple[tuple[tuple, int], ...] = ()
    ownership: PressureOwnership | None = None
    mock_memory: MockMemorySample | None = None
    mock_cache: MockCacheSample | None = None
    mock_remote_write: MockRemoteWriteSample | None = None
    remote_read_payload_bytes: int | None = None
    remote_write_acknowledged_payload_bytes: int | None = None
    # Canonical compact native handshake/configuration; never inferred from CLI.
    remote_runtime: str | None = None
    local_mock_targets: LocalMockTargets | None = None
    local_store: LocalStoreSample | None = None


@dataclass(frozen=True)
class PressureTarget:
    rank: int
    instance: str
    group_id: int
    device_pool_ids: tuple[str, ...]
    host_pool_id: str
    owner_rank: int | None = None
    owner_instance: str = ""
    owner_group_id: int | None = None
    pool_mapping: tuple[tuple[str, str], ...] = ()
    participants: tuple[tuple[int, str], ...] = ()
    publication_scope: int = 1
    ownership_mode: int = 1
    layout: tuple = ()


@dataclass(frozen=True)
class DecodedPressure:
    raw_wire: bytes
    samples: tuple[RankSample, ...]


def _id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("pool/service identity must be nonempty")
    return value


def _pool_map(pools):
    result = {}
    for pool in pools:
        _id(pool.pool_id)
        for name in (
            "capacity",
            "used",
            "pending_release",
            "host_resident",
            "free",
            "available",
        ):
            _integer(getattr(pool, name), name)
        # HOST/DISK and observe-only groups may disable eviction with {0, 0}.
        # Enabled, ordered thresholds are required only by the FULL GPU gate.
        _number(pool.low_ratio, "low_ratio", inclusive=True)
        _number(pool.high_ratio, "high_ratio", inclusive=True)
        if max(pool.low_ratio, pool.high_ratio) > 1:
            raise ValueError("watermark ratios must be <= 1")
        if not (
            0 <= pool.pending_release <= pool.used <= pool.capacity
            and 0 <= pool.free <= pool.capacity
            and 0 <= pool.available <= pool.capacity
            and pool.host_resident <= pool.used
        ):
            raise ValueError("invalid physical occupancy")
        if pool.pool_id in result and result[pool.pool_id] != pool:
            raise ValueError("conflicting physical pool records")
        result[pool.pool_id] = pool
    return result


def _validate_group(group, pools):
    _integer(group.group_id, "group_id")
    _integer(group.triggers, "triggers")
    _integer(group.gpu_host_published, "gpu_host_published")
    if group.group_type not in ("FULL", "SWA", "LINEAR"):
        raise ValueError("unknown group type")
    if not isinstance(group.device_pool_ids, tuple) or len(
        set(group.device_pool_ids)
    ) != len(group.device_pool_ids):
        raise ValueError("invalid device mapping")
    if not group.device_pool_ids or any(
        key not in pools for key in group.device_pool_ids
    ):
        raise ValueError("missing device pool")
    if group.host_pool_id and group.host_pool_id not in pools:
        raise ValueError("missing host pool")
    if group.host_pool_id in group.device_pool_ids:
        raise ValueError("host and device pools must differ")
    if group.group_type == "FULL" and (
        not group.host_pool_id or pools[group.host_pool_id].capacity <= 0
    ):
        raise ValueError("FULL requires positive host capacity")
    if group.group_type == "FULL":
        for key in group.device_pool_ids:
            pool = pools[key]
            if not 0 < pool.low_ratio < pool.high_ratio <= 1:
                raise ValueError("FULL device requires 0 < low < high <= 1")


def _layout(sample):
    return (
        tuple(
            sorted(
                (
                    p.pool_id,
                    p.pool_name,
                    p.tier,
                    p.capacity,
                    p.block_size_bytes,
                    p.low_ratio,
                    p.high_ratio,
                )
                for p in sample.pools
            )
        ),
        tuple(
            sorted(
                (
                    g.group_id,
                    g.group_type,
                    g.device_pool_ids,
                    g.host_pool_id,
                    g.group_tags,
                    g.disk_pool_id,
                    g.geometry,
                )
                for g in sample.groups
            )
        ),
    )


def _shared_layout(sample):
    pools = {p.pool_id: p for p in sample.pools}
    result = []
    tags = set()
    for group in sample.groups:
        identity = tuple(sorted(group.group_tags))
        if not identity or len(set(identity)) != len(identity) or identity in tags:
            raise ValueError("ambiguous shared group tags")
        tags.add(identity)
        tokens, payload, storage, _ = group.geometry
        if tokens <= 0 or payload <= 0 or storage < payload:
            raise ValueError("missing shared group geometry")
        members = []
        for key in (
            *group.device_pool_ids,
            *((group.host_pool_id,) if group.host_pool_id else ()),
            *((group.disk_pool_id,) if group.disk_pool_id else ()),
        ):
            pool = pools[key]
            if (
                not pool.pool_name
                or pool.block_size_bytes <= 0
                or pool.tier not in (1, 2, 3)
            ):
                raise ValueError("missing shared pool geometry")
            members.append(
                (pool.tier, pool.pool_name, pool.capacity, pool.block_size_bytes)
            )
        if len(set((tier, name) for tier, name, _, _ in members)) != len(members):
            raise ValueError("ambiguous shared pool tags")
        result.append(
            (identity, group.group_type, group.geometry, tuple(sorted(members)))
        )
    return tuple(sorted(result))


def _local_targets(
    samples: tuple[RankSample, ...], ranks: tuple[int, ...]
) -> tuple[PressureTarget, ...]:
    if not ranks or len(set(ranks)) != len(ranks):
        raise ValueError("expected ranks must be nonempty and unique")
    for rank in ranks:
        _integer(rank, "rank")
    if len(samples) != len(ranks) or {s.rank for s in samples} != set(ranks):
        raise ValueError("missing, duplicate, or unexpected rank")
    result = []
    for sample in sorted(samples, key=lambda s: s.rank):
        _integer(sample.rank, "rank")
        _id(sample.instance)
        _integer(sample.sequence, "sequence")
        for name in ("server_s", "client_send_s", "client_receive_s"):
            _number(getattr(sample, name), name, inclusive=True)
        if sample.client_receive_s < sample.client_send_s:
            raise ValueError("reversed client timing")
        pools = _pool_map(sample.pools)
        seen = set()
        full = []
        for group in sample.groups:
            _validate_group(group, pools)
            if group.group_id in seen:
                raise ValueError("duplicate group identity")
            seen.add(group.group_id)
            if group.group_type == "FULL":
                full.append(
                    PressureTarget(
                        sample.rank,
                        sample.instance,
                        group.group_id,
                        tuple(sorted(group.device_pool_ids)),
                        group.host_pool_id,
                    )
                )
        if not full:
            raise ValueError("rank has no FULL groups")
        result.extend(sorted(full, key=lambda target: target.group_id))
    return tuple(result)


def select_targets(
    samples: tuple[RankSample, ...], ranks: tuple[int, ...]
) -> tuple[PressureTarget, ...]:
    result = _local_targets(samples, ranks)
    by_rank = {sample.rank: sample for sample in samples}
    frozen = []
    for target in result:
        sample = by_rank[target.rank]
        declaration = sample.ownership
        owner = sample
        participants = (sample,)
        if declaration is not None:
            if declaration.mode == 1:
                if (
                    declaration.owner_rank != sample.rank
                    or declaration.participants != (sample.rank,)
                    or declaration.publication_scope != 1
                ):
                    raise ValueError("inconsistent independent ownership")
            elif declaration.mode == 2:
                if (
                    declaration.publication_scope != 2
                    or len(declaration.participants) < 2
                    or len(set(declaration.participants))
                    != len(declaration.participants)
                    or set(declaration.participants) != set(ranks)
                    or sample.rank not in declaration.participants
                    or declaration.owner_rank not in declaration.participants
                    or any(rank not in by_rank for rank in declaration.participants)
                ):
                    raise ValueError("incomplete shared ownership scope")
                participants = tuple(by_rank[rank] for rank in declaration.participants)
                if any(peer.ownership != declaration for peer in participants):
                    raise ValueError("inconsistent shared ownership declaration")
                owner = by_rank[declaration.owner_rank]
                layout = _shared_layout(owner)
                if any(_shared_layout(peer) != layout for peer in participants):
                    raise ValueError("incompatible shared group/pool layout")
            else:
                raise ValueError("unsupported pressure ownership")
        group = next(g for g in sample.groups if g.group_id == target.group_id)
        owner_group = group
        mapping = tuple(
            (key, key) for key in (*group.device_pool_ids, group.host_pool_id)
        )
        if owner is not sample:
            if not group.group_tags or len(set(group.group_tags)) != len(
                group.group_tags
            ):
                raise ValueError("shared group requires unique tags")
            matches = [
                g for g in owner.groups if set(g.group_tags) == set(group.group_tags)
            ]
            if len(matches) != 1 or matches[0].group_type != group.group_type:
                raise ValueError("ambiguous shared group layout")
            owner_group = matches[0]
            local_pools = {p.pool_id: p for p in sample.pools}
            owner_pools = {p.pool_id: p for p in owner.pools}
            mapped = []
            for key in (*group.device_pool_ids, group.host_pool_id):
                pool = local_pools[key]
                matches = [
                    owner_pools[k]
                    for k in (*owner_group.device_pool_ids, owner_group.host_pool_id)
                    if (owner_pools[k].tier, owner_pools[k].pool_name)
                    == (pool.tier, pool.pool_name)
                ]
                if (
                    not pool.pool_name
                    or pool.tier not in (1, 2)
                    or pool.block_size_bytes <= 0
                    or len(matches) != 1
                    or (matches[0].capacity, matches[0].block_size_bytes)
                    != (pool.capacity, pool.block_size_bytes)
                ):
                    raise ValueError("incompatible shared pool geometry/capacity")
                mapped.append((key, matches[0].pool_id))
            if len({v for _, v in mapped}) != len(mapped):
                raise ValueError("ambiguous shared pool mapping")
            mapping = tuple(mapped)
        # Freeze all tiers/groups, including observe-only layout, without counters.
        layout = tuple((peer.rank, *_layout(peer)) for peer in participants)
        frozen.append(
            replace(
                target,
                owner_rank=owner.rank,
                owner_instance=owner.instance,
                owner_group_id=owner_group.group_id,
                pool_mapping=mapping,
                participants=tuple((p.rank, p.instance) for p in participants),
                publication_scope=declaration.publication_scope if declaration else 1,
                ownership_mode=declaration.mode if declaration else 1,
                layout=layout,
            )
        )
    return tuple(frozen)


def validate_inventory(
    current: tuple[PressureTarget, ...], frozen: tuple[PressureTarget, ...]
) -> None:
    """Caller explicitly compares each later inventory with the epoch's first one."""
    if current != frozen:
        raise ValueError("pressure topology or service instance discontinuity")


def group_valid(
    group: GroupSample,
    pools: dict[str, PoolSample],
    baseline: GroupSample,
    config: PressureConfig,
) -> bool:
    validated = _pool_map(pools.values())
    if set(validated) != set(pools) or any(
        key != pool.pool_id for key, pool in pools.items()
    ):
        raise ValueError("pool keys differ from physical identities")
    _validate_group(group, validated)
    _validate_group(baseline, validated)
    if (
        group.group_id,
        group.group_type,
        group.device_pool_ids,
        group.host_pool_id,
    ) != (
        baseline.group_id,
        baseline.group_type,
        baseline.device_pool_ids,
        baseline.host_pool_id,
    ):
        raise ValueError("group topology discontinuity")
    if group.group_type != "FULL":
        raise ValueError("only FULL groups have a pressure predicate")
    if (
        group.triggers < baseline.triggers
        or group.gpu_host_published < baseline.gpu_host_published
    ):
        raise ValueError("pressure counter discontinuity")
    for key in group.device_pool_ids:
        if config.gpu_tolerance_blocks > math.floor(
            pools[key].capacity * pools[key].low_ratio
        ):
            raise ValueError("GPU tolerance exceeds selected low threshold")
        if pools[key].capacity <= 0:
            raise ValueError("unreachable device pressure target")
    device_ok = any(
        pools[key].used - pools[key].pending_release
        >= math.floor(pools[key].capacity * pools[key].low_ratio)
        - config.gpu_tolerance_blocks
        for key in group.device_pool_ids
    )
    host = pools[group.host_pool_id]
    return (
        device_ok
        and host.host_resident >= math.ceil(host.capacity * config.host_ratio)
        and group.triggers > baseline.triggers
        and group.gpu_host_published > baseline.gpu_host_published
    )


def raw_pressure_record(
    response,
    client_send_s: float,
    client_receive_s: float,
    *,
    to_dict: Callable[[Any], dict] = dict,
) -> dict:
    """Record JSON-safe evidence from a mapping or a backend message projection."""

    _number(client_send_s, "client_send_s", inclusive=True)
    _number(client_receive_s, "client_receive_s", inclusive=True)
    if client_receive_s < client_send_s:
        raise ValueError("reversed client timing")
    record = dict(
        client_send_s=client_send_s,
        client_receive_s=client_receive_s,
        response=to_dict(response),
    )

    if hasattr(response, "ranks"):
        record["rank_offset_bounds_s"] = {
            str(rank.world_rank): _wire_offset_bounds(
                response, rank, client_send_s, client_receive_s
            )
            for rank in response.ranks
            if not rank.error
        }

    # Reject protobuf nonfinite-double string representations as well as Python NaNs.
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return (
            value not in ("NaN", "Infinity", "-Infinity")
            if isinstance(value, str)
            else True
        )

    if not finite(record):
        raise ValueError("nonfinite raw pressure evidence")
    json.dumps(record, allow_nan=False)
    return record


def decode_pressure_response(
    response,
    client_send_s: float,
    client_receive_s: float,
    expected_ranks: tuple[int, ...],
    *,
    complete: bool = True,
    to_dict: Callable[[Any], dict] = dict,
) -> DecodedPressure:
    """Decode a duck-typed schema-1 snapshot and preserve its raw bytes.

    For native Remote runtime messages, pass the same message-to-dict projector
    used for raw evidence. The backend supplies wire field names and enum values.
    """
    if response.schema_version != 1:
        raise ValueError("unsupported pressure schema")
    if len(response.requested_ranks) != len(expected_ranks) or set(
        response.requested_ranks
    ) != set(expected_ranks):
        raise ValueError("response requested ranks mismatch")
    samples = []
    for rank in response.ranks:
        if rank.HasField("ownership") and not rank.ownership.HasField(
            "owner_world_rank"
        ):
            raise ValueError("missing explicit owner rank")
        if rank.error:
            raise ValueError(f"rank {rank.world_rank}: {rank.error}")
        raw_pools = {}
        for pool in rank.pools:
            if pool.tier not in (1, 2, 3, 4):
                raise ValueError("unknown pool tier")
            if pool.pool_id in raw_pools and pool != raw_pools[pool.pool_id]:
                raise ValueError("conflicting raw physical pool records")
            raw_pools[pool.pool_id] = pool
        watermarks = {}
        groups = []
        group_ids = {g.group_id for g in rank.groups}
        for transfer in rank.transfers:
            if transfer.group_id not in group_ids or transfer.pool_id not in raw_pools:
                raise ValueError("unknown transfer mapping")
            if (
                transfer.pool_tier != raw_pools[transfer.pool_id].tier
                or transfer.operation not in (1, 2, 3)
                or transfer.source_tier not in (1, 2, 3, 4, 5)
                or transfer.target_tier not in (1, 2, 3, 4, 5)
            ):
                raise ValueError("invalid transfer tier/operation")
        for group in rank.groups:
            if group.group_type not in (1, 2, 3):
                raise ValueError("unknown group type")
            host = group.host_pool_id.value if group.HasField("host_pool_id") else ""
            mappings = [(key, 1, "device_watermark") for key in group.device_pool_ids]
            if host:
                mappings.append((host, 2, "host_watermark"))
            if group.HasField("disk_pool_id"):
                mappings.append((group.disk_pool_id.value, 3, "disk_watermark"))
            for key, tier, field in mappings:
                if (
                    key not in raw_pools
                    or raw_pools[key].tier != tier
                    or not group.HasField(field)
                ):
                    raise ValueError("missing or mismatched pool/watermark mapping")
                watermark = getattr(group, field)
                pair = (watermark.low_ratio, watermark.high_ratio)
                if key in watermarks and watermarks[key] != pair:
                    raise ValueError("conflicting shared-pool watermarks")
                watermarks[key] = pair
            published = 0
            endpoints = set()
            for transfer in rank.transfers:
                if transfer.group_id != group.group_id:
                    continue
                if transfer.pool_id not in {key for key, _, _ in mappings}:
                    raise ValueError("transfer outside group pools")
                identity = (
                    transfer.pool_id,
                    transfer.operation,
                    transfer.source_tier,
                    transfer.target_tier,
                )
                if identity in endpoints:
                    raise ValueError("duplicate physical transfer endpoint")
                endpoints.add(identity)
                if (
                    transfer.pool_tier == 2
                    and transfer.pool_id == host
                    and transfer.source_tier == 1
                    and transfer.target_tier == 2
                    and transfer.operation in (2, 3)
                ):
                    published += _integer(transfer.published_blocks, "published_blocks")
            groups.append(
                GroupSample(
                    group.group_id,
                    {1: "FULL", 2: "SWA", 3: "LINEAR"}[group.group_type],
                    tuple(group.device_pool_ids),
                    host,
                    group.watermark_trigger_count,
                    published,
                    tuple(getattr(group, "group_tags", ())),
                    group.disk_pool_id.value if group.HasField("disk_pool_id") else "",
                    tuple(
                        getattr(group, field, 0)
                        for field in (
                            "block_tokens",
                            "payload_bytes",
                            "storage_bytes",
                            "crc_enabled",
                        )
                    ),
                )
            )
        pools = []
        for key, pool in raw_pools.items():
            if key not in watermarks:
                raise ValueError("physical pool has no group watermark mapping")
            pools.append(
                PoolSample(
                    key,
                    pool.total_blocks,
                    pool.used_blocks,
                    pool.pending_release_blocks,
                    *watermarks[key],
                    pool.published_host_blocks,
                    getattr(pool, "pool_name", ""),
                    pool.tier,
                    getattr(pool, "block_size_bytes", 0),
                    pool.free_blocks,
                    pool.available_blocks,
                    getattr(pool, "mock_source_ref_blocks", 0),
                    getattr(pool, "mock_source_ref_bytes", 0),
                    getattr(pool, "disk_read_bytes", 0),
                    getattr(pool, "disk_write_bytes", 0),
                )
            )
        _integer(rank.sampled_monotonic_ns, "sampled_monotonic_ns")
        _integer(rank.collect_duration_ns, "collect_duration_ns")
        samples.append(
            RankSample(
                rank.world_rank,
                rank.service_instance,
                rank.sequence,
                rank.sampled_monotonic_ns / 1e9,
                client_send_s,
                client_receive_s,
                tuple(pools),
                tuple(groups),
                _wire_offset_bounds(response, rank, client_send_s, client_receive_s),
                _wire_counters(rank),
                (
                    PressureOwnership(
                        rank.ownership.mode,
                        rank.ownership.owner_world_rank.value,
                        tuple(rank.ownership.participant_world_ranks),
                        rank.ownership.publication_scope,
                    )
                    if rank.HasField("ownership")
                    else None
                ),
                (
                    MockMemorySample(
                        *(
                            _integer(getattr(rank.mock_memory, field), field)
                            for field in (
                                "eligible_blocks",
                                "planned_host_blocks",
                                "successful_h2d_blocks",
                                "failed_h2d_blocks",
                                "d2h_submitted_blocks",
                                "d2h_completed_blocks",
                                "d2h_failed_blocks",
                            )
                        ),
                        terminal_outcome_version=getattr(
                            rank.mock_memory, "terminal_outcome_version", 0
                        ),
                    )
                    if "mock_memory" in rank.DESCRIPTOR.fields_by_name
                    and rank.HasField("mock_memory")
                    else None
                ),
                (
                    MockCacheSample(
                        **{
                            f.name: _integer(
                                getattr(rank.mock_cache, f.name, 0), f.name
                            )
                            for f in fields(MockCacheSample)
                        }
                    )
                    if "mock_cache" in rank.DESCRIPTOR.fields_by_name
                    and rank.HasField("mock_cache")
                    else None
                ),
                (
                    MockRemoteWriteSample(
                        **{
                            f.name: (
                                bool(getattr(rank.mock_remote_write, f.name))
                                if f.name == "faulted"
                                else _integer(
                                    getattr(rank.mock_remote_write, f.name), f.name
                                )
                            )
                            for f in fields(MockRemoteWriteSample)
                        }
                    )
                    if "mock_remote_write" in rank.DESCRIPTOR.fields_by_name
                    and rank.HasField("mock_remote_write")
                    else None
                ),
                getattr(rank, "remote_read_payload_bytes", None),
                getattr(rank, "remote_write_acknowledged_payload_bytes", None),
                _wire_remote_runtime(rank, to_dict),
                _wire_record(rank, "local_mock_targets", LocalMockTargets),
                _wire_record(rank, "local_store", LocalStoreSample),
            )
        )
    result = tuple(samples)
    if complete:
        select_targets(result, expected_ranks)
    else:
        # Runtime partial pages retain local evidence. The coverage accumulator
        # compares its ownership/layout/counters with the full frozen inventory.
        _local_targets(result, expected_ranks)
    return DecodedPressure(response.SerializeToString(), result)


def _wire_record(rank, name, record):
    if name in rank.DESCRIPTOR.fields_by_name and rank.HasField(name):
        value = getattr(rank, name)
        return record(**{f.name: getattr(value, f.name) for f in fields(record)})
    return None


def _wire_remote_runtime(rank, to_dict):
    if "remote_runtime" not in rank.DESCRIPTOR.fields_by_name or not rank.HasField(
        "remote_runtime"
    ):
        return None
    runtime = rank.remote_runtime
    required_options = {
        "mock_remote_cache",
        "mock_remote_cache_address",
        "mock_remote_cache_hit_ratio",
        "mock_remote_cache_source_blocks",
        "mock_remote_cache_read_delay_ms",
        "mock_remote_cache_write_delay_ms",
        "mock_remote_cache_callback_port",
        "mock_remote_cache_staging_size_mb",
        "mock_remote_cache_max_inflight",
        "mock_remote_cache_timeout_ms",
        "mock_memory_cache",
        "mock_memory_cache_hit_ratio",
        "mock_disk_cache",
        "mock_disk_cache_hit_ratio",
        "service_reads_per_connection",
        "service_writes_per_connection",
        "service_max_callback_endpoints",
        "service_max_timeout_ms",
    }
    required_limits = {
        "total_bytes",
        "source_bytes",
        "write_slot_bytes",
        "read_staging_bytes",
        "write_staging_bytes",
        "write_slots",
        "max_rpc_bytes",
        "max_blocks_per_rpc",
        "max_read_operations",
        "max_ranks",
        "max_groups_per_rank",
        "max_sources_per_group",
        "max_members_per_group",
        "replay_window",
        "max_delay_ms",
    }
    required_credits = {
        "local_read_bytes",
        "local_write_bytes",
        "service_read_bytes",
        "service_write_bytes",
        "read_operations",
        "write_operations",
        "first_write_slot",
        "write_slots",
        "max_slice_bytes",
        "max_batch_records",
        "max_write_slice_bytes",
    }
    if (
        not required_options <= runtime.options.keys()
        or not required_limits <= runtime.limits.keys()
        or not required_credits <= runtime.credits.keys()
        or any(
            not getattr(layout, name)
            for layout in runtime.layouts
            for name in (
                "payload_bytes",
                "storage_bytes",
                "stride_bytes",
                "source_count",
                "alignment",
                "tokens_per_block",
                "cp_world_size",
            )
        )
        or runtime.version != 1
        or runtime.options.get("mock_remote_cache") != "true"
        or not runtime.session_id
        or not runtime.server_epoch
        or not runtime.generation
        or not runtime.expected_ranks
        or not runtime.layout_hash
        or not runtime.callback_endpoint
        or not runtime.layouts
        or not runtime.credits.get("write_operations")
        or not runtime.credits.get("max_write_slice_bytes")
    ):
        raise ValueError(
            "Remote runtime capability/handshake is incomplete or unsupported"
        )
    return json.dumps(
        to_dict(runtime),
        sort_keys=True,
        separators=(",", ":"),
    )


Interval = tuple[float, float]


def merge_intervals(values) -> list[Interval]:
    merged = []
    for start, end in sorted(values):
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("interval endpoints must be finite")
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersect_intervals(left, right) -> list[Interval]:
    left, right = merge_intervals(left), merge_intervals(right)
    i = j = 0
    output = []
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            output.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return output


def coverage(values, start, end) -> tuple[float, float]:
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError("coverage requires a positive finite window")
    clipped = merge_intervals((max(a, start), min(b, end)) for a, b in values)
    valid = 0.0
    cursor = start
    longest = 0.0
    for a, b in clipped:
        longest = max(longest, a - cursor)
        valid += b - a
        cursor = b
    return valid / (end - start), max(longest, end - cursor)


def confirmed_segment(start_server, end_server, offset_low, offset_high):
    if (
        not all(
            math.isfinite(v)
            for v in (start_server, end_server, offset_low, offset_high)
        )
        or offset_low > offset_high
    ):
        raise ValueError("invalid clock bounds")
    start, end = start_server + offset_high, end_server + offset_low
    return (start, end) if end > start else None


def possible_event_segment(
    start_server: float,
    end_server: float,
    start_bounds: tuple[float, float],
    end_bounds: tuple[float, float],
) -> Interval:
    """Outer envelope of a cumulative increment between two sampled points."""
    return (
        start_server + min(start_bounds[0], end_bounds[0]),
        end_server + max(start_bounds[1], end_bounds[1]),
    )


def _failure_interval(old, current, now):
    bounds = tuple(
        sample.offset_bounds
        or (
            sample.client_send_s - sample.server_s,
            sample.client_receive_s - sample.server_s,
        )
        for sample in (old, current)
    )
    if (
        all(math.isfinite(v) for pair in bounds for v in pair)
        and all(low <= high for low, high in bounds)
        and old.server_s <= current.server_s
    ):
        return possible_event_segment(old.server_s, current.server_s, *bounds)
    # Unknown server-clock mapping cannot erase an observed failure.
    # Client RPC envelopes conservatively bound the observation gap.
    return (
        min(old.client_send_s, current.client_send_s),
        max(now, old.client_receive_s, current.client_receive_s),
    )


def _wire_offset_bounds(response, rank, client_send, client_receive):
    """Client minus rank clock; fan-out composes both transport uncertainties."""
    received = getattr(response, "coordinator_received_ns", 0)
    sent = getattr(response, "coordinator_sent_ns", 0)
    rank_send = getattr(rank, "coordinator_send_ns", 0)
    rank_receive = getattr(rank, "coordinator_receive_ns", 0)
    if not any((received, sent, rank_send, rank_receive)):
        return (
            client_send - rank.sampled_monotonic_ns / 1e9,
            client_receive - rank.sampled_monotonic_ns / 1e9,
        )
    if (
        min(received, sent, rank_send, rank_receive) <= 0
        or not received <= rank_send <= rank_receive <= sent
    ):
        raise ValueError("invalid coordinator timing bounds")
    # Outer request establishes client-coordinator offset; internal request
    # establishes coordinator-rank offset. No cross-process clocks are compared.
    server = rank.sampled_monotonic_ns / 1e9
    low = client_send - received / 1e9 + rank_send / 1e9 - server
    high = client_receive - sent / 1e9 + rank_receive / 1e9 - server
    if low > high:
        raise ValueError("inconsistent composed clock bounds")
    return low, high


@dataclass(frozen=True)
class ActivitySample:
    last_dispatch_s: float
    last_progress_s: float
    active_step_started_s: float | None
    running: int | None
    waiting: int | None


@dataclass(frozen=True)
class ActivityAssessment:
    valid: bool
    reasons: tuple[str, ...]


def assess_activity(sample, now, timeout_s) -> ActivityAssessment:
    _number(now, "now", inclusive=True)
    _number(timeout_s, "timeout_s")

    def recent(value):
        return (
            value is not None and math.isfinite(value) and 0 <= now - value <= timeout_s
        )

    if sample.running is None or sample.waiting is None:
        return ActivityAssessment(False, ("unknown_activity",))
    _integer(sample.running, "running")
    _integer(sample.waiting, "waiting")
    active = recent(sample.active_step_started_s)
    supplied = recent(sample.last_dispatch_s) or active or sample.waiting > 0
    progressing = recent(sample.last_progress_s) or active
    reasons = tuple(
        reason
        for valid, reason in (
            (supplied, "supply_timeout"),
            (progressing, "progress_timeout"),
        )
        if not valid
    )
    return ActivityAssessment(not reasons, reasons)


def activity_valid(sample, now, timeout_s) -> bool:
    return assess_activity(sample, now, timeout_s).valid


@dataclass(frozen=True)
class CoverageAssessment:
    """Window coverage with cumulative diagnostics from the entire epoch so far."""

    epoch_valid: bool
    max_segments_per_target: int
    ratio: float
    longest_invalid_s: float
    valid: bool
    intervals: tuple[Interval, ...]
    epoch_reasons: tuple[str, ...]


@dataclass(frozen=True)
class MigrationPublicationEvidence:
    interval: Interval
    before_blocks: int
    after_blocks: int
    endpoint_deltas: tuple[tuple[tuple, int, int], ...]
    zero_baseline_endpoints: tuple[tuple, ...] = ()
    source: str = "published_blocks"


@dataclass(frozen=True)
class MigrationGroupAssessment:
    owner_rank: int
    owner_group_id: int
    d2h_intervals: tuple[Interval, ...]
    h2d_intervals: tuple[Interval, ...]
    valid: bool
    d2h_evidence: tuple[MigrationPublicationEvidence, ...] = ()
    h2d_evidence: tuple[MigrationPublicationEvidence, ...] = ()
    d2h_failure_evidence: tuple[MigrationPublicationEvidence, ...] = ()
    h2d_failure_evidence: tuple[MigrationPublicationEvidence, ...] = ()


@dataclass(frozen=True)
class MigrationAssessment:
    epoch_valid: bool
    groups: tuple[MigrationGroupAssessment, ...]
    valid: bool
    reasons: tuple[str, ...]


class PressureCoverage:
    """Bounded streaming sampled evidence, shared by all diagnostic windows.

    Baselines are captured before fill. Call add once per poll, including empty
    failed polls; persist raw evidence separately. Historical segments beyond
    max_segments per target are dropped explicitly, so their time is unknown.
    """

    def __init__(self, baselines, config, max_segments=4096):
        _integer(max_segments, "max_segments", 1)
        self.config = config
        self.ranks = tuple(sorted(s.rank for s in baselines))
        self.targets = select_targets(tuple(baselines), self.ranks)
        self.baselines = {s.rank: s for s in baselines}
        if config.evidence_policy == "tree_then_mock_disk" and any(
            getattr(
                self.baselines[t.owner_rank].mock_cache, "terminal_outcome_version", 0
            )
            != 1
            for t in self.targets
        ):
            raise ValueError(
                "mock cache terminal outcome counters are unavailable or unsupported"
            )
        self.local_tiers = tuple(
            tier
            for tier in ("memory", "disk")
            if config.evidence_policy.startswith("tree_then_mock_")
            and mock_provider_enabled(config, tier)
        )
        self.local_reasons = set()
        self.local_previous = dict(self.baselines)
        self.local_final = None
        self.effective_ppm = mock_target_ppm(config)
        self.natural_memory_load = (
            config.evidence_policy == "tree_then_mock_memory"
            and not self.effective_ppm["memory"]
        )
        self.remote_final = None
        self.remote_request_reuse = None
        self.remote_reasons = set()
        self.remote_previous = dict(self.baselines)
        if config.evidence_policy == "tree_then_mock_remote":
            if any(not self._remote_capable(s) for s in baselines):
                raise ValueError(
                    "Remote completion capability/configuration is unavailable"
                )
            for sample in baselines:
                options = json.loads(sample.remote_runtime).get("options", {})
                for name, expected in config.mock_cache_settings:
                    if not (
                        name.startswith("mock_remote_")
                        or name
                        in {
                            "mock_memory_cache",
                            "mock_disk_cache",
                            "mock_memory_cache_hit_ratio",
                            "mock_disk_cache_hit_ratio",
                        }
                    ):
                        continue
                    raw = options.get(name)
                    if raw is None:
                        raise ValueError("Remote runtime option is missing: " + name)
                    actual = (
                        raw.lower() == "true"
                        if isinstance(expected, bool)
                        else type(expected)(raw)
                    )
                    if actual != expected:
                        raise ValueError(
                            "Remote effective engine option differs from frozen configuration: "
                            + name
                        )
            self._observe_remote(baselines)
        self.mock_cache_deltas = []
        if config.evidence_policy in {"mock_memory", "tree_then_mock_memory"} and any(
            self.baselines[target.owner_rank].mock_memory is None
            for target in self.targets
        ):
            raise ValueError("mock memory pressure counters are unavailable")
        if config.evidence_policy in {"mock_memory", "tree_then_mock_memory"} and any(
            getattr(
                self.baselines[target.owner_rank].mock_memory,
                "terminal_outcome_version",
                0,
            )
            != MOCK_MEMORY_TERMINAL_OUTCOME_VERSION
            for target in self.targets
        ):
            raise ValueError(
                "mock memory terminal outcome counters are unavailable or unsupported"
            )
        self.previous = {}
        self.last_seen = {}
        self.segments = {(t.rank, t.group_id): [] for t in self.targets}
        self.owner_groups = tuple(
            sorted(
                {(target.owner_rank, target.owner_group_id) for target in self.targets}
            )
        )
        self.migration_intervals = {
            (*key, direction): []
            for key in self.owner_groups
            for direction in ("d2h", "h2d")
        }
        self.migration_evidence = {key: [] for key in self.migration_intervals}
        self.migration_failure_evidence = {key: [] for key in self.migration_intervals}
        # Failure observations survive rejected/partial pages and freshness gaps.
        # Successful migration evidence still requires complete adjacency.
        self.failure_previous = {
            rank: self.baselines[rank] for rank, _ in self.owner_groups
        }
        self.migration_history_overflow = False
        self.open_segments = {}
        self.max_segments = max_segments
        self.reasons = set()
        self.epoch_valid = True

    def _break(self, rank):
        self.previous.pop(rank, None)
        for key in list(self.open_segments):
            if key[0] == rank:
                del self.open_segments[key]

    def add(self, samples, now, *, complete=True):
        _number(now, "now", inclusive=True)
        if not self.epoch_valid:
            return
        samples = tuple(samples)
        self._observe_local(samples)
        if self.config.evidence_policy == "tree_then_mock_remote":
            self._observe_remote(samples)
        # Validate every concrete payload even if another participant is absent.
        # Missing samples break adjacency; observed resets invalidate the epoch.
        for sample in samples:
            baseline = self.baselines.get(sample.rank)
            reference = self.last_seen.get(sample.rank, baseline)
            if baseline is None:
                self.epoch_valid = False
                self.reasons.add("inventory_discontinuity")
                continue
            if (
                self.config.evidence_policy in {"mock_memory", "tree_then_mock_memory"}
                and any(target.owner_rank == sample.rank for target in self.targets)
                and getattr(sample.mock_memory, "terminal_outcome_version", 0)
                != MOCK_MEMORY_TERMINAL_OUTCOME_VERSION
            ):
                self.epoch_valid = False
                self.reasons.add("mock_terminal_outcomes_unavailable")
            if (
                sample.instance != baseline.instance
                or sample.local_mock_targets != baseline.local_mock_targets
                or sample.ownership != baseline.ownership
                or _layout(sample) != _layout(baseline)
            ):
                self.epoch_valid = False
                self.reasons.add("inventory_discontinuity")
            counters = dict(sample.counters)
            if (
                self.config.evidence_policy == "tree_then_mock_disk"
                and sample.rank in {t.owner_rank for t in self.targets}
            ):
                if getattr(sample.mock_cache, "terminal_outcome_version", 0) != 1:
                    self.epoch_valid = False
                    self.reasons.add("mock_terminal_outcomes_unavailable")
                elif any(
                    getattr(sample.mock_cache, f.name)
                    < getattr(reference.mock_cache, f.name)
                    for f in fields(MockCacheSample)
                ):
                    self.epoch_valid = False
                    self.reasons.add("counter_reset")
                old_pools = {p.pool_id: p for p in reference.pools}
                if any(
                    getattr(p, field) < getattr(old_pools[p.pool_id], field)
                    for p in sample.pools
                    if p.pool_id in old_pools
                    for field in ("disk_read_bytes", "disk_write_bytes")
                ):
                    self.epoch_valid = False
                    self.reasons.add("counter_reset")
            old = dict(reference.counters)
            groups = {g.group_id: g for g in sample.groups}
            if (
                old
                and any(k not in counters or counters[k] < v for k, v in old.items())
            ) or any(
                g.group_id in groups
                and (
                    groups[g.group_id].triggers < g.triggers
                    or groups[g.group_id].gpu_host_published < g.gpu_host_published
                )
                for g in reference.groups
            ):
                self.epoch_valid = False
                self.reasons.add("counter_reset")
            if sample.sequence <= reference.sequence:
                self.epoch_valid = False
                self.reasons.add("sequence_reset")
        if self.epoch_valid:
            if (
                self.config.evidence_policy == "tree_pressure"
                or self.natural_memory_load
            ):
                self._observe_real_failures(samples, now)
            if self.config.evidence_policy != "tree_pressure":
                self._observe_mock_memory_failures(samples, now)
            self.failure_previous.update(
                (s.rank, s) for s in samples if s.rank in self.failure_previous
            )
        if not self.epoch_valid:
            for rank in self.ranks:
                self._break(rank)
            return
        if not complete:
            self.reasons.add("incomplete_sample")
            self.last_seen.update((sample.rank, sample) for sample in samples)
            for rank in self.ranks:
                self._break(rank)
            return
        try:
            validate_inventory(select_targets(samples, self.ranks), self.targets)
        except ValueError:
            self.reasons.add("inventory_discontinuity")
            # Rejected pages still contain observed cumulative counters. Retain
            # them for reset detection without establishing certified adjacency.
            self.last_seen.update((sample.rank, sample) for sample in samples)
            # A failed/missing poll breaks adjacency; changed service identity
            # invalidates this epoch and requires new pre-fill baselines.
            if any(
                s.rank in self.baselines
                and s.instance != self.baselines[s.rank].instance
                for s in samples
            ):
                self.epoch_valid = False
            for rank in self.ranks:
                self._break(rank)
            return
        adjacent = {}
        for sample in samples:
            rank = sample.rank
            bounds = sample.offset_bounds or (
                sample.client_send_s - sample.server_s,
                sample.client_receive_s - sample.server_s,
            )
            low, high = bounds
            reference = self.last_seen.get(rank, self.baselines[rank])
            old_counters = dict(reference.counters)
            counters = dict(sample.counters)
            old_groups = {g.group_id: g for g in reference.groups}
            reset = (
                old_counters
                and (
                    any(
                        k not in counters or counters[k] < v
                        for k, v in old_counters.items()
                    )
                )
            ) or any(
                g.triggers < old_groups[g.group_id].triggers
                or g.gpu_host_published < old_groups[g.group_id].gpu_host_published
                for g in sample.groups
                if g.group_id in old_groups
            )
            reason = (
                "counter_reset"
                if reset
                else (
                    "sequence_reset" if sample.sequence <= reference.sequence else None
                )
            )
            self.last_seen[rank] = sample
            if reason:
                self.reasons.add(reason)
                self.epoch_valid = False
                self._break(rank)
                continue
            if (
                not all(math.isfinite(v) for v in bounds)
                or low > high
                or now < sample.client_receive_s
                or now - (sample.server_s + low) > self.config.max_sample_age_s
            ):
                self.reasons.add("stale_or_unknown_age")
                self._break(rank)
                continue
            previous = self.previous.get(rank)
            if previous:
                old, old_bounds = previous
                reason = None
                if sample.sequence <= old.sequence:
                    reason = "sequence_reset"
                elif (
                    not 0
                    < sample.server_s - old.server_s
                    <= self.config.max_sample_gap_s
                    or not 0
                    <= sample.client_receive_s - old.client_receive_s
                    <= self.config.max_sample_gap_s
                ):
                    reason = "sample_gap"
                elif max(low, old_bounds[0]) > min(high, old_bounds[1]):
                    reason = "clock_discontinuity"
                old_groups = {g.group_id: g for g in old.groups}
                if any(
                    g.triggers < old_groups[g.group_id].triggers
                    or g.gpu_host_published < old_groups[g.group_id].gpu_host_published
                    for g in sample.groups
                    if g.group_id in old_groups
                ):
                    reason = "counter_reset"
                if reason:
                    self.reasons.add(reason)
                    if reason in ("counter_reset", "sequence_reset"):
                        self.epoch_valid = False
                    self._break(rank)
                    previous = None
            if not self.epoch_valid:
                continue
            if previous:
                adjacent[rank] = (*previous, sample, bounds)
            for target in (t for t in self.targets if t.rank == rank):
                key = (rank, target.group_id)
                try:
                    # Peer intervals establish freshness only. Every shared target
                    # predicate is evaluated on its owner's clock and counters.
                    valid = all(
                        self._target_valid(t, sample, samples)
                        for t in self.targets
                        if t.owner_rank == rank and t.owner_group_id == target.group_id
                    )
                except ValueError:
                    valid = False
                if not valid:
                    self.reasons.add("pressure_lost")
                    self.open_segments.pop(key, None)
                    continue
                # Track a valid point even without an interval; next adjacent
                # point extends it. Merge on the server clock before mapping.
                current = self.open_segments.get(key) if previous else None
                if current is None:
                    self.open_segments[key] = [
                        sample.server_s,
                        sample.server_s,
                        low,
                        high,
                    ]
                else:
                    current[1] = sample.server_s
                    current[2] = min(current[2], low)
                    current[3] = max(current[3], high)
                    if self.segments[key] and self.segments[key][-1] is current:
                        pass
                    else:
                        self.segments[key].append(current)
                        if len(self.segments[key]) > self.max_segments:
                            self.segments[key].pop(0)
                            self.reasons.add("history_overflow")
            self.previous[rank] = sample, bounds
        if self.epoch_valid:
            self._record_migration(adjacent)

    def _record_migration(self, adjacent):
        if self.config.evidence_policy == "tree_then_mock_disk":
            self._record_mock_cache_deltas(adjacent)
            return
        for owner_rank, owner_group_id in self.owner_groups:
            target = next(
                target
                for target in self.targets
                if (target.owner_rank, target.owner_group_id)
                == (owner_rank, owner_group_id)
            )
            # A complete page alone is insufficient: every declared participant
            # must have fresh, clock-compatible adjacent samples in this poll.
            if any(rank not in adjacent for rank, _ in target.participants):
                continue
            old, old_bounds, current, bounds = adjacent[owner_rank]
            interval = possible_event_segment(
                old.server_s, current.server_s, old_bounds, bounds
            )
            if self.config.evidence_policy in {"mock_memory", "tree_then_mock_memory"}:
                self._record_mock_memory_migration(
                    owner_rank, owner_group_id, old, current, interval
                )
                if not self.natural_memory_load:
                    continue
            old_counters = dict(old.counters)
            group = next(g for g in current.groups if g.group_id == owner_group_id)
            for direction in ("h2d",) if self.natural_memory_load else ("d2h", "h2d"):
                endpoints = []
                for key, value in current.counters:
                    if len(key) != 6 or key[:2] != ("published_blocks", owner_group_id):
                        continue
                    _, _, pool_id, operation, source, destination = key
                    if direction == "d2h":
                        matches = (
                            pool_id == group.host_pool_id
                            and operation in (2, 3)
                            and (source, destination) == (1, 2)
                        )
                    else:
                        matches = (
                            pool_id in group.device_pool_ids
                            and operation == 1
                            and (source, destination) == (2, 1)
                        )
                    if matches:
                        endpoints.append((key, old_counters.get(key, 0), value))
                before = sum(before for _, before, _ in endpoints)
                after = sum(after for _, _, after in endpoints)
                if after <= before:
                    continue
                key = owner_rank, owner_group_id, direction
                self.migration_intervals[key].append(interval)
                self.migration_evidence[key].append(
                    MigrationPublicationEvidence(
                        interval,
                        before,
                        after,
                        tuple(sorted(endpoints)),
                        tuple(
                            sorted(
                                endpoint
                                for endpoint, _, _ in endpoints
                                if endpoint not in old_counters
                            )
                        ),
                    )
                )
                if len(self.migration_intervals[key]) > self.max_segments:
                    self.migration_intervals[key].pop(0)
                    self.migration_evidence[key].pop(0)
                    self.migration_history_overflow = True
                    self.reasons.add("history_overflow")

    def _observe_real_failures(self, samples, now):
        for current in samples:
            old = self.failure_previous.get(current.rank)
            if old is None:
                continue
            interval = _failure_interval(old, current, now)
            old_counters = dict(old.counters)
            for endpoint, after in current.counters:
                if len(endpoint) != 6 or endpoint[0] != "failed_operations":
                    continue
                _, group, _, operation, source, destination = endpoint
                if (source, destination) == (1, 2) and operation in (2, 3):
                    direction = "d2h"
                elif (source, destination) == (2, 1) and operation == 1:
                    direction = "h2d"
                else:
                    continue
                evidence = self.migration_failure_evidence.get(
                    (current.rank, group, direction)
                )
                before = old_counters.get(endpoint, 0)
                if evidence is None or after <= before:
                    continue
                # The source and destination can both report a failed
                # operation. Retain the endpoint deltas without summing them.
                evidence.append(
                    MigrationPublicationEvidence(
                        interval,
                        before,
                        after,
                        ((endpoint, before, after),),
                        (endpoint,) if endpoint not in old_counters else (),
                        source="failed_operations",
                    )
                )
                if len(evidence) > self.max_segments:
                    evidence.pop(0)
                    self.migration_history_overflow = True
                    self.reasons.add("history_overflow")

    def _observe_mock_memory_failures(self, samples, now):
        if self.config.evidence_policy == "tree_then_mock_disk":
            for current in samples:
                old = self.failure_previous.get(current.rank)
                if old is None or current.mock_cache is None or old.mock_cache is None:
                    continue
                interval = (
                    min(old.client_send_s, current.client_send_s),
                    max(now, current.client_receive_s),
                )
                for owner, group in self.owner_groups:
                    if owner != current.rank:
                        continue
                    for direction, field in (
                        ("h2d", "failed_blocks"),
                        ("d2h", "disk_write_failed_records"),
                    ):
                        before, after = getattr(old.mock_cache, field), getattr(
                            current.mock_cache, field
                        )
                        if after > before:
                            target = self.migration_failure_evidence[
                                (owner, group, direction)
                            ]
                            target.append(
                                MigrationPublicationEvidence(
                                    interval,
                                    before,
                                    after,
                                    ((("mock_cache", field), before, after),),
                                    source="mock_cache_" + field,
                                )
                            )
                            if len(target) > self.max_segments:
                                target.pop(0)
                                self.migration_history_overflow = True
            return
        if self.config.evidence_policy not in {"mock_memory", "tree_then_mock_memory"}:
            return
        for current in samples:
            old = self.failure_previous.get(current.rank)
            if old is None or old.mock_memory is None or current.mock_memory is None:
                continue
            if any(
                getattr(current.mock_memory, field) < getattr(old.mock_memory, field)
                for field in ("failed_h2d_blocks", "d2h_failed_blocks")
            ):
                self.epoch_valid = False
                self.reasons.add("counter_reset")
                continue
            interval = _failure_interval(old, current, now)
            for owner_rank, owner_group_id in self.owner_groups:
                if owner_rank == current.rank:
                    self._record_mock_memory_migration(
                        owner_rank,
                        owner_group_id,
                        old,
                        current,
                        interval,
                        failures=True,
                    )

    def _record_mock_memory_migration(
        self, owner_rank, owner_group_id, old, current, interval, *, failures=False
    ):
        if old.mock_memory is None or current.mock_memory is None:
            return
        fields = {
            "h2d": "failed_h2d_blocks" if failures else "successful_h2d_blocks",
            "d2h": "d2h_failed_blocks" if failures else "d2h_completed_blocks",
        }
        evidence = (
            self.migration_failure_evidence if failures else self.migration_evidence
        )
        for direction, field in fields.items():
            if not failures and self.natural_memory_load and direction == "h2d":
                continue
            key = owner_rank, owner_group_id, direction
            destination = evidence[key]
            before = getattr(old.mock_memory, field)
            after = getattr(current.mock_memory, field)
            if after <= before:
                continue
            destination.append(
                MigrationPublicationEvidence(
                    interval,
                    before,
                    after,
                    ((("mock_memory", field), before, after),),
                    source=f"mock_memory_{field}",
                )
            )
            if len(destination) > self.max_segments:
                destination.pop(0)
                self.migration_history_overflow = True
                self.reasons.add("history_overflow")

    def _local_counts(self, sample, tier):
        if tier == "memory":
            record = sample.mock_memory
            names = (
                "d2h_submitted_blocks",
                "d2h_completed_blocks",
                "d2h_failed_blocks",
            )
        else:
            record = sample.mock_cache
            names = (
                "disk_write_submitted_records",
                "disk_write_completed_records",
                "disk_write_failed_records",
            )
        if getattr(record, "terminal_outcome_version", 0) == 1:
            return tuple(getattr(record, name) for name in names)
        return None

    def _observe_local(self, samples):
        if not self.local_tiers:
            return
        for current in samples:
            if current.rank not in {t.owner_rank for t in self.targets}:
                continue
            previous = self.local_previous[current.rank]
            baseline = self.baselines[current.rank]
            for tier in self.local_tiers:
                old = self._local_counts(previous, tier)
                new = self._local_counts(current, tier)
                if old is None or new is None:
                    self.local_reasons.add(f"{tier}_write_counters_missing")
                elif any(a < b for a, b in zip(new, old)):
                    self.local_reasons.add(f"{tier}_write_counter_reset")
                if old and new and new[2] > old[2]:
                    self.local_reasons.add(f"{tier}_write_failed")
            if baseline.local_store is not None:
                if current.local_store is None or previous.local_store is None:
                    self.local_reasons.add("local_store_capability_missing")
                else:
                    for field in (
                        "released_requests",
                        "abandoned_requests",
                        "rejected_host_stores",
                        "rejected_disk_stores",
                    ):
                        count = getattr(current.local_store, field)
                        if count < getattr(previous.local_store, field):
                            self.local_reasons.add("local_store_counter_reset")
                        if field != "released_requests" and count > getattr(
                            baseline.local_store, field
                        ):
                            self.local_reasons.add("local_store_" + field)
            self.local_previous[current.rank] = current

    def local_write_summary(self, samples, completed_requests):
        """Compare one stopped-ingress observation against the pre-fill owner baselines."""
        owners = {}
        reasons = set(self.local_reasons)
        if not self.epoch_valid:
            reasons.update(self.reasons)
        released = 0
        pending = False
        current_by_rank = {s.rank: s for s in samples}
        for owner in sorted({t.owner_rank for t in self.targets}):
            baseline = self.baselines[owner]
            current = current_by_rank.get(owner)
            if current is None or current.instance != baseline.instance:
                reasons.add("local_store_owner_missing")
                continue
            rows = owners[str(owner)] = {}
            old, new = baseline.local_store, current.local_store
            if old is None or new is None:
                reasons.add("local_store_capability_missing")
            else:
                if old.pending_requests:
                    reasons.add("local_store_baseline_pending")
                for name in (
                    "abandoned_requests",
                    "rejected_host_stores",
                    "rejected_disk_stores",
                ):
                    if getattr(new, name) != getattr(old, name):
                        reasons.add("local_store_" + name)

                count = new.released_requests - old.released_requests
                rows.update(
                    released_requests=count,
                    pending_requests=new.pending_requests,
                    abandoned_requests=new.abandoned_requests - old.abandoned_requests,
                    rejected_host_stores=new.rejected_host_stores
                    - old.rejected_host_stores,
                    rejected_disk_stores=new.rejected_disk_stores
                    - old.rejected_disk_stores,
                )
                released += count
                pending |= new.pending_requests != 0
            for tier in self.local_tiers:
                first, last = self._local_counts(baseline, tier), self._local_counts(
                    current, tier
                )
                if first is None or last is None:
                    reasons.add(f"{tier}_write_counters_missing")
                    continue
                submitted, completed, failed = (a - b for a, b in zip(last, first))
                rows[tier] = dict(
                    submitted=submitted,
                    completed=completed,
                    failed=failed,
                    pending=last[0] - last[1] - last[2],
                )
                if first[0] != first[1] + first[2]:
                    reasons.add(f"{tier}_write_baseline_pending")
                if failed:
                    reasons.add(f"{tier}_write_failed")
                if last[0] < last[1] + last[2]:
                    reasons.add(f"{tier}_write_counters_inconsistent")
                if last[0] > last[1] + last[2]:
                    pending = True
                if completed <= 0:
                    reasons.add(f"{tier}_write_without_physical_completion")
        if completed_requests is None or released != completed_requests:
            reasons.add("local_store_request_count_mismatch")
            if completed_requests is not None and released < completed_requests:
                pending = True
        if pending:
            reasons.add("local_completions_pending")
        return dict(
            valid=not reasons, pending=pending, reasons=sorted(reasons), owners=owners
        )

    def _remote_capable(self, sample):
        return (
            sample.remote_runtime is not None
            and sample.mock_remote_write is not None
            and sample.remote_read_payload_bytes is not None
            and sample.remote_write_acknowledged_payload_bytes is not None
            and (
                sample.rank not in {t.owner_rank for t in self.targets}
                or getattr(sample.mock_cache, "terminal_outcome_version", 0) == 1
            )
        )

    def _observe_remote(self, samples):
        owners = {t.owner_rank for t in self.targets}
        for current in samples:
            baseline = self.baselines.get(current.rank)
            if baseline is None or not self._remote_capable(current):
                self.remote_reasons.add("remote_capability_unavailable")
                continue
            previous = self.remote_previous[current.rank]
            if current.remote_runtime != baseline.remote_runtime:
                self.remote_reasons.add("remote_runtime_changed")
            for field in (
                "remote_read_payload_bytes",
                "remote_write_acknowledged_payload_bytes",
            ):
                if getattr(current, field) < getattr(previous, field):
                    self.remote_reasons.add("remote_counter_reset")
            if current.rank in owners:
                for field in fields(MockCacheSample):
                    if getattr(current.mock_cache, field.name) < getattr(
                        previous.mock_cache, field.name
                    ):
                        self.remote_reasons.add("remote_counter_reset")
                for field in ("submitted", "completed", "failed", "rejected"):
                    if getattr(current.mock_remote_write, field) < getattr(
                        previous.mock_remote_write, field
                    ):
                        self.remote_reasons.add("remote_counter_reset")
                if current.mock_cache.failed_blocks > baseline.mock_cache.failed_blocks:
                    self.remote_reasons.add("remote_read_failed")
                for field in ("failed", "rejected"):
                    if getattr(current.mock_remote_write, field) > getattr(
                        baseline.mock_remote_write, field
                    ):
                        self.remote_reasons.add("remote_write_" + field)
            if (
                current.mock_remote_write.faulted
                or current.mock_remote_write.quarantined_blocks
            ):
                self.remote_reasons.add("remote_restart_required")
            self.remote_previous[current.rank] = current

    def finish_remote_window(self, samples, *, request_reuse=None):
        """Only after stopped ingress, awaited responses and native reusable drain."""
        samples = tuple(samples)
        self._observe_remote(samples)
        self._observe_local(samples)
        if self.local_tiers:
            self.local_final = self.local_write_summary(
                samples, request_reuse and request_reuse.get("completed_requests")
            )
        try:
            validate_inventory(select_targets(samples, self.ranks), self.targets)
        except ValueError:
            self.remote_reasons.add("remote_final_inventory_invalid")
        self.remote_final = {s.rank: s for s in samples}
        self.remote_request_reuse = (
            dict(request_reuse) if request_reuse is not None else None
        )

    def mock_remote_summary(self):
        reasons = set(self.remote_reasons)
        effective = {tier: ppm / 1_000_000 for tier, ppm in self.effective_ppm.items()}
        if not self.epoch_valid:
            reasons.update(self.reasons)
        owners, ranks = {}, {}
        if self.remote_final is None:
            reasons.add("remote_completions_not_drained")
        for rank, baseline in self.baselines.items():
            current = (self.remote_final or {}).get(rank)
            if current is None or not self._remote_capable(current):
                reasons.add("remote_final_rank_missing")
                continue
            ranks[str(rank)] = {
                field: getattr(current, field) - getattr(baseline, field)
                for field in (
                    "remote_read_payload_bytes",
                    "remote_write_acknowledged_payload_bytes",
                )
            }
        for owner in sorted({t.owner_rank for t in self.targets}):
            baseline = self.baselines[owner]
            current = (self.remote_final or {}).get(owner)
            if current is None or not self._remote_capable(current):
                reasons.add("remote_owner_missing")
                continue
            logical = {
                f.name: getattr(current.mock_cache, f.name)
                - getattr(baseline.mock_cache, f.name)
                for f in fields(MockCacheSample)
                if f.name != "terminal_outcome_version"
            }
            logical.update(
                {
                    field: getattr(current.mock_remote_write, field)
                    - getattr(baseline.mock_remote_write, field)
                    for field in ("submitted", "completed", "failed", "rejected")
                }
            )
            logical.update(
                {
                    field: getattr(current.mock_remote_write, field)
                    for field in (
                        "pending_requests",
                        "pending_intents",
                        "retained_descriptors",
                        "faulted",
                        "quarantined_blocks",
                    )
                }
            )
            for tier in ("memory", "disk", "remote"):
                for prefix in ("planned", "successful"):
                    logical[f"{prefix}_{tier}_ratio"] = (
                        logical[f"{prefix}_{tier}_blocks"] / logical["eligible_blocks"]
                        if logical["eligible_blocks"]
                        else None
                    )
                if (
                    logical[f"planned_{tier}_blocks"]
                    != logical[f"successful_{tier}_blocks"]
                ):
                    reasons.add("remote_incomplete_" + tier + "_reuse")
            if not logical["eligible_blocks"] and (
                any(effective.values())
                or not self.remote_request_reuse
                or self.remote_request_reuse.get("reused_requests", 0) <= 0
                or self.remote_request_reuse.get("reused_tokens", 0) <= 0
            ):
                reasons.add("remote_missing_reuse_coverage")
            if (
                logical["submitted"] <= 0
                or logical["completed"] != logical["submitted"]
            ):
                reasons.add("remote_incomplete_write_coverage")
            if any(
                logical[field]
                for field in (
                    "pending_requests",
                    "pending_intents",
                    "retained_descriptors",
                )
            ):
                reasons.add("remote_completions_pending")
            participants = {
                rank
                for t in self.targets
                if t.owner_rank == owner
                for rank, _ in t.participants
            }
            for rank in participants:
                physical = ranks.get(str(rank), {})
                if logical["planned_remote_blocks"] and not physical.get(
                    "remote_read_payload_bytes", 0
                ):
                    reasons.add("remote_missing_read_payload")
                if logical["completed"] and not physical.get(
                    "remote_write_acknowledged_payload_bytes", 0
                ):
                    reasons.add("remote_missing_write_ack_payload")
            owners[str(owner)] = logical
        if self.remote_request_reuse is not None:
            requests = self.remote_request_reuse
            if (
                requests.get("completed_requests", 0) <= 0
                or requests.get("reused_requests") != requests.get("completed_requests")
                or sum(value["submitted"] for value in owners.values())
                != requests.get("completed_requests")
            ):
                reasons.add("remote_incomplete_cacheable_request_coverage")
        remote_valid = bool(owners) and not reasons
        if self.local_tiers:
            reasons.update(self.local_reasons)
            if self.local_final is None:
                reasons.add("local_completions_not_drained")
            else:
                reasons.update(self.local_final["reasons"])
        return dict(
            remote_drain_valid=remote_valid,
            valid=bool(owners) and not reasons,
            reasons=sorted(reasons),
            owners=owners,
            ranks=ranks,
            effective_target_ratios=effective,
            local_writes=self.local_final,
            request_reuse=self.remote_request_reuse,
            configured_target_ratios={
                tier: getattr(self.config, f"mock_{tier}_target_ratio")
                for tier in ("memory", "disk", "remote")
            },
            remote_read_payload_bytes=sum(
                v["remote_read_payload_bytes"] for v in ranks.values()
            ),
            remote_write_acknowledged_payload_bytes=sum(
                v["remote_write_acknowledged_payload_bytes"] for v in ranks.values()
            ),
            basis="post-startup baseline to native drained final snapshot; owner logical counts once; actual per-rank application bytes",
            output_correctness_valid=False,
        )

    def _record_mock_cache_deltas(self, adjacent):
        # Exactly one record per owner poll, regardless of groups/TP endpoints.
        for owner in sorted({rank for rank, _ in self.owner_groups}):
            targets = [t for t in self.targets if t.owner_rank == owner]
            if any(rank not in adjacent for t in targets for rank, _ in t.participants):
                continue
            old, old_bounds, current, bounds = adjacent[owner]
            if old.mock_cache is None or current.mock_cache is None:
                continue
            interval = possible_event_segment(
                old.server_s, current.server_s, old_bounds, bounds
            )
            delta = {
                f.name: getattr(current.mock_cache, f.name)
                - getattr(old.mock_cache, f.name)
                for f in fields(MockCacheSample)
                if f.name != "terminal_outcome_version"
            }
            old_pools = {p.pool_id: p for p in old.pools}
            disk = {
                p.pool_id: {
                    field: getattr(p, field) - getattr(old_pools[p.pool_id], field)
                    for field in ("disk_read_bytes", "disk_write_bytes")
                }
                for p in current.pools
                if p.tier == 3 and p.pool_id in old_pools
            }
            self.mock_cache_deltas.append((owner, interval, delta, disk))
            if len(self.mock_cache_deltas) > self.max_segments:
                self.mock_cache_deltas.pop(0)
                self.migration_history_overflow = True

    def mock_cache_summary(self, start, end):
        owners = {}
        for owner, interval, delta, disk in self.mock_cache_deltas:
            if not start <= interval[0] <= interval[1] <= end:
                continue
            value = owners.setdefault(
                str(owner),
                dict.fromkeys((*delta, "disk_read_bytes", "disk_write_bytes"), 0),
            )
            for field, amount in delta.items():
                value[field] += amount
            # Pool inventory is deduplicated; do not also sum transfer endpoints.
            for counters in disk.values():
                for field, amount in counters.items():
                    value[field] += amount
        for value in owners.values():
            for prefix in ("planned", "successful"):
                for tier in ("memory", "disk"):
                    value[f"{prefix}_{tier}_ratio"] = (
                        value[f"{prefix}_{tier}_blocks"] / value["eligible_blocks"]
                        if value["eligible_blocks"]
                        else None
                    )
        return {
            "owners": owners,
            "configured_target_ratios": {
                "memory": self.config.mock_memory_target_ratio,
                "disk": self.config.mock_disk_target_ratio,
            },
            "basis": "confirmed measurement deltas; logical owner counts; physical disk pool bytes",
            "output_correctness_valid": False,
        }

    def _assess_mock_cache_migration(self, start, end):
        reasons = set(self.reasons) if not self.epoch_valid else set()
        if self.local_final is not None:
            reasons.update(self.local_final["reasons"])
        elif self.local_tiers and any(s.local_store for s in self.baselines.values()):
            reasons.add("local_completions_not_drained")
        if self.migration_history_overflow:
            reasons.add("history_overflow")
        groups = []
        for owner, group_id in self.owner_groups:
            group = next(
                g for g in self.baselines[owner].groups if g.group_id == group_id
            )
            records = [
                r
                for r in self.mock_cache_deltas
                if r[0] == owner and start <= r[1][0] <= r[1][1] <= end
            ]
            planned = sum(
                r[2]["planned_memory_blocks"] + r[2]["planned_disk_blocks"]
                for r in records
            )
            planned_disk = sum(r[2]["planned_disk_blocks"] for r in records)
            reads = sum(
                r[3].get(group.disk_pool_id, {}).get("disk_read_bytes", 0)
                for r in records
            )
            writes = sum(
                r[3].get(group.disk_pool_id, {}).get("disk_write_bytes", 0)
                for r in records
            )
            evidence = {"h2d": [], "d2h": []}
            for _, interval, delta, _ in records:
                for direction, count in (
                    (
                        "h2d",
                        delta["successful_memory_blocks"]
                        + delta["successful_disk_blocks"],
                    ),
                    ("d2h", delta["disk_write_completed_records"]),
                ):
                    if count:
                        evidence[direction].append(
                            MigrationPublicationEvidence(
                                interval,
                                0,
                                count,
                                (),
                                source="mock_cache_aggregate_" + direction,
                            )
                        )
            failures = {
                d: tuple(
                    v
                    for v in self.migration_failure_evidence[(owner, group_id, d)]
                    if v.interval[0] <= end and start <= v.interval[1]
                )
                for d in evidence
            }
            h2d, d2h = tuple(v.interval for v in evidence["h2d"]), tuple(
                v.interval for v in evidence["d2h"]
            )
            missing = []
            if not records:
                missing.append("missing_mock_cache_samples")
            if planned and not h2d:
                missing.append("missing_h2d_load")
            if planned_disk and not reads:
                missing.append("missing_disk_read")
            if planned_disk and not sum(
                r[2]["successful_disk_blocks"] for r in records
            ):
                missing.append("missing_successful_disk_reuse")
            if sum(r[2]["planned_memory_blocks"] for r in records) and not sum(
                r[2]["successful_memory_blocks"] for r in records
            ):
                missing.append("missing_successful_memory_reuse")
            if not d2h or not writes:
                missing.append("missing_disk_write")
            for direction in evidence:
                if failures[direction]:
                    missing.append("failed_" + direction)
            reasons.update(f"{reason}:{owner}:{group_id}" for reason in missing)
            groups.append(
                MigrationGroupAssessment(
                    owner,
                    group_id,
                    d2h,
                    h2d,
                    not missing,
                    tuple(evidence["d2h"]),
                    tuple(evidence["h2d"]),
                    failures["d2h"],
                    failures["h2d"],
                )
            )
        return MigrationAssessment(
            self.epoch_valid,
            tuple(groups),
            self.epoch_valid and bool(groups) and not reasons,
            tuple(sorted(reasons)),
        )

    def assess_migration(self, start: float, end: float) -> MigrationAssessment:
        if self.config.evidence_policy == "tree_then_mock_remote":
            coverage([], start, end)
            summary = self.mock_remote_summary()
            return MigrationAssessment(
                self.epoch_valid, (), summary["valid"], tuple(summary["reasons"])
            )
        if self.config.evidence_policy == "tree_then_mock_disk":
            coverage([], start, end)
            return self._assess_mock_cache_migration(start, end)
        coverage([], start, end)
        reasons = set(self.reasons) if not self.epoch_valid else set()
        if self.local_final is not None:
            reasons.update(self.local_final["reasons"])
        elif self.local_tiers and any(s.local_store for s in self.baselines.values()):
            reasons.add("local_completions_not_drained")
        if self.migration_history_overflow:
            reasons.add("history_overflow")
        groups = []
        for owner_rank, owner_group_id in self.owner_groups:
            evidence = {}
            failures = {}
            for direction in ("d2h", "h2d"):
                evidence[direction] = tuple(
                    item
                    for item in self.migration_evidence[
                        (owner_rank, owner_group_id, direction)
                    ]
                    if start <= item.interval[0] and item.interval[1] <= end
                )
                failures[direction] = tuple(
                    item
                    for item in self.migration_failure_evidence[
                        (owner_rank, owner_group_id, direction)
                    ]
                    # A boundary-straddling failure may have occurred inside
                    # the window. Reject conservatively; containment is only
                    # sufficient for positive evidence.
                    if item.interval[0] <= end and start <= item.interval[1]
                )
            d2h = tuple(item.interval for item in evidence["d2h"])
            h2d = tuple(item.interval for item in evidence["h2d"])
            if not d2h:
                reasons.add(f"missing_d2h:{owner_rank}:{owner_group_id}")
            if not h2d:
                reasons.add(f"missing_h2d_load:{owner_rank}:{owner_group_id}")
            if failures["h2d"]:
                reasons.add(f"failed_h2d:{owner_rank}:{owner_group_id}")
            if failures["d2h"]:
                reasons.add(f"failed_d2h:{owner_rank}:{owner_group_id}")
            groups.append(
                MigrationGroupAssessment(
                    owner_rank,
                    owner_group_id,
                    d2h,
                    h2d,
                    bool(d2h and h2d and not failures["d2h"] and not failures["h2d"]),
                    evidence["d2h"],
                    evidence["h2d"],
                    failures["d2h"],
                    failures["h2d"],
                )
            )
        valid = (
            self.epoch_valid
            and not self.migration_history_overflow
            and bool(groups)
            and all(group.valid for group in groups)
            and not reasons
        )
        return MigrationAssessment(
            self.epoch_valid, tuple(groups), valid, tuple(sorted(reasons))
        )

    def _target_valid(self, target, owner, samples):
        if self.config.evidence_policy == "mock_memory":
            return owner.mock_memory is not None
        local = next(s for s in samples if s.rank == target.rank)
        local_pools = {p.pool_id: p for p in local.pools}
        owner_pools = {p.pool_id: p for p in owner.pools}
        pools = {}
        for local_id, owner_id in target.pool_mapping:
            source = owner_pools[owner_id]
            pools[local_id] = replace(
                local_pools[local_id],
                used=source.used,
                pending_release=source.pending_release,
                host_resident=source.host_resident,
            )
        source = next(g for g in owner.groups if g.group_id == target.owner_group_id)
        base = next(
            g
            for g in self.baselines[owner.rank].groups
            if g.group_id == target.owner_group_id
        )
        group = next(g for g in local.groups if g.group_id == target.group_id)
        return group_valid(
            replace(
                group,
                triggers=source.triggers,
                gpu_host_published=source.gpu_host_published,
            ),
            pools,
            replace(
                group,
                triggers=base.triggers,
                gpu_host_published=base.gpu_host_published,
            ),
            self.config,
        )

    def target_valid(self, target, samples):
        owner = next(s for s in samples if s.rank == target.owner_rank)
        return self._target_valid(target, owner, samples)

    def intervals(self, start, end):
        coverage([], start, end)  # validate window
        result = [(start, end)]
        for segments in self.segments.values():
            mapped = [confirmed_segment(*segment) for segment in segments]
            result = intersect_intervals(result, [v for v in mapped if v is not None])
        return result

    def assess(self, start, end) -> CoverageAssessment:
        values = self.intervals(start, end)
        ratio, longest = coverage(values, start, end)
        return CoverageAssessment(
            self.epoch_valid,
            self.max_segments,
            ratio,
            longest,
            self.epoch_valid
            and ratio >= self.config.min_coverage
            and longest <= self.config.max_invalid_s,
            tuple(values),
            tuple(sorted(self.reasons)),
        )


def _wire_counters(rank):
    """Preserve each cumulative endpoint and unit independently for reset checks."""
    output = []
    if "mock_cache" in rank.DESCRIPTOR.fields_by_name and rank.HasField("mock_cache"):
        for f in fields(MockCacheSample):
            if f.name != "terminal_outcome_version":
                output.append(
                    (
                        ("mock_cache", f.name),
                        _integer(getattr(rank.mock_cache, f.name, 0), f.name),
                    )
                )
    for pool in rank.pools:
        for field in ("disk_read_bytes", "disk_write_bytes"):
            output.append(
                ((field, pool.pool_id), _integer(getattr(pool, field, 0), field))
            )
    for group in rank.groups:
        output.append((("trigger", group.group_id), group.watermark_trigger_count))
    for transfer in rank.transfers:
        endpoint = (
            transfer.group_id,
            transfer.pool_id,
            transfer.operation,
            transfer.source_tier,
            transfer.target_tier,
        )
        for name in (
            "copied_blocks",
            "copied_bytes",
            "published_blocks",
            "published_bytes",
            "failed_operations",
        ):
            value = getattr(transfer, name, 0)
            output.append(((name,) + endpoint, _integer(value, name)))
    if "mock_memory" in rank.DESCRIPTOR.fields_by_name and rank.HasField("mock_memory"):
        for name in (
            "eligible_blocks",
            "planned_host_blocks",
            "successful_h2d_blocks",
            "failed_h2d_blocks",
            "d2h_submitted_blocks",
            "d2h_completed_blocks",
            "d2h_failed_blocks",
        ):
            output.append(
                (("mock_memory", name), _integer(getattr(rank.mock_memory, name), name))
            )
    return tuple(sorted(output))
