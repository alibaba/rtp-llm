# SPDX-License-Identifier: Apache-2.0
"""Engram storage modes, shard configuration and allocation limits."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from .placement import MAX_LOCAL_SEGMENTS, host_groups, parse_numa

HARD_MAX_ALLOCATION_BYTES = 128 << 30
HARD_MAX_TOTAL_ALLOCATION_BYTES = 256 << 30
DEFAULT_MAX_ALLOCATION_BYTES = HARD_MAX_ALLOCATION_BYTES
DEFAULT_MAX_TOTAL_ALLOCATION_BYTES = HARD_MAX_TOTAL_ALLOCATION_BYTES
MAX_ALLOCATION_ENV = "RTP_ENGRAM_MAX_ALLOCATION_BYTES"
MAX_TOTAL_ALLOCATION_ENV = "RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES"
_DECIMAL = re.compile(r"[1-9][0-9]{0,11}\Z")


class CapacityError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def validate_capacity(max_allocation_bytes, max_total_allocation_bytes):
    """Return validated local ceilings; a total smaller than per-allocation is legal."""
    for value, ceiling in (
        (max_allocation_bytes, HARD_MAX_ALLOCATION_BYTES),
        (max_total_allocation_bytes, HARD_MAX_TOTAL_ALLOCATION_BYTES),
    ):
        if type(value) is not int or not 0 < value <= ceiling:
            raise CapacityError("INVALID_CAPACITY_LIMIT")
    return max_allocation_bytes, max_total_allocation_bytes


def capacity_from_environ(environ=None):
    """Read each trusted local setting once; accept canonical decimal bytes only."""
    env = os.environ if environ is None else environ
    values = []
    missing = object()
    for key, default in (
        (MAX_ALLOCATION_ENV, DEFAULT_MAX_ALLOCATION_BYTES),
        (MAX_TOTAL_ALLOCATION_ENV, DEFAULT_MAX_TOTAL_ALLOCATION_BYTES),
    ):
        value = env.get(key, missing)
        if value is missing:
            values.append(default)
        elif type(value) is str and _DECIMAL.fullmatch(value):
            values.append(int(value))
        else:
            raise CapacityError("INVALID_CAPACITY_LIMIT")
    return validate_capacity(*values)


ENV_COUNT = "ENGRAM_SHARD_COUNT"
EXTERNAL_GRAPH_MODE_ENV = "RTP_ENGRAM_GRAPH_MODE"
MAX_SHARDS = 64  # manifest ABI limit, not an NVL rack size


class ShardConfigError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


class StorageMode(str, Enum):
    HOST = "host"
    HOST_SHARED = "host_shared"
    GPU_SHARED = "gpu_shared"


@dataclass(frozen=True)
class StoragePolicy:
    """One storage choice, independent of topology and physical placement."""

    mode: StorageMode

    @property
    def shared(self):
        return self.mode != StorageMode.HOST

    @property
    def memory_kind(self):
        return "gpu" if self.mode == StorageMode.GPU_SHARED else "host"

    @property
    def backend(self):
        return "fabric_gpu" if self.memory_kind == "gpu" else "fabric_egm"

    @property
    def default_graph_mode(self):
        return "process_lifetime" if self.shared else "eager"

    @classmethod
    def from_environ(cls, environ=None):
        env = os.environ if environ is None else environ
        if "RTP_ENGRAM_MEMORY" in env:
            raise ShardConfigError("REMOVED_ENGRAM_MEMORY_OPTION")
        if "RTP_ENGRAM_STORAGE" not in env and (
            ENV_COUNT in env or "RTP_ENGRAM_COHORT" in env
        ):
            raise ShardConfigError("ENGRAM_STORAGE_REQUIRED")
        try:
            return cls(StorageMode(env.get("RTP_ENGRAM_STORAGE", "host")))
        except (ValueError, TypeError):
            raise ShardConfigError("INVALID_ENGRAM_STORAGE_MODE") from None


def validate_count(value):
    if type(value) is not int or not 2 <= value <= MAX_SHARDS:
        raise ShardConfigError("STORAGE_HOST_COUNT_MUST_BE_2_TO_64")
    return value


@dataclass(frozen=True)
class ShardConfig:
    shard_count: int

    def __post_init__(self):
        validate_count(self.shard_count)

    @classmethod
    def from_environ(cls, environ=None, *, inferred_count=None):
        env = os.environ if environ is None else environ
        if any(
            key in env
            for key in (
                "ENGRAM_SHARD_INDEX",
                "ENGRAM_SHARD_MODE",
                "ENGRAM_SHARD_ENDPOINTS",
            )
        ):
            raise ShardConfigError("REMOVED_SHARD_DEPLOYMENT_OPTIONS")
        raw = env.get(ENV_COUNT)
        if raw is None:
            return cls(inferred_count)
        if type(raw) is not str or not raw.isascii() or not raw.isdecimal():
            raise ShardConfigError("INVALID_STORAGE_HOST_COUNT")
        return cls(int(raw))

    def validate_manifest(self, manifest):
        expected_hosts = None
        for tensor in manifest.tensors:
            groups = host_groups(tensor)
            if len(groups) != self.shard_count:
                raise ShardConfigError("STORAGE_HOST_COUNT_MISMATCH")
            hosts = tuple(g[0].host_id for g in groups)
            if len(set(hosts)) != self.shard_count:
                raise ShardConfigError("STORAGE_HOSTS_MUST_DIFFER")
            if any(len(group) > MAX_LOCAL_SEGMENTS for group in groups):
                raise ShardConfigError("INVALID_NUMA_SEGMENTS")
            if expected_hosts is None:
                expected_hosts = hosts
            elif hosts != expected_hosts:
                raise ShardConfigError("INCONSISTENT_STORAGE_HOSTS")
        if expected_hosts is None:
            raise ShardConfigError("EMPTY_STORAGE_GROUP")
        return self


@dataclass(frozen=True)
class EngramOptions:
    """Resolved worker options passed from startup to model attachment."""

    policy: StoragePolicy
    shards: ShardConfig
    graph_mode: str


def storage_host_count(manifest):
    return len(host_groups(manifest.tensors[0]))


ANNOTATION_ENV = "GANG_ANNOCATION_PATH"


DEFAULT_ANNOTATION = "/etc/podinfo/annotations"


ANNOTATION_KEY = "app.c2.io/biz-detail-ganginfo"


class GangInfoError(ValueError):
    """The annotation is absent, ambiguous or malformed."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def part_number(name):
    """Trailing ``_part<N>`` index, or None when the name has no such suffix."""
    head, separator, tail = name.rpartition("_part")
    if head and separator and tail.isdigit():
        return int(tail)
    return None


def family(name):
    """Gang name shared by every part of one deployment, or None."""
    number = part_number(name)
    return None if number is None else name[: name.rindex("_part")]


def read_parts(path=DEFAULT_ANNOTATION):
    """Return ``[(name, ip, port_or_None)]`` ordered by part number.

    Raises GangInfoError with a stable code instead of leaking OSError/JSON
    errors, so callers can decide whether a missing file is a misconfiguration
    or simply an environment without platform annotations.
    """
    try:
        content = Path(path).read_text()
    except OSError:
        raise GangInfoError("GANGINFO_UNREADABLE") from None
    lines = [line for line in content.splitlines() if ANNOTATION_KEY in line]
    if len(lines) != 1:
        raise GangInfoError("GANGINFO_NOT_UNIQUE")
    line = lines[0].rstrip().replace("\\", "")
    try:
        payload = json.loads(line[line.index("=") + 2 : -1])
    except (ValueError, IndexError):
        raise GangInfoError("GANGINFO_INVALID_JSON") from None
    if type(payload) is not dict or not payload:
        raise GangInfoError("GANGINFO_INVALID_JSON")
    parts = []
    for name, info in payload.items():
        if type(info) is not dict or type(info.get("ip")) is not str or not info["ip"]:
            raise GangInfoError("GANGINFO_INVALID_JSON")
        port = info.get("port")
        if port in (None, "", 0):
            port = None
        elif type(port) is int and 1 <= port <= 65535:
            pass
        elif type(port) is str and port.isdigit() and 1 <= int(port) <= 65535:
            port = int(port)
        else:
            raise GangInfoError("GANGINFO_INVALID_PORT")
        parts.append((name, info["ip"], port))
    if len({ip for _, ip, _ in parts}) != len(parts):
        raise GangInfoError("GANGINFO_DUPLICATE_ADDRESS")
    if all(part_number(name) is not None for name, _, _ in parts):
        parts.sort(key=lambda item: part_number(item[0]))
    return parts


COHORT_ENV = "RTP_ENGRAM_COHORT"


INDEX_ENV = "RTP_ENGRAM_INSTANCE_INDEX"


REVISION_ENV = "RTP_ENGRAM_MODEL_REVISION"


TIMEOUT_ENV = "RTP_ENGRAM_COHORT_TIMEOUT"


CHECKPOINT_ENV = "RTP_ENGRAM_CHECKPOINT"


COUNT_ENV = "ENGRAM_SHARD_COUNT"


RANKS_ENV = "RTP_ENGRAM_RANKS_PER_INSTANCE"


PORT_ENV = "RTP_ENGRAM_COHORT_PORT"


GANGINFO_ENV = "RTP_ENGRAM_GANGINFO_DIR"


DEFAULT_COHORT_PORT = 29483


DEFAULT_TIMEOUT = 1800


def _parse_members(raw, default_port):
    """Turn one "host[,host...]" string into validated (host, port) members."""
    if default_port is not None and (
        not default_port.isdigit() or not 1 <= int(default_port) <= 65535
    ):
        raise ShardConfigError("INVALID_COHORT")
    members = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        # Only the first address listens. Its explicit port overrides the
        # shared port; non-coordinator ports never affect the group identity.
        host, _, port = item.rpartition(":")
        if not host:
            host, port = item, ""
        if not port:
            port = default_port
        if not port:
            port = str(DEFAULT_COHORT_PORT)
        if not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ShardConfigError("INVALID_COHORT")
        members.append((host, int(port)))
    if not members:
        raise ShardConfigError("INVALID_COHORT")
    if len({host for host, _ in members}) != len(members):
        raise ShardConfigError("DISTINCT_STORAGE_HOST_ADDRESSES_REQUIRED")
    return members


def _cohort(env):
    raw = env.get(COHORT_ENV)
    if raw is None:
        return None
    return _parse_members(raw, env.get(PORT_ENV))


def _ganginfo_entries(root):
    """Member list from the directory the platform writes the topology into."""
    for name in ("cohort", "hosts", "members"):
        path = root / name
        if path.is_file():
            lines = [
                line.split("#", 1)[0].strip() for line in path.read_text().splitlines()
            ]
            return ",".join(line for line in lines if line) or None
    # Fall back to one file per part, ordered by the number in the name.
    numbered = []
    for path in root.iterdir():
        if not path.is_file():
            continue
        for prefix in ("rank-", "rank", "part-", "part", "node-", "node"):
            suffix = path.name[len(prefix) :]
            if path.name.startswith(prefix) and suffix.isdigit():
                numbered.append((int(suffix), path))
                break
    items = []
    for _, path in sorted(numbered):
        content = path.read_text().splitlines()
        if content and content[0].strip():
            items.append(content[0].strip())
    return ",".join(items) or None


def _ganginfo_cohort(env):
    directory = env.get(GANGINFO_ENV)
    if directory is None:
        return None
    root = Path(directory)
    if not root.is_dir():
        raise ShardConfigError("INVALID_GANGINFO")
    try:
        raw = _ganginfo_entries(root)
    except OSError as error:
        # A directory the deployment points at but cannot read is a
        # misconfiguration; silently serving with Engram off would hide it.
        raise ShardConfigError("INVALID_GANGINFO") from error
    if raw is None:
        raise ShardConfigError("EMPTY_GANGINFO")
    return _parse_members(raw, env.get(PORT_ENV))


def _annotation_cohort(env, parts):
    """Storage hosts from the platform's gang annotation, when unambiguous.

    Only a one-host instance can be read from addresses alone: every listed pod
    is then one storage host.  A multi-host instance keeps the existing
    derivation (the gang itself is the cohort), because the annotation does not
    say how several multi-host instances group, and guessing would silently
    change an existing per-instance layout.

    TODO(engram): multi-host instances sharing one storage cohort are out of
    scope for now.  Supporting them needs the platform to publish the
    pod-to-instance grouping (or an explicit index), not just a flat address
    list; until then such a gang keeps one shard per host of its own instance.
    """
    path = env.get(ANNOTATION_ENV)
    try:
        entries = read_parts(path or DEFAULT_ANNOTATION)
    except GangInfoError:
        # Deployments without platform annotations (manual containers) fall
        # through to the explicit inputs; a path the operator set is a
        # misconfiguration and must not be ignored.
        if path is None:
            return None
        raise ShardConfigError("INVALID_GANGINFO") from None
    if parts is None or len(parts) != 1:
        return None
    local = [entry for entry in entries if entry[1] == parts[0]]
    if len(local) != 1:
        raise ShardConfigError("LOCAL_HOST_NOT_IN_GANGINFO")
    name = family(local[0][0])
    listed = entries if name is None else [e for e in entries if family(e[0]) == name]
    default_port = env.get(PORT_ENV)
    if default_port is not None and (
        not default_port.isdigit() or not 1 <= int(default_port) <= 65535
    ):
        raise ShardConfigError("INVALID_COHORT")
    port = int(default_port) if default_port else DEFAULT_COHORT_PORT
    return [(host, explicit or port) for _, host, explicit in listed]


def _gang_parts(gang_hosts, world_size):
    """Use the completed DistributedServer topology, never rediscover NICs."""
    if gang_hosts is None:
        return None, None
    if len(gang_hosts) != world_size or any(
        type(h) is not str or not h for h in gang_hosts
    ):
        raise ShardConfigError("INVALID_GANG_TOPOLOGY")
    blocks = []
    for host in gang_hosts:
        if blocks and blocks[-1][0] == host:
            blocks[-1][1] += 1
        else:
            blocks.append([host, 1])
    if len({host for host, _ in blocks}) != len(blocks):
        raise ShardConfigError("NONCONTIGUOUS_GANG_HOST")
    if len({size for _, size in blocks}) != 1:
        raise ShardConfigError("UNEVEN_PART_SIZE")
    return [host for host, _ in blocks], blocks[0][1]


def _local_addresses():
    """IPv4 addresses this process can be reached on, for index discovery."""
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except OSError:
        pass
    return found


def _checkpoint_revision(checkpoint):
    """Checkpoint metadata namespace, NOT a weight-content checksum.

    Equal indices do not prove equal tensor bytes. Publication hashes and GPU
    sample validation remain separate; deployments must use one immutable model.
    """
    root = Path(checkpoint)
    for name in ("model.safetensors.index.json", "config.json"):
        path = root / name
        if path.is_file():
            return "ckpt@" + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    raise ShardConfigError("MODEL_REVISION_REQUIRED")


def _instance_index(env, cohort, local_host=None):
    raw = env.get(INDEX_ENV)
    if raw is not None:
        if not raw.isdigit() or not 0 <= int(raw) < len(cohort):
            raise ShardConfigError("INVALID_INSTANCE_INDEX")
        if local_host is not None and cohort[int(raw)][0] != local_host:
            raise ShardConfigError("INSTANCE_INDEX_HOST_MISMATCH")
        return int(raw)
    # The platform only has to hand over the address list: a host that appears
    # in it exactly once knows which shard it is.
    local = {local_host} if local_host is not None else _local_addresses()
    matches = [i for i, (host, _) in enumerate(cohort) if host in local]
    if not matches:
        raise ShardConfigError("INSTANCE_INDEX_REQUIRED")
    if len(matches) > 1:
        raise ShardConfigError("AMBIGUOUS_LOCAL_ADDRESS")
    return matches[0]


def _placement(env, name, default):
    raw = env.get(name)
    if raw is None:
        return default
    if not raw.isdigit() or int(raw) > 4096:
        raise ShardConfigError("INVALID_LOCAL_PLACEMENT")
    return int(raw)


def validate_group(group):
    """Public immutable startup contract; no credentials or local filesystem paths."""
    if type(group) is not dict or set(group) != {
        "generation",
        "model_revision",
        "members",
    }:
        raise ShardConfigError("INVALID_GROUP")
    try:
        if str(UUID(group["generation"])) != group["generation"]:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise ShardConfigError("INVALID_GENERATION") from None
    if (
        type(group["model_revision"]) is not str
        or not 0 < len(group["model_revision"]) <= 256
    ):
        raise ShardConfigError("INVALID_REVISION")
    members = group["members"]
    if type(members) is not list or not 1 <= len(members) <= 64:
        raise ShardConfigError("INVALID_MEMBERS")
    ids, hosts, indices = set(), set(), []
    for member in members:
        if type(member) is not dict or set(member) != {
            "instance_id",
            "world_size",
            "shards",
        }:
            raise ShardConfigError("INVALID_MEMBER")
        ident = member["instance_id"]
        if type(ident) is not str or not 0 < len(ident) <= 128 or ident in ids:
            raise ShardConfigError("INVALID_INSTANCE_ID")
        ids.add(ident)
        if (
            type(member["world_size"]) is not int
            or not 1 <= member["world_size"] <= 4096
        ):
            raise ShardConfigError("INVALID_WORLD_SIZE")
        if type(member["shards"]) is not list or not member["shards"]:
            raise ShardConfigError("EVERY_INSTANCE_MUST_HOLD_SHARDS")
        publishers = set()
        for shard in member["shards"]:
            if type(shard) is not dict or set(shard) != {
                "index",
                "host_id",
                "publisher_rank",
            }:
                raise ShardConfigError("INVALID_SHARD")
            if type(shard["index"]) is not int or shard["index"] < 0:
                raise ShardConfigError("INVALID_SHARD_INDEX")
            if (
                type(shard["host_id"]) is not str
                or not 0 < len(shard["host_id"]) <= 256
                or shard["host_id"] in hosts
            ):
                raise ShardConfigError("DUPLICATE_OR_INVALID_HOST")
            rank = shard["publisher_rank"]
            if (
                type(rank) is not int
                or not 0 <= rank < member["world_size"]
                or rank in publishers
            ):
                raise ShardConfigError("INVALID_PUBLISHER_RANK")
            hosts.add(shard["host_id"])
            publishers.add(rank)
            indices.append(shard["index"])
    validate_count(len(indices))
    if sorted(indices) != list(range(len(indices))):
        raise ShardConfigError("INCOMPLETE_SHARD_SET")


def build_group(
    env, world_size, *, rank=0, gang_hosts=None, checkpoint=None, policy=None
):
    """Derive membership from explicit sharing inputs and the completed gang.

    The address list may come from the environment, from the topology directory
    the platform writes, from the gang annotation, or from the gang itself when
    one gang spans the hosts.
    """
    # Ordinary deployments must not opt in merely because /etc/ganginfo exists.
    if policy is None:
        policy = StoragePolicy.from_environ(env)
    if not policy.shared:
        return None
    if type(world_size) is not int or world_size <= 0 or not 0 <= rank < world_size:
        raise ShardConfigError("INVALID_WORLD_SIZE")
    parts, discovered = _gang_parts(gang_hosts, world_size)
    source = "cohort"
    cohort = _cohort(env)
    if cohort is None:
        source = "ganginfo"
        cohort = _ganginfo_cohort(env)
    if cohort is None:
        source = "annotation"
        cohort = _annotation_cohort(env, parts)
    if cohort is None:
        source = "gang"
        if parts is None:
            raise ShardConfigError("GANG_TOPOLOGY_REQUIRED")
        if len(parts) < 2:
            raise ShardConfigError("COHORT_REQUIRED")
        cohort = _parse_members(",".join(parts), env.get(PORT_ENV))
    if parts is not None and not set(parts) <= {host for host, _ in cohort}:
        raise ShardConfigError("GANG_HOST_NOT_IN_COHORT")
    instances = len(cohort)
    # One shard per storage host: the cohort list is the topology, so the count
    # is only needed as a cross-check when a launcher still sets it.
    raw_count = env.get(COUNT_ENV)
    if raw_count is None:
        p = validate_count(instances)
    elif raw_count.isdigit():
        p = validate_count(int(raw_count))
        if p != instances:
            raise ShardConfigError("STORAGE_HOST_COUNT_MUST_MATCH_COHORT")
    else:
        raise ShardConfigError("STORAGE_HOST_COUNT_REQUIRED")
    # The launcher resolves an oss:// checkpoint to its local fuse mount before
    # the engine starts; prefer that over the raw URI, which is not readable here.
    checkpoint = env.get(CHECKPOINT_ENV) or checkpoint or env.get("CHECKPOINT_PATH")
    if not checkpoint:
        raise ShardConfigError("CHECKPOINT_REQUIRED")
    revision = env.get(REVISION_ENV) or _checkpoint_revision(checkpoint)
    index = _instance_index(
        env, cohort, gang_hosts[rank] if gang_hosts is not None else None
    )
    raw_ranks = env.get(RANKS_ENV)
    if raw_ranks is not None and (not raw_ranks.isdigit() or int(raw_ranks) <= 0):
        raise ShardConfigError("INVALID_RANKS_PER_INSTANCE")
    # Internal topology namespace, NOT a deployment input or restart token.
    # Store attempt counters below isolate repeated/partial startup attempts.
    generation = str(
        uuid5(
            NAMESPACE_URL,
            "engram|"
            + revision
            + "|"
            + ",".join(host for host, _ in cohort)
            + "|"
            + str(cohort[0][1])
            + "|"
            + str(p),
        )
    )
    if (
        raw_ranks is not None
        and discovered is not None
        and int(raw_ranks) != discovered
    ):
        raise ShardConfigError("RANKS_PER_INSTANCE_MISMATCH")
    if raw_ranks is not None:
        local_world_size = int(raw_ranks)
    else:
        local_world_size = world_size if discovered is None else discovered
    if world_size % local_world_size or instances % (world_size // local_world_size):
        raise ShardConfigError("WORLD_SIZE_NOT_DIVISIBLE_BY_INSTANCES")
    members = [
        dict(
            instance_id=f"i{i}",
            world_size=local_world_size,
            shards=[dict(index=i, host_id=host, publisher_rank=0)],
        )
        for i, (host, _) in enumerate(cohort)
    ]
    group = dict(generation=generation, model_revision=revision, members=members)
    validate_group(group)
    group_digest = hashlib.sha256(
        json.dumps(group, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    local = members[index]
    raw_timeout = env.get(TIMEOUT_ENV)
    timeout = DEFAULT_TIMEOUT if raw_timeout is None else float(raw_timeout)
    if not 0 < timeout <= 86400:
        raise ShardConfigError("INVALID_TIMEOUT")
    # Reuse DistributedServer.store when its world covers every storage host.
    reuse_world_store = parts is not None and len(parts) == instances
    # One line that answers "which topology did this pod pick, and why".
    logging.getLogger(__name__).info(
        "engram topology source=%s instances=%d p=%d index=%d coordinator=%s:%d "
        "local=%s reuse_world_store=%s",
        source,
        instances,
        p,
        index,
        cohort[0][0],
        cohort[0][1],
        parts[0] if parts else "-",
        reuse_world_store,
    )
    return dict(
        group=group,
        group_digest=group_digest,
        instance_id=local["instance_id"],
        index=index,
        checkpoint=checkpoint,
        coordinator=dict(host=cohort[0][0], port=cohort[0][1]),
        timeout_seconds=timeout,
        rank_base=(
            (rank // local_world_size) * local_world_size
            if parts is not None
            else (index % (world_size // local_world_size)) * local_world_size
        ),
        reuse_world_store=reuse_world_store,
    )


def build_config(
    env, world_size, *, rank=0, gang_hosts=None, checkpoint=None, local_device=0
):
    """Resolve allocation policy; the TCPStore uses the existing trusted network."""
    policy = StoragePolicy.from_environ(env)
    config = build_group(
        env,
        world_size,
        rank=rank,
        gang_hosts=gang_hosts,
        checkpoint=checkpoint,
        policy=policy,
    )
    if config is None:
        return None
    if "RTP_ENGRAM_LOCAL_DESCRIPTORS" in env:
        raise ShardConfigError("REMOVED_LOCAL_DESCRIPTOR_MODE")
    if env.get("RTP_ENGRAM_TLS_DIR"):
        raise ShardConfigError("REMOVED_ENGRAM_TLS_CONFIG")
    config["local_device"] = _placement(env, "RTP_ENGRAM_LOCAL_DEVICE", local_device)
    config["local_numa"] = parse_numa(env.get("RTP_ENGRAM_LOCAL_NUMA"))
    shard_count = sum(len(member["shards"]) for member in config["group"]["members"])
    config["options"] = EngramOptions(
        policy,
        ShardConfig.from_environ(env, inferred_count=shard_count),
        env.get(EXTERNAL_GRAPH_MODE_ENV, policy.default_graph_mode),
    )
    config["gpu_devices"] = tuple(
        range(config["group"]["members"][config["index"]]["world_size"])
    )
    reserve = (
        env.get("RTP_ENGRAM_GPU_RESERVE_BYTES", str(32 << 30))
        if policy.memory_kind == "gpu"
        else str(32 << 30)
    )
    if (
        type(reserve) is not str
        or not reserve.isascii()
        or not reserve.isdecimal()
        or not 0 < int(reserve) < 2**63
    ):
        raise ShardConfigError("INVALID_ENGRAM_GPU_RESERVE")
    config["gpu_reserve_bytes"] = int(reserve)
    config["release_page_cache"] = env.get("RTP_ENGRAM_RELEASE_INPUT_PAGE_CACHE") == "1"
    config["max_allocation_bytes"], config["budget_bytes"] = capacity_from_environ(env)
    return config
